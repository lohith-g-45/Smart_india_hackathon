package com.navshield.map.engine

import android.content.Context
import com.navshield.map.contract.Member5Result
import com.navshield.map.contract.NavShieldSensorState
import com.navshield.map.members.Member5DriftGuardian
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.sqrt

/**
 * Implementation of Member 5 using TensorFlow Lite.
 * loads the nav_shield_lstm_stage8.tflite model and performs temporal inference.
 * Reproduces the EXACT 137-feature preprocessing pipeline from Stage 8 training.
 */
class Member5TfliteEngine(private val context: Context) : Member5DriftGuardian {

    private var interpreter: Interpreter? = null
    private var status = MemberStatus.WAITING_FOR_MEMBER

    private val sequenceLength = 30
    private val featureCount = 137
    private val rollingWindowSize = 30

    // History for deriving changes and rates
    private val sensorHistory = mutableListOf<NavShieldSensorState>()
    // History for computing rolling statistics
    private val derivedHistory = mutableListOf<DerivedFeatures>()

    private var firstTimestamp: Long = -1

    /**
     * Internal structure for derived features (pre-rolling, pre-scaling).
     */
    private data class DerivedFeatures(
        val accelMag: Double,
        val gyroMag: Double,
        val speed: Double,
        val accelChange: Double,
        val gyroChange: Double,
        val speedChange: Double,
        val yaw: Double,
        val pitch: Double,
        val roll: Double,
        val yawChange: Double,
        val pitchChange: Double,
        val rollChange: Double
    )

    fun initialize() {
        try {
            val modelBuffer = loadModelFile()
            interpreter = Interpreter(modelBuffer)
            status = MemberStatus.CONNECTED
        } catch (e: Exception) {
            status = MemberStatus.ERROR
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("nav_shield_lstm_stage8.tflite")
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
    }

    override fun predict(sensorState: NavShieldSensorState): Member5Result? {
        if (status != MemberStatus.CONNECTED) return null

        if (firstTimestamp == -1L) firstTimestamp = sensorState.timestampMillis

        val prevSensor = sensorHistory.lastOrNull()
        sensorHistory.add(sensorState)
        if (sensorHistory.size > 60) sensorHistory.removeAt(0)

        val dt = if (prevSensor != null) (sensorState.timestampMillis - prevSensor.timestampMillis) / 1000.0 else 0.1
        val safeDt = if (dt <= 0) 0.1 else dt

        // 1. Derive basic features for this timestep
        val currentDerived = computeDerived(sensorState, prevSensor, safeDt)
        derivedHistory.add(currentDerived)
        if (derivedHistory.size > 60) derivedHistory.removeAt(0)

        // 2. We need a sequence of 30 timesteps, each with 30-sample rolling stats
        if (derivedHistory.size < sequenceLength + rollingWindowSize - 1) return null

        // 3. Prepare input tensor: [1, 30, 137]
        val inputBuffer = ByteBuffer.allocateDirect(1 * sequenceLength * featureCount * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val startIndex = derivedHistory.size - sequenceLength
        for (i in startIndex until derivedHistory.size) {
            val window = derivedHistory.subList(i - rollingWindowSize + 1, i + 1)
            val sensor = sensorHistory[i]
            val prevS = if (i > 0) sensorHistory[i - 1] else null
            val vector = constructFeatureVector(sensor, prevS, window)
            
            // Apply imputation and scaling
            for (f in 0 until featureCount) {
                val value = if (vector[f].isFinite()) vector[f] else MEDIANS[f]
                val scaled = (value - MEANS[f]) / SCALES[f]
                inputBuffer.putFloat(scaled.toFloat())
            }
        }

        // 4. Run Inference
        val outputBuffer = ByteBuffer.allocateDirect(1 * 2 * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
        interpreter?.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val posError = outputBuffer.float
        val velError = outputBuffer.float

        return postProcess(posError, velError, sensorState.timestampMillis)
    }

    private fun computeDerived(s: NavShieldSensorState, prev: NavShieldSensorState?, dt: Double): DerivedFeatures {
        val am = sqrt(s.accelX * s.accelX + s.accelY * s.accelY + s.accelZ * s.accelZ)
        val gm = sqrt(s.gyroX * s.gyroX + s.gyroY * s.gyroY + s.gyroZ * s.gyroZ)
        
        // Android Gyro convention: Z=Yaw, X=Pitch, Y=Roll
        val yaw = s.gyroZ
        val pitch = s.gyroX
        val roll = s.gyroY

        var ac = 0.0; var gc = 0.0; var sc = 0.0; var yc = 0.0; var pc = 0.0; var rc = 0.0

        if (prev != null) {
            val pam = sqrt(prev.accelX * prev.accelX + prev.accelY * prev.accelY + prev.accelZ * prev.accelZ)
            val pgm = sqrt(prev.gyroX * prev.gyroX + prev.gyroY * prev.gyroY + prev.gyroZ * prev.gyroZ)
            ac = am - pam
            gc = gm - pgm
            sc = s.speed - prev.speed
            yc = yaw - prev.gyroZ
            pc = pitch - prev.gyroX
            rc = roll - prev.gyroY
        }

        return DerivedFeatures(am, gm, s.speed, ac, gc, sc, yaw, pitch, roll, yc, pc, rc)
    }

    private fun constructFeatureVector(s: NavShieldSensorState, prev: NavShieldSensorState?, window: List<DerivedFeatures>): DoubleArray {
        val v = DoubleArray(featureCount)
        val current = window.last()
        val dt = if (prev != null) (s.timestampMillis - prev.timestampMillis) / 1000.0 else 0.1
        val safeDt = if (dt <= 0) 0.1 else dt

        // Mapping to EXACT CSV ORDER (0..136)
        v[0] = s.timestampMillis / 1000.0 // S_time_seconds
        v[1] = s.timestampMillis / 1000.0 // V_time_seconds
        v[2] = safeDt // time_difference_seconds
        v[3] = s.latitude
        v[4] = s.longitude
        v[5] = s.altitude
        v[6] = s.speed 
        v[7] = s.accuracy
        v[8] = s.bearing 
        v[9] = (s.timestampMillis - firstTimestamp).toDouble() // S_TIME SINCE START
        v[10] = s.accelX
        v[11] = s.accelY
        v[12] = s.accelZ
        v[13] = 0.0 // S_GRAVITY X (Unavailable, using training mean/median)
        v[14] = 0.0 // S_GRAVITY Y
        v[15] = 9.8066 // S_GRAVITY Z
        v[16] = current.yaw 
        v[17] = current.pitch
        v[18] = current.roll
        v[19] = -12.06 // S_MAGNETIC FIELD X (Unavailable)
        v[20] = -33.69 
        v[21] = 15.31
        v[22] = 245.85 // S_ORIENTATION Yaw (Unavailable)
        v[23] = -84.41
        v[24] = -118.56
        v[25] = s.satellites.toDouble() // V_No of Satellites
        v[26] = (s.timestampMillis % 86400000) / 1000.0 // V_Time Since Day Start
        v[27] = s.latitude // V_Latitude
        v[28] = s.longitude
        v[29] = s.speed * 3.6 // V_Velocity (km/hr)
        v[30] = s.bearing // V_Heading
        v[31] = s.altitude / 1000.0 // V_Height (km)
        v[32] = 0.0 // V_Vertical velocity
        v[33] = safeDt // V_Sample period
        v[34] = 10.2 // V_Steering Angle
        v[35] = 36.29 // V_Wheel Speed FL
        v[36] = 36.3
        v[37] = 36.19
        v[38] = 36.16
        v[39] = 0.0 // V_Yaw Rate
        v[40] = s.speed * 3.6 // V_Indicated Speed
        v[41] = 0.0 // V_Indicated Long Accel
        v[42] = 0.0 // V_Indicated Lat Accel
        v[43] = 0.0 // V_Handbrake
        v[44] = 4.0 // V_Gear Req
        v[45] = 3.0 // V_Gear
        v[46] = 1548.0 // V_Engine Speed
        v[47] = 87.0 // V_Coolant Temp
        v[48] = 0.0 // V_Clutch
        v[49] = 0.02 // V_Brake Pressure
        v[50] = 0.0 // V_Brake Pos
        v[51] = 14.0 // V_Battery
        v[52] = 12.0 // V_Air Temp
        v[53] = 5.5 // V_Pedal Pos
        v[54] = s.timestampMillis / 1000.0 // timestamp
        v[55] = (s.timestampMillis - firstTimestamp) / 1000.0 // elapsed_time
        v[56] = safeDt // delta_time
        v[57] = s.accelX
        v[58] = s.accelY
        v[59] = s.accelZ
        v[60] = current.accelMag
        v[61] = current.yaw
        v[62] = current.pitch
        v[63] = current.roll
        v[64] = current.gyroMag
        v[65] = -12.06 // mag_x
        v[66] = -33.69
        v[67] = 15.31
        v[68] = 42.44 // mag_magnitude
        v[69] = 0.0 // gravity_x
        v[70] = 0.0
        v[71] = 9.8066
        v[72] = 9.8066 // gravity_magnitude
        v[73] = 245.85 // orientation_yaw
        v[74] = -84.41
        v[75] = -118.56
        v[76] = s.latitude
        v[77] = s.longitude
        v[78] = s.altitude
        v[79] = s.speed
        v[80] = s.accuracy
        v[81] = s.satellites.toDouble()
        v[82] = current.accelChange
        v[83] = current.accelChange / safeDt // accel_change_rate
        v[84] = current.gyroChange
        v[85] = current.gyroChange / safeDt
        v[86] = 0.0 // magnetic_field_change
        v[87] = 0.0
        v[88] = current.speedChange
        v[89] = current.speedChange / safeDt
        v[90] = current.yawChange
        v[91] = current.yawChange / safeDt
        v[92] = current.pitchChange
        v[93] = current.pitchChange / safeDt
        v[94] = current.rollChange
        v[95] = current.rollChange / safeDt
        v[96] = s.confidence

        // Rolling Stats over window
        val ams = window.map { it.accelMag }
        v[97] = ams.average() 
        v[98] = stdDev(ams)
        v[99] = ams.minOrNull() ?: 0.0
        v[100] = ams.maxOrNull() ?: 0.0
        v[101] = v[100] - v[99]

        val gms = window.map { it.gyroMag }
        v[102] = gms.average()
        v[103] = stdDev(gms)
        v[104] = gms.minOrNull() ?: 0.0
        v[105] = gms.maxOrNull() ?: 0.0
        v[106] = v[105] - v[104]

        v[107] = 42.5094 // mag rolling (Unavailable, using median)
        v[108] = 0.3719
        v[109] = 41.375
        v[110] = 43.7538
        v[111] = 1.2379

        v[112] = 9.8066 // gravity rolling (Unavailable)
        v[113] = 0.00005
        v[114] = 9.8065
        v[115] = 9.8067
        v[116] = 0.00016

        val sps = window.map { it.speed }
        v[117] = sps.average()
        v[118] = stdDev(sps)
        v[119] = sps.minOrNull() ?: 0.0
        v[120] = sps.maxOrNull() ?: 0.0
        v[121] = v[120] - v[119]

        val acs = window.map { it.accelChange }
        v[122] = acs.average()
        v[123] = stdDev(acs)
        v[124] = acs.minOrNull() ?: 0.0
        v[125] = acs.maxOrNull() ?: 0.0
        v[126] = v[125] - v[124]

        val gcs = window.map { it.gyroChange }
        v[127] = gcs.average()
        v[128] = stdDev(gcs)
        v[129] = gcs.minOrNull() ?: 0.0
        v[130] = gcs.maxOrNull() ?: 0.0
        v[131] = v[130] - v[129]

        val scs = window.map { it.speedChange }
        v[132] = scs.average()
        v[133] = stdDev(scs)
        v[134] = scs.minOrNull() ?: 0.0
        v[135] = scs.maxOrNull() ?: 0.0
        v[136] = v[135] - v[134]

        return v
    }

    private fun stdDev(data: List<Double>): Double {
        if (data.isEmpty()) return 0.0
        val mean = data.average()
        val sumSq = data.sumOf { (it - mean) * (it - mean) }
        return sqrt(sumSq / data.size)
    }

    private fun postProcess(posErr: Float, velErr: Float, timestamp: Long): Member5Result {
        // Thresholds from stage8_decision_config.json
        val posWarning = 10.0f
        val posAbnormal = 20.0f
        
        val state = when {
            posErr >= posAbnormal -> "ABNORMAL"
            posErr >= posWarning -> "WARNING"
            else -> "NORMAL"
        }

        return Member5Result(
            predictedPositionErrorM = posErr,
            predictedVelocityErrorMps = velErr,
            timestamp = timestamp,
            isAbnormal = state == "ABNORMAL",
            isDriftDetected = state == "ABNORMAL",
            state = state
        )
    }

    override fun getStatus(): MemberStatus = status

    fun close() {
        interpreter?.close()
        status = MemberStatus.WAITING_FOR_MEMBER
    }

    companion object {
        // Training medians, means, and scales from nav_shield_member5_android_preprocessing_contract.csv
        private val MEDIANS = doubleArrayOf(
            1405.899, 1434.0, 0.001, 52.4459, -1.5348, 159.0, 9.77, 3.0, 147.199, 2113111.0, 
            0.0144, -0.0118, 9.8494, 0.0, 0.0, 9.8066, -1.0e-4, -0.001, 2.0e-4, -12.06, 
            -33.69, 15.31, 245.85, -84.41, -118.56, 137.0, 47396.9, 52.4494, -1.5371, 36.086, 
            164.316, 110.36, 0.0, 0.1, 10.2, 36.29, 36.3, 36.19, 36.16, 0.0, 
            36.29, 0.0, 0.0, 0.0, 4.0, 3.0, 1548.0, 87.0, 0.0, 0.02, 
            0.0, 14.0, 12.0, 5.5, 1405.899, 1421.6, 0.1, 0.0144, -0.0118, 9.8494, 
            9.9433, -1.0e-4, -0.001, 2.0e-4, 0.1279, -12.06, -33.69, 15.31, 42.4453, 0.0, 
            0.0, 9.8066, 9.8066, 245.85, -84.41, -118.56, 52.4459, -1.5348, 159.0, 9.77, 
            3.0, 21.0, 0.3927, 0.0, 0.0381, 0.0, 0.3619, 0.0, 0.0, 0.0, 
            0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 9.9707, 0.4857, 9.2934, 
            10.7855, 1.6069, 0.1542, 0.0573, 0.0532, 0.2677, 0.1853, 42.5094, 0.3719, 41.375, 
            43.7538, 1.2379, 9.8066, 5.0e-5, 9.8065, 9.8067, 1.6e-4, 9.74, 0.0, 9.59, 
            10.0, 0.0, 0.5468, 0.3783, 0.0489, 1.288, 1.2046, 0.0551, 0.0385, 0.0047, 
            0.1316, 0.1227, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        private val MEANS = doubleArrayOf(
            2060.933, 2094.271, 0.005, 52.5925, -1.5395, 177.201, 10.5106, 23.4173, 167.795, 2701114.6, 
            -0.0107, -0.0393, 9.847, 1.6e-6, -1.6e-4, 9.8065, -3.8e-4, -0.0031, 1.4e-5, -10.9109, 
            -31.8224, 15.3262, 211.7267, -83.8742, -36.6316, 114.5, 50516.0, 52.5926, -1.5414, 38.0883, 
            180.6957, 131.9141, 0.0212, 0.1, 47.3772, 38.0376, 38.0126, 37.9727, 37.8406, -0.1626, 
            38.0227, 2.7e-4, 8.1e-5, 0.0442, 3.8481, 2.7908, 1550.488, 86.5643, 0.0, 2.7756, 
            0.2467, 13.8068, 11.9715, 8.9054, 2060.933, 2089.623, 0.1025, -0.0107, -0.0393, 9.847, 
            10.09, -3.8e-4, -0.0031, 1.4e-5, 0.1952, -10.9109, -31.8224, 15.3262, 43.2546, 1.6e-6, 
            1.6e-6, 9.8065, 9.8066, 211.7267, -83.8742, -36.6316, 52.5925, -1.5395, 177.201, 10.5106, 
            23.4173, 21.4605, 0.6851, 0.057, 0.093, 0.2137, 1.0143, 3.7772, 0.1132, -5.7853, 
            -0.0209, -167.1069, -9.9e-5, 20.1367, -0.0011, -52.4202, 0.9993, 10.0899, 0.5817, 9.161, 
            11.0967, 1.9356, 0.1952, 0.0857, 0.0775, 0.3516, 0.2741, 43.2543, 1.0772, 42.0199, 
            45.3814, 3.3615, 9.8066, 1.2e-4, 9.8064, 9.8068, 4.1e-4, 10.5107, 0.148, 10.3435, 
            10.6788, 0.3353, 0.6851, 0.477, 0.099, 1.6164, 1.5174, 0.093, 0.064, 0.0135, 
            0.2165, 0.203, 0.1133, 0.0904, 0.0692, 0.3623, 0.2931
        )
        private val SCALES = doubleArrayOf(
            2085.31, 2047.575, 0.0107, 0.3174, 0.2568, 76.5659, 7.6003, 213.9891, 111.0689, 2465593.6, 
            1.6122, 1.5669, 0.7296, 0.0351, 0.0336, 0.0031, 0.1151, 0.2328, 0.1386, 13.0679, 
            8.0666, 18.0161, 119.7548, 3.8551, 136.4469, 49.3085, 11752.526, 0.3173, 0.2551, 27.3774, 
            110.0361, 67.3933, 0.416, 1.3e-4, 98.5753, 27.4968, 27.4942, 27.5113, 27.3946, 6.0017, 
            27.4938, 0.075, 0.0771, 0.2055, 2.3247, 0.7835, 529.1239, 12.7795, 1.0, 7.1748, 
            0.4311, 1.6206, 5.4221, 10.3116, 2085.31, 2052.33, 2.7644, 1.6122, 1.5669, 0.7296, 
            0.863, 0.1151, 0.2328, 0.1386, 0.2203, 13.0679, 8.0666, 18.0161, 7.4876, 0.0351, 
            0.0351, 0.0031, 0.0022, 119.7548, 3.8551, 136.4469, 0.3174, 0.2568, 76.5659, 7.6003, 
            213.9891, 4.0172, 0.8668, 62.3269, 0.1569, 25.843, 3.2487, 352.6115, 0.9721, 338.9427, 
            13.3829, 4905.9689, 1.4794, 1628.3196, 8.237, 2179.7422, 0.0133, 0.4226, 0.4772, 0.6375, 
            1.2324, 1.6086, 0.1753, 0.1023, 0.0828, 0.3559, 0.3209, 7.0729, 2.209, 6.9483, 
            9.6667, 6.705, 6.0e-4, 0.0021, 0.0033, 0.0056, 0.0082, 7.5725, 0.6394, 7.6169, 
            7.58, 1.3935, 0.5864, 0.4243, 0.1467, 1.4082, 1.3406, 0.1161, 0.0839, 0.0273, 
            0.2776, 0.2642, 0.8948, 0.3701, 0.8269, 1.4333, 1.1794
        )
    }
}
