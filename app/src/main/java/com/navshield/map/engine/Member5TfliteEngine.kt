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
import kotlin.math.sqrt

/**
 * Implementation of Member 5 using TensorFlow Lite.
 * Loads the nav_shield_lstm_android_107.tflite model and performs temporal inference.
 * Reproduces the EXACT 107-feature preprocessing pipeline from Stage 8 training.
 */
class Member5TfliteEngine(private val context: Context) : Member5DriftGuardian {

    private var interpreter: Interpreter? = null
    private var status = MemberStatus.WAITING_FOR_MEMBER

    private val sequenceLength = 30
    private val featureCount = 107
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
        val magMag: Double,
        val gravityMag: Double,
        val speed: Double,
        val accelChange: Double,
        val gyroChange: Double,
        val magChange: Double,
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
        val assetFileDescriptor = context.assets.openFd("nav_shield_lstm_android_107.tflite")
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

        // 3. Prepare input tensor: [1, 30, 107]
        val inputBuffer = ByteBuffer.allocateDirect(1 * sequenceLength * featureCount * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val startIndex = derivedHistory.size - sequenceLength
        for (i in startIndex until derivedHistory.size) {
            // Window for rolling stats ending at index i
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

        val rawPosError = outputBuffer.float
        val rawVelError = outputBuffer.float

        // Target Inverse Scaling
        val posError = rawPosError * TARGET_SCALES[0] + TARGET_MEANS[0]
        val velError = rawVelError * TARGET_SCALES[1] + TARGET_MEANS[1]

        return postProcess(posError.toFloat(), velError.toFloat(), sensorState.timestampMillis)
    }

    private fun computeDerived(s: NavShieldSensorState, prev: NavShieldSensorState?, dt: Double): DerivedFeatures {
        val am = sqrt(s.accelX * s.accelX + s.accelY * s.accelY + s.accelZ * s.accelZ)
        val gm = sqrt(s.gyroX * s.gyroX + s.gyroY * s.gyroY + s.gyroZ * s.gyroZ)
        val mm = if (s.magneticFieldX.isNaN()) 0.0 else sqrt(s.magneticFieldX * s.magneticFieldX + s.magneticFieldY * s.magneticFieldY + s.magneticFieldZ * s.magneticFieldZ)
        val gvm = if (s.gravityX.isNaN()) 9.8066 else sqrt(s.gravityX * s.gravityX + s.gravityY * s.gravityY + s.gravityZ * s.gravityZ)

        // Android Z is Yaw, X is Pitch, Y is Roll
        val yaw = s.gyroZ
        val pitch = s.gyroX
        val roll = s.gyroY

        var ac = 0.0; var gc = 0.0; var mc = 0.0; var sc = 0.0; var yc = 0.0; var pc = 0.0; var rc = 0.0

        if (prev != null) {
            val pam = sqrt(prev.accelX * prev.accelX + prev.accelY * prev.accelY + prev.accelZ * prev.accelZ)
            val pgm = sqrt(prev.gyroX * prev.gyroX + prev.gyroY * prev.gyroY + prev.gyroZ * prev.gyroZ)
            val pmm = if (prev.magneticFieldX.isNaN()) 0.0 else sqrt(prev.magneticFieldX * prev.magneticFieldX + prev.magneticFieldY * prev.magneticFieldY + prev.magneticFieldZ * prev.magneticFieldZ)
            
            ac = am - pam
            gc = gm - pgm
            mc = mm - pmm
            sc = s.speed - prev.speed
            yc = yaw - prev.gyroZ
            pc = pitch - prev.gyroX
            rc = roll - prev.gyroY
        }

        return DerivedFeatures(am, gm, mm, gvm, s.speed, ac, gc, mc, sc, yaw, pitch, roll, yc, pc, rc)
    }

    private fun constructFeatureVector(s: NavShieldSensorState, prev: NavShieldSensorState?, window: List<DerivedFeatures>): DoubleArray {
        val v = DoubleArray(featureCount)
        val current = window.last()
        val dt = if (prev != null) (s.timestampMillis - prev.timestampMillis) / 1000.0 else 0.1
        val safeDt = if (dt <= 0) 0.1 else dt

        // 1-24: Direct and immediate derived
        v[0] = s.timestampMillis / 1000.0 // 1 S_time_seconds
        v[1] = safeDt // 2 time_difference_seconds
        v[2] = s.latitude // 3
        v[3] = s.longitude // 4
        v[4] = s.altitude // 5
        v[5] = s.speed * 3.6 // 6 S_GPS SPEED (Kmh)
        v[6] = s.accuracy // 7
        v[7] = s.bearing // 8 S_GPS ORIENTATION
        v[8] = (s.timestampMillis - firstTimestamp).toDouble() // 9 S_TIME SINCE START
        v[9] = s.accelX // 10
        v[10] = s.accelY // 11
        v[11] = s.accelZ // 12
        v[12] = s.gravityX // 13
        v[13] = s.gravityY // 14
        v[14] = s.gravityZ // 15
        v[15] = current.yaw // 16
        v[16] = current.pitch // 17
        v[17] = current.roll // 18
        v[18] = s.magneticFieldX // 19
        v[19] = s.magneticFieldY // 20
        v[20] = s.magneticFieldZ // 21
        v[21] = s.orientationYaw // 22
        v[22] = s.orientationPitch // 23
        v[23] = s.orientationRoll // 24

        // 25-67: Temporal and calculated changes
        v[24] = s.timestampMillis / 1000.0 // 25 timestamp
        v[25] = (s.timestampMillis - firstTimestamp) / 1000.0 // 26 elapsed_time
        v[26] = safeDt // 27 delta_time
        v[27] = s.accelX // 28
        v[28] = s.accelY // 29
        v[29] = s.accelZ // 30
        v[30] = current.accelMag // 31
        v[31] = current.yaw // 32
        v[32] = current.pitch // 33
        v[33] = current.roll // 34
        v[34] = current.gyroMag // 35
        v[35] = s.magneticFieldX // 36
        v[36] = s.magneticFieldY // 37
        v[37] = s.magneticFieldZ // 38
        v[38] = current.magMag // 39
        v[39] = s.gravityX // 40
        v[40] = s.gravityY // 41
        v[41] = s.gravityZ // 42
        v[42] = current.gravityMag // 43
        v[43] = s.orientationYaw // 44
        v[44] = s.orientationPitch // 45
        v[45] = s.orientationRoll // 46
        v[46] = s.latitude // 47
        v[47] = s.longitude // 48
        v[48] = s.altitude // 49
        v[49] = s.speed // 50 gps_speed
        v[50] = s.accuracy // 51
        v[51] = s.satellites.toDouble() // 52
        v[52] = current.accelChange // 53
        v[53] = current.accelChange / safeDt // 54 accel_change_rate
        v[54] = current.gyroChange // 55
        v[55] = current.gyroChange / safeDt // 56 gyro_change_rate
        v[56] = current.magChange // 57
        v[57] = current.magChange / safeDt // 58 magnetic_field_change_rate
        v[58] = current.speedChange * 3.6 // 59 speed_change (Kmh)
        v[59] = (current.speedChange * 3.6) / safeDt // 60 speed_change_rate
        v[60] = current.yawChange // 61
        v[61] = current.yawChange / safeDt // 62
        v[62] = current.pitchChange // 63
        v[63] = current.pitchChange / safeDt // 64
        v[64] = current.rollChange // 65
        v[65] = current.rollChange / safeDt // 66
        v[66] = s.confidence // 67 sensor_quality_score

        // 68-107: Rolling Stats
        val ams = window.map { it.accelMag }
        v[67] = ams.average(); v[68] = stdDev(ams); v[69] = ams.minOrNull() ?: 0.0; v[70] = ams.maxOrNull() ?: 0.0; v[71] = v[70] - v[69]

        val gms = window.map { it.gyroMag }
        v[72] = gms.average(); v[73] = stdDev(gms); v[74] = gms.minOrNull() ?: 0.0; v[75] = gms.maxOrNull() ?: 0.0; v[76] = v[75] - v[74]

        val mms = window.map { it.magMag }
        v[77] = mms.average(); v[78] = stdDev(mms); v[79] = mms.minOrNull() ?: 0.0; v[80] = mms.maxOrNull() ?: 0.0; v[81] = v[80] - v[79]

        val gvms = window.map { it.gravityMag }
        v[82] = gvms.average(); v[83] = stdDev(gvms); v[84] = gvms.minOrNull() ?: 0.0; v[85] = gvms.maxOrNull() ?: 0.0; v[86] = v[85] - v[84]

        val sps = window.map { it.speed * 3.6 }
        v[87] = sps.average(); v[88] = stdDev(sps); v[89] = sps.minOrNull() ?: 0.0; v[90] = sps.maxOrNull() ?: 0.0; v[91] = v[90] - v[89]

        val acs = window.map { it.accelChange }
        v[92] = acs.average(); v[93] = stdDev(acs); v[94] = acs.minOrNull() ?: 0.0; v[95] = acs.maxOrNull() ?: 0.0; v[96] = v[95] - v[94]

        val gcs = window.map { it.gyroChange }
        v[97] = gcs.average(); v[98] = stdDev(gcs); v[99] = gcs.minOrNull() ?: 0.0; v[100] = gcs.maxOrNull() ?: 0.0; v[101] = v[100] - v[99]

        val scs = window.map { it.speedChange * 3.6 }
        v[102] = scs.average(); v[103] = stdDev(scs); v[104] = scs.minOrNull() ?: 0.0; v[105] = scs.maxOrNull() ?: 0.0; v[106] = v[105] - v[104]

        return v
    }

    private fun stdDev(data: List<Double>): Double {
        if (data.isEmpty()) return 0.0
        val mean = data.average()
        val sumSq = data.sumOf { (it - mean) * (it - mean) }
        return sqrt(sumSq / data.size)
    }

    private fun postProcess(posErr: Float, velErr: Float, timestamp: Long): Member5Result {
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
        // Training medians, means, and scales for 107 features
        private val MEDIANS = doubleArrayOf(
            1405.899, 0.001, 52.4459, -1.5348, 159.0, 9.77, 3.0, 147.199, 2113111.0, 0.0143,
            -0.0118, 9.849, 0.0, 0.0, 9.806, -9.9e-05, -0.001, 0.00019, -12.06, -33.68,
            15.31, 245.85, -84.41, -118.55, 1405.89, 1421.59, 0.1, 0.0143, -0.0118, 9.849,
            9.943, -9.9e-05, -0.001, 0.00019, 0.1279, -12.06, -33.68, 15.31, 42.445, 0.0,
            0.0, 9.806, 9.806, 245.85, -84.41, -118.55, 52.44, -1.53, 159.0, 9.77,
            3.0, 21.0, 0.39, 0.0, 0.038, 0.0, 0.36, 0.0, 0.0, 0.0,
            0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 9.97, 0.48, 9.29,
            10.78, 1.60, 0.15, 0.05, 0.05, 0.26, 0.18, 42.50, 0.37, 41.37,
            43.75, 1.23, 9.80, 5e-05, 9.80, 9.80, 0.00016, 9.73, 0.0, 9.59,
            10.0, 0.0, 0.54, 0.37, 0.04, 1.28, 1.20, 0.05, 0.03, 0.004,
            0.13, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        private val MEANS = doubleArrayOf(
            2060.933, 0.0049, 52.59, -1.53, 177.20, 10.51, 23.41, 167.79, 2701114.6, -0.01,
            -0.03, 9.84, 1.62e-06, -0.00016, 9.80, -0.00038, -0.003, 1.43e-05, -10.91, -31.82,
            15.32, 211.72, -83.87, -36.63, 2060.93, 2089.62, 0.10, -0.01, -0.03, 9.84,
            10.08, -0.00038, -0.003, 1.43e-05, 0.19, -10.91, -31.82, 15.32, 43.25, 1.62e-06,
            1.62e-06, 9.80, 9.80, 211.72, -83.87, -36.63, 52.59, -1.53, 177.20, 10.51,
            23.41, 21.46, 0.68, 0.05, 0.09, 0.21, 1.01, 3.77, 0.11, -5.78,
            -0.02, -167.10, -9.85e-05, 20.13, -0.001, -52.42, 0.99, 10.08, 0.58, 9.16,
            11.09, 1.93, 0.19, 0.08, 0.07, 0.35, 0.27, 43.25, 1.07, 42.01,
            45.38, 3.36, 9.80, 0.0001, 9.80, 9.80, 0.0004, 10.51, 0.14, 10.34,
            10.67, 0.33, 0.68, 0.47, 0.09, 1.61, 1.51, 0.09, 0.06, 0.01,
            0.21, 0.20, 0.11, 0.09, 0.06, 0.36, 0.29
        )
        private val SCALES = doubleArrayOf(
            2085.31, 0.0107, 0.317, 0.25, 76.56, 7.60, 213.98, 111.06, 2465593.5, 1.61,
            1.56, 0.72, 0.035, 0.033, 0.003, 0.11, 0.23, 0.13, 13.06, 8.06,
            18.01, 119.75, 3.85, 136.44, 2085.31, 2052.32, 2.76, 1.61, 1.56, 0.72,
            0.86, 0.11, 0.23, 0.13, 0.22, 13.06, 8.06, 18.01, 7.48, 0.035,
            0.035, 0.003, 0.002, 119.75, 3.85, 136.44, 0.31, 0.25, 76.56, 7.60,
            213.98, 4.01, 0.86, 62.32, 0.15, 25.84, 3.24, 352.61, 0.97, 338.94,
            13.38, 4905.96, 1.47, 1628.31, 8.23, 2179.74, 0.013, 0.42, 0.47, 0.63,
            1.23, 1.60, 0.17, 0.10, 0.08, 0.35, 0.32, 7.07, 2.20, 6.94,
            9.66, 6.70, 0.0006, 0.002, 0.003, 0.005, 0.008, 7.57, 0.63, 7.61,
            7.58, 1.39, 0.58, 0.42, 0.14, 1.40, 1.34, 0.11, 0.08, 0.02,
            0.27, 0.26, 0.89, 0.37, 0.82, 1.43, 1.17
        )

        // Target inverse scaling parameters
        private val TARGET_MEANS = doubleArrayOf(142.462, 7.809)
        private val TARGET_SCALES = doubleArrayOf(308.372, 5.637)
    }
}
