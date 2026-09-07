package com.example.nav_shield_1

import android.app.Activity
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * NAV-SHIELD Member 3 - IMU / Dead-Reckoning core.
 *
 * This single-file version implements the requested Phase A-I work while keeping
 * clean public input/output interfaces for the other members.
 *
 * IMPORTANT:
 * - Do NOT feed a fake GNSS heading automatically. Dead reckoning waits for the
 *   first real heading through setGnssHeading().
 * - The 5-second accelerometer stationary mean is retained as a diagnostic
 *   gravity-inclusive stationary offset. It is NOT blindly subtracted from the
 *   acceleration used for dead reckoning, because that would remove gravity and
 *   hide real vehicle dynamics.
 * - Member 4 remains responsible for sensor fusion/UKF. This class exposes reset
 *   methods so an externally corrected state can be fed back into the IMU engine.
 */
class MainActivity : Activity(), SensorEventListener {

    companion object {
        private const val G = 9.80665f
        private const val DEG_TO_RAD = 0.017453292519943295f
        private const val RAD_TO_DEG = 57.29577951308232f
        private const val NS_TO_SEC = 1e-9f

        private const val CALIBRATION_MS = 5_000L
        private const val ACCEL_ALPHA = 0.8f
        private const val GYRO_HP_ALPHA = 0.98f

        private const val ACCEL_RING_SIZE = 100
        private const val MOTION_WINDOW_SIZE = 50
        private const val LOG_INTERVAL_NS = 50_000_000L // ~20 Hz max file logging

        private const val STATIONARY_GYRO_THRESHOLD = 0.08f       // rad/s
        private const val STATIONARY_ACCEL_ERROR = 0.35f          // m/s² from 1 g
        private const val ENGINE_VIBRATION_STD = 0.70f            // m/s²
        private const val LOW_RATE_HZ = 20f
        private const val DISTURBANCE_GYRO_THRESHOLD = 4.0f       // rad/s
        private const val POTHOLE_DEVIATION = 5.0f                // m/s²
        private const val POTHOLE_JERK = 25.0f                    // m/s³
        private const val HARD_BRAKING_THRESHOLD = -5.0f          // m/s²
        private const val TURN_RATE_THRESHOLD = 0.25f             // rad/s
        private const val ACCELERATION_THRESHOLD = 1.0f           // m/s²
        private const val LONG_DR_SECONDS = 300f
    }

    // ============================================================
    // UI / ANDROID SENSOR API
    // ============================================================

    private lateinit var sensorManager: SensorManager
    private lateinit var textView: TextView

    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var rotationVector: Sensor? = null

    private var accelAvailable = false
    private var gyroAvailable = false
    private var imuAvailable = false

    // ============================================================
    // PHASE A - ACCELEROMETER ACQUISITION
    // ============================================================

    private var rawAccelX = 0f
    private var rawAccelY = 0f
    private var rawAccelZ = 0f

    // Required alpha = 0.8 low-pass filter.
    private var filteredAccelX = 0f
    private var filteredAccelY = 0f
    private var filteredAccelZ = 0f
    private var accelFilterInitialized = false

    // Circular ring buffer.
    private val accelRingX = FloatArray(ACCEL_RING_SIZE)
    private val accelRingY = FloatArray(ACCEL_RING_SIZE)
    private val accelRingZ = FloatArray(ACCEL_RING_SIZE)
    private var accelRingIndex = 0
    private var accelRingCount = 0

    // ============================================================
    // PHASE B - GYROSCOPE ACQUISITION
    // ============================================================

    private var rawGyroX = 0f
    private var rawGyroY = 0f
    private var rawGyroZ = 0f

    // Required simple high-pass IIR.
    private var filteredGyroX = 0f
    private var filteredGyroY = 0f
    private var filteredGyroZ = 0f
    private var previousGyroX = 0f
    private var previousGyroY = 0f
    private var previousGyroZ = 0f
    private var gyroFilterInitialized = false

    // ============================================================
    // PHASE C - 5 SECOND CALIBRATION
    // ============================================================

    private var calibrationStartNs = 0L
    private var accelCalibrationSamples = 0
    private var gyroCalibrationSamples = 0

    private var accelSumX = 0.0
    private var accelSumY = 0.0
    private var accelSumZ = 0.0
    private var gyroSumX = 0.0
    private var gyroSumY = 0.0
    private var gyroSumZ = 0.0

    // Stationary mean values for the contract/diagnostics.
    private var accelBiasX = 0f
    private var accelBiasY = 0f
    private var accelBiasZ = 0f
    private var gyroBiasX = 0f
    private var gyroBiasY = 0f
    private var gyroBiasZ = 0f
    private var isCalibrated = false

    // ============================================================
    // PHASE D - PHONE -> VEHICLE ALIGNMENT
    // ============================================================

    private var phoneYawDeg = 0f
    private var phonePitchDeg = 0f
    private var phoneRollDeg = 0f

    // GNSS heading provided by Member 2. NaN means unavailable.
    private var gnssHeadingDeg = Float.NaN

    // Fixed phone -> vehicle mounting alignment.
    private var yawOffsetDeg = 0f
    private var alignmentPitchDeg = 0f
    private var alignmentRollDeg = 0f
    private var hasAlignment = false

    private var phoneToVehicleMatrix = identity3()

    // ============================================================
    // PHASE E - CUSTOM ORIENTATION / COMPLEMENTARY FILTER
    // ============================================================

    // Quaternion q maps phone/body coordinates into the Android world frame.
    // Stored as w, x, y, z.
    private var qW = 1f
    private var qX = 0f
    private var qY = 0f
    private var qZ = 0f
    private var quaternionInitialized = false

    private var orientationYawDeg = 0f
    private var orientationPitchDeg = 0f
    private var orientationRollDeg = 0f

    // 98% gyro integration + 2% accelerometer gravity correction.
    private val gyroWeight = 0.98f
    private val accelCorrectionWeight = 0.02f

    private var lastGyroTimestampNs = 0L
    private var lastAccelTimestampNs = 0L
    private var currentDtSec = 0f

    // Reference orientation supplied by Android rotation-vector sensor.
    private var rotationVectorReferenceReady = false
    private val rotationReferenceMatrix = FloatArray(9)

    // ============================================================
    // PHASE F - VEHICLE ACCELERATION / VELOCITY / DISPLACEMENT
    // ============================================================

    private var vehicleAccelX = 0f
    private var vehicleAccelY = 0f
    private var vehicleAccelZ = 0f

    private var linearAccelPhoneX = 0f
    private var linearAccelPhoneY = 0f
    private var linearAccelPhoneZ = 0f

    private var velocityX = 0f
    private var velocityY = 0f
    private var velocityZ = 0f
    private var speedMps = 0f

    private var displacementX = 0f
    private var displacementY = 0f
    private var displacementZ = 0f
    private var displacementMeters = 0f

    private var zuptApplied = false

    // Member 1 can feed road grade. Default is 0 degrees.
    private var roadGradeDeg = 0f

    // ============================================================
    // PHASE G - DEAD-RECKONING POSITION
    // ============================================================

    private var lastKnownLatitude = Double.NaN
    private var lastKnownLongitude = Double.NaN
    private var drLatitude = Double.NaN
    private var drLongitude = Double.NaN

    // Position at the start of current local displacement.
    private var originLatitude = Double.NaN
    private var originLongitude = Double.NaN

    // ============================================================
    // PHASE H - MOTION / VIBRATION CLASSIFICATION
    // ============================================================

    enum class MotionState {
        STATIONARY,
        NORMAL,
        ACCELERATING,
        BRAKING,
        TURNING,
        POTHOLE,
        PHONE_DISTURBANCE
    }

    private var motionState = MotionState.STATIONARY
    private var vibrationLevel = 0f

    private val accelMagnitudeWindow = FloatArray(MOTION_WINDOW_SIZE)
    private var motionWindowIndex = 0
    private var motionWindowCount = 0
    private var previousAccelMagnitude = G

    private var lowSampleRate = false
    private var estimatedSampleRateHz = 0f

    // ============================================================
    // PHASE I - CONFIDENCE / LOGGING / DIAGNOSTICS
    // ============================================================

    private var imuConfidence = 0f
    private var gnssOutageStartNs = 0L
    private var longDrMode = false

    private var logFile: File? = null
    private var lastLogTimestampNs = 0L
    private var logHeaderWritten = false

    // Bias stability diagnostics.
    private var previousGyroBiasMagnitude = 0f
    private var biasStability = 1f

    // ============================================================
    // ANDROID LIFECYCLE
    // ============================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        textView = TextView(this)
        textView.textSize = 15f
        textView.setPadding(24, 36, 24, 24)
        textView.text = "NAV-SHIELD IMU\nStarting sensors..."

        // The A-I diagnostic screen is longer than a phone display.
        // Wrap it in a ScrollView so Phase H and Phase I can be viewed
        // by scrolling to the bottom.
        val scrollView = ScrollView(this)
        scrollView.addView(textView)
        setContentView(scrollView)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        accelAvailable = accelerometer != null
        gyroAvailable = gyroscope != null
        imuAvailable = accelAvailable && gyroAvailable

        val now = System.nanoTime()
        calibrationStartNs = now
        gnssOutageStartNs = now

        logFile = File(filesDir, "nav_shield_imu.csv")
        writeCsvHeader()
        updateDisplay()
    }

    override fun onResume() {
        super.onResume()

        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        rotationVector?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    // ============================================================
    // SENSOR CALLBACK
    // ============================================================

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> processAccelerometer(event)
            Sensor.TYPE_GYROSCOPE -> processGyroscope(event)
            Sensor.TYPE_ROTATION_VECTOR -> processRotationVector(event)
        }

        classifyMotion()
        updateConfidence(event.timestamp)
        writeCsvSampleIfDue(event.timestamp)
        updateDisplay()
    }

    // ============================================================
    // PHASE A
    // ============================================================

    private fun processAccelerometer(event: SensorEvent) {
        rawAccelX = event.values[0]
        rawAccelY = event.values[1]
        rawAccelZ = event.values[2]

        val previousAccelTimestampNs = lastAccelTimestampNs
        lastAccelTimestampNs = event.timestamp

        if (!isCalibrated) {
            accelSumX += rawAccelX.toDouble()
            accelSumY += rawAccelY.toDouble()
            accelSumZ += rawAccelZ.toDouble()
            accelCalibrationSamples++
        }

        accelRingX[accelRingIndex] = rawAccelX
        accelRingY[accelRingIndex] = rawAccelY
        accelRingZ[accelRingIndex] = rawAccelZ
        accelRingIndex = (accelRingIndex + 1) % ACCEL_RING_SIZE
        accelRingCount = min(accelRingCount + 1, ACCEL_RING_SIZE)

        if (!accelFilterInitialized) {
            filteredAccelX = rawAccelX
            filteredAccelY = rawAccelY
            filteredAccelZ = rawAccelZ
            accelFilterInitialized = true
        } else {
            filteredAccelX = ACCEL_ALPHA * filteredAccelX + (1f - ACCEL_ALPHA) * rawAccelX
            filteredAccelY = ACCEL_ALPHA * filteredAccelY + (1f - ACCEL_ALPHA) * rawAccelY
            filteredAccelZ = ACCEL_ALPHA * filteredAccelZ + (1f - ACCEL_ALPHA) * rawAccelZ
        }

        val magnitude = sqrt(rawAccelX * rawAccelX + rawAccelY * rawAccelY + rawAccelZ * rawAccelZ)
        pushMotionMagnitude(magnitude)
        currentDtSec = if (previousAccelTimestampNs > 0L) {
            val dt = (event.timestamp - previousAccelTimestampNs) * NS_TO_SEC
            if (dt in 0.0001f..0.2f) dt else currentDtSec
        } else {
            currentDtSec
        }

        if (isCalibrated) {
            // Dead-reckoning acceleration comes from raw specific force minus the
            // gravity vector estimated from orientation. Do not subtract the
            // stationary mean here because its Z component contains gravity.
            updateLinearAccelerationAndKinematics(event.timestamp)
        }

        finishCalibrationIfReady(event.timestamp)
    }

    // ============================================================
    // PHASE B
    // ============================================================

    private fun processGyroscope(event: SensorEvent) {
        rawGyroX = event.values[0]
        rawGyroY = event.values[1]
        rawGyroZ = event.values[2]

        if (!isCalibrated) {
            gyroSumX += rawGyroX.toDouble()
            gyroSumY += rawGyroY.toDouble()
            gyroSumZ += rawGyroZ.toDouble()
            gyroCalibrationSamples++
        }

        if (!gyroFilterInitialized) {
            previousGyroX = rawGyroX
            previousGyroY = rawGyroY
            previousGyroZ = rawGyroZ
            filteredGyroX = 0f
            filteredGyroY = 0f
            filteredGyroZ = 0f
            gyroFilterInitialized = true
        } else {
            filteredGyroX = GYRO_HP_ALPHA * (filteredGyroX + rawGyroX - previousGyroX)
            filteredGyroY = GYRO_HP_ALPHA * (filteredGyroY + rawGyroY - previousGyroY)
            filteredGyroZ = GYRO_HP_ALPHA * (filteredGyroZ + rawGyroZ - previousGyroZ)

            previousGyroX = rawGyroX
            previousGyroY = rawGyroY
            previousGyroZ = rawGyroZ
        }

        if (lastGyroTimestampNs == 0L) {
            lastGyroTimestampNs = event.timestamp
            return
        }

        var dt = (event.timestamp - lastGyroTimestampNs) * NS_TO_SEC
        lastGyroTimestampNs = event.timestamp
        if (dt <= 0f || dt > 0.2f) dt = 1f / 50f
        currentDtSec = dt

        if (isCalibrated) {
            val correctedX = rawGyroX - gyroBiasX
            val correctedY = rawGyroY - gyroBiasY
            val correctedZ = rawGyroZ - gyroBiasZ
            integrateGyroscope(correctedX, correctedY, correctedZ, dt)
        }
    }

    // ============================================================
    // PHASE C
    // ============================================================

    private fun finishCalibrationIfReady(timestampNs: Long) {
        if (isCalibrated) return
        if (timestampNs - calibrationStartNs < CALIBRATION_MS * 1_000_000L) return
        if (accelCalibrationSamples == 0 || gyroCalibrationSamples == 0) return

        accelBiasX = (accelSumX / accelCalibrationSamples).toFloat()
        accelBiasY = (accelSumY / accelCalibrationSamples).toFloat()
        accelBiasZ = (accelSumZ / accelCalibrationSamples).toFloat()

        gyroBiasX = (gyroSumX / gyroCalibrationSamples).toFloat()
        gyroBiasY = (gyroSumY / gyroCalibrationSamples).toFloat()
        gyroBiasZ = (gyroSumZ / gyroCalibrationSamples).toFloat()

        isCalibrated = true
        previousGyroBiasMagnitude = gyroBiasMagnitude()
        biasStability = 1f

        // Require real GNSS heading before dead reckoning, per project contract.
        updateAlignment()
    }

    // ============================================================
    // PHASE D - ROTATION VECTOR REFERENCE + ALIGNMENT
    // ============================================================

    private fun processRotationVector(event: SensorEvent) {
        SensorManager.getRotationMatrixFromVector(rotationReferenceMatrix, event.values)
        val orientation = FloatArray(3)
        SensorManager.getOrientation(rotationReferenceMatrix, orientation)

        phoneYawDeg = orientation[0] * RAD_TO_DEG
        phonePitchDeg = orientation[1] * RAD_TO_DEG
        phoneRollDeg = orientation[2] * RAD_TO_DEG

        // Use the phone's rotation-vector orientation as the initial reference only.
        // After initialization the orientation is driven by the custom gyro path.
        if (!quaternionInitialized) {
            setQuaternionFromRotationMatrix(rotationReferenceMatrix)
            quaternionInitialized = true
            rotationVectorReferenceReady = true
            updateOrientationEuler()
            updateAlignment()
        }
    }

    /**
     * Member 2 -> Member 3 interface.
     * Heading is degrees clockwise from North, normalized to [-180, 180].
     */
    fun setGnssHeading(headingDegrees: Float) {
        if (!headingDegrees.isFinite()) return
        gnssHeadingDeg = normalizeAngle(headingDegrees)
        gnssOutageStartNs = System.nanoTime()
        longDrMode = false
        updateAlignment()
        updateConfidence(System.nanoTime())
        updateDisplay()
    }

    /**
     * Set the current known position used as the DR origin.
     */
    fun setInitialPosition(latitude: Double, longitude: Double) {
        if (!validLatitude(latitude) || !validLongitude(longitude)) return
        lastKnownLatitude = latitude
        lastKnownLongitude = longitude
        drLatitude = latitude
        drLongitude = longitude
        originLatitude = latitude
        originLongitude = longitude
    }

    /**
     * Complete external-state initialization from the rest of the system.
     */
    fun setInitialState(
        latitude: Double,
        longitude: Double,
        headingDegrees: Float,
        initialVelocityMps: Float = 0f
    ) {
        if (!validLatitude(latitude) || !validLongitude(longitude)) return
        setInitialPosition(latitude, longitude)
        if (headingDegrees.isFinite()) setGnssHeading(headingDegrees)
        setVelocity(initialVelocityMps)
    }

    /**
     * Member 1 -> Member 3 interface for road grade correction.
     */
    fun setRoadGrade(degrees: Float) {
        if (degrees.isFinite()) roadGradeDeg = degrees.coerceIn(-45f, 45f)
    }

    /**
     * Member 4 -> Member 3 feedback interface after UKF/GNSS correction.
     * This resets the local DR origin without automatically deciding how fusion
     * should occur.
     */
    fun resetFromExternalState(
        latitude: Double,
        longitude: Double,
        headingDegrees: Float? = null,
        velocityMps: Float? = null
    ) {
        if (!validLatitude(latitude) || !validLongitude(longitude)) return

        lastKnownLatitude = latitude
        lastKnownLongitude = longitude
        drLatitude = latitude
        drLongitude = longitude
        originLatitude = latitude
        originLongitude = longitude

        displacementX = 0f
        displacementY = 0f
        displacementZ = 0f
        displacementMeters = 0f

        if (headingDegrees != null && headingDegrees.isFinite()) {
            setGnssHeading(headingDegrees)
        }
        if (velocityMps != null && velocityMps.isFinite()) {
            setVelocity(max(0f, velocityMps))
        }
    }

    private fun updateAlignment() {
        if (!quaternionInitialized) return

        updateOrientationEuler()

        if (gnssHeadingDeg.isNaN()) {
            hasAlignment = false
            yawOffsetDeg = 0f
            alignmentPitchDeg = orientationPitchDeg
            alignmentRollDeg = orientationRollDeg
            phoneToVehicleMatrix = identity3()
            return
        }

        // GNSS gives vehicle heading; quaternion gives phone yaw. Their difference
        // is the yaw mounting offset. Pitch/roll are captured at alignment time.
        yawOffsetDeg = normalizeAngle(gnssHeadingDeg - orientationYawDeg)
        if (!hasAlignment) {
            alignmentPitchDeg = orientationPitchDeg
            alignmentRollDeg = orientationRollDeg
        }
        hasAlignment = true
        phoneToVehicleMatrix = createRotationMatrix(
            yawOffsetDeg,
            -alignmentPitchDeg,
            -alignmentRollDeg
        )
    }

    // ============================================================
    // PHASE E - QUATERNION + COMPLEMENTARY FILTER
    // ============================================================

    private fun integrateGyroscope(gx: Float, gy: Float, gz: Float, dt: Float) {
        if (!quaternionInitialized) {
            quaternionInitialized = true
        }

        // Quaternion derivative integration using the current angular velocity.
        // Small-angle form is stable at Android IMU sample rates.
        val halfDt = 0.5f * dt
        val dw = 1f
        val dx = gx * halfDt
        val dy = gy * halfDt
        val dz = gz * halfDt

        // q_new = q * dq, with dq ~= [1, wx*dt/2, wy*dt/2, wz*dt/2]
        val nw = qW * dw - qX * dx - qY * dy - qZ * dz
        val nx = qW * dx + qX * dw + qY * dz - qZ * dy
        val ny = qW * dy - qX * dz + qY * dw + qZ * dx
        val nz = qW * dz + qX * dy - qY * dx + qZ * dw

        qW = nw
        qX = nx
        qY = ny
        qZ = nz
        normalizeQuaternion()

        updateOrientationEuler()

        // Accelerometer gravity direction supplies roll/pitch correction only.
        val correctedAccelMagnitude = sqrt(
            rawAccelX * rawAccelX + rawAccelY * rawAccelY + rawAccelZ * rawAccelZ
        )

        if (correctedAccelMagnitude in (G * 0.65f)..(G * 1.35f)) {
            val accelRoll = atan2(rawAccelY.toDouble(), rawAccelZ.toDouble()).toFloat() * RAD_TO_DEG
            val accelPitch = atan2(
                -rawAccelX.toDouble(),
                sqrt((rawAccelY * rawAccelY + rawAccelZ * rawAccelZ).toDouble())
            ).toFloat() * RAD_TO_DEG

            // Explicit 98/2 complementary blend.
            orientationRollDeg = gyroWeight * orientationRollDeg +
                    accelCorrectionWeight * accelRoll
            orientationPitchDeg = gyroWeight * orientationPitchDeg +
                    accelCorrectionWeight * accelPitch

            // Preserve yaw from gyro integration; accelerometer has no absolute yaw information.
            qW = 1f
            qX = 0f
            qY = 0f
            qZ = 0f
            setQuaternionFromEuler(
                orientationYawDeg,
                orientationPitchDeg,
                orientationRollDeg
            )
        }

        if (!gnssHeadingDeg.isNaN() && hasAlignment) {
            // Re-anchor vehicle heading from GNSS only when the explicit external
            // heading update arrives. Do not overwrite yaw continuously.
            phoneYawDeg = orientationYawDeg
        }
    }

    private fun updateOrientationEuler() {
        // roll (x-axis)
        val sinr = 2f * (qW * qX + qY * qZ)
        val cosr = 1f - 2f * (qX * qX + qY * qY)
        orientationRollDeg = atan2(sinr.toDouble(), cosr.toDouble()).toFloat() * RAD_TO_DEG

        // pitch (y-axis)
        val sinp = 2f * (qW * qY - qZ * qX)
        orientationPitchDeg = if (abs(sinp) >= 1f) {
            if (sinp >= 0f) 90f else -90f
        } else {
            kotlin.math.asin(sinp.toDouble()).toFloat() * RAD_TO_DEG
        }

        // yaw (z-axis)
        val siny = 2f * (qW * qZ + qX * qY)
        val cosy = 1f - 2f * (qY * qY + qZ * qZ)
        orientationYawDeg = atan2(siny.toDouble(), cosy.toDouble()).toFloat() * RAD_TO_DEG
        orientationYawDeg = normalizeAngle(orientationYawDeg)

        phoneYawDeg = orientationYawDeg
        phonePitchDeg = orientationPitchDeg
        phoneRollDeg = orientationRollDeg
    }

    // ============================================================
    // PHASE F - GRAVITY REMOVAL, VEHICLE FRAME, VELOCITY, ZUPT
    // ============================================================

    private fun updateLinearAccelerationAndKinematics(timestampNs: Long) {
        if (!hasAlignment) return
        if (lastGyroTimestampNs == 0L) return

        // Estimate gravity in the phone frame from the current orientation.
        val gravityPhone = rotateVectorByQuaternionInverse(0f, 0f, G)

        linearAccelPhoneX = rawAccelX - gravityPhone[0]
        linearAccelPhoneY = rawAccelY - gravityPhone[1]
        linearAccelPhoneZ = rawAccelZ - gravityPhone[2]

        val vehicle = multiply3x3Vector(
            phoneToVehicleMatrix,
            linearAccelPhoneX,
            linearAccelPhoneY,
            linearAccelPhoneZ
        )

        vehicleAccelX = vehicle[0]
        vehicleAccelY = vehicle[1]
        vehicleAccelZ = vehicle[2]

        // Road-grade compensation on longitudinal axis supplied by Member 1.
        if (abs(roadGradeDeg) > 0.01f) {
            vehicleAccelX -= G * kotlin.math.sin(roadGradeDeg * DEG_TO_RAD)
        }

        val dt = currentDtSec.coerceIn(0.001f, 0.2f)

        zuptApplied = motionState == MotionState.STATIONARY && vibrationLevel < 0.45f
        if (zuptApplied) {
            velocityX = 0f
            velocityY = 0f
            velocityZ = 0f
        } else {
            velocityX += vehicleAccelX * dt
            velocityY += vehicleAccelY * dt
            velocityZ += vehicleAccelZ * dt

            // Projected ground-speed is used for the navigation output.
            // Do not let hard braking produce a negative forward speed.
            if (vehicleAccelX < HARD_BRAKING_THRESHOLD && velocityX < 0f) {
                velocityX = 0f
            }
        }

        speedMps = sqrt(velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ)
        if (speedMps < 0.05f && zuptApplied) speedMps = 0f

        displacementX += velocityX * dt
        displacementY += velocityY * dt
        displacementZ += velocityZ * dt
        displacementMeters = sqrt(
            displacementX * displacementX +
                    displacementY * displacementY +
                    displacementZ * displacementZ
        )

        updateDeadReckoningPosition()
    }

    // ============================================================
    // PHASE G - HAVERSINE / FORWARD PROJECTION
    // ============================================================

    private fun updateDeadReckoningPosition() {
        if (originLatitude.isNaN() || originLongitude.isNaN()) return
        if (gnssHeadingDeg.isNaN() || !hasAlignment) return

        // Vehicle X is the forward axis. Vehicle Y is lateral.
        // Combine local displacement with current vehicle heading.
        val headingRad = gnssHeadingDeg * DEG_TO_RAD
        val northMeters = displacementX * cos(headingRad.toDouble()) -
                displacementY * kotlin.math.sin(headingRad.toDouble())
        val eastMeters = displacementX * kotlin.math.sin(headingRad.toDouble()) +
                displacementY * cos(headingRad.toDouble())

        val projected = haversineForward(
            originLatitude,
            originLongitude,
            northMeters,
            eastMeters
        )

        drLatitude = projected.first
        drLongitude = projected.second
        lastKnownLatitude = drLatitude
        lastKnownLongitude = drLongitude
    }

    private fun haversineForward(
        latitude: Double,
        longitude: Double,
        northMeters: Double,
        eastMeters: Double
    ): Pair<Double, Double> {
        val earthRadius = 6_378_137.0
        val lat1 = latitude * DEG_TO_RAD
        val lon1 = longitude * DEG_TO_RAD

        val angularDistance = sqrt(northMeters * northMeters + eastMeters * eastMeters) / earthRadius
        if (angularDistance < 1e-12) return latitude to longitude

        val bearing = atan2(eastMeters, northMeters)
        val lat2 = kotlin.math.asin(
            kotlin.math.sin(lat1) * cos(angularDistance) +
                    kotlin.math.cos(lat1) * kotlin.math.sin(angularDistance) * cos(bearing)
        )
        val lon2 = lon1 + atan2(
            kotlin.math.sin(bearing) * kotlin.math.sin(angularDistance) * kotlin.math.cos(lat1),
            cos(angularDistance) - kotlin.math.sin(lat1) * kotlin.math.sin(lat2)
        )

        return (lat2 * RAD_TO_DEG) to normalizeLongitude(lon2 * RAD_TO_DEG)
    }

    // ============================================================
    // PHASE H - MOTION / VIBRATION CLASSIFICATION
    // ============================================================

    private fun pushMotionMagnitude(magnitude: Float) {
        accelMagnitudeWindow[motionWindowIndex] = magnitude
        motionWindowIndex = (motionWindowIndex + 1) % MOTION_WINDOW_SIZE
        motionWindowCount = min(motionWindowCount + 1, MOTION_WINDOW_SIZE)
    }

    private fun classifyMotion() {
        if (!imuAvailable) {
            motionState = MotionState.STATIONARY
            vibrationLevel = 0f
            return
        }

        val accelMagnitude = sqrt(
            rawAccelX * rawAccelX + rawAccelY * rawAccelY + rawAccelZ * rawAccelZ
        )
        val gyroMagnitude = sqrt(
            rawGyroX * rawGyroX + rawGyroY * rawGyroY + rawGyroZ * rawGyroZ
        )

        val variance = windowVariance()
        val stdDev = sqrt(max(0f, variance))
        vibrationLevel = (stdDev / 2.5f).coerceIn(0f, 1f)

        val jerk = if (currentDtSec > 0.001f) {
            abs(accelMagnitude - previousAccelMagnitude) / currentDtSec
        } else {
            0f
        }
        previousAccelMagnitude = accelMagnitude

        if (gyroMagnitude > DISTURBANCE_GYRO_THRESHOLD && abs(accelMagnitude - G) > 2.5f) {
            motionState = MotionState.PHONE_DISTURBANCE
            return
        }

        if (abs(accelMagnitude - G) > POTHOLE_DEVIATION && jerk > POTHOLE_JERK) {
            motionState = MotionState.POTHOLE
            return
        }

        val stationary = abs(accelMagnitude - G) < STATIONARY_ACCEL_ERROR &&
                gyroMagnitude < STATIONARY_GYRO_THRESHOLD &&
                stdDev < ENGINE_VIBRATION_STD

        if (stationary) {
            motionState = MotionState.STATIONARY
            return
        }

        if (vehicleAccelX < HARD_BRAKING_THRESHOLD) {
            motionState = MotionState.BRAKING
            return
        }

        if (abs(rawGyroZ) > TURN_RATE_THRESHOLD) {
            motionState = MotionState.TURNING
            return
        }

        if (vehicleAccelX > ACCELERATION_THRESHOLD) {
            motionState = MotionState.ACCELERATING
            return
        }

        motionState = MotionState.NORMAL
    }

    private fun windowVariance(): Float {
        if (motionWindowCount < 2) return 0f

        var sum = 0.0
        for (i in 0 until motionWindowCount) sum += accelMagnitudeWindow[i]
        val mean = sum / motionWindowCount

        var sq = 0.0
        for (i in 0 until motionWindowCount) {
            val d = accelMagnitudeWindow[i] - mean
            sq += d * d
        }
        return (sq / (motionWindowCount - 1)).toFloat()
    }

    // ============================================================
    // PHASE I - CONFIDENCE / RATE / LOGGING
    // ============================================================

    private fun updateConfidence(timestampNs: Long) {
        if (!imuAvailable) {
            imuConfidence = 0f
            return
        }

        val elapsedSinceCalibration = max(
            0f,
            (timestampNs - calibrationStartNs) * NS_TO_SEC
        )

        val noisePenalty = vibrationLevel.coerceIn(0f, 1f)
        val ratePenalty = if (lowSampleRate) 0.45f else 0f
        val biasPenalty = 1f - biasStability

        var confidence = 1f
        confidence *= (1f - 0.45f * noisePenalty)
        confidence *= (1f - 0.35f * biasPenalty)
        confidence *= (1f - ratePenalty)

        when (motionState) {
            MotionState.PHONE_DISTURBANCE -> confidence = min(confidence, 0.1f)
            MotionState.POTHOLE -> confidence *= 0.6f
            MotionState.TURNING -> confidence *= 0.9f
            else -> Unit
        }

        if (gnssHeadingDeg.isNaN()) {
            // No valid initial heading = no trusted DR solution.
            confidence = 0f
        }

        val outageSeconds = if (gnssHeadingDeg.isNaN()) {
            elapsedSinceCalibration
        } else {
            (timestampNs - gnssOutageStartNs) * NS_TO_SEC
        }

        if (outageSeconds > LONG_DR_SECONDS) {
            longDrMode = true
            confidence *= exp(-(outageSeconds - LONG_DR_SECONDS) / LONG_DR_SECONDS)
        }

        imuConfidence = confidence.coerceIn(0f, 1f)

        if (lastGyroTimestampNs > 0L && currentDtSec > 0f) {
            estimatedSampleRateHz = 1f / currentDtSec
            lowSampleRate = estimatedSampleRateHz < LOW_RATE_HZ
        }

        val biasMagnitude = gyroBiasMagnitude()
        val biasDelta = abs(biasMagnitude - previousGyroBiasMagnitude)
        biasStability = (1f - (biasDelta / 0.05f)).coerceIn(0f, 1f)
        previousGyroBiasMagnitude = biasMagnitude
    }

    private fun writeCsvHeader() {
        val file = logFile ?: return
        if (logHeaderWritten) return
        if (!file.exists() || file.length() == 0L) {
            file.appendText(
                "timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z," +
                        "vehicle_ax,vehicle_ay,vehicle_az,velocity_mps,displacement_m," +
                        "heading_deg,motion_state,vibration,imu_confidence,zupt\n"
            )
        }
        logHeaderWritten = true
    }

    private fun writeCsvSampleIfDue(timestampNs: Long) {
        if (lastLogTimestampNs != 0L && timestampNs - lastLogTimestampNs < LOG_INTERVAL_NS) return
        lastLogTimestampNs = timestampNs

        val file = logFile ?: return
        writeCsvHeader()

        val heading = currentVehicleHeadingDegrees()
        val row = String.format(
            Locale.US,
            "%d,%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f,%.5f,%.5f,%.3f,%s,%.4f,%.4f,%b\n",
            timestampNs,
            rawAccelX, rawAccelY, rawAccelZ,
            rawGyroX, rawGyroY, rawGyroZ,
            vehicleAccelX, vehicleAccelY, vehicleAccelZ,
            speedMps, displacementMeters,
            heading,
            motionState.name,
            vibrationLevel,
            imuConfidence,
            zuptApplied
        )
        file.appendText(row)
    }

    /** Returns the path of the internal CSV log for sharing/export by the app. */
    fun getCsvLogFilePath(): String = logFile?.absolutePath.orEmpty()

    fun clearCsvLog() {
        logFile?.writeText("")
        logHeaderWritten = false
        writeCsvHeader()
    }

    // ============================================================
    // OUTPUT CONTRACT
    // ============================================================

    data class ImuBiasEstimate(
        val accelX: Float,
        val accelY: Float,
        val accelZ: Float,
        val gyroX: Float,
        val gyroY: Float,
        val gyroZ: Float
    )

    data class ImuOutput(
        val timestampMs: Long,
        val imuAvailable: Boolean,
        val imuLatitude: Double,
        val imuLongitude: Double,
        val imuHeadingDeg: Float,
        val imuVelocityMps: Float,
        val imuDisplacementM: Float,
        val imuBiasEstimate: ImuBiasEstimate,
        val imuConfidence: Float,
        val motionState: MotionState,
        val vibrationLevel: Float,
        val zuptApplied: Boolean,
        val quaternionW: Float,
        val quaternionX: Float,
        val quaternionY: Float,
        val quaternionZ: Float,
        val phoneToVehicleMatrix: FloatArray,
        val lowSampleRate: Boolean,
        val longDrMode: Boolean
    )

    fun getOutput(): ImuOutput {
        return ImuOutput(
            timestampMs = System.currentTimeMillis(),
            imuAvailable = imuAvailable,
            imuLatitude = drLatitude,
            imuLongitude = drLongitude,
            imuHeadingDeg = currentVehicleHeadingDegrees(),
            imuVelocityMps = speedMps,
            imuDisplacementM = displacementMeters,
            imuBiasEstimate = ImuBiasEstimate(
                accelBiasX,
                accelBiasY,
                accelBiasZ,
                gyroBiasX,
                gyroBiasY,
                gyroBiasZ
            ),
            imuConfidence = imuConfidence,
            motionState = motionState,
            vibrationLevel = vibrationLevel,
            zuptApplied = zuptApplied,
            quaternionW = qW,
            quaternionX = qX,
            quaternionY = qY,
            quaternionZ = qZ,
            phoneToVehicleMatrix = phoneToVehicleMatrix.copyOf(),
            lowSampleRate = lowSampleRate,
            longDrMode = longDrMode
        )
    }

    // ============================================================
    // DISPLAY
    // ============================================================

    private fun updateDisplay() {
        val correctedGyroX = rawGyroX - gyroBiasX
        val correctedGyroY = rawGyroY - gyroBiasY
        val correctedGyroZ = rawGyroZ - gyroBiasZ

        val gnssText = if (gnssHeadingDeg.isNaN()) "WAITING FOR MEMBER 2" else "%.2f°".format(Locale.US, gnssHeadingDeg)
        val positionText = if (drLatitude.isNaN()) {
            "NOT INITIALIZED"
        } else {
            "%.6f, %.6f".format(Locale.US, drLatitude, drLongitude)
        }

        val status = when {
            !imuAvailable -> "IMU UNAVAILABLE"
            !isCalibrated -> "CALIBRATING - KEEP PHONE STILL"
            gnssHeadingDeg.isNaN() -> "WAITING FOR MEMBER 2 GNSS HEADING"
            else -> "RUNNING"
        }

        textView.text = """
            NAV-SHIELD IMU ENGINE
            STATUS: $status

            PHASE A - ACCEL
            RAW      X:${"%.3f".format(Locale.US, rawAccelX)} Y:${"%.3f".format(Locale.US, rawAccelY)} Z:${"%.3f".format(Locale.US, rawAccelZ)}
            LOW-PASS X:${"%.3f".format(Locale.US, filteredAccelX)} Y:${"%.3f".format(Locale.US, filteredAccelY)} Z:${"%.3f".format(Locale.US, filteredAccelZ)}
            RING BUFFER $accelRingCount / $ACCEL_RING_SIZE

            PHASE B - GYRO
            RAW      X:${"%.4f".format(Locale.US, rawGyroX)} Y:${"%.4f".format(Locale.US, rawGyroY)} Z:${"%.4f".format(Locale.US, rawGyroZ)}
            HIGH-PASS X:${"%.4f".format(Locale.US, filteredGyroX)} Y:${"%.4f".format(Locale.US, filteredGyroY)} Z:${"%.4f".format(Locale.US, filteredGyroZ)}
            CORRECTED X:${"%.4f".format(Locale.US, correctedGyroX)} Y:${"%.4f".format(Locale.US, correctedGyroY)} Z:${"%.4f".format(Locale.US, correctedGyroZ)}

            PHASE C - CALIBRATION
            READY: $isCalibrated
            ACCEL MEAN X:${"%.3f".format(Locale.US, accelBiasX)} Y:${"%.3f".format(Locale.US, accelBiasY)} Z:${"%.3f".format(Locale.US, accelBiasZ)}
            GYRO BIAS  X:${"%.4f".format(Locale.US, gyroBiasX)} Y:${"%.4f".format(Locale.US, gyroBiasY)} Z:${"%.4f".format(Locale.US, gyroBiasZ)}

            PHASE D - ALIGNMENT
            PHONE YPR ${"%.2f".format(Locale.US, phoneYawDeg)}° / ${"%.2f".format(Locale.US, phonePitchDeg)}° / ${"%.2f".format(Locale.US, phoneRollDeg)}°
            GNSS HEADING $gnssText
            YAW OFFSET ${"%.2f".format(Locale.US, yawOffsetDeg)}°
            ALIGNMENT READY $hasAlignment
            R=[${"%.2f".format(Locale.US, phoneToVehicleMatrix[0])} ${"%.2f".format(Locale.US, phoneToVehicleMatrix[1])} ${"%.2f".format(Locale.US, phoneToVehicleMatrix[2])}]
              [${"%.2f".format(Locale.US, phoneToVehicleMatrix[3])} ${"%.2f".format(Locale.US, phoneToVehicleMatrix[4])} ${"%.2f".format(Locale.US, phoneToVehicleMatrix[5])}]
              [${"%.2f".format(Locale.US, phoneToVehicleMatrix[6])} ${"%.2f".format(Locale.US, phoneToVehicleMatrix[7])} ${"%.2f".format(Locale.US, phoneToVehicleMatrix[8])}]

            PHASE E - ORIENTATION
            VEHICLE HEADING ${"%.2f".format(Locale.US, currentVehicleHeadingDegrees())}°
            QUAT [${"%.4f".format(Locale.US, qW)}, ${"%.4f".format(Locale.US, qX)}, ${"%.4f".format(Locale.US, qY)}, ${"%.4f".format(Locale.US, qZ)}]
            PITCH ${"%.2f".format(Locale.US, orientationPitchDeg)}° ROLL ${"%.2f".format(Locale.US, orientationRollDeg)}°

            PHASE F - DEAD RECKONING
            A_vehicle ${"%.2f".format(Locale.US, vehicleAccelX)}, ${"%.2f".format(Locale.US, vehicleAccelY)}, ${"%.2f".format(Locale.US, vehicleAccelZ)} m/s²
            VELOCITY ${"%.3f".format(Locale.US, speedMps)} m/s
            DISPLACEMENT ${"%.3f".format(Locale.US, displacementMeters)} m
            ZUPT $zuptApplied

            PHASE G - POSITION
            $positionText

            PHASE H - MOTION
            STATE ${motionState.name}
            VIBRATION ${"%.2f".format(Locale.US, vibrationLevel)}

            PHASE I - DIAGNOSTICS
            CONFIDENCE ${"%.2f".format(Locale.US, imuConfidence)}
            RATE ${"%.1f".format(Locale.US, estimatedSampleRateHz)} Hz  LOW_RATE=$lowSampleRate
            LONG_DR_MODE=$longDrMode
            CSV=${logFile?.name ?: "N/A"}
        """.trimIndent()
    }

    // ============================================================
    // QUATERNION HELPERS
    // ============================================================

    private fun normalizeQuaternion() {
        val norm = sqrt(
            qW * qW + qX * qX + qY * qY + qZ * qZ
        )
        if (norm < 1e-9f) {
            qW = 1f
            qX = 0f
            qY = 0f
            qZ = 0f
            return
        }
        qW /= norm
        qX /= norm
        qY /= norm
        qZ /= norm
    }

    private fun setQuaternionFromRotationMatrix(m: FloatArray) {
        // Standard matrix -> quaternion conversion.
        val trace = m[0] + m[4] + m[8]
        if (trace > 0f) {
            val s = sqrt(trace + 1f) * 2f
            qW = 0.25f * s
            qX = (m[7] - m[5]) / s
            qY = (m[2] - m[6]) / s
            qZ = (m[3] - m[1]) / s
        } else if (m[0] > m[4] && m[0] > m[8]) {
            val s = sqrt(1f + m[0] - m[4] - m[8]) * 2f
            qW = (m[7] - m[5]) / s
            qX = 0.25f * s
            qY = (m[1] + m[3]) / s
            qZ = (m[2] + m[6]) / s
        } else if (m[4] > m[8]) {
            val s = sqrt(1f + m[4] - m[0] - m[8]) * 2f
            qW = (m[2] - m[6]) / s
            qX = (m[1] + m[3]) / s
            qY = 0.25f * s
            qZ = (m[5] + m[7]) / s
        } else {
            val s = sqrt(1f + m[8] - m[0] - m[4]) * 2f
            qW = (m[3] - m[1]) / s
            qX = (m[2] + m[6]) / s
            qY = (m[5] + m[7]) / s
            qZ = 0.25f * s
        }
        normalizeQuaternion()
    }

    private fun setQuaternionFromEuler(yawDeg: Float, pitchDeg: Float, rollDeg: Float) {
        val yaw = yawDeg * DEG_TO_RAD * 0.5f
        val pitch = pitchDeg * DEG_TO_RAD * 0.5f
        val roll = rollDeg * DEG_TO_RAD * 0.5f

        val cy = cos(yaw.toDouble()).toFloat()
        val sy = kotlin.math.sin(yaw.toDouble()).toFloat()
        val cp = cos(pitch.toDouble()).toFloat()
        val sp = kotlin.math.sin(pitch.toDouble()).toFloat()
        val cr = cos(roll.toDouble()).toFloat()
        val sr = kotlin.math.sin(roll.toDouble()).toFloat()

        qW = cr * cp * cy + sr * sp * sy
        qX = sr * cp * cy - cr * sp * sy
        qY = cr * sp * cy + sr * cp * sy
        qZ = cr * cp * sy - sr * sp * cy
        normalizeQuaternion()
    }

    private fun rotateVectorByQuaternionInverse(x: Float, y: Float, z: Float): FloatArray {
        // q^-1 * v * q. q maps phone/body -> world, so this converts the
        // world gravity vector [0, 0, g] back into the phone frame.
        val iw = -qX * x - qY * y - qZ * z
        val ix = qW * x + qY * z - qZ * y
        val iy = qW * y + qZ * x - qX * z
        val iz = qW * z + qX * y - qY * x

        val rx = ix * qW + iw * -qX + iy * -qZ - iz * -qY
        val ry = iy * qW + iw * -qY + iz * -qX - ix * -qZ
        val rz = iz * qW + iw * -qZ + ix * -qY - iy * -qX
        return floatArrayOf(rx, ry, rz)
    }

    // ============================================================
    // MATRIX / ANGLE HELPERS
    // ============================================================

    private fun createRotationMatrix(
        yawDegrees: Float,
        pitchDegrees: Float,
        rollDegrees: Float
    ): FloatArray {
        val yaw = yawDegrees * DEG_TO_RAD
        val pitch = pitchDegrees * DEG_TO_RAD
        val roll = rollDegrees * DEG_TO_RAD

        val cy = cos(yaw.toDouble()).toFloat()
        val sy = kotlin.math.sin(yaw.toDouble()).toFloat()
        val cp = cos(pitch.toDouble()).toFloat()
        val sp = kotlin.math.sin(pitch.toDouble()).toFloat()
        val cr = cos(roll.toDouble()).toFloat()
        val sr = kotlin.math.sin(roll.toDouble()).toFloat()

        return floatArrayOf(
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr
        )
    }

    private fun multiply3x3Vector(m: FloatArray, x: Float, y: Float, z: Float): FloatArray {
        return floatArrayOf(
            m[0] * x + m[1] * y + m[2] * z,
            m[3] * x + m[4] * y + m[5] * z,
            m[6] * x + m[7] * y + m[8] * z
        )
    }

    private fun normalizeAngle(angle: Float): Float {
        var result = angle
        while (result > 180f) result -= 360f
        while (result < -180f) result += 360f
        return result
    }

    private fun normalizeLongitude(longitude: Double): Double {
        var result = longitude
        while (result > 180.0) result -= 360.0
        while (result < -180.0) result += 360.0
        return result
    }

    private fun currentVehicleHeadingDegrees(): Float {
        return if (gnssHeadingDeg.isNaN()) {
            normalizeAngle(orientationYawDeg + yawOffsetDeg)
        } else {
            normalizeAngle(orientationYawDeg + yawOffsetDeg)
        }
    }

    private fun setVelocity(mps: Float) {
        velocityX = mps.coerceAtLeast(0f)
        velocityY = 0f
        velocityZ = 0f
        speedMps = velocityX
    }

    private fun gyroBiasMagnitude(): Float {
        return sqrt(
            gyroBiasX * gyroBiasX +
                    gyroBiasY * gyroBiasY +
                    gyroBiasZ * gyroBiasZ
        )
    }

    private fun validLatitude(latitude: Double): Boolean = latitude in -90.0..90.0

    private fun validLongitude(longitude: Double): Boolean = longitude in -180.0..180.0

    private fun identity3(): FloatArray = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    )

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No special action required; the confidence score already represents
        // runtime trust in the IMU stream.
    }
}
