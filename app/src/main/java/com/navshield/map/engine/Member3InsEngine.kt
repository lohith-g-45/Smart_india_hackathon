package com.navshield.map.engine

import com.navshield.map.contract.Member3Result
import com.navshield.map.contract.NavShieldSensorState
import java.util.Locale
import kotlin.math.*

/**
 * PRODUCTION-GRADE INERTIAL NAVIGATION SYSTEM (INS) ENGINE.
 * Integrated from Member 3 Module.
 */
class Member3InsEngine {

    companion object {
        private const val G = 9.80665f
        private const val DEG_TO_RAD = (PI / 180.0).toFloat()
        private const val RAD_TO_DEG = (180.0 / PI).toFloat()
        private const val NS_TO_SEC = 1.0f / 1_000_000_000.0f
        
        private const val CALIBRATION_MS = 3000L
        private const val ACCEL_ALPHA = 0.9f
        private const val MOTION_WINDOW_SIZE = 50
        
        private const val STATIONARY_GYRO_THRESHOLD = 0.05f
        private const val STATIONARY_ACCEL_ERROR = 0.15f
        private const val ENGINE_VIBRATION_STD = 0.04f
        private const val DISTURBANCE_GYRO_THRESHOLD = 1.5f
        private const val POTHOLE_DEVIATION = 5.0f
        private const val POTHOLE_JERK = 150.0f
        private const val HARD_BRAKING_THRESHOLD = -2.5f
        private const val TURN_RATE_THRESHOLD = 0.4f
        private const val ACCELERATION_THRESHOLD = 1.5f
        private const val LONG_DR_SECONDS = 60.0f
        
        private const val EARTH_RADIUS = 6378137.0
    }

    private var status = MemberStatus.WAITING_FOR_MEMBER
    private var isCalibrated = false
    private var calibrationStartNs = 0L
    
    private var curAccelX = 0f; private var curAccelY = 0f; private var curAccelZ = 0f
    private var curGyroX = 0f; private var curGyroY = 0f; private var curGyroZ = 0f

    private var accelCalibrationSamples = 0
    private var gyroCalibrationSamples = 0
    private var accelSumX = 0.0; private var accelSumY = 0.0; private var accelSumZ = 0.0
    private var gyroSumX = 0.0; private var gyroSumY = 0.0; private var gyroSumZ = 0.0
    
    private var accelBiasX = 0f; private var accelBiasY = 0f; private var accelBiasZ = 0f
    private var gyroBiasX = 0f; private var gyroBiasY = 0f; private var gyroBiasZ = 0f
    
    private var filteredAccelX = 0f; private var filteredAccelY = 0f; private var filteredAccelZ = 0f
    private var accelFilterInitialized = false
    private var gyroFilterInitialized = false

    private var qW = 1f; private var qX = 0f; private var qY = 0f; private var qZ = 0f
    private var quaternionInitialized = false
    private var orientationYawDeg = 0f; private var orientationPitchDeg = 0f; private var orientationRollDeg = 0f
    
    private var hasAlignment = false
    private var gnssHeadingDeg = Float.NaN
    private var yawOffsetDeg = 0f
    private var alignmentPitchDeg = 0f
    private var alignmentRollDeg = 0f
    private var phoneToVehicleMatrix = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
    
    private var vehicleAccelX = 0f; private var vehicleAccelY = 0f; private var vehicleAccelZ = 0f
    private var velocityX = 0f; private var velocityY = 0f; private var velocityZ = 0f
    private var speedMps = 0f
    private var displacementX = 0f; private var displacementY = 0f; private var displacementZ = 0f
    private var displacementMeters = 0f
    
    private var drLatitude = Double.NaN
    private var drLongitude = Double.NaN
    private var originLatitude = Double.NaN
    private var originLongitude = Double.NaN
    
    private var lastAccelTimestampNs = 0L
    private var lastGyroTimestampNs = 0L
    private var currentDtSec = 0f
    
    private var motionState = "STATIONARY"
    private var vibrationLevel = 0f
    private val accelMagnitudeWindow = FloatArray(MOTION_WINDOW_SIZE)
    private var motionWindowIndex = 0
    private var motionWindowCount = 0
    private var previousAccelMagnitude = G
    private var zuptApplied = false
    private var roadGradeDeg = 0f
    private var imuConfidence = 0f
    private var gnssOutageStartNs = 0L
    private var longDrMode = false
    private var biasStability = 1f
    private var previousGyroBiasMagnitude = 0f

    fun initialize(timestampMs: Long) {
        calibrationStartNs = timestampMs * 1_000_000L
        gnssOutageStartNs = calibrationStartNs
        status = MemberStatus.CONNECTED
    }

    fun update(sensorState: NavShieldSensorState): Member3Result? {
        if (status != MemberStatus.CONNECTED) return null
        val timestampNs = sensorState.timestampMillis * 1_000_000L
        curAccelX = sensorState.accelX.toFloat(); curAccelY = sensorState.accelY.toFloat(); curAccelZ = sensorState.accelZ.toFloat()
        curGyroX = sensorState.gyroX.toFloat(); curGyroY = sensorState.gyroY.toFloat(); curGyroZ = sensorState.gyroZ.toFloat()

        processAccelerometer(timestampNs)
        processGyroscope(timestampNs)
        processRotationVector(sensorState)
        classifyMotion()
        updateConfidence(timestampNs)
        return getOutput()
    }

    fun setGnssHeading(headingDegrees: Float, timestampNs: Long = System.nanoTime()) {
        if (!headingDegrees.isFinite()) return
        gnssHeadingDeg = normalizeAngle(headingDegrees)
        gnssOutageStartNs = timestampNs
        longDrMode = false
        updateAlignment()
    }

    fun setInitialPosition(latitude: Double, longitude: Double) {
        drLatitude = latitude; drLongitude = longitude
        originLatitude = latitude; originLongitude = longitude
        displacementX = 0f; displacementY = 0f; displacementZ = 0f; displacementMeters = 0f
    }

    fun resetFromExternalState(lat: Double, lon: Double, heading: Float?, velocityMps: Float?, timestampMs: Long = System.currentTimeMillis()) {
        setInitialPosition(lat, lon)
        if (heading != null) setGnssHeading(heading, timestampMs * 1_000_000L)
        if (velocityMps != null) {
            velocityX = velocityMps.coerceAtLeast(0f); velocityY = 0f; velocityZ = 0f; speedMps = velocityX
        }
    }

    fun setRoadGrade(degrees: Float) {
        if (degrees.isFinite()) roadGradeDeg = degrees.coerceIn(-45f, 45f)
    }

    private fun processAccelerometer(timestampNs: Long) {
        if (!isCalibrated) {
            accelSumX += curAccelX; accelSumY += curAccelY; accelSumZ += curAccelZ; accelCalibrationSamples++
        }
        if (!accelFilterInitialized) {
            filteredAccelX = curAccelX; filteredAccelY = curAccelY; filteredAccelZ = curAccelZ; accelFilterInitialized = true
        } else {
            filteredAccelX = ACCEL_ALPHA * filteredAccelX + (1f - ACCEL_ALPHA) * curAccelX
            filteredAccelY = ACCEL_ALPHA * filteredAccelY + (1f - ACCEL_ALPHA) * curAccelY
            filteredAccelZ = ACCEL_ALPHA * filteredAccelZ + (1f - ACCEL_ALPHA) * curAccelZ
        }
        val magnitude = sqrt(curAccelX * curAccelX + curAccelY * curAccelY + curAccelZ * curAccelZ)
        pushMotionMagnitude(magnitude)
        if (lastAccelTimestampNs > 0L) {
            val dt = (timestampNs - lastAccelTimestampNs) * NS_TO_SEC
            if (dt in 0.0001f..0.2f) currentDtSec = dt
        }
        lastAccelTimestampNs = timestampNs
        if (isCalibrated) updateLinearAccelerationAndKinematics()
        finishCalibrationIfReady(timestampNs)
    }

    private fun processGyroscope(timestampNs: Long) {
        if (!isCalibrated) {
            gyroSumX += curGyroX; gyroSumY += curGyroY; gyroSumZ += curGyroZ; gyroCalibrationSamples++
        }
        if (lastGyroTimestampNs != 0L) {
            var dt = (timestampNs - lastGyroTimestampNs) * NS_TO_SEC
            if (dt <= 0f || dt > 0.2f) dt = 1f / 50f
            if (isCalibrated) integrateGyroscope(curGyroX - gyroBiasX, curGyroY - gyroBiasY, curGyroZ - gyroBiasZ, dt)
        }
        lastGyroTimestampNs = timestampNs
        gyroFilterInitialized = true
    }

    private fun processRotationVector(s: NavShieldSensorState) {
        if (!quaternionInitialized && !s.orientationYaw.isNaN()) {
            orientationYawDeg = s.orientationYaw.toFloat()
            orientationPitchDeg = s.orientationPitch.toFloat()
            orientationRollDeg = s.orientationRoll.toFloat()
            setQuaternionFromEuler(orientationYawDeg, orientationPitchDeg, orientationRollDeg)
            quaternionInitialized = true; updateAlignment()
        }
    }

    private fun updateAlignment() {
        if (!quaternionInitialized) return
        updateOrientationEuler()
        if (gnssHeadingDeg.isNaN()) {
            hasAlignment = false; yawOffsetDeg = 0f; phoneToVehicleMatrix = floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f)
            return
        }
        yawOffsetDeg = normalizeAngle(gnssHeadingDeg - orientationYawDeg)
        if (!hasAlignment) { alignmentPitchDeg = orientationPitchDeg; alignmentRollDeg = orientationRollDeg }
        hasAlignment = true; phoneToVehicleMatrix = createRotationMatrix(yawOffsetDeg, -alignmentPitchDeg, -alignmentRollDeg)
    }

    private fun integrateGyroscope(gx: Float, gy: Float, gz: Float, dt: Float) {
        val halfDt = 0.5f * dt
        val dw = 1f; val dx = gx * halfDt; val dy = gy * halfDt; val dz = gz * halfDt
        val nw = qW * dw - qX * dx - qY * dy - qZ * dz; val nx = qW * dx + qX * dw + qY * dz - qZ * dy
        val ny = qW * dy - qX * dz + qY * dw + qZ * dx; val nz = qW * dz + qX * dy - qY * dx + qZ * dw
        qW = nw; qX = nx; qY = ny; qZ = nz; normalizeQuaternion(); updateOrientationEuler()
    }

    private fun updateOrientationEuler() {
        val sinr = 2f * (qW * qX + qY * qZ); val cosr = 1f - 2f * (qX * qX + qY * qY)
        orientationRollDeg = atan2(sinr.toDouble(), cosr.toDouble()).toFloat() * RAD_TO_DEG
        val sinp = 2f * (qW * qY - qZ * qX)
        orientationPitchDeg = if (abs(sinp) >= 1f) (if (sinp >= 0f) 90f else -90f) else asin(sinp.toDouble()).toFloat() * RAD_TO_DEG
        val siny = 2f * (qW * qZ + qX * qY); val cosy = 1f - 2f * (qY * qY + qZ * qZ)
        orientationYawDeg = normalizeAngle(atan2(siny.toDouble(), cosy.toDouble()).toFloat() * RAD_TO_DEG)
    }

    private fun updateLinearAccelerationAndKinematics() {
        if (!hasAlignment) return
        val gPhone = rotateVectorByQuaternionInverse(0f, 0f, G)
        val v = multiply3x3Vector(phoneToVehicleMatrix, curAccelX - gPhone[0], curAccelY - gPhone[1], curAccelZ - gPhone[2])
        vehicleAccelX = v[0]; vehicleAccelY = v[1]; vehicleAccelZ = v[2]
        if (abs(roadGradeDeg) > 0.01f) vehicleAccelX -= G * sin(roadGradeDeg * DEG_TO_RAD)
        val dt = currentDtSec.coerceIn(0.001f, 0.2f)
        zuptApplied = (motionState == "STATIONARY" && vibrationLevel < 0.45f)
        if (zuptApplied) { velocityX = 0f; velocityY = 0f; velocityZ = 0f } else {
            velocityX += vehicleAccelX * dt; velocityY += vehicleAccelY * dt; velocityZ += vehicleAccelZ * dt
            if (vehicleAccelX < HARD_BRAKING_THRESHOLD && velocityX < 0f) velocityX = 0f
        }
        speedMps = sqrt(velocityX*velocityX + velocityY*velocityY + velocityZ*velocityZ)
        if (speedMps < 0.05f && zuptApplied) speedMps = 0f
        displacementX += velocityX * dt; displacementY += velocityY * dt; displacementZ += velocityZ * dt
        displacementMeters = sqrt(displacementX*displacementX + displacementY*displacementY + displacementZ*displacementZ)
        updateDeadReckoningPosition()
    }

    private fun updateDeadReckoningPosition() {
        if (originLatitude.isNaN() || originLongitude.isNaN() || gnssHeadingDeg.isNaN() || !hasAlignment) return
        val vehicleHeadingDeg = normalizeAngle(orientationYawDeg + yawOffsetDeg)
        val hRad = vehicleHeadingDeg * DEG_TO_RAD
        val northM = displacementX * cos(hRad.toDouble()) - displacementY * sin(hRad.toDouble())
        val eastM = displacementX * sin(hRad.toDouble()) + displacementY * cos(hRad.toDouble())
        val projected = haversineForward(originLatitude, originLongitude, northM, eastM)
        drLatitude = projected.first; drLongitude = projected.second
    }

    private fun haversineForward(lat: Double, lon: Double, dN: Double, dE: Double): Pair<Double, Double> {
        val lat1 = lat * DEG_TO_RAD.toDouble(); val lon1 = lon * DEG_TO_RAD.toDouble()
        val dist = sqrt(dN*dN + dE*dE); if (dist < 1e-12) return lat to lon
        val angDist = dist / EARTH_RADIUS; val bearing = atan2(dE, dN)
        val lat2 = asin(sin(lat1) * cos(angDist) + cos(lat1) * sin(angDist) * cos(bearing))
        val lon2 = lon1 + atan2(sin(bearing) * sin(angDist) * cos(lat1), cos(angDist) - sin(lat1) * sin(lat2))
        return (lat2 * RAD_TO_DEG.toDouble()) to normalizeLongitude(lon2 * RAD_TO_DEG.toDouble())
    }

    private fun pushMotionMagnitude(m: Float) {
        accelMagnitudeWindow[motionWindowIndex] = m; motionWindowIndex = (motionWindowIndex + 1) % MOTION_WINDOW_SIZE; motionWindowCount = min(motionWindowCount + 1, MOTION_WINDOW_SIZE)
    }

    private fun classifyMotion() {
        val accMag = sqrt(curAccelX*curAccelX + curAccelY*curAccelY + curAccelZ*curAccelZ)
        val stdDev = if (motionWindowCount < 2) 0f else {
            val avg = accelMagnitudeWindow.take(motionWindowCount).average()
            sqrt(accelMagnitudeWindow.take(motionWindowCount).sumOf { (it - avg).pow(2) } / (motionWindowCount - 1)).toFloat()
        }
        vibrationLevel = (stdDev / 2.5f).coerceIn(0f, 1f)
        val stationary = abs(accMag - G) < STATIONARY_ACCEL_ERROR && sqrt(curGyroX*curGyroX + curGyroY*curGyroY + curGyroZ*curGyroZ) < STATIONARY_GYRO_THRESHOLD && stdDev < ENGINE_VIBRATION_STD
        motionState = when {
            sqrt(curGyroX*curGyroX + curGyroY*curGyroY + curGyroZ*curGyroZ) > DISTURBANCE_GYRO_THRESHOLD && abs(accMag - G) > 2.5f -> "PHONE_DISTURBANCE"
            abs(accMag - G) > POTHOLE_DEVIATION && (if(currentDtSec>0.001f) abs(accMag-previousAccelMagnitude)/currentDtSec else 0f) > POTHOLE_JERK -> "POTHOLE"
            stationary -> "STATIONARY"
            vehicleAccelX < HARD_BRAKING_THRESHOLD -> "BRAKING"
            abs(curGyroZ) > TURN_RATE_THRESHOLD -> "TURNING"
            vehicleAccelX > ACCELERATION_THRESHOLD -> "ACCELERATING"
            else -> "NORMAL"
        }
        previousAccelMagnitude = accMag
    }

    private fun updateConfidence(tsNs: Long) {
        var conf = 1f; conf *= (1f - 0.45f * vibrationLevel.coerceIn(0f, 1f)); conf *= (1f - 0.35f * (1f - biasStability))
        if (gnssHeadingDeg.isNaN()) conf = 0f
        val outageS = (tsNs - gnssOutageStartNs) * NS_TO_SEC
        if (outageS > LONG_DR_SECONDS) { longDrMode = true; conf *= exp(-(outageS - LONG_DR_SECONDS) / LONG_DR_SECONDS) }
        imuConfidence = conf.coerceIn(0f, 1f)
        val biasMag = sqrt(gyroBiasX*gyroBiasX + gyroBiasY*gyroBiasY + gyroBiasZ*gyroBiasZ)
        biasStability = (1f - (abs(biasMag - previousGyroBiasMagnitude) / 0.05f)).coerceIn(0f, 1f); previousGyroBiasMagnitude = biasMag
    }

    private fun finishCalibrationIfReady(tsNs: Long) {
        if (isCalibrated || tsNs - calibrationStartNs < CALIBRATION_MS * 1_000_000L) return
        if (accelCalibrationSamples == 0 || gyroCalibrationSamples == 0) return
        accelBiasX = (accelSumX / accelCalibrationSamples).toFloat(); accelBiasY = (accelSumY / accelCalibrationSamples).toFloat(); accelBiasZ = (accelSumZ / accelCalibrationSamples).toFloat()
        gyroBiasX = (gyroSumX / gyroCalibrationSamples).toFloat(); gyroBiasY = (gyroSumY / gyroCalibrationSamples).toFloat(); gyroBiasZ = (gyroSumZ / gyroCalibrationSamples).toFloat()
        isCalibrated = true; previousGyroBiasMagnitude = sqrt(gyroBiasX*gyroBiasX + gyroBiasY*gyroBiasY + gyroBiasZ*gyroBiasZ); updateAlignment()
    }

    private fun setQuaternionFromEuler(y: Float, p: Float, r: Float) {
        val yaw = y * DEG_TO_RAD * 0.5f; val pitch = p * DEG_TO_RAD * 0.5f; val roll = r * DEG_TO_RAD * 0.5f
        val cy = cos(yaw.toDouble()).toFloat(); val sy = sin(yaw.toDouble()).toFloat()
        val cp = cos(pitch.toDouble()).toFloat(); val sp = sin(pitch.toDouble()).toFloat()
        val cr = cos(roll.toDouble()).toFloat(); val sr = sin(roll.toDouble()).toFloat()
        qW = cr * cp * cy + sr * sp * sy; qX = sr * cp * cy - cr * sp * sy; qY = cr * sp * cy + sr * cp * sy; qZ = cr * cp * sy - sr * sp * cy; normalizeQuaternion()
    }

    private fun rotateVectorByQuaternionInverse(x: Float, y: Float, z: Float): FloatArray {
        val iw = -qX * x - qY * y - qZ * z; val ix = qW * x + qY * z - qZ * y; val iy = qW * y + qZ * x - qX * z; val iz = qW * z + qX * y - qY * x
        val rx = ix * qW + iw * -qX + iy * -qZ - iz * -qY; val ry = iy * qW + iw * -qY + iz * -qX - ix * -qZ; val rz = iz * qW + iw * -qZ + ix * -qY - iy * -qX
        return floatArrayOf(rx, ry, rz)
    }

    private fun createRotationMatrix(yD: Float, pD: Float, rD: Float): FloatArray {
        val y = yD * DEG_TO_RAD; val p = pD * DEG_TO_RAD; val r = rD * DEG_TO_RAD
        val cy = cos(y.toDouble()).toFloat(); val sy = sin(y.toDouble()).toFloat(); val cp = cos(p.toDouble()).toFloat(); val sp = sin(p.toDouble()).toFloat(); val cr = cos(r.toDouble()).toFloat(); val sr = sin(r.toDouble()).toFloat()
        return floatArrayOf(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, -sp, cp*sr, cp*cr)
    }

    private fun multiply3x3Vector(m: FloatArray, x: Float, y: Float, z: Float): FloatArray {
        return floatArrayOf(m[0]*x + m[1]*y + m[2]*z, m[3]*x + m[4]*y + m[5]*z, m[6]*x + m[7]*y + m[8]*z)
    }

    private fun normalizeQuaternion() {
        val n = sqrt(qW*qW + qX*qX + qY*qY + qZ*qZ)
        if (n < 1e-9f) { qW=1f; qX=0f; qY=0f; qZ=0f } else { qW/=n; qX/=n; qY/=n; qZ/=n }
    }

    private fun normalizeAngle(a: Float): Float {
        var r = a; while (r > 180f) r -= 360f; while (r < -180f) r += 360f; return r
    }

    private fun normalizeLongitude(l: Double): Double {
        var r = l; while (r > 180.0) r -= 360.0; while (r < -180.0) r += 360.0; return r
    }

    private fun getOutput() = Member3Result(
        timestampMs = System.currentTimeMillis(),
        imuAvailable = accelFilterInitialized && gyroFilterInitialized,
        imuLatitude = drLatitude, imuLongitude = drLongitude,
        imuHeadingDeg = normalizeAngle(orientationYawDeg + yawOffsetDeg),
        imuVelocityMps = speedMps, imuDisplacementM = displacementMeters,
        imuConfidence = imuConfidence, motionState = motionState, vibrationLevel = vibrationLevel,
        zuptApplied = zuptApplied, accelBiasX = accelBiasX, accelBiasY = accelBiasY, accelBiasZ = accelBiasZ,
        gyroBiasX = gyroBiasX, gyroBiasY = gyroBiasY, gyroBiasZ = gyroBiasZ
    )
}
