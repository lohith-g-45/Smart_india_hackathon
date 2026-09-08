package com.navshield.map.engine

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import androidx.core.content.ContextCompat
import android.content.pm.PackageManager
import android.Manifest
import com.navshield.map.contract.NavShieldSensorState
import com.navshield.map.members.SensorFusionSource
import kotlin.math.abs

/**
 * Production Android implementation of Member 2: Sensor Fusion (GNSS + IMU).
 * Uses LocationManager for GNSS and SensorManager for IMU.
 */
class AndroidSensorFusionSource(private val context: Context) : SensorFusionSource, LocationListener, SensorEventListener {

    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    
    private var lastLocation: Location? = null
    private var satelliteCount: Int = 0
    
    private var lastAccel: FloatArray = FloatArray(3)
    private var lastGyro: FloatArray = FloatArray(3)
    private var lastMag: FloatArray = FloatArray(3) { Float.NaN }
    private var lastGravity: FloatArray = FloatArray(3) { Float.NaN }
    private var lastOrientation: FloatArray = FloatArray(3) { Float.NaN }
    
    private var status: MemberStatus = MemberStatus.WAITING_FOR_MEMBER

    private val gnssStatusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            var count = 0
            for (i in 0 until status.satelliteCount) {
                if (status.usedInFix(i)) count++
            }
            satelliteCount = count
        }
    }

    /**
     * Checks for necessary permissions and starts sensor updates.
     */
    @SuppressLint("MissingPermission")
    fun start() {
        if (!hasLocationPermission()) {
            status = MemberStatus.ERROR
            return
        }

        try {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100L, 0f, this)
            locationManager.registerGnssStatusCallback(gnssStatusCallback, null)
            
            val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
            val mag = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
            val gravity = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
            val rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            
            sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_FASTEST)
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST)
            
            if (mag != null) sensorManager.registerListener(this, mag, SensorManager.SENSOR_DELAY_FASTEST)
            if (gravity != null) sensorManager.registerListener(this, gravity, SensorManager.SENSOR_DELAY_FASTEST)
            if (rotationVector != null) sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_FASTEST)
            
            status = MemberStatus.CONNECTED
        } catch (e: Exception) {
            status = MemberStatus.ERROR
        }
    }

    fun stop() {
        locationManager.removeUpdates(this)
        locationManager.unregisterGnssStatusCallback(gnssStatusCallback)
        sensorManager.unregisterListener(this)
        status = MemberStatus.WAITING_FOR_MEMBER
    }

    override fun getCurrentState(): NavShieldSensorState? {
        val loc = lastLocation ?: return null
        
        val anomalyReasons = mutableListOf<String>()
        
        // Phase E: Plausibility Checks
        if (loc.accuracy > 50.0) {
            anomalyReasons.add("accuracy_exceeds_50m")
        }
        if (satelliteCount == 0) {
            anomalyReasons.add("satellite_lock_lost")
        }
        
        // Speed inconsistency check (simplified)
        // In production, we'd compare loc.speed with implied speed from displacement
        
        val isAnomalous = anomalyReasons.isNotEmpty()
        
        // Determine GNSS Mode (Phase F)
        val mode = when {
            loc.accuracy > 50.0 -> "DENIED"
            isAnomalous -> "ANOMALOUS"
            loc.accuracy < 10.0 -> "STRONG"
            else -> "WEAK"
        }

        return NavShieldSensorState(
            latitude = loc.latitude,
            longitude = loc.longitude,
            altitude = loc.altitude,
            speed = loc.speed.toDouble(),
            bearing = loc.bearing.toDouble(),
            accuracy = loc.accuracy.toDouble(),
            satellites = satelliteCount,
            confidence = calculateConfidence(loc, isAnomalous),
            isAnomalous = isAnomalous,
            anomalyReasons = anomalyReasons,
            navigationMode = mode,
            accelX = lastAccel[0].toDouble(),
            accelY = lastAccel[1].toDouble(),
            accelZ = lastAccel[2].toDouble(),
            gyroX = lastGyro[0].toDouble(),
            gyroY = lastGyro[1].toDouble(),
            gyroZ = lastGyro[2].toDouble(),
            magneticFieldX = lastMag[0].toDouble(),
            magneticFieldY = lastMag[1].toDouble(),
            magneticFieldZ = lastMag[2].toDouble(),
            gravityX = lastGravity[0].toDouble(),
            gravityY = lastGravity[1].toDouble(),
            gravityZ = lastGravity[2].toDouble(),
            orientationYaw = lastOrientation[0].toDouble(),
            orientationPitch = lastOrientation[1].toDouble(),
            orientationRoll = lastOrientation[2].toDouble(),
            timestampMillis = loc.time
        )
    }

    private fun calculateConfidence(loc: Location, isAnomalous: Boolean): Double {
        var conf = 1.0
        if (isAnomalous) conf -= 0.3
        val accPenalty = (loc.accuracy / 50.0).coerceIn(0.0, 1.0)
        conf -= accPenalty * 0.4
        return conf.coerceIn(0.0, 1.0)
    }

    override fun getStatus(): MemberStatus = status

    // --- LocationListener ---

    override fun onLocationChanged(location: Location) {
        lastLocation = location
    }

    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {
        if (provider == LocationManager.GPS_PROVIDER) {
            status = MemberStatus.ERROR
        }
    }

    // --- SensorEventListener ---

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, lastAccel, 0, 3)
            }
            Sensor.TYPE_GYROSCOPE -> {
                System.arraycopy(event.values, 0, lastGyro, 0, 3)
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                System.arraycopy(event.values, 0, lastMag, 0, 3)
            }
            Sensor.TYPE_GRAVITY -> {
                System.arraycopy(event.values, 0, lastGravity, 0, 3)
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                val rotationMatrix = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                val orientationValues = FloatArray(3)
                SensorManager.getOrientation(rotationMatrix, orientationValues)
                
                // Convert radians to degrees for Member 5 contract
                lastOrientation[0] = Math.toDegrees(orientationValues[0].toDouble()).toFloat() // Azimuth / Yaw
                lastOrientation[1] = Math.toDegrees(orientationValues[1].toDouble()).toFloat() // Pitch
                lastOrientation[2] = Math.toDegrees(orientationValues[2].toDouble()).toFloat() // Roll
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
    }
}
