package com.navshield.map.contract

/**
 * Complete Member 2 sensor output representing the state of the vehicle's sensors.
 * This class is used by the Trust Brain (Member 4) for high-fidelity state estimation.
 */
data class NavShieldSensorState(
    // GNSS
    val latitude: Double,
    val longitude: Double,
    val altitude: Double,
    val speed: Double,
    val bearing: Double,
    val accuracy: Double,
    val satellites: Int,

    // GNSS quality
    val confidence: Double,
    val isAnomalous: Boolean,
    val anomalyReasons: List<String>,
    val navigationMode: String,

    // IMU
    val accelX: Double,
    val accelY: Double,
    val accelZ: Double,
    val gyroX: Double,
    val gyroY: Double,
    val gyroZ: Double,

    // Magnetometer (µT)
    val magneticFieldX: Double = Double.NaN,
    val magneticFieldY: Double = Double.NaN,
    val magneticFieldZ: Double = Double.NaN,

    // Gravity (m/s²)
    val gravityX: Double = Double.NaN,
    val gravityY: Double = Double.NaN,
    val gravityZ: Double = Double.NaN,

    // Orientation (degrees) - calculated from Rotation Vector
    val orientationYaw: Double = Double.NaN,
    val orientationPitch: Double = Double.NaN,
    val orientationRoll: Double = Double.NaN,

    // Metadata
    val timestampMillis: Long
) {
    /**
     * Derives the backward-compatible MapMatchQuery required by Member 1.
     */
    fun toMapMatchQuery(): MapMatchQuery {
        return MapMatchQuery(
            latitude = latitude,
            longitude = longitude,
            heading = bearing,
            speed = speed
        )
    }
}
