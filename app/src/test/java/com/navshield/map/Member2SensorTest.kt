package com.navshield.map

import com.navshield.map.contract.NavShieldSensorState
import com.navshield.map.contract.MapMatchQuery
import org.junit.Assert.*
import org.junit.Test

class Member2SensorTest {

    @Test
    fun `NavShieldSensorState converts to MapMatchQuery correctly`() {
        val sensorState = NavShieldSensorState(
            latitude = 12.9716,
            longitude = 77.5946,
            altitude = 920.0,
            speed = 15.5,
            bearing = 45.0,
            accuracy = 3.2,
            satellites = 12,
            confidence = 0.95,
            isAnomalous = false,
            anomalyReasons = emptyList(),
            navigationMode = "STRONG",
            accelX = 0.1, accelY = 0.2, accelZ = 9.8,
            gyroX = 0.01, gyroY = 0.02, gyroZ = 0.03,
            timestampMillis = 1625097600000L
        )

        val query = sensorState.toMapMatchQuery()

        assertEquals(12.9716, query.latitude, 1e-6)
        assertEquals(77.5946, query.longitude, 1e-6)
        assertEquals(45.0, query.heading, 1e-6)
        assertEquals(15.5, query.speed, 1e-6)
    }

    @Test
    fun `NavShieldSensorState supports extended sensors`() {
        val sensorState = NavShieldSensorState(
            latitude = 0.0, longitude = 0.0, altitude = 0.0, speed = 0.0, bearing = 0.0,
            accuracy = 0.0, satellites = 0, confidence = 1.0, isAnomalous = false,
            anomalyReasons = emptyList(), navigationMode = "STRONG",
            accelX = 1.0, accelY = 2.0, accelZ = 3.0,
            gyroX = 0.1, gyroY = 0.2, gyroZ = 0.3,
            magneticFieldX = 10.0, magneticFieldY = 20.0, magneticFieldZ = 30.0,
            gravityX = 0.0, gravityY = 0.0, gravityZ = 9.8,
            orientationYaw = 45.0, orientationPitch = 10.0, orientationRoll = -5.0,
            timestampMillis = 1000L
        )

        assertEquals(10.0, sensorState.magneticFieldX, 1e-6)
        assertEquals(9.8, sensorState.gravityZ, 1e-6)
        assertEquals(45.0, sensorState.orientationYaw, 1e-6)
    }

    @Test
    fun `NavShieldSensorState defaults new fields to NaN`() {
        val sensorState = NavShieldSensorState(
            latitude = 0.0, longitude = 0.0, altitude = 0.0, speed = 0.0, bearing = 0.0,
            accuracy = 0.0, satellites = 0, confidence = 1.0, isAnomalous = false,
            anomalyReasons = emptyList(), navigationMode = "STRONG",
            accelX = 0.0, accelY = 0.0, accelZ = 0.0,
            gyroX = 0.0, gyroY = 0.0, gyroZ = 0.0,
            timestampMillis = 0L
        )

        assertTrue(sensorState.magneticFieldX.isNaN())
        assertTrue(sensorState.gravityX.isNaN())
        assertTrue(sensorState.orientationYaw.isNaN())
    }
}
