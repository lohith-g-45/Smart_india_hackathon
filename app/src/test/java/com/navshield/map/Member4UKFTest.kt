package com.navshield.map

import com.navshield.map.contract.CandidateRoad
import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult
import com.navshield.map.contract.NavShieldSensorState
import com.navshield.map.engine.Member4Pipeline
import com.navshield.map.engine.math.GeoUtils
import org.junit.Assert.*
import org.junit.Test

class Member4UKFTest {

    private fun createTestSensorState(lat: Double, lon: Double, heading: Double, speed: Double) = NavShieldSensorState(
        latitude = lat,
        longitude = lon,
        altitude = 0.0,
        speed = speed,
        bearing = heading,
        accuracy = 5.0,
        satellites = 10,
        confidence = 0.9,
        isAnomalous = false,
        anomalyReasons = emptyList(),
        navigationMode = "GNSS_DOMINANT",
        accelX = 0.0, accelY = 0.0, accelZ = 9.8,
        gyroX = 0.0, gyroY = 0.0, gyroZ = 0.0,
        timestampMillis = 1000L
    )

    @Test
    fun `UKF converges toward strong map measurement`() {
        val brain = Member4Pipeline()
        val t0 = 1000L
        
        // Starting at (12.0, 77.0)
        val sensorState = createTestSensorState(12.0, 77.0, 0.0, 10.0)
        brain.update(sensorState, null, null, null, t0)
        
        // Map says we are actually at (12.0001, 77.0001) with high confidence
        val mapResult = MapMatchResult(
            matchedLatitude = 12.0001,
            matchedLongitude = 77.0001,
            roadSegmentId = "R1",
            roadHeading = 0.0,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 0.9,
            candidateRoads = listOf(
                CandidateRoad("R1", 0.9, 12.0001, 77.0001, 0.0, 0.0, 0.0)
            )
        )
        
        var trustState = brain.update(sensorState, mapResult, null, null, t0 + 1000)
        
        // Run a few more cycles to allow hypothesis weight to grow and state to converge
        for (i in 1..5) {
            trustState = brain.update(sensorState, mapResult, null, null, t0 + 1000 + i * 100)
        }
        
        // State should have moved toward map position
        assertTrue("Latitude should increase toward map", trustState.final_latitude > 12.0)
        assertTrue("Longitude should increase toward map", trustState.final_longitude > 77.0)
        assertEquals("R1", trustState.active_hypothesis)
    }

    @Test
    fun `UKF ignores low-confidence map measurements`() {
        val brain = Member4Pipeline()
        val t0 = 1000L
        
        val sensorState = createTestSensorState(12.0, 77.0, 0.0, 0.0)
        brain.update(sensorState, null, null, null, t0)
        
        // Map says we are far away but with very low confidence
        val mapResult = MapMatchResult(
            matchedLatitude = 12.1,
            matchedLongitude = 77.1,
            roadSegmentId = "R2",
            roadHeading = 0.0,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 0.01, // Very low
            candidateRoads = listOf(
                CandidateRoad("R2", 0.01, 12.1, 77.1, 0.0, 0.0, 0.0)
            )
        )
        
        val trustState = brain.update(sensorState, mapResult, null, null, t0 + 1000)
        
        // Latitude should be very close to 12.0, not 12.1
        assertEquals(12.0, trustState.final_latitude, 0.001)
    }

    @Test
    fun `UKF maintains recursive state across updates`() {
        val brain = Member4Pipeline()
        
        // T=0
        val s1 = createTestSensorState(12.0, 77.0, 90.0, 10.0) // Heading North
        brain.update(s1, null, null, null, 0)
        
        // T=1s, no map, only prediction
        val state1 = brain.update(s1, null, null, null, 1000)
        
        // Vehicle should have moved North (Latitude increases)
        assertTrue(state1.final_latitude > 12.0)
        
        // T=2s
        val state2 = brain.update(s1, null, null, null, 2000)
        assertTrue(state2.final_latitude > state1.final_latitude)
    }

    @Test
    fun `UKF reset clears state`() {
        val brain = Member4Pipeline()
        val s1 = createTestSensorState(12.0, 77.0, 0.0, 10.0)
        brain.update(s1, null, null, null, 0)
        
        brain.reset()
        
        // Next update will be a fresh initialization at new position
        val s2 = createTestSensorState(13.0, 78.0, 0.0, 0.0)
        val state = brain.update(s2, null, null, null, 1000)
        assertEquals(13.0, state.final_latitude, 0.0001)
    }
}
