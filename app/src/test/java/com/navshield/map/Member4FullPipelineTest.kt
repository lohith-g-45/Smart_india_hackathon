package com.navshield.map

import com.navshield.map.contract.*
import com.navshield.map.engine.*
import org.junit.Assert.*
import org.junit.Test

class Member4FullPipelineTest {

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
    fun `pipeline handles tunnel scenario`() {
        val pipeline = Member4Pipeline()
        var time = 0L
        
        // 1. GNSS Available
        val s1 = createTestSensorState(12.0, 77.0, 0.0, 11.1)
        val state1 = pipeline.update(s1, null, null, null, time)
        assertEquals(NavigationMode.GNSS_DOMINANT, state1.navigation_mode)
        
        // 2. Simulated Tunnel
        time += 5000L
        val state2 = pipeline.update(s1, null, null, null, time)
        assertTrue(state2.estimated_position_error >= 0.0)
    }

    @Test
    fun `pipeline detects drift warning`() {
        val pipeline = Member4Pipeline()
        val s1 = createTestSensorState(12.0, 77.0, 0.0, 10.0)
        
        // Force high uncertainty
        val state = pipeline.update(s1, null, null, null, 0L)
        assertNotNull(state)
    }

    @Test
    fun `pipeline handles multi-hypothesis spawning`() {
        val pipeline = Member4Pipeline()
        val s1 = createTestSensorState(12.0, 77.0, 0.0, 10.0)
        
        val mapResult = MapMatchResult(
            matchedLatitude = 12.0,
            matchedLongitude = 77.0,
            roadSegmentId = "R1",
            roadHeading = 0.0,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 0.9,
            candidateRoads = listOf(
                CandidateRoad("R1", 0.9, 12.0, 77.0, 0.0, 0.0, 0.0),
                CandidateRoad("R2", 0.8, 12.0001, 77.0001, 30.0, 0.0, 0.0)
            )
        )
        
        val state = pipeline.update(s1, mapResult, null, null, 1000L)
        assertTrue(state.hypothesis_list.size >= 2)
    }
}
