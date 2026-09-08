package com.navshield.map

import com.navshield.map.contract.*
import com.navshield.map.engine.*
import org.junit.Assert.*
import org.junit.Test

class IntegrationTest {

    private fun createSensorState(
        confidence: Double = 1.0,
        isAnomalous: Boolean = false,
        navMode: String = "STRONG"
    ) = NavShieldSensorState(
        latitude = 12.9716,
        longitude = 77.5946,
        altitude = 0.0,
        speed = 10.0,
        bearing = 90.0,
        accuracy = 5.0,
        satellites = 10,
        confidence = confidence,
        isAnomalous = isAnomalous,
        anomalyReasons = if (isAnomalous) listOf("test_anomaly") else emptyList(),
        navigationMode = navMode,
        accelX = 0.0, accelY = 0.0, accelZ = 9.8,
        gyroX = 0.0, gyroY = 0.0, gyroZ = 0.0,
        timestampMillis = 1000L
    )

    private fun createMapResult(segmentId: String = "R1") = MapMatchResult(
        matchedLatitude = 12.9717,
        matchedLongitude = 77.5947,
        roadSegmentId = segmentId,
        roadHeading = 90.0,
        roadCurvature = 0.0,
        roadSlope = 0.0,
        mapConfidence = 0.9,
        candidateRoads = listOf(
            CandidateRoad(segmentId, 0.9, 12.9717, 77.5947, 90.0, 0.0, 0.0)
        )
    )

    @Test
    fun `TEST 1 - Normal GNSS integration`() {
        val pipeline = Member4Pipeline()
        val sensor = createSensorState(confidence = 1.0, isAnomalous = false, navMode = "STRONG")
        val map = createMapResult()
        
        // Run a few cycles to allow the hypothesis to become best
        var result: NavShieldTrustState? = null
        for (i in 1..5) {
            result = pipeline.update(sensor, map, null, 1000L + i * 100L)
        }
        
        assertNotNull(result)
        assertEquals(NavigationMode.GNSS_DOMINANT, result!!.navigation_mode)
        assertEquals("R1", result.active_hypothesis)
        assertEquals(1.0, result.sensor_weights["gnss"]!!, 1e-6)
    }

    @Test
    fun `TEST 2 - GNSS anomaly integration`() {
        val pipeline = Member4Pipeline()
        // Provide normal GNSS first to initialize
        pipeline.update(createSensorState(), createMapResult(), null, 1000L)
        
        // Now provide anomalous GNSS
        val anomalousSensor = createSensorState(confidence = 0.2, isAnomalous = true, navMode = "ANOMALOUS")
        val result = pipeline.update(anomalousSensor, createMapResult(), null, 2000L)
        
        assertNotEquals(NavigationMode.GNSS_DOMINANT, result.navigation_mode)
        // Weight should be reduced
        assertTrue(result.sensor_weights["gnss"]!! < 1.0)
    }

    @Test
    fun `TEST 3 - GNSS recovery integration`() {
        val pipeline = Member4Pipeline()
        
        // 1. GNSS Denied
        val deniedSensor = createSensorState(confidence = 0.0, isAnomalous = true, navMode = "DENIED")
        var result = pipeline.update(deniedSensor, createMapResult(), null, 1000L)
        assertEquals(NavigationMode.GNSS_DENIED, result.navigation_mode)
        assertEquals(0.0, result.sensor_weights["gnss"]!!, 1e-6)
        
        val startLat = result.final_latitude
        val startLon = result.final_longitude

        // 2. Start recovery
        val healthySensor = createSensorState(confidence = 0.9, isAnomalous = false, navMode = "STRONG")
        
        // Cycle through recovery (10 cycles required by RECOVERY_RAMP_LIMIT in Member4Pipeline)
        var lastLat = startLat
        var lastLon = startLon
        
        for (i in 1..5) {
            result = pipeline.update(healthySensor, createMapResult(), null, 1000L + i * 1000L)
            assertEquals(NavigationMode.RECOVERY, result.navigation_mode)
            
            // Verify smooth recovery (no huge jump)
            // Distance between cycles should be reasonable
            val dist = Math.sqrt(Math.pow(result.final_latitude - lastLat, 2.0) + Math.pow(result.final_longitude - lastLon, 2.0))
            assertTrue("Jump too large at cycle $i: $dist", dist < 0.001)
            
            lastLat = result.final_latitude
            lastLon = result.final_longitude
        }
    }

    @Test
    fun `TEST 4 - Member 1 to Member 4 propagation`() {
        val sensor = createSensorState()
        val map = createMapResult("SEG_XYZ")
        
        val pipeline = Member4Pipeline()
        // Run multiple cycles to stabilize
        var result = pipeline.update(sensor, map, null, 1000L)
        for (i in 1..5) {
            result = pipeline.update(sensor, map, null, 1000L + i * 100L)
        }
        
        assertEquals("SEG_XYZ", result.active_hypothesis)
        
        // Verify map result influence by changing road position
        val shiftedMap = MapMatchResult(
            matchedLatitude = 12.98, // Shifted
            matchedLongitude = 77.60,
            roadSegmentId = "SEG_XYZ",
            roadHeading = 0.0,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 1.0,
            candidateRoads = listOf(CandidateRoad("SEG_XYZ", 1.0, 12.98, 77.60, 0.0, 0.0, 0.0))
        )
        
        val result2 = pipeline.update(sensor, shiftedMap, null, 2000L)
        // The final position should move towards the map result
        assertTrue("Position should move towards map. Lat1: ${result.final_latitude}, Lat2: ${result2.final_latitude}", 
            result2.final_latitude > result.final_latitude)
    }

    @Test
    fun `TEST 5 - Real confidence propagation`() {
        val pipelineHigh = Member4Pipeline()
        val pipelineLow = Member4Pipeline()
        
        // Initialize both
        val initSensor = createSensorState(confidence = 1.0)
        pipelineHigh.update(initSensor, null, null, 1000L)
        pipelineLow.update(initSensor, null, null, 1000L)
        
        // Move GNSS position significantly
        val movedSensorHigh = createSensorState(confidence = 1.0).copy(latitude = 12.98, longitude = 77.60)
        val movedSensorLow = createSensorState(confidence = 0.01).copy(latitude = 12.98, longitude = 77.60)
        
        val resHigh = pipelineHigh.update(movedSensorHigh, null, null, 2000L)
        val resLow = pipelineLow.update(movedSensorLow, null, null, 2000L)
        
        // High confidence should track GNSS more closely
        val diffHigh = Math.abs(resHigh.final_latitude - movedSensorHigh.latitude)
        val diffLow = Math.abs(resLow.final_latitude - movedSensorLow.latitude)
        
        assertTrue("High confidence should track GNSS closer than low confidence. High diff: $diffHigh, Low diff: $diffLow", diffHigh < diffLow)
    }

    @Test
    fun `TEST 6 - Member 5 AI prediction integration`() {
        val pipeline = Member4Pipeline()
        val sensor = createSensorState()
        
        // 1. Normal AI prediction
        val normalAi = Member5Result(1.0f, 0.1f, 1000L, false, false, "NORMAL")
        val res1 = pipeline.update(sensor, null, normalAi, 1000L)
        assertEquals(1.0, res1.sensor_weights["ai"]!!, 0.1)
        assertFalse(res1.drift_warning)

        // 2. Abnormal AI prediction (Drift detected)
        val abnormalAi = Member5Result(25.0f, 2.5f, 2000L, true, true, "ABNORMAL")
        val res2 = pipeline.update(sensor, null, abnormalAi, 2000L)
        
        // AI weight should decrease because of high predicted error
        assertTrue("AI weight should decrease. Got: ${res2.sensor_weights["ai"]}", res2.sensor_weights["ai"]!! < 0.5)
        // Drift warning should be triggered
        assertTrue(res2.drift_warning)
    }
}
