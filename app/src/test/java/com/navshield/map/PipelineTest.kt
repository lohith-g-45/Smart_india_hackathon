package com.navshield.map

import com.navshield.map.contract.CandidateRoad
import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult
import com.navshield.map.contract.Member5Result
import com.navshield.map.contract.NavShieldSensorState
import com.navshield.map.engine.MapMatchingEngine
import com.navshield.map.engine.MemberStatus
import com.navshield.map.engine.NavShieldPipeline
import com.navshield.map.members.Member5DriftGuardian
import com.navshield.map.members.RoutingEngine
import com.navshield.map.members.SensorFusionSource
import org.junit.Assert.*
import org.junit.Test

class PipelineTest {

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

    // --- MOCK IMPLEMENTATIONS FOR TESTING ---
    class MockEngine(private val ready: Boolean, private val result: MapMatchResult?) : MapMatchingEngine {
        override fun match(query: MapMatchQuery): MapMatchResult = result!!
        override fun isReady(): Boolean = ready
    }

    class MockSensors(private val state: NavShieldSensorState?) : SensorFusionSource {
        override fun getCurrentState(): NavShieldSensorState? = state
        override fun getStatus(): MemberStatus = MemberStatus.CONNECTED
    }

    class MockRouter : RoutingEngine {
        var lastGuidance: Triple<Double, Double, String>? = null
        override fun updateGuidance(lat: Double, lon: Double, segmentId: String) {
            lastGuidance = Triple(lat, lon, segmentId)
        }
        override fun getStatus(): MemberStatus = MemberStatus.CONNECTED
    }

    class MockDriftGuardian : Member5DriftGuardian {
        override fun predict(sensorState: NavShieldSensorState): Member5Result? = null
        override fun getStatus(): MemberStatus = MemberStatus.CONNECTED
    }

    @Test
    fun `pipeline cycle propagates data correctly`() {
        val sensorState = createTestSensorState(12.0, 77.0, 0.0, 10.0)
        val mapMatch = MapMatchResult(
            matchedLatitude = 12.0001,
            matchedLongitude = 77.0001,
            roadSegmentId = "S1",
            roadHeading = 0.0,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 0.9,
            candidateRoads = listOf(
                CandidateRoad("S1", 0.9, 12.0001, 77.0001, 0.0, 0.0, 0.0)
            )
        )
        
        val router = MockRouter()
        val pipeline = NavShieldPipeline(
            MockEngine(true, mapMatch),
            MockSensors(sensorState),
            router,
            MockDriftGuardian()
        )
        
        var result = pipeline.processCycle(1000L)
        assertNotNull(result)
        
        // Run a few more cycles
        for (i in 1..5) {
            result = pipeline.processCycle(1000L + i * 100L)
        }

        // Verify active hypothesis matches the map input
        assertEquals("S1", result!!.trustState.active_hypothesis)
        assertNotNull(router.lastGuidance)
    }

    @Test
    fun `pipeline returns null if sensors are unavailable`() {
        val pipeline = NavShieldPipeline(
            MockEngine(true, null),
            MockSensors(null),
            MockRouter(),
            MockDriftGuardian()
        )
        
        val result = pipeline.processCycle(1000L)
        assertNull(result)
    }
}
