package com.navshield.map

import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult
import com.navshield.map.engine.MapMatchingEngine
import com.navshield.map.engine.MemberStatus
import com.navshield.map.engine.NavShieldPipeline
import com.navshield.map.members.PerceptionModule
import com.navshield.map.members.RoutingEngine
import com.navshield.map.members.SensorFusionSource
import org.junit.Assert.*
import org.junit.Test

class PipelineTest {

    // --- MOCK IMPLEMENTATIONS FOR TESTING ---
    class MockEngine(private val ready: Boolean, private val result: MapMatchResult?) : MapMatchingEngine {
        override fun match(query: MapMatchQuery): MapMatchResult = result!!
        override fun isReady(): Boolean = ready
    }

    class MockSensors(private val query: MapMatchQuery?) : SensorFusionSource {
        override fun getCurrentQuery(): MapMatchQuery? = query
        override fun getStatus(): MemberStatus = MemberStatus.CONNECTED
    }

    class MockRouter : RoutingEngine {
        var lastGuidance: Triple<Double, Double, String>? = null
        override fun updateGuidance(lat: Double, lon: Double, segmentId: String) {
            lastGuidance = Triple(lat, lon, segmentId)
        }
        override fun getStatus(): MemberStatus = MemberStatus.CONNECTED
    }

    class MockPerception : PerceptionModule {
        override fun getVisualCues(): List<String> = emptyList()
        override fun getStatus(): MemberStatus = MemberStatus.CONNECTED
    }

    @Test
    fun `pipeline cycle propagates data correctly`() {
        val sensorQuery = MapMatchQuery(12.0, 77.0, 0.0, 10.0)
        val mapMatch = MapMatchResult(12.0001, 77.0001, "S1", 0.0, 0.0, 0.0, 0.9)
        
        val router = MockRouter()
        val pipeline = NavShieldPipeline(
            MockEngine(true, mapMatch),
            MockSensors(sensorQuery),
            router,
            MockPerception()
        )
        
        val result = pipeline.processCycle(1000)
        assertNotNull(result)
        assertEquals(0.9, result!!.trustState.mapMatchingStatus.confidence, 0.001)
        assertEquals("S1", result.trustState.mapMatchingStatus.roadSegmentId)
        assertNotNull(router.lastGuidance)
    }

    @Test
    fun `pipeline returns null if sensors are unavailable`() {
        val pipeline = NavShieldPipeline(
            MockEngine(true, null),
            MockSensors(null),
            MockRouter(),
            MockPerception()
        )
        
        val result = pipeline.processCycle(1000)
        assertNull(result)
    }
}
