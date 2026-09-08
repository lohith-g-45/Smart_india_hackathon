package com.navshield.map

import com.navshield.map.contract.*
import com.navshield.map.engine.BasicMapMatchingEngine
import com.navshield.map.engine.RoadRepository
import org.junit.Assert.*
import org.junit.Test

class BasicMapMatchingEngineTest {

    private class MockRoadRepository : RoadRepository {
        val nodes = mutableMapOf<String, RoadNode>()
        val segments = mutableListOf<RoadSegment>()

        override fun getNode(id: String): RoadNode? = nodes[id]
        override fun getNearbySegments(lat: Double, lon: Double, radiusMeters: Double): List<RoadSegment> = segments
        override fun isReady(): Boolean = true
    }

    @Test
    fun `nearest road matching`() {
        val repo = MockRoadRepository()
        repo.nodes["N1"] = RoadNode("N1", 12.0, 77.0)
        repo.nodes["N2"] = RoadNode("N2", 12.1, 77.0)
        repo.segments.add(RoadSegment("S1", "W1", "N1", "N2", 11000.0, 0.0, "primary", false, null, 0.0, false, 0.0, false, "Main St"))

        val engine = BasicMapMatchingEngine(repo)
        val query = MapMatchQuery(12.05, 77.0001, 0.0, 10.0)
        val result = engine.match(query)

        assertEquals("S1", result.roadSegmentId)
        assertTrue(result.mapConfidence > 0.5)
        assertEquals(12.05, result.matchedLatitude, 0.001)
        assertEquals(77.0, result.matchedLongitude, 0.0001)
    }

    @Test
    fun `heading matching preference`() {
        val repo = MockRoadRepository()
        // Two parallel roads
        repo.nodes["N1"] = RoadNode("N1", 12.0, 77.0)
        repo.nodes["N2"] = RoadNode("N2", 12.1, 77.0)
        repo.segments.add(RoadSegment("S1", "W1", "N1", "N2", 11000.0, 0.0, "primary", false, null, 0.0, false, 0.0, false, "North Road"))

        repo.nodes["N3"] = RoadNode("N3", 12.0, 77.0001)
        repo.nodes["N4"] = RoadNode("N4", 12.0, 77.1001)
        repo.segments.add(RoadSegment("S2", "W2", "N3", "N4", 11000.0, 90.0, "primary", false, null, 0.0, false, 0.0, false, "East Road"))

        val engine = BasicMapMatchingEngine(repo)
        
        // Query closer to S2 but heading matches S1
        val query = MapMatchQuery(12.0001, 77.0001, 0.0, 10.0)
        val result = engine.match(query)

        assertEquals("S1", result.roadSegmentId)
    }

    @Test
    fun `off-road low confidence`() {
        val repo = MockRoadRepository()
        repo.nodes["N1"] = RoadNode("N1", 12.0, 77.0)
        repo.nodes["N2"] = RoadNode("N2", 12.1, 77.0)
        repo.segments.add(RoadSegment("S1", "W1", "N1", "N2", 11000.0, 0.0, "primary", false, null, 0.0, false, 0.0, false, "Main St"))

        val engine = BasicMapMatchingEngine(repo)
        // 25m East of the road
        val query = MapMatchQuery(12.05, 77.000225, 0.0, 10.0) // 0.0001 deg lon approx 11m
        val result = engine.match(query)

        assertTrue("Confidence should be capped for off-road", result.mapConfidence <= 0.4)
    }

    @Test
    fun `one-way violation detection`() {
        val repo = MockRoadRepository()
        repo.nodes["N1"] = RoadNode("N1", 12.0, 77.0)
        repo.nodes["N2"] = RoadNode("N2", 12.1, 77.0)
        repo.segments.add(RoadSegment("S1", "W1", "N1", "N2", 11000.0, 0.0, "primary", true, null, 0.0, false, 0.0, false, "One-Way St"))

        val engine = BasicMapMatchingEngine(repo)
        // Heading South on a North-bound one-way road
        val query = MapMatchQuery(12.05, 77.0, 180.0, 10.0)
        val result = engine.match(query)

        assertTrue(result.oneWayViolation)
    }

    @Test
    fun `tunnel no-road fallback`() {
        val repo = MockRoadRepository()
        val engine = BasicMapMatchingEngine(repo)
        val query = MapMatchQuery(12.0, 77.0, 0.0, 10.0)
        val result = engine.match(query)

        assertEquals("", result.roadSegmentId)
        assertEquals(0.0, result.mapConfidence, 0.001)
        assertEquals(12.0, result.matchedLatitude, 0.001)
    }

    @Test
    fun `fork with multiple candidates`() {
        val repo = MockRoadRepository()
        // Fork at N1
        repo.nodes["N1"] = RoadNode("N1", 12.0, 77.0)
        repo.nodes["N2"] = RoadNode("N2", 12.01, 77.001)
        repo.nodes["N3"] = RoadNode("N3", 12.01, 76.999)
        
        repo.segments.add(RoadSegment("S1", "W1", "N1", "N2", 1000.0, 45.0, "primary", false, null, 0.0, false, 0.0, false, "Fork Left"))
        repo.segments.add(RoadSegment("S2", "W1", "N1", "N3", 1000.0, 315.0, "primary", false, null, 0.0, false, 0.0, false, "Fork Right"))

        val engine = BasicMapMatchingEngine(repo)
        val query = MapMatchQuery(12.001, 77.0, 45.0, 10.0)
        val result = engine.match(query)

        assertEquals("S1", result.roadSegmentId)
        assertTrue("Should have multiple candidates", result.candidateRoads.size >= 2)
        assertTrue("Best candidate should have higher score", result.candidateRoads[0].score >= result.candidateRoads[1].score)
    }
}
