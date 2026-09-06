package com.navshield.map

import com.navshield.map.contract.CandidateRoad
import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult
import com.navshield.map.engine.Member4Pipeline
import com.navshield.map.engine.math.GeoUtils
import org.junit.Assert.*
import org.junit.Test

class Member4UKFTest {

    @Test
    fun `UKF converges toward strong map measurement`() {
        val brain = Member4Pipeline()
        val t0 = 1000L
        
        // Starting at (12.0, 77.0)
        val query = MapMatchQuery(12.0, 77.0, 0.0, 10.0)
        brain.update(query, null, t0)
        
        // Map says we are actually at (12.0001, 77.0001) with high confidence
        val mapResult = MapMatchResult(
            matchedLatitude = 12.0001,
            matchedLongitude = 77.0001,
            roadSegmentId = "R1",
            roadHeading = 0.0,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 0.9
        )
        
        val trustState = brain.update(query, mapResult, t0 + 1000)
        
        // State should have moved toward map position
        assertTrue("Latitude should increase toward map", trustState.latitude > 12.0)
        assertTrue("Longitude should increase toward map", trustState.longitude > 77.0)
        assertEquals(0.9, trustState.mapMatchingStatus.confidence, 0.001)
    }

    @Test
    fun `UKF ignores low-confidence map measurements`() {
        val brain = Member4Pipeline()
        val t0 = 1000L
        
        val query = MapMatchQuery(12.0, 77.0, 0.0, 0.0)
        brain.update(query, null, t0)
        
        // Map says we are far away but with very low confidence
        val mapResult = MapMatchResult(
            matchedLatitude = 12.1,
            matchedLongitude = 77.1,
            roadSegmentId = "R2",
            roadHeading = 0.0,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 0.01 // Very low
        )
        
        val trustState = brain.update(query, mapResult, t0 + 1000)
        
        // Latitude should be very close to 12.0, not 12.1
        assertEquals(12.0, trustState.latitude, 0.001)
    }

    @Test
    fun `UKF maintains recursive state across updates`() {
        val brain = Member4Pipeline()
        
        // T=0
        val q1 = MapMatchQuery(12.0, 77.0, 90.0, 10.0) // Heading North
        brain.update(q1, null, 0)
        
        // T=1s, no map, only prediction
        val state1 = brain.update(q1, null, 1000)
        
        // Vehicle should have moved North (Latitude increases)
        assertTrue(state1.latitude > 12.0)
        
        // T=2s
        val state2 = brain.update(q1, null, 2000)
        assertTrue(state2.latitude > state1.latitude)
    }

    @Test
    fun `UKF reset clears state`() {
        val brain = Member4Pipeline()
        brain.update(MapMatchQuery(12.0, 77.0, 0.0, 10.0), null, 0)
        
        brain.reset()
        
        // Next update will be a fresh initialization at new position
        val state = brain.update(MapMatchQuery(13.0, 78.0, 0.0, 0.0), null, 1000)
        assertEquals(13.0, state.latitude, 0.0001)
    }
}
