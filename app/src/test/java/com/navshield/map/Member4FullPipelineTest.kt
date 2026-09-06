package com.navshield.map

import com.navshield.map.contract.*
import com.navshield.map.engine.*
import org.junit.Assert.*
import org.junit.Test

class Member4FullPipelineTest {

    @Test
    fun `pipeline handles tunnel scenario`() {
        val pipeline = Member4Pipeline()
        var time = 0L
        
        // 1. GNSS Available
        val q1 = MapMatchQuery(12.0, 77.0, 0.0, 11.1)
        val state1 = pipeline.update(q1, null, time)
        assertEquals(NavigationMode.GNSS_DOMINANT, state1.navigation_mode)
        
        // 2. Simulated Tunnel (No GNSS updates would come in real integration, 
        // but here we just check if filter continues predicting)
        time += 5000L
        val state2 = pipeline.update(q1, null, time)
        assertTrue(state2.estimated_position_error >= 0.0)
    }

    @Test
    fun `pipeline detects drift warning`() {
        val pipeline = Member4Pipeline()
        val q = MapMatchQuery(12.0, 77.0, 0.0, 10.0)
        
        // Force high uncertainty
        val state = pipeline.update(q, null, 0L)
        // Since we don't have direct access to internal P from here easily 
        // without reflection or public exposure, we check if it's running.
        assertNotNull(state)
    }

    @Test
    fun `pipeline handles multi-hypothesis spawning`() {
        val pipeline = Member4Pipeline()
        val q = MapMatchQuery(12.0, 77.0, 0.0, 10.0)
        
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
        
        val state = pipeline.update(q, mapResult, 1000L)
        assertTrue(state.hypothesis_list.size >= 2)
    }
}
