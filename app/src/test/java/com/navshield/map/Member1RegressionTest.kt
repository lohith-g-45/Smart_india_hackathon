package com.navshield.map

import com.navshield.map.contract.CandidateRoadDto
import com.navshield.map.contract.Member1ResultDto
import com.navshield.map.engine.Member1ResultAdapter
import org.junit.Assert.*
import org.junit.Test

/**
 * MEMBER 1 SYNTHETIC REGRESSION FIXTURE.
 * 
 * Verifies that the Android-side implementation correctly consumes Member 1's 
 * already verified synthetic outputs from the Python reference module.
 */
class Member1RegressionTest {

    @Test
    fun `Regression Scenario 1 - Hairpin Apex Match`() {
        // Data from ONE_COMPLETE_EXECUTION_TRACE.md
        val dto = Member1ResultDto(
            matched_latitude = 12.972131395316328,
            matched_longitude = 77.5946961532261,
            road_segment_id = "W_hairpin_2:N2->N3",
            road_heading = 140.0,
            road_curvature = 130.0,
            road_slope = 83.3333333282419,
            map_confidence = 0.269,
            candidate_roads = listOf(
                CandidateRoadDto("W_hairpin_2:N2->N3", 0.269, 12.972131395316328, 77.5946961532261, 140.0, 130.0, 83.33),
                CandidateRoadDto("W_hairpin_1:N2->N1", 0.257, 12.972131395316328, 77.5946961532261, 190.0, 130.0, -66.67)
            ),
            road_slope_available = true,
            one_way_violation = false
        )

        val result = Member1ResultAdapter.fromDto(dto)

        assertEquals("W_hairpin_2:N2->N3", result.roadSegmentId)
        assertEquals(12.972131395316328, result.matchedLatitude, 1e-10)
        assertEquals(140.0, result.roadHeading, 0.1)
        assertEquals(0.269, result.mapConfidence, 0.001)
        assertEquals(2, result.candidateRoads.size)
        assertEquals(83.33, result.roadSlope, 0.01)
    }

    @Test
    fun `Regression Scenario 7 - Fork Ambiguity`() {
        // Data from EDGE_CASE_RESULTS.md
        val dto = Member1ResultDto(
            matched_latitude = 12.9750,
            matched_longitude = 77.5946,
            road_segment_id = "fork_edge_1",
            road_heading = 27.0,
            road_curvature = 0.0,
            road_slope = 0.0,
            map_confidence = 0.315,
            candidate_roads = listOf(
                CandidateRoadDto("fork_edge_1", 0.315, 12.9750, 77.5946, 27.0, 0.0, 0.0),
                CandidateRoadDto("fork_edge_2", 0.285, 12.9750, 77.5946, 45.0, 0.0, 0.0)
            )
        )

        val result = Member1ResultAdapter.fromDto(dto)
        assertEquals(0.315, result.mapConfidence, 0.001)
        assertEquals(2, result.candidateRoads.size)
        // Multi-candidate engine flags fork in Phase G
        assertTrue(result.candidateRoads.size >= 2)
    }

    @Test
    fun `Regression Scenario 8 - Off-Road Point`() {
        val dto = Member1ResultDto(
            matched_latitude = 12.9716,
            matched_longitude = 77.5946,
            road_segment_id = "near_parking",
            road_heading = 90.0,
            road_curvature = 0.0,
            road_slope = 0.0,
            map_confidence = 0.4 // Capped per spec Section 6
        )

        val result = Member1ResultAdapter.fromDto(dto)
        assertEquals(0.4, result.mapConfidence, 0.001)
    }

    @Test
    fun `Regression Scenario 15 - Tunnel Gap Fallback`() {
        val dto = Member1ResultDto(
            matched_latitude = 12.9716,
            matched_longitude = 77.5946,
            road_segment_id = "W_hairpin_2",
            road_heading = 140.0,
            road_curvature = 130.0,
            road_slope = 83.33,
            map_confidence = 0.3 // Sticky fallback confidence
        )

        val result = Member1ResultAdapter.fromDto(dto)
        assertEquals(0.3, result.mapConfidence, 0.001)
    }

    @Test
    fun `Regression Scenario 5 - One-Way Violation`() {
        val dto = Member1ResultDto(
            matched_latitude = 12.9716,
            matched_longitude = 77.5946,
            road_segment_id = "one_way_street",
            road_heading = 0.0,
            road_curvature = 0.0,
            road_slope = 0.0,
            map_confidence = 0.1,
            one_way_violation = true
        )

        val result = Member1ResultAdapter.fromDto(dto)
        assertTrue(result.oneWayViolation)
    }
}
