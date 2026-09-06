package com.navshield.map

import com.navshield.map.contract.CandidateRoad
import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult
import org.junit.Assert.*
import org.junit.Test

class Member1ContractTest {

    @Test
    fun `MapMatchQuery holds expected fields`() {
        val query = MapMatchQuery(12.9716, 77.5946, 10.0, 8.0)
        assertEquals(12.9716, query.latitude, 0.0001)
        assertEquals(77.5946, query.longitude, 0.0001)
        assertEquals(10.0, query.heading, 0.0001)
        assertEquals(8.0, query.speed, 0.0001)
    }

    @Test
    fun `MapMatchResult handles nullable uncertainty`() {
        val resultWithoutUncertainty = MapMatchResult(
            matchedLatitude = 12.9717,
            matchedLongitude = 77.5947,
            roadSegmentId = "seg_001",
            roadHeading = 12.0,
            roadCurvature = 0.5,
            roadSlope = 2.0,
            mapConfidence = 0.9,
            positionUncertainty = null
        )
        assertNull(resultWithoutUncertainty.positionUncertainty)

        val resultWithUncertainty = resultWithoutUncertainty.copy(positionUncertainty = 5.0)
        assertEquals(5.0, resultWithUncertainty.positionUncertainty!!, 0.0001)
    }

    @Test
    fun `MapMatchResult handles empty candidates list`() {
        val result = MapMatchResult(
            matchedLatitude = 12.9717,
            matchedLongitude = 77.5947,
            roadSegmentId = "seg_001",
            roadHeading = 12.0,
            roadCurvature = 0.5,
            roadSlope = 2.0,
            mapConfidence = 0.9
        )
        assertTrue(result.candidateRoads.isEmpty())
    }

    @Test
    fun `CandidateRoad conversion validation`() {
        val candidate = CandidateRoad(
            segmentId = "seg_002",
            score = 0.8,
            lat = 12.9718,
            lon = 77.5948,
            heading = 15.0,
            curvature = 0.1,
            slope = 1.5
        )
        assertEquals("seg_002", candidate.segmentId)
        assertEquals(0.8, candidate.score, 0.0001)
    }
}
