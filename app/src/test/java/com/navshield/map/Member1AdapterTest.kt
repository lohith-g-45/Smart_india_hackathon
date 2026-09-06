package com.navshield.map

import com.navshield.map.contract.CandidateRoadDto
import com.navshield.map.contract.Member1ResultDto
import com.navshield.map.engine.Member1ResultAdapter
import org.junit.Assert.*
import org.junit.Test

class Member1AdapterTest {

    // --- TEST FIXTURES ---
    private fun createValidDto() = Member1ResultDto(
        matched_latitude = 12.9717,
        matched_longitude = 77.5947,
        road_segment_id = "seg_001",
        road_heading = 12.0,
        road_curvature = 0.5,
        road_slope = 2.0,
        map_confidence = 0.9,
        candidate_roads = listOf(
            CandidateRoadDto("seg_002", 0.8, 12.9718, 77.5948, 15.0, 0.1, 1.5)
        )
    )

    @Test
    fun `adapter converts valid DTO correctly`() {
        val dto = createValidDto()
        val result = Member1ResultAdapter.fromDto(dto)
        
        assertEquals(12.9717, result.matchedLatitude, 0.0001)
        assertEquals("seg_001", result.roadSegmentId)
        assertEquals(0.9, result.mapConfidence, 0.0001)
        assertEquals(1, result.candidateRoads.size)
        assertEquals("seg_002", result.candidateRoads[0].segmentId)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `adapter rejects missing mandatory field`() {
        val invalidDto = createValidDto().copy(matched_latitude = null)
        Member1ResultAdapter.fromDto(invalidDto)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `adapter rejects out of range confidence`() {
        val invalidDto = createValidDto().copy(map_confidence = 1.5)
        Member1ResultAdapter.fromDto(invalidDto)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `adapter rejects invalid latitude`() {
        val invalidDto = createValidDto().copy(matched_latitude = 100.0)
        Member1ResultAdapter.fromDto(invalidDto)
    }

    @Test
    fun `adapter handles empty candidate list`() {
        val dto = createValidDto().copy(candidate_roads = null)
        val result = Member1ResultAdapter.fromDto(dto)
        assertTrue(result.candidateRoads.isEmpty())
    }

    @Test(expected = IllegalArgumentException::class)
    fun `adapter rejects candidate missing mandatory field`() {
        val invalidCandidate = CandidateRoadDto("seg_002", null, 12.0, 77.0, 0.0, 0.0, 0.0)
        val dto = createValidDto().copy(candidate_roads = listOf(invalidCandidate))
        Member1ResultAdapter.fromDto(dto)
    }
}
