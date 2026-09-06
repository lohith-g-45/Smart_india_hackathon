package com.navshield.map.engine

import com.navshield.map.contract.CandidateRoad
import com.navshield.map.contract.CandidateRoadDto
import com.navshield.map.contract.MapMatchResult
import com.navshield.map.contract.Member1ResultDto

/**
 * Adapter responsible for converting Member 1's raw output (DTO) 
 * into the internal Kotlin MapMatchResult contract.
 */
object Member1ResultAdapter {

    /**
     * Converts a DTO to the internal contract.
     * Performs strict validation on required fields and ranges.
     * 
     * @throws IllegalArgumentException if mandatory fields are missing or invalid.
     */
    fun fromDto(dto: Member1ResultDto): MapMatchResult {
        // 1. Validate Mandatory Fields
        val matchedLat = dto.matched_latitude ?: throw IllegalArgumentException("Missing matched_latitude")
        val matchedLon = dto.matched_longitude ?: throw IllegalArgumentException("Missing matched_longitude")
        val segmentId = dto.road_segment_id ?: throw IllegalArgumentException("Missing road_segment_id")
        val heading = dto.road_heading ?: throw IllegalArgumentException("Missing road_heading")
        val curvature = dto.road_curvature ?: throw IllegalArgumentException("Missing road_curvature")
        val slope = dto.road_slope ?: throw IllegalArgumentException("Missing road_slope")
        val confidence = dto.map_confidence ?: throw IllegalArgumentException("Missing map_confidence")

        // 2. Range Validation
        require(matchedLat in -90.0..90.0) { "Invalid latitude: $matchedLat" }
        require(matchedLon in -180.0..180.0) { "Invalid longitude: $matchedLon" }
        require(confidence in 0.0..1.0) { "Confidence out of range: $confidence" }

        // 3. Convert Candidate List
        val candidates = dto.candidate_roads?.map { mapCandidate(it) } ?: emptyList()

        // 4. Construct Result
        return MapMatchResult(
            matchedLatitude = matchedLat,
            matchedLongitude = matchedLon,
            roadSegmentId = segmentId,
            roadHeading = heading,
            roadCurvature = curvature,
            roadSlope = slope,
            mapConfidence = confidence,
            candidateRoads = candidates,
            roadSlopeAvailable = dto.road_slope_available ?: true,
            oneWayViolation = dto.one_way_violation ?: false,
            positionUncertainty = dto.position_uncertainty
        )
    }

    private fun mapCandidate(dto: CandidateRoadDto): CandidateRoad {
        val segmentId = dto.segment_id ?: throw IllegalArgumentException("Candidate missing segment_id")
        val score = dto.score ?: throw IllegalArgumentException("Candidate missing score")
        val lat = dto.lat ?: throw IllegalArgumentException("Candidate missing lat")
        val lon = dto.lon ?: throw IllegalArgumentException("Candidate missing lon")
        val heading = dto.heading ?: throw IllegalArgumentException("Candidate missing heading")
        val curvature = dto.curvature ?: throw IllegalArgumentException("Candidate missing curvature")
        val slope = dto.slope ?: throw IllegalArgumentException("Candidate missing slope")

        return CandidateRoad(
            segmentId = segmentId,
            score = score,
            lat = lat,
            lon = lon,
            heading = heading,
            curvature = curvature,
            slope = slope
        )
    }
}
