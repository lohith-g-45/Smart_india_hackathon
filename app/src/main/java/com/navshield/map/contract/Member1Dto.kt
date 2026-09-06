package com.navshield.map.contract

/**
 * Data Transfer Object (DTO) matching Member 1's Python output exactly.
 * Used for receiving results over a bridge (e.g., JSON from a Python service 
 * or ONNX Runtime output).
 * 
 * Field names match the snake_case used in contract.py.
 */
data class Member1ResultDto(
    val matched_latitude: Double?,
    val matched_longitude: Double?,
    val road_segment_id: String?,
    val road_heading: Double?,
    val road_curvature: Double?,
    val road_slope: Double?,
    val map_confidence: Double?,
    val candidate_roads: List<CandidateRoadDto>? = null,
    val road_slope_available: Boolean? = null,
    val one_way_violation: Boolean? = null,
    val position_uncertainty: Double? = null
)

data class CandidateRoadDto(
    val segment_id: String?,
    val score: Double?,
    val lat: Double?,
    val lon: Double?,
    val heading: Double?,
    val curvature: Double?,
    val slope: Double?
)
