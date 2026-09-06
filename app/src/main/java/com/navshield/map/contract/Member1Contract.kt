package com.navshield.map.contract

/**
 * Kotlin representation of Member 1's MapMatchQuery (from contract.py).
 * Used as input to the Map Matching engine.
 */
data class MapMatchQuery(
    val latitude: Double,
    val longitude: Double,
    val heading: Double,   // Degrees
    val speed: Double      // m/s
)

/**
 * Kotlin representation of Member 1's CandidateRoad (from contract.py).
 */
data class CandidateRoad(
    val segmentId: String,
    val score: Double,     // 0.0 - 1.0
    val lat: Double,
    val lon: Double,
    val heading: Double,
    val curvature: Double,
    val slope: Double
)

/**
 * Kotlin representation of Member 1's MapMatchResult (from contract.py).
 * This is the ACTUAL output contract provided by Member 1.
 */
data class MapMatchResult(
    val matchedLatitude: Double,
    val matchedLongitude: Double,
    val roadSegmentId: String,
    val roadHeading: Double,
    val roadCurvature: Double,
    val roadSlope: Double,            // Percent (%)
    val mapConfidence: Double,         // 0.0 - 1.0
    val candidateRoads: List<CandidateRoad> = emptyList(),
    val roadSlopeAvailable: Boolean = true,
    val oneWayViolation: Boolean = false,
    val positionUncertainty: Double? = null // Metres, optional
)
