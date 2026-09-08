package com.navshield.map.engine

import com.navshield.map.contract.*
import com.navshield.map.engine.math.GeoUtils
import kotlin.math.exp

class BasicMapMatchingEngine(private val repository: RoadRepository) : MapMatchingEngine {

    private val MAX_MATCH_DISTANCE_M = 30.0
    private val OFF_ROAD_THRESHOLD_M = 20.0
    private val HEADING_WEIGHT = 0.15
    private val ONE_WAY_PENALTY = 50.0

    override fun match(query: MapMatchQuery): MapMatchResult {
        if (!repository.isReady()) {
            return emptyResult(query)
        }

        val candidates = repository.getNearbySegments(query.latitude, query.longitude, MAX_MATCH_DISTANCE_M)
        
        if (candidates.isEmpty()) {
            return emptyResult(query)
        }

        val scoredCandidates = candidates.map { segment ->
            scoreCandidate(query, segment)
        }.filter { it.distance <= MAX_MATCH_DISTANCE_M }
         .sortedBy { it.score }

        if (scoredCandidates.isEmpty()) {
            return emptyResult(query)
        }

        val best = scoredCandidates[0]
        val confidences = attachConfidences(scoredCandidates)
        
        val candidateRoads = scoredCandidates.mapIndexed { index, scored ->
            val segment = scored.segment
            CandidateRoad(
                segmentId = segment.segmentId,
                score = confidences[index],
                lat = scored.snappedLat,
                lon = scored.snappedLon,
                heading = segment.bearing,
                curvature = segment.curvature,
                slope = segment.slope
            )
        }

        var confidence = confidences[0]
        if (best.distance > OFF_ROAD_THRESHOLD_M) {
            confidence = 0.4.coerceAtMost(confidence)
        }

        return MapMatchResult(
            matchedLatitude = best.snappedLat,
            matchedLongitude = best.snappedLon,
            roadSegmentId = best.segment.segmentId,
            roadHeading = best.segment.bearing,
            roadCurvature = best.segment.curvature,
            roadSlope = best.segment.slope,
            mapConfidence = confidence,
            candidateRoads = candidateRoads,
            roadSlopeAvailable = best.segment.slopeAvailable,
            oneWayViolation = best.oneWayViolation
        )
    }

    override fun isReady(): Boolean = repository.isReady()

    private fun scoreCandidate(query: MapMatchQuery, segment: RoadSegment): ScoredCandidate {
        val nodeU = repository.getNode(segment.u) ?: throw IllegalStateException("Node ${segment.u} not found")
        val nodeV = repository.getNode(segment.v) ?: throw IllegalStateException("Node ${segment.v} not found")

        val distance = GeoUtils.pointToSegmentDistance(
            query.latitude, query.longitude,
            nodeU.lat, nodeU.lon,
            nodeV.lat, nodeV.lon
        )

        val headingDiff = GeoUtils.angularDifference(query.heading, segment.bearing)
        val rank = getRoadTypeRank(segment.roadType)
        
        var oneWayViolation = false
        if (segment.oneWay) {
            // If heading difference is more than 90 degrees, it's a likely one-way violation
            if (headingDiff > 90.0) {
                oneWayViolation = true
            }
        }

        val score = distance * 1.0 + headingDiff * HEADING_WEIGHT - rank + (if (oneWayViolation) ONE_WAY_PENALTY else 0.0)

        // Snapping logic: find the closest point on segment
        // Simple snapping for now: midpoint or proportional to t from GeoUtils?
        // Let's improve GeoUtils.pointToSegmentDistance to return snapped point.
        val snapped = snapToSegment(query.latitude, query.longitude, nodeU, nodeV)

        return ScoredCandidate(segment, score, distance, snapped.first, snapped.second, oneWayViolation)
    }

    private fun snapToSegment(px: Double, py: Double, u: RoadNode, v: RoadNode): Pair<Double, Double> {
        val p = GeoUtils.project(px, py, u.lat, u.lon)
        val s1 = GeoUtils.project(u.lat, u.lon, u.lat, u.lon)
        val s2 = GeoUtils.project(v.lat, v.lon, u.lat, u.lon)
        
        val dx = s2.first - s1.first
        val dy = s2.second - s1.second
        val magSq = dx * dx + dy * dy
        
        if (magSq == 0.0) return Pair(u.lat, u.lon)
        
        val t = ((p.first - s1.first) * dx + (p.second - s1.second) * dy) / magSq
        val constrainedT = t.coerceIn(0.0, 1.0)
        
        val snappedX = s1.first + constrainedT * dx
        val snappedY = s1.second + constrainedT * dy
        
        return GeoUtils.inverseProject(snappedX, snappedY, u.lat, u.lon)
    }

    private fun getRoadTypeRank(type: String): Double {
        return when (type) {
            "motorway" -> 6.0
            "trunk" -> 5.0
            "primary" -> 4.0
            "secondary" -> 3.0
            "tertiary" -> 2.0
            "unclassified", "residential" -> 1.0
            else -> 0.0
        }
    }

    private fun attachConfidences(candidates: List<ScoredCandidate>): List<Double> {
        // Softmin approach as in Member 1
        val scores = candidates.map { it.score }
        val minScore = scores.minOrNull() ?: 0.0
        val exps = scores.map { exp(-(it - minScore) / 10.0) }
        val sum = exps.sum()
        return exps.map { it / sum }
    }

    private fun emptyResult(query: MapMatchQuery): MapMatchResult {
        return MapMatchResult(
            matchedLatitude = query.latitude,
            matchedLongitude = query.longitude,
            roadSegmentId = "",
            roadHeading = query.heading,
            roadCurvature = 0.0,
            roadSlope = 0.0,
            mapConfidence = 0.0,
            candidateRoads = emptyList(),
            roadSlopeAvailable = false,
            oneWayViolation = false
        )
    }

    private data class ScoredCandidate(
        val segment: RoadSegment,
        val score: Double,
        val distance: Double,
        val snappedLat: Double,
        val snappedLon: Double,
        val oneWayViolation: Boolean
    )
}
