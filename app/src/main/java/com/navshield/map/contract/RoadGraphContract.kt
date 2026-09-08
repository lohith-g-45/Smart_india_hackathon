package com.navshield.map.contract

/**
 * Representation of a node in the road graph.
 */
data class RoadNode(
    val id: String,
    val lat: Double,
    val lon: Double,
    val elevation: Double? = null
)

/**
 * Representation of a directed road segment in the road graph.
 */
data class RoadSegment(
    val segmentId: String,
    val wayId: String,
    val u: String, // Start node ID
    val v: String, // End node ID
    val length: Double, // meters
    val bearing: Double, // degrees
    val roadType: String,
    val oneWay: Boolean,
    val maxSpeed: Double?,
    val slope: Double, // percent
    val slopeAvailable: Boolean,
    val curvature: Double, // degrees
    val junction: Boolean,
    val name: String?
)
