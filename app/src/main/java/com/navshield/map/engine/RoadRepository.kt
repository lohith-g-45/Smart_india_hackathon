package com.navshield.map.engine

import com.navshield.map.contract.RoadNode
import com.navshield.map.contract.RoadSegment

interface RoadRepository {
    fun getNode(id: String): RoadNode?
    fun getNearbySegments(lat: Double, lon: Double, radiusMeters: Double): List<RoadSegment>
    fun isReady(): Boolean
}
