package com.navshield.map.engine

import com.navshield.map.contract.MapMatchResult
import com.navshield.map.contract.NavShieldTrustState
import com.navshield.map.members.PerceptionModule
import com.navshield.map.members.RoutingEngine
import com.navshield.map.members.SensorFusionSource

/**
 * The final output state of the NAV-SHIELD system.
 * Explicitly tracks which components contributed to the result.
 */
data class NavShieldResult(
    val trustState: NavShieldTrustState,
    val statusMap: Map<Int, MemberStatus>
)

/**
 * Orchestration layer for the complete NAV-SHIELD system.
 * Connects Member 1 (Map) -> Member 4 (Trust Brain) -> Future Members.
 */
class NavShieldPipeline(
    private val mapEngine: MapMatchingEngine,
    private val sensors: SensorFusionSource,
    private val router: RoutingEngine,
    private val perception: PerceptionModule,
    private val trustBrain: Member4Pipeline = Member4Pipeline()
) {
    /**
     * Executes one cycle of the NAV-SHIELD processing pipeline.
     */
    fun processCycle(timestamp: Long): NavShieldResult? {
        // 1. Get input from Member 2 (Sensors)
        val query = sensors.getCurrentQuery() ?: return null

        // 2. Get input from Member 1 (Map)
        val mapResult = if (mapEngine.isReady()) {
            mapEngine.match(query)
        } else {
            null
        }

        // 3. Process in Member 4 (Trust Brain / UKF)
        val trustState = trustBrain.update(query, mapResult, timestamp)

        // 4. Update Member 3 (Router)
        router.updateGuidance(trustState.latitude, trustState.longitude, trustState.mapMatchingStatus.roadSegmentId)

        return NavShieldResult(
            trustState = trustState,
            statusMap = mapOf(
                1 to NavShieldRegistry.member1Status,
                2 to sensors.getStatus(),
                3 to router.getStatus(),
                4 to NavShieldRegistry.member4Status,
                5 to perception.getStatus()
            )
        )
    }
}
