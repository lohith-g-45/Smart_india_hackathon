package com.navshield.map.engine

import com.navshield.map.contract.MapMatchResult
import com.navshield.map.contract.Member5Result
import com.navshield.map.contract.NavShieldTrustState
import com.navshield.map.members.Member5DriftGuardian
import com.navshield.map.members.RoutingEngine
import com.navshield.map.members.SensorFusionSource

/**
 * The final output state of the NAV-SHIELD system.
 * Explicitly tracks which components contributed to the result.
 */
data class NavShieldResult(
    val trustState: NavShieldTrustState,
    val member5Result: Member5Result?,
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
    private val driftGuardian: Member5DriftGuardian,
    private val trustBrain: Member4Pipeline = Member4Pipeline()
) {
    /**
     * Executes one cycle of the NAV-SHIELD processing pipeline.
     */
    fun processCycle(timestamp: Long): NavShieldResult? {
        // 1. Get input from Member 2 (Sensors)
        val sensorState = sensors.getCurrentState() ?: return null

        // 2. Get input from Member 5 (AI Drift Guardian)
        val aiPrediction = driftGuardian.predict(sensorState)

        // 3. Get input from Member 1 (Map)
        val mapResult = if (mapEngine.isReady()) {
            mapEngine.match(sensorState.toMapMatchQuery())
        } else {
            null
        }

        // 4. Process in Member 4 (Trust Brain / UKF)
        val trustState = trustBrain.update(sensorState, mapResult, aiPrediction, timestamp)

        // 5. Update Member 3 (Router)
        router.updateGuidance(trustState.final_latitude, trustState.final_longitude, trustState.active_hypothesis)

        return NavShieldResult(
            trustState = trustState,
            member5Result = aiPrediction,
            statusMap = mapOf(
                1 to NavShieldRegistry.member1Status,
                2 to sensors.getStatus(),
                3 to router.getStatus(),
                4 to NavShieldRegistry.member4Status,
                5 to driftGuardian.getStatus()
            )
        )
    }
}
