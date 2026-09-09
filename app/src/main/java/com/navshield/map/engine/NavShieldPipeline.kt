package com.navshield.map.engine

import android.util.Log
import com.navshield.map.contract.MapMatchResult
import com.navshield.map.contract.Member3Result
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
    val imuResult: Member3Result?,
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
    private val insEngine: Member3InsEngine = Member3InsEngine(),
    private val trustBrain: Member4Pipeline = Member4Pipeline()
) {
    /**
     * Executes one cycle of the NAV-SHIELD processing pipeline.
     */
    fun processCycle(timestamp: Long): NavShieldResult? {
        // 1. Get input from Member 2 (Sensors)
        val sensorState = sensors.getCurrentState() ?: return null
        Log.d("NavShield", "M2: GNSS=${sensorState.navigationMode} conf=${"%.2f".format(sensorState.confidence)} anomalous=${sensorState.isAnomalous}")

        // 2. Process in Member 3 (INS / Dead Reckoning)
        val imuResult = insEngine.update(sensorState)
        if (imuResult != null) {
            Log.d("NavShield", "M3: state=${imuResult.motionState} conf=${"%.2f".format(imuResult.imuConfidence)}")
        }

        // 3. Get input from Member 5 (AI Drift Guardian)
        val aiPrediction = driftGuardian.predict(sensorState)
        if (aiPrediction != null) {
            Log.d("NavShield", "M5: posError=${"%.2f".format(aiPrediction.predictedPositionErrorM)}m velError=${"%.2f".format(aiPrediction.predictedVelocityErrorMps)}m/s state=${aiPrediction.state}")
        }

        // 4. Get input from Member 1 (Map)
        val mapResult = if (mapEngine.isReady()) {
            mapEngine.match(sensorState.toMapMatchQuery())
        } else {
            null
        }
        if (mapResult != null) {
            Log.d("NavShield", "M1: segment=${mapResult.roadSegmentId} conf=${"%.2f".format(mapResult.mapConfidence)}")
            // Feed road grade back to INS
            insEngine.setRoadGrade(mapResult.roadSlope.toFloat())
        }

        // 5. Process in Member 4 (Trust Brain / UKF)
        val trustState = trustBrain.update(sensorState, mapResult, imuResult, aiPrediction, timestamp)
        Log.d("NavShield", "M4: mode=${trustState.navigation_mode} pos_conf=${"%.2f".format(trustState.position_confidence)}")

        // Handle GNSS Recovery: If GNSS is healthy, sync INS state
        if (!sensorState.isAnomalous && sensorState.navigationMode == "STRONG") {
            insEngine.resetFromExternalState(
                sensorState.latitude,
                sensorState.longitude,
                sensorState.bearing.toFloat(),
                sensorState.speed.toFloat(),
                timestamp
            )
        }

        // 6. Update Member 3 (Router)
        router.updateGuidance(trustState.final_latitude, trustState.final_longitude, trustState.active_hypothesis)

        return NavShieldResult(
            trustState = trustState,
            imuResult = imuResult,
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
