package com.navshield.map.members

import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.Member5Result
import com.navshield.map.contract.NavShieldSensorState
import com.navshield.map.engine.MemberStatus

/**
 * Interface for Member 2: Sensor Fusion (GNSS + IMU).
 * Provides the complete fused sensor state for map matching and state estimation.
 */
interface SensorFusionSource {
    /**
     * Provides the current fused sensor state.
     * Returns null if sensors are unavailable or status is not CONNECTED.
     */
    fun getCurrentState(): NavShieldSensorState?
    
    fun getStatus(): MemberStatus
}

/**
 * Interface for Member 3: Routing and Guidance.
 * NOT AVAILABLE YET.
 */
interface RoutingEngine {
    /**
     * Receives the "Trust Brain" state from Member 4 and updates path guidance.
     */
    fun updateGuidance(lat: Double, lon: Double, segmentId: String)
    
    fun getStatus(): MemberStatus
}

/**
 * Interface for Member 5: AI Drift Guardian (LSTM).
 */
interface Member5DriftGuardian {
    /**
     * Analyzes sensor sequences to predict navigation drift/errors.
     */
    fun predict(sensorState: NavShieldSensorState): Member5Result?

    fun getStatus(): MemberStatus
}
