package com.navshield.map.members

import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.engine.MemberStatus

/**
 * Interface for Member 2: Sensor Fusion (GNSS + IMU).
 * NOT AVAILABLE YET.
 */
interface SensorFusionSource {
    /**
     * Provides the current fused sensor state for map matching.
     * Returns null if sensors are unavailable or status is not CONNECTED.
     */
    fun getCurrentQuery(): MapMatchQuery?
    
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
 * Interface for Member 5: Perception / Visual SLAM.
 * NOT AVAILABLE YET.
 */
interface PerceptionModule {
    /**
     * Provides visual confirmation of road features (e.g. traffic signs, lane count).
     */
    fun getVisualCues(): List<String>
    
    fun getStatus(): MemberStatus
}
