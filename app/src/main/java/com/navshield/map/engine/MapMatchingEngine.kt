package com.navshield.map.engine

import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult

/**
 * Clean interface for Member 1 -> Member 4 integration.
 * This allows Member 4 (Trust Brain/UKF) to receive ACTUAL Member 1 results.
 */
interface MapMatchingEngine {
    /**
     * Executes map matching for the given GPS/Sensor query.
     * Returns an ACTUAL Member 1 result structure.
     */
    fun match(query: MapMatchQuery): MapMatchResult

    /**
     * Check if the engine is ready (e.g., offline data loaded, models ready).
     */
    fun isReady(): Boolean
}

/**
 * Representation of the overall system integration state,
 * tracking the availability of different module members.
 */
enum class MemberStatus {
    CONNECTED,           // Actual implementation is wired and working
    WAITING_FOR_MEMBER,  // Member has not provided final implementation
    NOT_AVAILABLE,       // Module is disabled or missing
    ERROR                // Technical failure in integration
}

/**
 * Global Registry for NAV-SHIELD Member Status.
 * Used to explicitly represent unavailable members without faking data.
 */
object NavShieldRegistry {
    var member1Status: MemberStatus = MemberStatus.WAITING_FOR_MEMBER
    var member2Status: MemberStatus = MemberStatus.WAITING_FOR_MEMBER
    var member3Status: MemberStatus = MemberStatus.WAITING_FOR_MEMBER
    var member5Status: MemberStatus = MemberStatus.WAITING_FOR_MEMBER
    
    // Member 4 is the current implementation (Self)
    val member4Status: MemberStatus = MemberStatus.CONNECTED
}
