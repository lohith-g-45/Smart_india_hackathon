package com.navshield.map.contract

import com.navshield.map.engine.NavigationMode

data class NavShieldTrustState(
    val final_latitude: Double,
    val final_longitude: Double,
    val final_altitude: Double,
    val final_heading: Double,
    val final_velocity: Double,
    val position_confidence: Double,
    val estimated_position_error: Double,
    val navigation_mode: NavigationMode,
    val sensor_weights: Map<String, Double>,
    val active_hypothesis: String,
    val hypothesis_list: List<String>,
    val drift_warning: Boolean,
    val self_healing_active: Boolean,
    val one_way_violation: Boolean = false
)
