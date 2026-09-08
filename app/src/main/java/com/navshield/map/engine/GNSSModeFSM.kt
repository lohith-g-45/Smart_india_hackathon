package com.navshield.map.engine

import com.navshield.map.contract.NavShieldSensorState

enum class NavigationMode {
    GNSS_DOMINANT,
    HYBRID,
    GNSS_DEGRADED,
    GNSS_DENIED,
    RECOVERY
}

class GNSSModeFSM {
    var currentMode = NavigationMode.GNSS_DOMINANT
        private set

    private var recoveryCycleCount = 0

    fun update(sensorState: NavShieldSensorState, positionError: Double): NavigationMode {
        val prevMode = currentMode
        val gnssConfidence = sensorState.confidence.coerceIn(0.0, 1.0)
        val isAnomalous = sensorState.isAnomalous
        val navMode = sensorState.navigationMode
        
        currentMode = when (currentMode) {
            NavigationMode.GNSS_DOMINANT -> {
                if (navMode == "DENIED" || gnssConfidence < 0.05) NavigationMode.GNSS_DENIED
                else if (isAnomalous || navMode == "ANOMALOUS" || gnssConfidence < 0.3) NavigationMode.GNSS_DEGRADED
                else if (gnssConfidence < 0.7) NavigationMode.HYBRID
                else NavigationMode.GNSS_DOMINANT
            }
            NavigationMode.HYBRID -> {
                if (navMode == "DENIED" || gnssConfidence < 0.05) NavigationMode.GNSS_DENIED
                else if (isAnomalous || navMode == "ANOMALOUS" || gnssConfidence < 0.2) NavigationMode.GNSS_DEGRADED
                else if (gnssConfidence > 0.8 && !isAnomalous) NavigationMode.GNSS_DOMINANT
                else NavigationMode.HYBRID
            }
            NavigationMode.GNSS_DEGRADED -> {
                if (navMode == "DENIED" || gnssConfidence < 0.05) NavigationMode.GNSS_DENIED
                else if (!isAnomalous && gnssConfidence > 0.4) NavigationMode.HYBRID
                else NavigationMode.GNSS_DEGRADED
            }
            NavigationMode.GNSS_DENIED -> {
                if (navMode != "DENIED" && gnssConfidence > 0.1) NavigationMode.RECOVERY
                else NavigationMode.GNSS_DENIED
            }
            NavigationMode.RECOVERY -> {
                if (recoveryCycleCount >= 10) {
                    recoveryCycleCount = 0
                    if (isAnomalous || gnssConfidence < 0.3) NavigationMode.GNSS_DEGRADED
                    else if (gnssConfidence > 0.8) NavigationMode.GNSS_DOMINANT 
                    else NavigationMode.HYBRID
                } else {
                    recoveryCycleCount++
                    NavigationMode.RECOVERY
                }
            }
        }
        
        if (prevMode == NavigationMode.GNSS_DENIED && currentMode == NavigationMode.RECOVERY) {
            recoveryCycleCount = 0
        }

        return currentMode
    }
}
