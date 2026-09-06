package com.navshield.map.engine

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

    fun update(gnssConfidence: Double, positionError: Double): NavigationMode {
        val prevMode = currentMode
        
        currentMode = when (currentMode) {
            NavigationMode.GNSS_DOMINANT -> {
                if (gnssConfidence < 0.3) NavigationMode.GNSS_DEGRADED
                else if (gnssConfidence < 0.7) NavigationMode.HYBRID
                else NavigationMode.GNSS_DOMINANT
            }
            NavigationMode.HYBRID -> {
                if (gnssConfidence > 0.8) NavigationMode.GNSS_DOMINANT
                else if (gnssConfidence < 0.2) NavigationMode.GNSS_DEGRADED
                else NavigationMode.HYBRID
            }
            NavigationMode.GNSS_DEGRADED -> {
                if (gnssConfidence < 0.05) NavigationMode.GNSS_DENIED
                else if (gnssConfidence > 0.4) NavigationMode.HYBRID
                else NavigationMode.GNSS_DEGRADED
            }
            NavigationMode.GNSS_DENIED -> {
                if (gnssConfidence > 0.1) NavigationMode.RECOVERY
                else NavigationMode.GNSS_DENIED
            }
            NavigationMode.RECOVERY -> {
                if (recoveryCycleCount >= 10) {
                    recoveryCycleCount = 0
                    if (gnssConfidence > 0.8) NavigationMode.GNSS_DOMINANT else NavigationMode.HYBRID
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
