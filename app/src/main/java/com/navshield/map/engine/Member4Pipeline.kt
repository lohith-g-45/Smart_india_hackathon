package com.navshield.map.engine

import com.navshield.map.contract.*
import com.navshield.map.engine.math.Matrix
import com.navshield.map.engine.math.GeoUtils
import kotlin.math.sqrt
import kotlin.math.abs

class Member4Pipeline {
    private val fsm = GNSSModeFSM(); private val mhe = MultiHypothesisEngine()
    private var isInitialized = false; private var refLat: Double = 0.0; private var refLon: Double = 0.0; private var lastTimestamp: Long = 0
    private var w_gnss: Double = 1.0; private var w_imu: Double = 1.0; private var w_map: Double = 1.0; private var w_ai: Double = 1.0
    private var gnssRecoveryCycles = 0; private val RECOVERY_RAMP_LIMIT = 10

    fun reset() { isInitialized = false }

    fun update(sensorInput: NavShieldSensorState, mapResult: MapMatchResult?, imuResult: Member3Result?, aiPrediction: Member5Result?, timestamp: Long): NavShieldTrustState {
        if (!isInitialized) { refLat = sensorInput.latitude; refLon = sensorInput.longitude; isInitialized = true; lastTimestamp = timestamp }
        val dt = ((timestamp - lastTimestamp) / 1000.0).coerceIn(0.0, 1.0); lastTimestamp = timestamp
        val mode = fsm.update(sensorInput, 1.0)
        w_gnss = when (mode) {
            NavigationMode.RECOVERY -> { gnssRecoveryCycles++; (gnssRecoveryCycles.toDouble() / RECOVERY_RAMP_LIMIT).coerceIn(0.0, 1.0) }
            NavigationMode.GNSS_DENIED -> { gnssRecoveryCycles = 0; 0.0 }
            else -> { gnssRecoveryCycles = 0; if (sensorInput.isAnomalous) 0.5 else 1.0 }
        }
        val bestHypo = mhe.update(sensorInput, mapResult, dt, refLat, refLon, w_gnss); val ukf = bestHypo.ukf
        val geo = GeoUtils.inverseProject(ukf.x[0, 0], ukf.x[1, 0], refLat, refLon)
        val posErr = sqrt(ukf.P.trace())
        var fLat = geo.first; var fLon = geo.second
        if (mode == NavigationMode.GNSS_DENIED && imuResult != null && imuResult.imuAvailable && !imuResult.imuLatitude.isNaN()) {
            fLat = imuResult.imuLatitude; fLon = imuResult.imuLongitude
        }
        w_ai = if (aiPrediction != null) (1.0 / (1.0 + aiPrediction.predictedPositionErrorM / 10.0)).coerceIn(0.0, 1.0) else 1.0
        return NavShieldTrustState(fLat, fLon, 0.0, GeoUtils.normalizeHeading(Math.toDegrees(ukf.x[3, 0])), ukf.x[2, 0], (1.0 / (1.0 + posErr)).coerceIn(0.0, 1.0), posErr, mode, mapOf("gnss" to w_gnss, "imu" to w_imu, "map" to w_map, "ai" to w_ai), bestHypo.roadSegmentId, mhe.getHypothesesList().map { it.roadSegmentId }, ukf.P.trace() > 50.0, mode == NavigationMode.RECOVERY, mapResult?.oneWayViolation ?: false)
    }
}
