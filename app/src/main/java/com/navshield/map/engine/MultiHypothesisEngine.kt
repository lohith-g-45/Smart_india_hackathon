package com.navshield.map.engine

import com.navshield.map.contract.*
import com.navshield.map.engine.math.Matrix
import com.navshield.map.engine.math.GeoUtils
import kotlin.math.*

class Hypothesis(val id: String, val ukf: UKF, var weight: Double, val roadSegmentId: String, var distanceTravelled: Double = 0.0)

class MultiHypothesisEngine {
    private var hypotheses = mutableListOf<Hypothesis>()
    fun update(sensorInput: NavShieldSensorState, mapResult: MapMatchResult?, dt: Double, refLat: Double, refLon: Double, gnssWeight: Double = 1.0): Hypothesis {
        if (hypotheses.isEmpty()) {
            val p = GeoUtils.project(sensorInput.latitude, sensorInput.longitude, refLat, refLon)
            val ukf = UKF(4); ukf.x[0,0]=p.first; ukf.x[1,0]=p.second; ukf.x[2,0]=sensorInput.speed; ukf.x[3,0]=Math.toRadians(GeoUtils.normalizeHeading(sensorInput.bearing))
            hypotheses.add(Hypothesis("main", ukf, 1.0, ""))
        }
        hypotheses.forEach { it.ukf.predict(dt); it.distanceTravelled += it.ukf.x[2, 0] * dt }
        hypotheses.forEach { h ->
            val p = GeoUtils.project(sensorInput.latitude, sensorInput.longitude, refLat, refLon)
            val zS = Matrix(4, 1); zS[0,0]=p.first; zS[1,0]=p.second; zS[2,0]=sensorInput.speed; zS[3,0]=Math.toRadians(GeoUtils.normalizeHeading(sensorInput.bearing))
            val rS = Matrix.identity(4) * (sensorInput.accuracy.coerceAtLeast(1.0) / (sensorInput.confidence.coerceIn(1e-6, 1.0) * gnssWeight.coerceIn(1e-6, 1.0)))
            h.ukf.update(zS, rS) { s -> val m = Matrix(4, 1); m[0,0]=s[0,0]; m[1,0]=s[1,0]; m[2,0]=s[2,0]; m[3,0]=s[3,0]; m }
        }
        return getBestHypothesis()
    }
    fun getBestHypothesis(): Hypothesis = hypotheses.maxByOrNull { it.weight } ?: hypotheses[0]
    fun getHypothesesList(): List<Hypothesis> = hypotheses
}
