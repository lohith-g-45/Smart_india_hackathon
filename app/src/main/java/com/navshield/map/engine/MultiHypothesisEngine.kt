package com.navshield.map.engine

import com.navshield.map.contract.CandidateRoad
import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult
import com.navshield.map.engine.math.Matrix
import com.navshield.map.engine.math.GeoUtils
import kotlin.math.exp
import kotlin.math.sqrt

class Hypothesis(
    val id: String,
    val ukf: UKF,
    var weight: Double,
    val roadSegmentId: String,
    var distanceTravelled: Double = 0.0
)

class MultiHypothesisEngine {
    private var hypotheses = mutableListOf<Hypothesis>()
    private val maxHypotheses = 5

    fun update(query: MapMatchQuery, mapResult: MapMatchResult?, dt: Double, refLat: Double, refLon: Double): Hypothesis {
        if (hypotheses.isEmpty()) {
            val initialUkf = UKF(4)
            initialUkf.x[0, 0] = 0.0
            initialUkf.x[1, 0] = 0.0
            initialUkf.x[2, 0] = query.speed
            initialUkf.x[3, 0] = Math.toRadians(GeoUtils.normalizeHeading(query.heading))
            hypotheses.add(Hypothesis("main", initialUkf, 1.0, ""))
        }

        hypotheses.forEach { 
            it.ukf.predict(dt)
            it.distanceTravelled += it.ukf.x[2, 0] * dt
        }

        if (mapResult != null && mapResult.candidateRoads.size > 1) {
            spawnHypotheses(mapResult)
        }

        hypotheses.forEach { hypo ->
            // Sensor update
            val pos = GeoUtils.project(query.latitude, query.longitude, refLat, refLon)
            val zSensor = Matrix(3, 1)
            zSensor[0, 0] = pos.first
            zSensor[1, 0] = pos.second
            zSensor[2, 0] = Math.toRadians(GeoUtils.normalizeHeading(query.heading))
            
            val rSensor = Matrix.identity(3) * 5.0
            hypo.ukf.update(zSensor, rSensor) { s -> 
                val m = Matrix(3, 1)
                m[0,0] = s[0,0]; m[1,0] = s[1,0]; m[2,0] = s[3,0]
                m
            }

            // Map update
            val road = mapResult?.candidateRoads?.find { it.segmentId == hypo.roadSegmentId }
            if (road != null) {
                val roadPos = GeoUtils.project(road.lat, road.lon, refLat, refLon)
                val zMap = Matrix(3, 1)
                zMap[0, 0] = roadPos.first
                zMap[1, 0] = roadPos.second
                zMap[2, 0] = Math.toRadians(GeoUtils.normalizeHeading(road.heading))
                
                val rMap = Matrix.identity(3) * (2.0 / (road.score + 0.01))
                hypo.ukf.update(zMap, rMap) { s -> 
                    val m = Matrix(3, 1)
                    m[0,0] = s[0,0]; m[1,0] = s[1,0]; m[2,0] = s[3,0]
                    m
                }
                
                // Phase F requirement: correct hypothesis weight must grow
                hypo.weight *= (1.0 + road.score * 0.2)
            } else if (hypo.roadSegmentId.isNotEmpty()) {
                hypo.weight *= 0.8
            }
        }

        normalizeWeights()
        pruneHypotheses()

        return getBestHypothesis()
    }

    private fun spawnHypotheses(mapResult: MapMatchResult) {
        mapResult.candidateRoads.forEach { candidate ->
            if (hypotheses.none { it.roadSegmentId == candidate.segmentId }) {
                if (hypotheses.size < maxHypotheses) {
                    val best = getBestHypothesis()
                    val newUkf = UKF(4)
                    newUkf.x = best.ukf.x.copy()
                    newUkf.P = best.ukf.P.copy()
                    hypotheses.add(Hypothesis(candidate.segmentId, newUkf, candidate.score * 0.5, candidate.segmentId))
                }
            }
        }
    }

    private fun normalizeWeights() {
        val total = hypotheses.sumOf { it.weight }
        if (total > 0) {
            hypotheses.forEach { it.weight /= total }
        }
    }

    private fun pruneHypotheses() {
        if (hypotheses.size > 1) {
            hypotheses.removeIf { it.weight < 0.05 }
        }
    }

    fun getBestHypothesis(): Hypothesis = hypotheses.maxByOrNull { it.weight } ?: hypotheses[0]
    
    fun getHypothesesList(): List<Hypothesis> = hypotheses
}
