package com.navshield.map

import com.navshield.map.contract.*
import com.navshield.map.engine.*
import com.navshield.map.engine.math.GeoUtils
import com.navshield.map.members.*
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.*
import kotlin.math.*

class BlackoutValidationTest {
    private val START_LAT = 12.9716; private val START_LON = 77.5946; private val VEL = 10.0; private val DT = 0.1; private val ER = 6378137.0; private val rng = Random(42)

    private fun sim(name: String, dist: Double, turn: Boolean = false): Map<String, Any> {
        val results = mutableListOf<Double>(); var tDist = 0.0
        for (run in 1..3) {
            val ins = Member3InsEngine(); val brain = Member4Pipeline(); val m5 = MockM5()
            ins.initialize(0L)
            var time = 0.0; var gtLat = START_LAT; var gtLon = START_LON; var head = 0.0; var captured = false; var lastImuLat = START_LAT; var lastImuLon = START_LON
            val bDur = dist / VEL; val tTime = 5.0 + bDur / 2.0
            while (time < 5.0 + bDur + 2.0) {
                val isB = time >= 5.0 && time <= 5.0 + bDur
                if (turn && isB && time > tTime && time <= tTime + (0.1 * bDur)) head += (PI / 2.0) * (DT / (0.1 * bDur))
                val d = VEL * DT; val latR = gtLat * PI / 180.0
                gtLat += (d * cos(head) / ER) * (180.0 / PI)
                gtLon += (d * sin(head) / (ER * cos(latR))) * (180.0 / PI)
                if (isB && run == 1) tDist += d
                val s = NavShieldSensorState(gtLat, gtLon, 0.0, VEL, Math.toDegrees(head), if (isB) 100.0 else 1.0, 12, if (isB) 0.01 else 1.0, isB, emptyList(), if (isB) "DENIED" else "STRONG", 0.0, 0.0, 9.8, 0.0, 0.0, if (turn && isB && time > tTime) 1.0 else 0.0, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Math.toDegrees(head), 0.0, 0.0, (time * 1000).toLong())
                if (!isB && time < 5.0) ins.resetFromExternalState(s.latitude, s.longitude, s.bearing.toFloat(), s.speed.toFloat(), (time * 1000).toLong())
                val imu = ins.update(s)
                if (imu != null && !imu.imuLatitude.isNaN()) { lastImuLat = imu.imuLatitude; lastImuLon = imu.imuLongitude }
                val tState = brain.update(s, null, imu, m5.predict(s), (time * 1000).toLong())
                if (time >= 5.0 + bDur && !captured) { results.add(GeoUtils.haversineDistance(tState.final_latitude, tState.final_longitude, gtLat, gtLon)); captured = true }
                time += DT
            }
        }
        return mapOf("name" to name, "distance" to tDist, "runErrors" to results, "avg" to results.average())
    }

    class MockM5 : Member5DriftGuardian { override fun predict(s: NavShieldSensorState) = Member5Result(if (s.isAnomalous) 10f else 0.5f, 0.1f, s.timestampMillis, s.isAnomalous, s.isAnomalous, "N")
        override fun getStatus() = MemberStatus.CONNECTED }

    @Test
    fun `VALIDATE SIH REQ - Full M1-M5 Pipeline`() {
        val scens = listOf(sim("50m Straight", 50.0), sim("100m Straight", 100.0), sim("1km Straight", 1000.0), sim("1km Turn", 1000.0, true))
        val sb = StringBuilder(); sb.append("Software-controlled simulated IMU / GNSS blackout test.\n\n| Test | Distance | Run 1 | Run 2 | Run 3 | Avg Error | Drift % | Result |\n|------|----------|-------|-------|-------|-----------|---------|--------|\n")
        var allPassed = true
        for (s in scens) {
            val dist = s["distance"] as Double; val errs = s["runErrors"] as List<Double>; val avg = s["avg"] as Double; val pct = (avg / dist) * 100.0; val lim = dist * 0.1; val pass = avg < lim
            if (!pass) allPassed = false
            sb.append("| ${s["name"]} | ${"%.0f".format(dist)}m | ${"%.2f".format(errs[0])}m | ${"%.2f".format(errs[1])}m | ${"%.2f".format(errs[2])}m | ${"%.2f".format(avg)}m | ${"%.2f".format(pct)}% | ${if(pass) "PASS" else "FAIL"} |\n")
        }
        File("D:/Smart_india_hackathon/blackout_report.txt").writeText(sb.toString()); println(sb.toString())
        assertTrue("SIH benchmark failed!", allPassed)
    }
}
