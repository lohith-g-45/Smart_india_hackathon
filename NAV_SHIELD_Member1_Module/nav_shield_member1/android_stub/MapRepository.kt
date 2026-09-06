/*
 * MapRepository.kt
 *
 * Android-side integration stub for Member 1's module. This is scaffolding,
 * not a finished implementation -- it shows the two things the spec commits
 * you to on the Android side:
 *   1. A tile-serving path for Member 6's MapLibre SDK to render the
 *      offline .mbtiles file.
 *   2. A SQLite query surface for on-device nearest-segment lookups, for
 *      cases where the Python backend result is being cached/queried
 *      locally instead of recomputed every cycle.
 *
 * The heavy lifting (graph construction, spatial indexing, GNN inference)
 * stays in the Python pipeline / ONNX Runtime Mobile; this class is the
 * thin Android-side data-access layer around its outputs.
 */

package com.navshield.map

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import java.io.File

data class CandidateRoad(
    val segmentId: String,
    val score: Double,
    val lat: Double,
    val lon: Double,
    val heading: Double,
    val curvature: Double,
    val slope: Double
)

data class MapMatchResult(
    val matchedLatitude: Double,
    val matchedLongitude: Double,
    val roadSegmentId: String,
    val roadHeading: Double,
    val roadCurvature: Double,
    val roadSlope: Double,
    val mapConfidence: Double,
    val candidateRoads: List<CandidateRoad>
)

class MapRepository(private val context: Context) {

    private val dbPath: String by lazy {
        File(context.filesDir, "navshield_roads.sqlite").absolutePath
    }

    private val mbtilesPath: String by lazy {
        File(context.filesDir, "navshield_map.mbtiles").absolutePath
    }

    /** Exposed to Member 6's MapLibre setup: MapLibre reads .mbtiles files
     * directly given a file path, no HTTP server needed for fully offline
     * rendering. */
    fun getOfflineTilePath(): String = mbtilesPath

    /**
     * On-device nearest-segment lookup against the pre-built spatial index
     * table (populated from Phase D's output when the offline DB is
     * downloaded/updated). This mirrors phase_e_map_matching.score_candidates
     * but as a lightweight on-device query for when you don't want a round
     * trip to the Python/ONNX pipeline for every single fix.
     */
    fun queryNearbySegments(lat: Double, lon: Double, radiusDeg: Double = 0.01, limit: Int = 8): List<CandidateRoad> {
        val db = SQLiteDatabase.openDatabase(dbPath, null, SQLiteDatabase.OPEN_READONLY)
        val results = mutableListOf<CandidateRoad>()
        db.use {
            val cursor = it.rawQuery(
                """
                SELECT segment_id, lat, lon, heading, curvature, slope, road_type
                FROM road_segments
                WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
                LIMIT ?
                """.trimIndent(),
                arrayOf(
                    (lat - radiusDeg).toString(), (lat + radiusDeg).toString(),
                    (lon - radiusDeg).toString(), (lon + radiusDeg).toString(),
                    limit.toString()
                )
            )
            cursor.use { c ->
                while (c.moveToNext()) {
                    results.add(
                        CandidateRoad(
                            segmentId = c.getString(0),
                            score = 0.0, // scored on the Python/ONNX side, not here
                            lat = c.getDouble(1),
                            lon = c.getDouble(2),
                            heading = c.getDouble(3),
                            curvature = c.getDouble(4),
                            slope = c.getDouble(5)
                        )
                    )
                }
            }
        }
        return results
    }

    /** Startup integrity check per Section 6: "Offline DB file corrupted /
     * incomplete download -- Detect on startup. Alert user. Do not silently
     * serve wrong data." */
    fun verifyOfflineDataIntegrity(): Boolean {
        val dbFile = File(dbPath)
        val tilesFile = File(mbtilesPath)
        if (!dbFile.exists() || !tilesFile.exists()) return false
        return try {
            val db = SQLiteDatabase.openDatabase(dbPath, null, SQLiteDatabase.OPEN_READONLY)
            val cursor = db.rawQuery("SELECT COUNT(*) FROM road_segments", null)
            val ok = cursor.use { it.moveToFirst() && it.getInt(0) > 0 }
            db.close()
            ok
        } catch (e: Exception) {
            false
        }
    }
}
