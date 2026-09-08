package com.navshield.map.engine

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.navshield.map.contract.RoadNode
import com.navshield.map.contract.RoadSegment

open class SqliteRoadRepository(context: Context) : RoadRepository {

    private val dbHelper = RoadDbHelper(context)

    override fun getNode(id: String): RoadNode? {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            "nodes",
            arrayOf("id", "lat", "lon", "elevation"),
            "id = ?",
            arrayOf(id),
            null, null, null
        )
        
        return cursor.use {
            if (it.moveToFirst()) {
                RoadNode(
                    id = it.getString(0),
                    lat = it.getDouble(1),
                    lon = it.getDouble(2),
                    elevation = if (it.isNull(3)) null else it.getDouble(3)
                )
            } else null
        }
    }

    override fun getNearbySegments(lat: Double, lon: Double, radiusMeters: Double): List<RoadSegment> {
        val db = dbHelper.readableDatabase
        // Rough bounding box for efficiency before filtering by exact distance
        val delta = radiusMeters / 111320.0 // approx degrees
        
        val cursor = db.query(
            "segments",
            null,
            "mid_lat BETWEEN ? AND ? AND mid_lon BETWEEN ? AND ?",
            arrayOf(
                (lat - delta).toString(), (lat + delta).toString(),
                (lon - delta).toString(), (lon + delta).toString()
            ),
            null, null, null
        )

        val results = mutableListOf<RoadSegment>()
        cursor.use {
            while (it.moveToNext()) {
                results.add(it.toRoadSegment())
            }
        }
        return results
    }

    override fun isReady(): Boolean {
        return try {
            val db = dbHelper.readableDatabase
            val cursor = db.rawQuery("SELECT count(*) FROM nodes", null)
            val ready = cursor.use { it.moveToFirst() && it.getInt(0) > 0 }
            ready
        } catch (e: Exception) {
            false
        }
    }

    open fun insertRoadData(nodes: List<RoadNode>?, segments: List<Pair<RoadSegment, Pair<Double, Double>>>?) {
        if (nodes == null || segments == null) return
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            nodes.forEach { node ->
                val cv = android.content.ContentValues().apply {
                    put("id", node.id)
                    put("lat", node.lat)
                    put("lon", node.lon)
                    put("elevation", node.elevation)
                }
                db.insertWithOnConflict("nodes", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
            }
            
            segments.forEach { (segment, mid) ->
                val cv = android.content.ContentValues().apply {
                    put("segment_id", segment.segmentId)
                    put("way_id", segment.wayId)
                    put("u", segment.u)
                    put("v", segment.v)
                    put("mid_lat", mid.first)
                    put("mid_lon", mid.second)
                    put("length", segment.length)
                    put("bearing", segment.bearing)
                    put("road_type", segment.roadType)
                    put("one_way", if (segment.oneWay) 1 else 0)
                    put("max_speed", segment.maxSpeed)
                    put("slope", segment.slope)
                    put("slope_available", if (segment.slopeAvailable) 1 else 0)
                    put("curvature", segment.curvature)
                    put("junction", if (segment.junction) 1 else 0)
                    put("name", segment.name)
                }
                db.insertWithOnConflict("segments", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    private fun android.database.Cursor.toRoadSegment(): RoadSegment {
        return RoadSegment(
            segmentId = getString(getColumnIndexOrThrow("segment_id")),
            wayId = getString(getColumnIndexOrThrow("way_id")),
            u = getString(getColumnIndexOrThrow("u")),
            v = getString(getColumnIndexOrThrow("v")),
            length = getDouble(getColumnIndexOrThrow("length")),
            bearing = getDouble(getColumnIndexOrThrow("bearing")),
            roadType = getString(getColumnIndexOrThrow("road_type")),
            oneWay = getInt(getColumnIndexOrThrow("one_way")) == 1,
            maxSpeed = if (isNull(getColumnIndexOrThrow("max_speed"))) null else getDouble(getColumnIndexOrThrow("max_speed")),
            slope = getDouble(getColumnIndexOrThrow("slope")),
            slopeAvailable = getInt(getColumnIndexOrThrow("slope_available")) == 1,
            curvature = getDouble(getColumnIndexOrThrow("curvature")),
            junction = getInt(getColumnIndexOrThrow("junction")) == 1,
            name = getString(getColumnIndexOrThrow("name"))
        )
    }

    private class RoadDbHelper(context: Context) : SQLiteOpenHelper(context, "navshield_map.db", null, 1) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL("""
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    elevation REAL
                )
            """)
            db.execSQL("""
                CREATE TABLE segments (
                    segment_id TEXT PRIMARY KEY,
                    way_id TEXT NOT NULL,
                    u TEXT NOT NULL,
                    v TEXT NOT NULL,
                    mid_lat REAL NOT NULL,
                    mid_lon REAL NOT NULL,
                    length REAL NOT NULL,
                    bearing REAL NOT NULL,
                    road_type TEXT NOT NULL,
                    one_way INTEGER NOT NULL,
                    max_speed REAL,
                    slope REAL NOT NULL,
                    slope_available INTEGER NOT NULL,
                    curvature REAL NOT NULL,
                    junction INTEGER NOT NULL,
                    name TEXT,
                    FOREIGN KEY(u) REFERENCES nodes(id),
                    FOREIGN KEY(v) REFERENCES nodes(id)
                )
            """)
            db.execSQL("CREATE INDEX idx_segments_pos ON segments(mid_lat, mid_lon)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            // Placeholder
        }
    }
}
