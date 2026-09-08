package com.navshield.map.engine

import android.content.Context
import com.google.gson.Gson
import com.navshield.map.contract.RoadNode
import com.navshield.map.contract.RoadSegment
import com.navshield.map.engine.math.GeoUtils

/**
 * Utility for ingesting OSM JSON road data into the SQLite repository.
 */
class RoadDataIngestor(private val context: Context, private val repository: SqliteRoadRepository) {

    /**
     * Parses the road graph JSON and populates the repository.
     */
    fun ingest(jsonFileName: String) {
        val jsonString = context.assets.open(jsonFileName).bufferedReader().use { it.readText() }
        ingestJson(jsonString)
    }

    fun ingestJson(jsonString: String) {
        val gson = Gson()
        val data = gson.fromJson(jsonString, MapData::class.java) ?: return

        val nodes = data.nodes?.map { (id, node) ->
            RoadNode(id, node.lat, node.lon, data.elevation_m?.get(id))
        } ?: emptyList()

        val segments = mutableListOf<Pair<RoadSegment, Pair<Double, Double>>>()

        data.ways?.forEach { way ->
            val wayNodes = way.nodes ?: return@forEach
            for (i in 0 until wayNodes.size - 1) {
                val uId = wayNodes[i]
                val vId = wayNodes[i + 1]
                val u = data.nodes?.get(uId) ?: continue
                val v = data.nodes?.get(vId) ?: continue

                val length = GeoUtils.haversineDistance(u.lat, u.lon, v.lat, v.lon)
                val bearing = GeoUtils.initialBearing(u.lat, u.lon, v.lat, v.lon)
                val midLat = (u.lat + v.lat) / 2.0
                val midLon = (u.lon + v.lon) / 2.0

                val elevationU = data.elevation_m?.get(uId) ?: 0.0
                val elevationV = data.elevation_m?.get(vId) ?: 0.0
                val slope = if (length > 0) (elevationV - elevationU) / length * 100.0 else 0.0

                val curvature = 0.0 

                val segment = RoadSegment(
                    segmentId = "${way.id}_$i",
                    wayId = way.id ?: "",
                    u = uId,
                    v = vId,
                    length = length,
                    bearing = bearing,
                    roadType = way.tags?.get("highway") ?: "unclassified",
                    oneWay = way.tags?.get("oneway") == "yes",
                    maxSpeed = way.tags?.get("maxspeed")?.toDoubleOrNull(),
                    slope = slope,
                    slopeAvailable = data.elevation_m?.containsKey(uId) == true && data.elevation_m?.containsKey(vId) == true,
                    curvature = curvature,
                    junction = way.tags?.get("junction") == "roundabout",
                    name = way.tags?.get("name")
                )
                segments.add(segment to Pair(midLat, midLon))
                
                if (way.tags?.get("oneway") != "yes") {
                     val revSegment = RoadSegment(
                        segmentId = "${way.id}_${i}_rev",
                        wayId = way.id ?: "",
                        u = vId,
                        v = uId,
                        length = length,
                        bearing = GeoUtils.normalizeHeading(bearing + 180.0),
                        roadType = way.tags?.get("highway") ?: "unclassified",
                        oneWay = false,
                        maxSpeed = way.tags?.get("maxspeed")?.toDoubleOrNull(),
                        slope = -slope,
                        slopeAvailable = data.elevation_m?.containsKey(uId) == true && data.elevation_m?.containsKey(vId) == true,
                        curvature = curvature,
                        junction = way.tags?.get("junction") == "roundabout",
                        name = way.tags?.get("name")
                    )
                    segments.add(revSegment to Pair(midLat, midLon))
                }
            }
        }

        repository.insertRoadData(nodes, segments)
    }

    internal data class MapData(
        val nodes: Map<String, NodeJson>? = null,
        val ways: List<WayJson>? = null,
        val elevation_m: Map<String, Double>? = null
    )

    internal data class NodeJson(val lat: Double = 0.0, val lon: Double = 0.0)
    internal data class WayJson(val id: String? = null, val nodes: List<String>? = null, val tags: Map<String, String>? = null)
}
