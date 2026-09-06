package com.navshield.map.engine.math

import kotlin.math.cos
import kotlin.math.PI

/**
 * Geographic utility functions for local coordinate projection.
 */
object GeoUtils {
    private const val EARTH_RADIUS = 6371000.0 // meters

    /**
     * Projects Latitude/Longitude to local meters relative to a reference point.
     * Uses a local flat-earth approximation (equirectangular projection).
     * 
     * @return Pair(x_meters, y_meters) where x is East and y is North.
     */
    fun project(lat: Double, lon: Double, refLat: Double, refLon: Double): Pair<Double, Double> {
        val dLat = Math.toRadians(lat - refLat)
        val dLon = Math.toRadians(lon - refLon)
        val latRad = Math.toRadians(refLat)
        
        val y = dLat * EARTH_RADIUS
        val x = dLon * EARTH_RADIUS * cos(latRad)
        return Pair(x, y)
    }

    /**
     * Inverse projection from local meters back to Latitude/Longitude.
     */
    fun inverseProject(x: Double, y: Double, refLat: Double, refLon: Double): Pair<Double, Double> {
        val latRad = Math.toRadians(refLat)
        val dLat = y / EARTH_RADIUS
        val dLon = x / (EARTH_RADIUS * cos(latRad))
        
        val lat = Math.toDegrees(dLat) + refLat
        val lon = Math.toDegrees(dLon) + refLon
        return Pair(lat, lon)
    }
    
    fun normalizeHeading(heading: Double): Double {
        var h = heading % 360.0
        if (h < 0) h += 360.0
        return h
    }
}
