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

    /**
     * Calculates the shortest distance from a point to a line segment.
     * 
     * @param px Point latitude
     * @param py Point longitude
     * @param x1 Segment start latitude
     * @param y1 Segment start longitude
     * @param x2 Segment end latitude
     * @param y2 Segment end longitude
     * @return Distance in meters
     */
    fun pointToSegmentDistance(px: Double, py: Double, x1: Double, y1: Double, x2: Double, y2: Double): Double {
        // Project to local meters for accurate distance calculation
        val p = project(px, py, x1, y1)
        val s1 = project(x1, y1, x1, y1) // (0,0)
        val s2 = project(x2, y2, x1, y1)
        
        val dx = s2.first - s1.first
        val dy = s2.second - s1.second
        
        if (dx == 0.0 && dy == 0.0) {
            return kotlin.math.sqrt(p.first * p.first + p.second * p.second)
        }
        
        val t = ((p.first - s1.first) * dx + (p.second - s1.second) * dy) / (dx * dx + dy * dy)
        
        return if (t < 0) {
            kotlin.math.sqrt((p.first - s1.first) * (p.first - s1.first) + (p.second - s1.second) * (p.second - s1.second))
        } else if (t > 1) {
            kotlin.math.sqrt((p.first - s2.first) * (p.first - s2.first) + (p.second - s2.second) * (p.second - s2.second))
        } else {
            val projX = s1.first + t * dx
            val projY = s1.second + t * dy
            kotlin.math.sqrt((p.first - projX) * (p.first - projX) + (p.second - projY) * (p.second - projY))
        }
    }

    fun angularDifference(a: Double, b: Double): Double {
        var diff = kotlin.math.abs(a - b) % 360.0
        if (diff > 180.0) diff = 360.0 - diff
        return diff
    }

    fun haversineDistance(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                Math.sin(dLon / 2) * Math.sin(dLon / 2)
        val c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
        return EARTH_RADIUS * c
    }

    fun initialBearing(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val phi1 = Math.toRadians(lat1)
        val phi2 = Math.toRadians(lat2)
        val deltaLambda = Math.toRadians(lon2 - lon1)
        
        val y = Math.sin(deltaLambda) * Math.cos(phi2)
        val x = Math.cos(phi1) * Math.sin(phi2) -
                Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda)
        
        return normalizeHeading(Math.toDegrees(Math.atan2(y, x)))
    }
}
