"""
geo_utils.py
Pure-math geographic helper functions shared by every phase of Member 1's module.

Why pure math instead of geopy/pyproj?
The formulas below (haversine distance, initial bearing, destination point,
angular difference) are exactly what geopy/pyproj compute internally for
short-range navigation distances (< a few km), and implementing them directly
means this module has zero heavyweight geo-library dependency on the Android
side or in constrained CI environments. If your team prefers geopy/pyproj for
consistency with the spec's tech-stack list, swap the function bodies below --
the signatures are what the rest of the module calls, so nothing else changes.
"""

import math

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing (0-360, 0=North) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def destination_point(lat1, lon1, bearing_deg, distance_m):
    """Given a start point, bearing and distance, compute the resulting lat/lon."""
    delta = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)

    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), (math.degrees(lambda2) + 540) % 360 - 180


def angular_diff_deg(a, b):
    """Smallest absolute difference between two compass bearings, 0-180."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def point_to_segment_distance_m(plat, plon, alat, alon, blat, blon):
    """
    Approximate perpendicular distance in metres from point P to the line
    segment A-B. Uses an equirectangular local projection, which is accurate
    enough for segments a few hundred metres long (typical road-graph edges).
    """
    # Local flat-earth projection centred on segment midpoint
    lat0 = math.radians((alat + blat) / 2.0)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat0)
    m_per_deg_lon = 111412.84 * math.cos(lat0)

    def to_xy(lat, lon):
        return (lon - alon) * m_per_deg_lon, (lat - alat) * m_per_deg_lat

    ax, ay = 0.0, 0.0
    bx, by = to_xy(blat, blon)
    px, py = to_xy(plat, plon)

    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def curvature_deg(bearing_in, bearing_out):
    """
    Curvature of a road at a node, defined as the turn angle (0-180 degrees)
    between the incoming and outgoing edge bearings. >60 deg is treated as a
    sharp bend/hairpin per the module's validation spec.
    """
    return angular_diff_deg(bearing_in, bearing_out)
