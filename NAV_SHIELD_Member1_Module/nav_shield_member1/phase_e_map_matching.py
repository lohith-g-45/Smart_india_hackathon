"""
Phase E -- Basic Map Matching

Given (lat, lon, heading, speed), scores nearby road-graph edges by:
  (a) perpendicular distance to the segment
  (b) angular difference between vehicle heading and road bearing
  (c) road-type plausibility (motorway ranks above service road etc., unless
      heading strongly disagrees -- handles the "motorway with parallel
      service road" edge case)

Returns the top-N ranked candidates. This is the fallback/baseline matcher
that Phase F's GNN is meant to improve on for path-level (not just
segment-level) reasoning -- and it is also what keeps the module functional
on its own if the GNN model isn't ready in time (a legitimate hackathon
fallback plan).
"""

from geo_utils import point_to_segment_distance_m, angular_diff_deg

ROAD_TYPE_RANK = {
    "motorway": 6, "trunk": 5, "primary": 5, "secondary": 4,
    "tertiary": 3, "residential": 2, "unclassified": 2, "service": 1,
    "living_street": 1,
}

# Distance beyond which we don't even consider a segment a match candidate.
MAX_MATCH_DISTANCE_M = 30.0
# Distance beyond which we treat the vehicle as genuinely off-road
# (parking lot / driveway edge case), rather than just noisy GPS.
OFF_ROAD_THRESHOLD_M = 20.0


def score_candidates(graph, spatial_index, lat, lon, heading, speed, top_k=5):
    nearby = spatial_index.query_nearby_edges(lat, lon, k=50)
    scored = []

    for u, v, data in nearby:
        lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        dist_m = point_to_segment_distance_m(lat, lon, lat_u, lon_u, lat_v, lon_v)
        if dist_m > MAX_MATCH_DISTANCE_M:
            continue

        heading_diff = angular_diff_deg(heading, data["bearing"])
        # a one-way segment traversed against its allowed direction is a
        # near-180 deg heading mismatch -- flag explicitly per spec.
        one_way_violation = data["one_way"] and heading_diff > 120

        type_rank = ROAD_TYPE_RANK.get(data["road_type"], 1)

        # Composite score: lower is better. Distance dominates, heading is a
        # strong secondary signal, road-type rank is a tie-breaker (e.g. a
        # motorway is preferred over its parallel service road *unless*
        # heading strongly disagrees, in which case distance/heading already
        # dominate the score and the tie-breaker doesn't matter).
        score = (dist_m * 1.0) + (heading_diff * 0.15) - (type_rank * 0.5)
        if one_way_violation:
            score += 50  # heavy penalty, not an automatic exclusion

        scored.append({
            "segment_id": data["segment_id"],
            "u": u, "v": v,
            "distance_m": round(dist_m, 2),
            "heading_diff_deg": round(heading_diff, 2),
            "road_type": data["road_type"],
            "curvature_deg": round(data["curvature_deg"], 2),
            "slope_pct": data["slope_pct"],
            "slope_available": data["slope_available"],
            "one_way_violation": one_way_violation,
            "raw_score": score,
        })

    scored.sort(key=lambda c: c["raw_score"])
    top = scored[:top_k]
    _attach_confidences(top)
    return top


def _attach_confidences(candidates):
    """Converts raw scores into 0-1 confidences via a simple softmin, and
    applies the off-road / no-candidate edge-case rules from Section 6."""
    if not candidates:
        return
    best_dist = candidates[0]["distance_m"]
    if best_dist > OFF_ROAD_THRESHOLD_M:
        # off-road (parking lot, driveway): don't force-snap, low confidence
        for c in candidates:
            c["map_confidence"] = 0.4 if c is candidates[0] else 0.0
        return

    scores = [c["raw_score"] for c in candidates]
    min_s = min(scores)
    weights = [1.0 / (1.0 + (s - min_s)) for s in scores]
    total = sum(weights)
    for c, w in zip(candidates, weights):
        c["map_confidence"] = round(w / total, 3)


if __name__ == "__main__":
    from phase_c_road_graph import load_raw, build_road_graph
    from phase_d_spatial_index import SpatialIndex

    raw = load_raw("sample_data/sample_hill_roads.json")
    g = build_road_graph(raw)
    idx = SpatialIndex(g)

    # Query right at the hairpin apex, heading roughly matching the outbound leg
    test = raw["test_reference_points"]
    apex_node = g.nodes[test["hairpin_apex_node"]]
    result = score_candidates(g, idx, apex_node["lat"], apex_node["lon"], heading=140, speed=8)
    print("Top candidates at hairpin apex:")
    for c in result:
        print(" ", c)
