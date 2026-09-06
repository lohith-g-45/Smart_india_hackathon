"""
Phase G -- Multi-Candidate Output

When multiple roads are plausible (fork, junction, roundabout entry), emits
ALL plausible candidates with honest, distinct confidence scores for
Member 4's Multi-Hypothesis engine, instead of collapsing to a single
"best guess" too early. Ambiguity is high when 2+ candidates are within
AMBIGUITY_DISTANCE_M of each other (spec: "within 15 m").
"""

AMBIGUITY_DISTANCE_M = 15.0


def build_candidate_roads_output(graph, ranked_candidates):
    """
    Converts ranked candidates (from Phase E or Phase F) into the exact
    `candidate_roads[]` contract format Member 4 expects:
        segment_id, score, lat, lon, heading, curvature, slope
    `lat`/`lon` here are the projected match point isn't computed at this
    stage (Phase E doesn't currently return the projected point, only
    distance) -- for the contract we report the segment's start-node
    position as a stable reference point; swap in the true projected point
    if/when Phase E is extended to return it.
    """
    output = []
    for c in ranked_candidates:
        u = c["u"]
        node = graph.nodes[u]
        output.append({
            "segment_id": c["segment_id"],
            "score": c["map_confidence"],
            "lat": node["lat"],
            "lon": node["lon"],
            "heading": graph.edges[c["u"], c["v"]]["bearing"],
            "curvature": c["curvature_deg"],
            "slope": c["slope_pct"],
        })
    return output


def is_ambiguous(ranked_candidates):
    """True when 2+ top candidates are close enough in distance that Member 4
    should treat this as a genuine multi-hypothesis situation rather than a
    confident single match."""
    close = [c for c in ranked_candidates if c["distance_m"] <= AMBIGUITY_DISTANCE_M]
    return len(close) >= 2


if __name__ == "__main__":
    import json
    from phase_c_road_graph import build_road_graph
    from phase_d_spatial_index import SpatialIndex
    from phase_e_map_matching import score_candidates

    with open("sample_data/sample_hill_roads.json") as f:
        raw = json.load(f)
    g = build_road_graph(raw)
    idx = SpatialIndex(g)

    fork_point = raw["test_reference_points"]["fork_test_point"]
    candidates = score_candidates(g, idx, fork_point["lat"], fork_point["lon"], heading=27, speed=10)
    print("Ambiguous (fork) situation detected:", is_ambiguous(candidates))
    for row in build_candidate_roads_output(g, candidates):
        print(" ", row)
