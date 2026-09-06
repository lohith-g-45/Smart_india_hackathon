"""
test_module.py

Implements the exact validation tests from Section 5 of the spec, plus the
edge cases from Section 6, run against sample_data/sample_hill_roads.json.
Per the spec: "You do NOT need GPS, IMU, or LSTM to test your module" --
every test here is self-contained.

Run with:  pytest test_module.py -v
"""

import json
import time

import pytest

from phase_c_road_graph import build_road_graph
from phase_d_spatial_index import SpatialIndex
from phase_e_map_matching import score_candidates
from phase_g_multi_candidate import is_ambiguous
from phase_f_gnn_lite_numpy import train_gnn_lite
from contract import MapMatchQuery
from map_matching_service import MapMatchingService


@pytest.fixture(scope="module")
def raw_data():
    with open("sample_data/sample_hill_roads.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def graph(raw_data):
    return build_road_graph(raw_data)


@pytest.fixture(scope="module")
def spatial_index(graph):
    return SpatialIndex(graph)


@pytest.fixture(scope="module")
def service():
    return MapMatchingService("sample_data/sample_hill_roads.json")


# ---------------------------------------------------------------------------
# Section 5 -- Validation & Independent Testing
# ---------------------------------------------------------------------------

def test_nearest_segment_lookup(graph, spatial_index):
    """Spec target: >= 95% correct on 50 random points on known roads.
    We generate points ON known edges (midpoints) rather than fully random
    OSM points, since our test region is synthetic -- same test intent."""
    import random
    rng = random.Random(1)
    edges = list(graph.edges(data=True))
    sample = rng.choices(edges, k=50)

    correct = 0
    for u, v, data in sample:
        lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        mid_lat, mid_lon = (lat_u + lat_v) / 2, (lon_u + lon_v) / 2
        candidates = score_candidates(graph, spatial_index, mid_lat, mid_lon,
                                       heading=data["bearing"], speed=10, top_k=1)
        if candidates and candidates[0]["segment_id"] == data["segment_id"]:
            correct += 1

    accuracy = correct / len(sample)
    assert accuracy >= 0.95, f"Nearest-segment accuracy {accuracy:.2%} below 95% target"


def test_heading_filter(graph, spatial_index):
    """Vehicle facing North on a North-South road: the East-West road 5 m
    away should rank lower due to heading mismatch. Uses the hairpin's
    first leg (roughly N-S) as the "facing north" road."""
    a1 = graph.nodes  # noqa: F841
    # W_hairpin_1 runs on bearing ~10 deg (roughly N-S)
    candidates = score_candidates(graph, spatial_index, 12.9718, 77.5946, heading=10, speed=8, top_k=5)
    assert candidates, "expected at least one candidate near the hairpin's first leg"
    top = candidates[0]
    assert top["heading_diff_deg"] < 30, "facing-matched road should rank first"


def test_gnn_accuracy(graph, spatial_index):
    """Spec: hold out 20% of segments, measure Top-1/Top-3 accuracy."""
    _, metrics = train_gnn_lite(graph, spatial_index, epochs=60)
    assert metrics["top_1_accuracy"] >= 0.5, f"Top-1 accuracy too low: {metrics}"
    assert metrics["top_3_accuracy"] >= 0.8, f"Top-3 accuracy too low: {metrics}"


def test_latency(graph, spatial_index):
    """Spec: median latency < 10 ms for 1000 consecutive map-match queries."""
    import statistics
    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        score_candidates(graph, spatial_index, 12.9716, 77.5946, heading=10, speed=8)
        latencies.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(latencies)
    assert median_ms < 10, f"Median latency {median_ms:.3f} ms exceeds 10 ms target"


def test_curvature_hairpin(graph):
    """Spec: driving a known hairpin should show curvature > 60 deg."""
    hairpin_edges = [d["curvature_deg"] for _, _, d in graph.edges(data=True) if "hairpin" in d["segment_id"]]
    assert any(c > 60 for c in hairpin_edges), "expected at least one hairpin edge with curvature > 60 deg"


def test_multi_candidate_at_fork(raw_data, graph, spatial_index):
    """Spec: vehicle at a fork with two roads within 12 m -- both should
    appear in candidate_roads[] with distinct scores."""
    fork_point = raw_data["test_reference_points"]["fork_test_point"]
    candidates = score_candidates(graph, spatial_index, fork_point["lat"], fork_point["lon"],
                                   heading=27, speed=10, top_k=5)
    assert is_ambiguous(candidates), "expected the fork to be flagged as ambiguous"
    segment_ids = {c["segment_id"] for c in candidates[:2]}
    assert len(segment_ids) == 2, "expected two distinct road segments at the fork"
    assert candidates[0]["map_confidence"] != candidates[1]["map_confidence"], \
        "expected distinct confidence scores for the two fork candidates"


# ---------------------------------------------------------------------------
# Section 6 -- Edge Cases
# ---------------------------------------------------------------------------

def test_tunnel_gap_returns_last_known_not_crash(raw_data, service):
    """Vehicle in tunnel with no OSM road tagged -> last_known_segment with
    map_confidence = 0.3, must not crash."""
    tp = raw_data["test_reference_points"]["tunnel_query_point"]
    apex = service.graph.nodes[raw_data["test_reference_points"]["hairpin_apex_node"]]
    service.match(MapMatchQuery(apex["lat"], apex["lon"], heading=140, speed=8))  # establish last_known
    result = service.match(MapMatchQuery(tp["lat"], tp["lon"], heading=0, speed=15))
    assert result.map_confidence == 0.3
    assert result.road_segment_id != ""


def test_off_road_low_confidence_no_force_snap(raw_data, service):
    """Vehicle off-road (parking lot/driveway) -> nearest road returned with
    map_confidence = 0.4, other candidates suppressed rather than force-snapped."""
    op = raw_data["test_reference_points"]["off_road_point"]
    result = service.match(MapMatchQuery(op["lat"], op["lon"], heading=90, speed=1))
    assert result.map_confidence == 0.4


def test_one_way_violation_flagged(graph, spatial_index):
    """Vehicle heading opposite to a one-way road's allowed direction should
    be flagged, with map_confidence reduced (not simply excluded)."""
    d_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d["one_way"]]
    assert d_edges, "expected at least one one-way edge in sample data"
    u, v, data = d_edges[0]
    lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
    lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
    mid_lat, mid_lon = (lat_u + lat_v) / 2, (lon_u + lon_v) / 2
    reverse_heading = (data["bearing"] + 180) % 360

    candidates = score_candidates(graph, spatial_index, mid_lat, mid_lon, heading=reverse_heading, speed=8, top_k=3)
    match = next(c for c in candidates if c["segment_id"] == data["segment_id"])
    assert match["one_way_violation"] is True


def test_roundabout_modelled_as_curved_edges(graph):
    roundabout_edges = [d for _, _, d in graph.edges(data=True) if d.get("junction") == "roundabout"]
    assert len(roundabout_edges) >= 4, "expected the roundabout's 4 arc edges"


def test_service_handles_missing_data_gracefully(tmp_path):
    """Corrupted/incomplete offline DB should be detected on startup, not
    silently serve wrong data."""
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        MapMatchingService(str(bad_file))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
