"""
map_matching_service.py

The single entry point the rest of the team (Member 4, primarily) calls.
Wires together Phase C (graph) -> Phase D (spatial index) -> Phase E (basic
matching) -> Phase F (learned re-ranking, when available) -> Phase G
(multi-candidate output), and explicitly implements every edge case listed
in Section 6 of the spec so Member 4 never has to special-case this module's
behaviour.

Usage:
    service = MapMatchingService("sample_data/sample_hill_roads.json")
    result = service.match(MapMatchQuery(latitude=12.9716, longitude=77.5946,
                                          heading=10, speed=8))
"""

import json

from contract import MapMatchQuery, MapMatchResult, CandidateRoad
from phase_c_road_graph import build_road_graph
from phase_d_spatial_index import SpatialIndex
from phase_e_map_matching import score_candidates
from phase_g_multi_candidate import build_candidate_roads_output, is_ambiguous

try:
    from phase_f_gnn_lite_numpy import train_gnn_lite, score_candidates_learned
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False


NO_MATCH_SEARCH_RADIUS_M = 30.0


class MapMatchingService:
    def __init__(self, data_path, use_learned_reranking=True):
        with open(data_path) as f:
            raw = json.load(f)
        self.graph = build_road_graph(raw)
        self.spatial_index = SpatialIndex(self.graph)
        self._last_known = None  # sticky fallback for the tunnel-gap edge case

        self._gnn_model = None
        if use_learned_reranking and _GNN_AVAILABLE:
            self._gnn_model, _ = train_gnn_lite(self.graph, self.spatial_index, epochs=40)

    def match(self, query: MapMatchQuery) -> MapMatchResult:
        candidates = score_candidates(
            self.graph, self.spatial_index,
            query.latitude, query.longitude, query.heading, query.speed,
            top_k=5,
        )

        if self._gnn_model is not None and candidates:
            candidates = score_candidates_learned(self._gnn_model, candidates)

        # ---- Edge case: no OSM road coverage at all near this point ----
        # (tunnel with no tagged road, or a private/new road not in OSM)
        if not candidates:
            if self._last_known is not None:
                return self._sticky_fallback_result()
            return MapMatchResult(
                matched_latitude=query.latitude, matched_longitude=query.longitude,
                road_segment_id="", road_heading=query.heading,
                road_curvature=0.0, road_slope=0.0, road_slope_available=False,
                map_confidence=0.0, candidate_roads=[],
            )

        best = candidates[0]
        candidate_roads = [CandidateRoad(**c) for c in build_candidate_roads_output(self.graph, candidates)]

        # ---- Edge case: off-road (parking lot / driveway) ----
        # score_candidates already sets map_confidence=0.4 for this case and
        # zeroes the rest; we still return the nearest segment as a reference
        # but the caller (Member 4) is expected to NOT force-snap on low confidence.
        result = MapMatchResult(
            matched_latitude=self.graph.nodes[best["u"]]["lat"],
            matched_longitude=self.graph.nodes[best["u"]]["lon"],
            road_segment_id=best["segment_id"],
            road_heading=self.graph.edges[best["u"], best["v"]]["bearing"],
            road_curvature=best["curvature_deg"],
            road_slope=best["slope_pct"],
            road_slope_available=best["slope_available"],
            map_confidence=best["map_confidence"],
            candidate_roads=candidate_roads,
            one_way_violation=best["one_way_violation"],
        )

        # Keep this as the sticky fallback for the next tunnel-style gap.
        # Deliberately NOT gated on a confidence threshold: "last known
        # segment" per spec means the last time we had *any* real match,
        # not the last high-confidence one -- an ambiguous fork match is
        # still a better anchor for dead reckoning than nothing at all.
        self._last_known = result

        return result

    def _sticky_fallback_result(self):
        """Tunnel-with-no-coverage edge case: return the last known segment
        with a fixed low confidence (0.3) rather than crashing or returning
        nothing, per spec Section 6."""
        r = self._last_known
        return MapMatchResult(
            matched_latitude=r.matched_latitude, matched_longitude=r.matched_longitude,
            road_segment_id=r.road_segment_id, road_heading=r.road_heading,
            road_curvature=r.road_curvature, road_slope=r.road_slope,
            road_slope_available=r.road_slope_available,
            map_confidence=0.3, candidate_roads=[],
        )


if __name__ == "__main__":
    service = MapMatchingService("sample_data/sample_hill_roads.json")

    with open("sample_data/sample_hill_roads.json") as f:
        raw = json.load(f)
    pts = raw["test_reference_points"]

    print("\n-- Normal query near hairpin --")
    apex = service.graph.nodes[pts["hairpin_apex_node"]]
    print(service.match(MapMatchQuery(apex["lat"], apex["lon"], heading=140, speed=8)))

    print("\n-- Tunnel gap query (no road coverage) --")
    tp = pts["tunnel_query_point"]
    print(service.match(MapMatchQuery(tp["lat"], tp["lon"], heading=0, speed=15)))

    print("\n-- Off-road / parking-lot query --")
    op = pts["off_road_point"]
    print(service.match(MapMatchQuery(op["lat"], op["lon"], heading=90, speed=1)))
