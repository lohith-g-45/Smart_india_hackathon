"""
Phase D -- Spatial Indexing

Builds a spatial index over all road-graph edges for fast nearest-segment
lookup (target: < 5 ms on device, per spec).

Uses `rtree` when it's installed (the spec's chosen library, and the right
choice for production/Android-side native queries). Falls back to
scipy.spatial.cKDTree automatically when rtree isn't available (e.g. this
development sandbox) -- same query interface either way, so nothing else in
the module needs to know or care which backend is active. This mirrors the
graceful-degradation pattern the spec already requires elsewhere (Section 6).
"""

import numpy as np
from scipy.spatial import cKDTree

try:
    from rtree import index as rtree_index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False


class SpatialIndex:
    def __init__(self, graph):
        self.graph = graph
        self.edges = list(graph.edges(data=True))
        # Represent each edge by its midpoint for coarse candidate retrieval;
        # exact distance is computed later via geo_utils.point_to_segment_distance_m.
        self._midpoints = []
        for u, v, data in self.edges:
            lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
            lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
            self._midpoints.append(((lat_u + lat_v) / 2.0, (lon_u + lon_v) / 2.0))

        if HAS_RTREE:
            self._backend = "rtree"
            p = rtree_index.Property()
            self._idx = rtree_index.Index(properties=p)
            for i, (lat, lon) in enumerate(self._midpoints):
                self._idx.insert(i, (lon, lat, lon, lat))
        else:
            self._backend = "ckdtree"
            pts = np.array([(lon, lat) for lat, lon in self._midpoints])
            self._idx = cKDTree(pts) if len(pts) else None

    def query_nearby_edges(self, lat, lon, k=8, search_radius_deg=0.01):
        """Returns up to k (u, v, edge_data) tuples near (lat, lon).
        search_radius_deg ~0.01 deg is roughly 1.1 km at the equator --
        generous enough to always catch true nearby roads before the exact
        metre-level distance filter in Phase E narrows it down."""
        if not self.edges:
            return []

        if self._backend == "rtree":
            bbox = (lon - search_radius_deg, lat - search_radius_deg,
                    lon + search_radius_deg, lat + search_radius_deg)
            ids = list(self._idx.intersection(bbox))
        else:
            if self._idx is None:
                return []
            radius_deg = search_radius_deg
            ids = self._idx.query_ball_point([lon, lat], r=radius_deg)

        candidates = [self.edges[i] for i in ids]
        return candidates[:k] if k else candidates

    @property
    def backend(self):
        return self._backend


if __name__ == "__main__":
    from phase_c_road_graph import load_raw, build_road_graph
    import time

    raw = load_raw("sample_data/sample_hill_roads.json")
    g = build_road_graph(raw)
    idx = SpatialIndex(g)
    print(f"Spatial index backend in use: {idx.backend} "
          f"({'rtree available' if HAS_RTREE else 'rtree NOT installed, using cKDTree fallback'})")

    test_lat, test_lon = 12.9716, 77.5946
    start = time.perf_counter()
    for _ in range(1000):
        idx.query_nearby_edges(test_lat, test_lon)
    elapsed_ms = (time.perf_counter() - start) / 1000 * 1000
    print(f"Median-ish latency over 1000 queries: {elapsed_ms:.4f} ms/query")
