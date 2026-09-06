"""
run_tests_manual.py

test_module.py is written as standard pytest (the right tool for your
team's CI). This sandbox has no network access to install pytest, so this
script calls the exact same test functions directly with manually-built
fixtures, purely to prove the test logic itself is correct right now.
Once your team runs `pip install pytest`, use test_module.py directly --
delete this file at that point, it's a sandbox-only stand-in.
"""

import json
import sys
import tempfile
import traceback
import types
import contextlib

# --- minimal pytest shim, since pytest can't be installed in this sandbox ---
if "pytest" not in sys.modules:
    fake_pytest = types.ModuleType("pytest")
    fake_pytest.fixture = lambda *a, **kw: (lambda f: f)

    @contextlib.contextmanager
    def _raises(exc_type):
        try:
            yield
        except exc_type:
            return
        else:
            raise AssertionError(f"expected {exc_type} to be raised")

    fake_pytest.raises = _raises
    fake_pytest.main = lambda *a, **kw: 0
    sys.modules["pytest"] = fake_pytest
# -----------------------------------------------------------------------------

from phase_c_road_graph import build_road_graph
from phase_d_spatial_index import SpatialIndex
from map_matching_service import MapMatchingService
import test_module as tm

with open("sample_data/sample_hill_roads.json") as f:
    raw_data = json.load(f)
graph = build_road_graph(raw_data)
spatial_index = SpatialIndex(graph)
service = MapMatchingService("sample_data/sample_hill_roads.json")


class FakeTmpPath:
    def __truediv__(self, name):
        import pathlib
        return pathlib.Path(tempfile.gettempdir()) / name


tests = [
    ("test_nearest_segment_lookup", lambda: tm.test_nearest_segment_lookup(graph, spatial_index)),
    ("test_heading_filter", lambda: tm.test_heading_filter(graph, spatial_index)),
    ("test_gnn_accuracy", lambda: tm.test_gnn_accuracy(graph, spatial_index)),
    ("test_latency", lambda: tm.test_latency(graph, spatial_index)),
    ("test_curvature_hairpin", lambda: tm.test_curvature_hairpin(graph)),
    ("test_multi_candidate_at_fork", lambda: tm.test_multi_candidate_at_fork(raw_data, graph, spatial_index)),
    ("test_tunnel_gap_returns_last_known_not_crash", lambda: tm.test_tunnel_gap_returns_last_known_not_crash(raw_data, service)),
    ("test_off_road_low_confidence_no_force_snap", lambda: tm.test_off_road_low_confidence_no_force_snap(raw_data, service)),
    ("test_one_way_violation_flagged", lambda: tm.test_one_way_violation_flagged(graph, spatial_index)),
    ("test_roundabout_modelled_as_curved_edges", lambda: tm.test_roundabout_modelled_as_curved_edges(graph)),
    ("test_service_handles_missing_data_gracefully", lambda: tm.test_service_handles_missing_data_gracefully(FakeTmpPath())),
]

passed, failed = 0, 0
for name, fn in tests:
    try:
        fn()
        print(f"PASS  {name}")
        passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {e}")
        traceback.print_exc(limit=1)
        failed += 1

print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
sys.exit(1 if failed else 0)
