"""
generate_sample_data.py

Produces sample_hill_roads.json: a synthetic OSM-lite road network standing
in for a real Overpass API download. This exists ONLY because this
development sandbox has no network access to hit the real Overpass API
(Phase A). On your machines, phase_a_osm_ingest.py will pull real data for
Bengaluru + Western Ghats and this file becomes unnecessary -- everything
downstream (Phase C onward) reads the exact same schema either way, so
nothing else in the module needs to change when you switch to real data.

The network below is centred near Bengaluru (12.9716 N, 77.5946 E) and
deliberately includes one instance of every edge case listed in Section 6
of the spec, so test_module.py can validate against all of them:

  - hairpin_road      : sharp >60 deg bend (curvature test)
  - fork_junction     : two roads within 12 m of a point (multi-candidate test)
  - roundabout        : modelled as 4 short curved edges
  - one_way_road      : oneway=yes tag
  - motorway_service  : motorway + parallel service road 10 m apart
  - tunnel_gap        : a deliberate missing link (no OSM way tagged) simulating
                        a tunnel with no road coverage
  - off_road_point    : used directly in tests, needs no dedicated way
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geo_utils import destination_point  # noqa: E402

nodes = {}
ways = []
_next_node_id = [1]


def add_node(lat, lon):
    nid = f"N{_next_node_id[0]}"
    _next_node_id[0] += 1
    nodes[nid] = {"lat": lat, "lon": lon}
    return nid


def add_way(way_id, node_ids, tags):
    ways.append({"id": way_id, "nodes": node_ids, "tags": tags})


# ---------------------------------------------------------------------------
# 1. Hairpin hill road: A1 -> A2 -> A3, sharp turn at A2 (~130 deg change)
# ---------------------------------------------------------------------------
a1 = add_node(12.9716, 77.5946)
lat, lon = destination_point(12.9716, 77.5946, 10, 60)   # 60 m heading N (10 deg)
a2 = add_node(lat, lon)
lat, lon = destination_point(lat, lon, 140, 60)          # sharp hairpin turn to bearing 140
a3 = add_node(lat, lon)
add_way("W_hairpin_1", [a1, a2], {"highway": "trunk", "maxspeed": "30", "name": "Ghat Road"})
add_way("W_hairpin_2", [a2, a3], {"highway": "trunk", "maxspeed": "20", "name": "Ghat Road"})

# ---------------------------------------------------------------------------
# 2. Fork junction: B1 splits into B2 (main) and B3 (alt), ~10 m apart near B1
# ---------------------------------------------------------------------------
b1 = add_node(12.9750, 77.5946)
lat, lon = destination_point(12.9750, 77.5946, 20, 80)
b2 = add_node(lat, lon)
lat, lon = destination_point(12.9750, 77.5946, 35, 80)   # diverges at a shallow angle
b3 = add_node(lat, lon)
add_way("W_fork_main", [b1, b2], {"highway": "secondary", "maxspeed": "40", "name": "Main Fork Road"})
add_way("W_fork_alt", [b1, b3], {"highway": "tertiary", "maxspeed": "30", "name": "Alt Fork Road"})

# ---------------------------------------------------------------------------
# 3. Roundabout: 4 short curved edges around a centre point
# ---------------------------------------------------------------------------
rc_lat, rc_lon = 12.9770, 77.5960
r_nodes = []
for bearing in (0, 90, 180, 270):
    lat, lon = destination_point(rc_lat, rc_lon, bearing, 15)
    r_nodes.append(add_node(lat, lon))
for i in range(4):
    add_way(f"W_roundabout_{i}", [r_nodes[i], r_nodes[(i + 1) % 4]],
            {"highway": "primary", "junction": "roundabout", "maxspeed": "20"})

# ---------------------------------------------------------------------------
# 4. One-way road
# ---------------------------------------------------------------------------
d1 = add_node(12.9790, 77.5946)
lat, lon = destination_point(12.9790, 77.5946, 0, 100)
d2 = add_node(lat, lon)
add_way("W_oneway", [d1, d2], {"highway": "residential", "oneway": "yes", "maxspeed": "30"})

# ---------------------------------------------------------------------------
# 5. Motorway with parallel service road ~10 m away
# ---------------------------------------------------------------------------
e1 = add_node(12.9810, 77.5946)
lat, lon = destination_point(12.9810, 77.5946, 0, 120)
e2 = add_node(lat, lon)
add_way("W_motorway", [e1, e2], {"highway": "motorway", "maxspeed": "100"})

lat_s1, lon_s1 = destination_point(12.9810, 77.5946, 90, 10)
lat_s2, lon_s2 = destination_point(lat, lon, 90, 10)
e1s = add_node(lat_s1, lon_s1)
e2s = add_node(lat_s2, lon_s2)
add_way("W_service", [e1s, e2s], {"highway": "service", "maxspeed": "20"})

# ---------------------------------------------------------------------------
# 6. Tunnel gap: F1 and F2 exist but there is NO way connecting them --
#    simulates a tunnel segment with zero OSM road coverage.
# ---------------------------------------------------------------------------
f1 = add_node(12.9830, 77.5946)
lat, lon = destination_point(12.9830, 77.5946, 0, 300)   # 300 m away, deliberately unconnected
f2 = add_node(lat, lon)
# NOTE: intentionally no add_way() call here -- this is the "tunnel gap"

# ---------------------------------------------------------------------------
# Elevation (slope) -- attach to a couple of ways to exercise the
# "elevation data unavailable" edge case for the rest.
# ---------------------------------------------------------------------------
elevation_m = {a1: 900, a2: 940, a3: 990, b1: 850, b2: 852, b3: 848}

out = {
    "nodes": nodes,
    "ways": ways,
    "elevation_m": elevation_m,
    "test_reference_points": {
        # used directly by test_module.py
        "hairpin_apex_node": a2,
        "fork_test_point": {"lat": destination_point(12.9750, 77.5946, 27, 25)[0],
                             "lon": destination_point(12.9750, 77.5946, 27, 25)[1]},
        "tunnel_query_point": {"lat": destination_point(12.9830, 77.5946, 0, 150)[0],
                                "lon": destination_point(12.9830, 77.5946, 0, 150)[1]},
        "off_road_point": {"lat": destination_point(12.9716, 77.5946, 90, 25)[0],
                            "lon": destination_point(12.9716, 77.5946, 90, 25)[1]},
        "private_road_point": {"lat": 13.5000, "lon": 78.2000},
    },
}

out_path = os.path.join(os.path.dirname(__file__), "sample_hill_roads.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"Generated {out_path} with {len(nodes)} nodes and {len(ways)} ways.")
