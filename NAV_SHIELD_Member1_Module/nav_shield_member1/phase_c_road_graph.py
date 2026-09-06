"""
Phase C -- Road Graph Construction

Parses the {"nodes", "ways"} schema (from Phase A or sample_data/) into a
NetworkX directed graph:
    - node    = intersection (OSM node id), with lat/lon attributes
    - edge    = road segment, with attributes: length, bearing, curvature,
                road_type, max_speed, slope, one_way

This is fully runnable in any environment (only depends on networkx, which
is already installed) and is tested end-to-end against sample_data/ in
test_module.py.
"""

import json
import pickle

import networkx as nx

from geo_utils import haversine_m, initial_bearing_deg, curvature_deg


def load_raw(path):
    with open(path) as f:
        return json.load(f)


def build_road_graph(raw):
    """
    raw: dict with "nodes" (id -> {lat, lon}), "ways" (list of {id, nodes, tags}),
         optional "elevation_m" (node id -> metres).
    Returns: networkx.DiGraph
    """
    nodes = raw["nodes"]
    ways = raw["ways"]
    elevation = raw.get("elevation_m", {})

    g = nx.DiGraph()
    for nid, attrs in nodes.items():
        g.add_node(nid, lat=attrs["lat"], lon=attrs["lon"],
                   elevation_m=elevation.get(nid))

    for way in ways:
        tags = way.get("tags", {})
        road_type = tags.get("highway", "unclassified")
        one_way = tags.get("oneway", "no") in ("yes", "1", "true")
        max_speed = _parse_maxspeed(tags.get("maxspeed"))
        junction = tags.get("junction")
        name = tags.get("name")

        way_nodes = way["nodes"]
        for i in range(len(way_nodes) - 1):
            u, v = way_nodes[i], way_nodes[i + 1]
            if u not in nodes or v not in nodes:
                continue
            _add_edge(g, u, v, way["id"], road_type, one_way, max_speed, junction, name, elevation)
            if not one_way:
                _add_edge(g, v, u, way["id"], road_type, one_way, max_speed, junction, name, elevation)

    _annotate_curvature(g)
    return g


def _parse_maxspeed(raw_val):
    if raw_val is None:
        return None
    try:
        return float("".join(ch for ch in raw_val if ch.isdigit() or ch == "."))
    except ValueError:
        return None


def _add_edge(g, u, v, way_id, road_type, one_way, max_speed, junction, name, elevation):
    lat1, lon1 = g.nodes[u]["lat"], g.nodes[u]["lon"]
    lat2, lon2 = g.nodes[v]["lat"], g.nodes[v]["lon"]
    length = haversine_m(lat1, lon1, lat2, lon2)
    bearing = initial_bearing_deg(lat1, lon1, lat2, lon2)

    slope = None
    slope_available = False
    e1, e2 = elevation.get(u), elevation.get(v)
    if e1 is not None and e2 is not None and length > 0:
        slope = 100.0 * (e2 - e1) / length
        slope_available = True

    g.add_edge(
        u, v,
        segment_id=f"{way_id}:{u}->{v}",
        way_id=way_id,
        length_m=length,
        bearing=bearing,
        road_type=road_type,
        one_way=one_way,
        max_speed=max_speed,
        junction=junction,
        name=name,
        slope_pct=slope if slope_available else 0.0,
        slope_available=slope_available,
        curvature_deg=0.0,  # filled in by _annotate_curvature
    )


def _annotate_curvature(g):
    """
    Curvature at a segment is defined here as the turn angle between the
    incoming edge (into the segment's start node) and the segment's own
    bearing -- i.e. "how sharply do you turn onto this segment". This
    matches the spec's hairpin test: curvature_deg should exceed 60 for a
    genuine hairpin bend.
    """
    for u, v, data in g.edges(data=True):
        incoming_bearings = [
            g.edges[p, u]["bearing"] for p in g.predecessors(u) if p != v
        ]
        if incoming_bearings:
            # use the incoming edge with the least turn (the "through" road),
            # curvature is the max turn required to continue onto this edge
            data["curvature_deg"] = max(
                curvature_deg(b_in, data["bearing"]) for b_in in incoming_bearings
            )
        else:
            data["curvature_deg"] = 0.0


def save_graph(g, path):
    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    raw = load_raw("sample_data/sample_hill_roads.json")
    g = build_road_graph(raw)
    save_graph(g, "road_graph.gpickle")
    print(f"Built graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} directed edges")
    hairpin_edges = [(u, v, d["curvature_deg"]) for u, v, d in g.edges(data=True) if d["curvature_deg"] > 60]
    print("Edges with curvature > 60 deg (hairpins):", hairpin_edges)
