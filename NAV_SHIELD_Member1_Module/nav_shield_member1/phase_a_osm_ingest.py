"""
Phase A -- OSM Data Acquisition

Downloads road-network data for the target region via the Overpass API and
saves it in the same {"nodes", "ways"} schema used by sample_data/, so
phase_c_road_graph.py can consume either real or synthetic data identically.

NOTE ON THIS SANDBOX: this file cannot be executed here (no network access to
overpass-api.de). It is written as real, runnable code for your own machine --
run `python3 phase_a_osm_ingest.py` once you're online and it will produce
osm_region_raw.json in the same schema as sample_data/sample_hill_roads.json.

For production, also keep the raw .pbf/.osm option (via pyosmium) mentioned in
the spec for larger regions -- Overpass is fine for a single test region
(e.g. one hill-road stretch) but rate-limits on large-area bulk downloads.
"""

import json
import time
import urllib.request
import urllib.parse

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding box: south, west, north, east.
# Default below is a small Western Ghats hill-road stretch near Bengaluru --
# replace with your team's actual target test region.
DEFAULT_BBOX = (12.90, 77.50, 13.05, 77.70)

HIGHWAY_TYPES = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "service", "unclassified", "living_street",
)


def build_query(bbox):
    south, west, north, east = bbox
    highway_filter = "|".join(HIGHWAY_TYPES)
    return f"""
    [out:json][timeout:60];
    (
      way["highway"~"^({highway_filter})$"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """


def fetch_overpass(bbox=DEFAULT_BBOX, retries=3):
    query = build_query(bbox)
    data = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OVERPASS_URL, data=data)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"Overpass request failed ({e}), retrying in 5s...")
            time.sleep(5)


def to_internal_schema(osm_json):
    """Converts raw Overpass JSON into the {"nodes", "ways"} schema shared
    across the whole module."""
    nodes = {}
    ways = []

    node_elements = {el["id"]: el for el in osm_json["elements"] if el["type"] == "node"}
    for nid, el in node_elements.items():
        nodes[f"N{nid}"] = {"lat": el["lat"], "lon": el["lon"]}

    for el in osm_json["elements"]:
        if el["type"] != "way":
            continue
        node_ids = [f"N{n}" for n in el["nodes"] if n in node_elements]
        if len(node_ids) < 2:
            continue
        ways.append({
            "id": f"W{el['id']}",
            "nodes": node_ids,
            "tags": el.get("tags", {}),
        })

    return {"nodes": nodes, "ways": ways, "elevation_m": {}}


if __name__ == "__main__":
    print("Fetching OSM road data via Overpass API...")
    raw = fetch_overpass()
    converted = to_internal_schema(raw)
    with open("osm_region_raw.json", "w") as f:
        json.dump(converted, f, indent=2)
    print(f"Saved osm_region_raw.json: {len(converted['nodes'])} nodes, "
          f"{len(converted['ways'])} ways.")
    print("Next: run phase_b_mbtiles.py to build offline tiles, and "
          "phase_c_road_graph.py to build the routable graph.")
