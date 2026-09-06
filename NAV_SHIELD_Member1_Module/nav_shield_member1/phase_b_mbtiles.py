"""
Phase B -- Offline Tile Storage

Converts the raw OSM data into an MBTiles file for offline rendering by
MapLibre Android SDK (consumed by Member 6). This wraps the `tilemaker`
CLI tool rather than reimplementing vector-tile generation in Python --
tilemaker is the standard, well-maintained tool for exactly this job and
reimplementing it would be reinventing a solved problem.

REQUIRES (install once, on your own machine -- not available in this sandbox):
  - tilemaker: https://github.com/systemed/tilemaker  (brew install tilemaker /
    apt install tilemaker / build from source)
  - An .osm.pbf extract of your target region (Geofabrik regional extracts,
    or convert Phase A's Overpass JSON output using `osmium` -- see
    osm_json_to_pbf() below for the conversion path if you only have JSON).

Usage:
    python3 phase_b_mbtiles.py --input region.osm.pbf --output navshield_map.mbtiles
"""

import argparse
import subprocess
import sys


def build_mbtiles(pbf_path: str, output_path: str, min_zoom: int = 10, max_zoom: int = 17):
    """
    Shells out to tilemaker. tilemaker's default config already handles
    highway=* tagging correctly for road rendering; a custom config.json /
    process.lua pair can be supplied via --config if your team wants custom
    styling (e.g. highlighting hairpin/curvature data), but the defaults are
    suffient for an offline road-network basemap.
    """
    cmd = [
        "tilemaker",
        pbf_path,
        f"--output={output_path}",
        f"--zoom={min_zoom},{max_zoom}",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("tilemaker failed:", result.stderr, file=sys.stderr)
        raise RuntimeError("MBTiles generation failed -- check tilemaker is installed and on PATH")
    print(f"Wrote {output_path} (zoom {min_zoom}-{max_zoom})")
    print("Verify tile rendering with: https://github.com/maplibre/maplibre-gl-inspector "
          "or simply load the .mbtiles path directly in the Android MapLibre SDK (Member 6).")


def osm_json_to_pbf_note():
    print(
        "If Phase A produced JSON (not .osm.pbf), convert first:\n"
        "  1. osmium/pyosmium can write .osm.pbf from raw Overpass XML output "
        "(use the Overpass 'out xml' response instead of 'out json' if you need this path), or\n"
        "  2. Simpler for a hackathon: download a pre-made regional .osm.pbf extract directly "
        "from Geofabrik (https://download.geofabrik.de/asia/india.html -> Karnataka extract) "
        "and skip Phase A's live query entirely for the demo region."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .osm.pbf extract")
    parser.add_argument("--output", default="navshield_map.mbtiles")
    parser.add_argument("--min-zoom", type=int, default=10)
    parser.add_argument("--max-zoom", type=int, default=17)
    args = parser.parse_args()
    build_mbtiles(args.input, args.output, args.min_zoom, args.max_zoom)
