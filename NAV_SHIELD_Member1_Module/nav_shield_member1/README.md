# NAV-SHIELD · Member 1 — Offline Map + Road Graph + GNN Map Matching

Implementation of the Geographical Intelligence Layer module, covering
Phases A–G from the spec. This README is honest about what's fully tested
right now vs. what's production-reference code you'll run once you have
network/GPU access.

## What's fully working and tested in this environment right now

| Phase | File | Status |
|---|---|---|
| C — Road graph construction | `phase_c_road_graph.py` | ✅ Runs, builds graph from sample data |
| D — Spatial indexing | `phase_d_spatial_index.py` | ✅ Runs, 0.006ms/query (target <5ms) |
| E — Basic map matching | `phase_e_map_matching.py` | ✅ Runs, correctly scores all edge cases |
| F — GNN map matching | `phase_f_gnn_lite_numpy.py` | ✅ Runs, trains, 84% Top-1 / 100% Top-3 on held-out data |
| G — Multi-candidate output | `phase_g_multi_candidate.py` | ✅ Runs, correctly flags fork ambiguity |
| Unified service | `map_matching_service.py` | ✅ Runs, all edge cases verified |
| Tests | `test_module.py` + `run_tests_manual.py` | ✅ 11/11 passing |

## What's reference-only (needs network/GPU you don't have in a locked-down sandbox, but you will have on your own machines)

| Phase | File | Why it can't run here |
|---|---|---|
| A — OSM ingestion | `phase_a_osm_ingest.py` | Needs live network access to Overpass API |
| B — MBTiles generation | `phase_b_mbtiles.py` | Needs the `tilemaker` binary installed |
| F — Production GNN | `phase_f_gnn_map_matching_pyg.py` | Needs `torch` + `torch_geometric` (no network to install) |

**Why a "sample data" layer exists at all:** Phase A needs live internet
access to Overpass API, which this dev environment doesn't have. So
`sample_data/generate_sample_data.py` produces a synthetic road network in
the *exact same schema* Phase A would produce from real OSM data —
deliberately built to include one instance of every edge case in Section 6
of the spec (hairpin, fork, roundabout, one-way, motorway+service-road,
tunnel gap). Everything from Phase C onward is therefore genuinely tested,
not just written and hoped-for. When you run Phase A for real against
Bengaluru/Western Ghats, nothing downstream changes — same schema in, same
schema out.

**Why two versions of Phase F exist:** the spec calls for PyTorch Geometric,
which needs network access to install and (realistically) a GPU/Colab to
train at any real scale — neither available here. `phase_f_gnn_lite_numpy.py`
is a legitimate engineering fallback: same function interface
(`score_candidates_learned`), same training signal, same accuracy tests —
just a plain MLP instead of graph convolution, so it has no message-passing
over neighbouring roads. Be upfront about this distinction if a judge asks
whether you're running "a real GNN" — you have both, and can honestly
describe which one is live in the demo.

## Quick start

```bash
pip install -r requirements.txt   # rtree/torch/torch_geometric optional --
                                   # everything below works without them
python3 sample_data/generate_sample_data.py   # only needed once
python3 map_matching_service.py               # end-to-end demo
python3 -m pytest test_module.py -v           # full test suite
```

If `pytest` isn't installed, `python3 run_tests_manual.py` runs the exact
same test logic without the dependency.

## Fixed I/O contract (Section 4)

Implemented in `contract.py` as `MapMatchQuery` / `MapMatchResult` /
`CandidateRoad` dataclasses — field names match the spec exactly. Member 4
should import from `contract.py` directly rather than re-declaring these
shapes, so the two sides of the interface can never silently drift apart.

## Switching from sample data to real data

1. Get online, run `python3 phase_a_osm_ingest.py` for your real target
   region (edit `DEFAULT_BBOX`), or download a Geofabrik `.osm.pbf` extract
   directly.
2. Run `phase_b_mbtiles.py` against that `.pbf` to produce the real
   `navshield_map.mbtiles` (requires `tilemaker` installed).
3. Point `MapMatchingService(...)` at the new JSON instead of
   `sample_data/sample_hill_roads.json` — nothing else changes.
4. Once you have GPU/Colab access, train `phase_f_gnn_map_matching_pyg.py`
   on real driving traces from Member 5's data pipeline, export to ONNX,
   and swap it in for `phase_f_gnn_lite_numpy.py` in
   `map_matching_service.py`.

## Known limitations to disclose to your team / judges

- Curvature is computed as the turn angle at each node from the graph
  topology, not from a smoothed road centerline — sharp digitisation of a
  gentle curve in real OSM data can occasionally register as higher
  curvature than the road actually has. Validate against a few real
  known-hairpin roads once you're on real data.
- `phase_e_map_matching.py`'s off-road/no-match thresholds (20m / 30m) are
  reasonable starting points, not tuned against real GPS noise
  characteristics — expect to adjust once real device GPS traces are
  available.
- The GNN-lite model's 84%/100% accuracy is on synthetic, noise-injected
  data from a small (18-node) network — treat it as "the harness
  works," not as a claim about real-world accuracy at city scale.
