# Member 1 Actual Output Contract

Based on `contract.py` and `map_matching_service.py` in `D:/NAV_SHIELD_Member1_Module/nav_shield_member1/`.

## 1. MapMatchQuery (Input to Member 1)
| Field | Type | Unit | Description |
|---|---|---|---|
| `latitude` | `float` | Decimal Degrees | Current GPS latitude |
| `longitude` | `float` | Decimal Degrees | Current GPS longitude |
| `heading` | `float` | Degrees | Current vehicle heading |
| `speed` | `float` | m/s | Current vehicle speed |

## 2. MapMatchResult (Output from Member 1)
| Field | Type | Unit | Nullable? | Description |
|---|---|---|---|---|
| `matched_latitude` | `float` | Decimal Degrees | No | Snapped latitude on the road segment |
| `matched_longitude` | `float` | Decimal Degrees | No | Snapped longitude on the road segment |
| `road_segment_id` | `str` | UUID/String | No (Empty string if no match) | Unique identifier for the road segment |
| `road_heading` | `float` | Degrees | No | Orientation of the road segment |
| `road_curvature` | `float` | Degrees | No | Local curvature of the road |
| `road_slope` | `float` | Percent (%) | No | Local slope/gradient of the road |
| `map_confidence` | `float` | 0.0 - 1.0 | No | Probability/Quality of the match |
| `candidate_roads` | `List[CandidateRoad]` | N/A | No (Empty list if none) | Alternative road segments nearby |
| `road_slope_available`| `bool` | N/A | No | Flag if slope data is actual or estimated |
| `one_way_violation` | `bool` | N/A | No | True if heading contradicts road direction |
| `position_uncertainty`| `float` | Metres | **Yes** | Echoed uncertainty estimate |

## 3. CandidateRoad (Nested in MapMatchResult)
| Field | Type | Unit | Description |
|---|---|---|---|
| `segment_id` | `str` | N/A | Unique identifier |
| `score` | `float` | 0.0 - 1.0 | Individual candidate match score |
| `lat` | `float` | Degrees | Candidate point latitude |
| `lon` | `float` | Degrees | Candidate point longitude |
| `heading` | `float` | Degrees | Candidate road heading |
| `curvature` | `float` | Degrees | Candidate road curvature |
| `slope` | `float` | Percent | Candidate road slope |

## 4. Error Behaviour & Edge Cases
* **No Match (Nearness):** Returns result with empty `road_segment_id`, `map_confidence` = 0.0, and coordinates equal to query coordinates.
* **Tunnel/Gap:** Returns "sticky" last known match with `map_confidence` fixed at 0.3.
* **Off-Road:** Returns nearest segment with `map_confidence` around 0.4.
