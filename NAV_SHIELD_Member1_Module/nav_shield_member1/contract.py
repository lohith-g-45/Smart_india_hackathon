"""
contract.py

The fixed input/output contract with Member 4 (UKF/Trust Brain), exactly as
specified in Section 4 of the module spec. Field names here must never
change once agreed with the rest of the team -- everyone else writes code
against these exact names.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MapMatchQuery:
    """INPUT from Member 4."""
    latitude: float
    longitude: float
    heading: float       # degrees
    speed: float          # m/s


@dataclass
class CandidateRoad:
    segment_id: str
    score: float
    lat: float
    lon: float
    heading: float
    curvature: float
    slope: float


@dataclass
class MapMatchResult:
    """OUTPUT to Member 4."""
    matched_latitude: float
    matched_longitude: float
    road_segment_id: str
    road_heading: float
    road_curvature: float
    road_slope: float                 # percent
    map_confidence: float             # 0-1
    candidate_roads: List[CandidateRoad] = field(default_factory=list)
    road_slope_available: bool = True
    one_way_violation: bool = False
    position_uncertainty: Optional[float] = None  # metres, echoed from Member 4's own estimate if provided
