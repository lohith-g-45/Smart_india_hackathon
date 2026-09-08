from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3_5_audit import audit_sequence, feature_status, missing_reason


def test_gyro_missingness_is_identified_as_stage3_bug(tmp_path):
    path = tmp_path / "sample_features.csv"
    frame = pd.DataFrame(
        {
            "sequence_id": ["sample", "sample"],
            "timestamp": [0.0, 0.1],
            "S_GYROSCOPE Yaw (rad/s)": [0.1, 0.2],
            "S_GYROSCOPE Pitch (rad/s)": [0.1, 0.2],
            "S_GYROSCOPE Roll (rad/s)": [0.1, 0.2],
            "gyro_yaw": [0.1, 0.2],
            "gyro_pitch": [0.1, 0.2],
            "gyro_roll": [0.1, 0.2],
            "gyro_magnitude": [np.nan, np.nan],
        }
    )
    frame.to_csv(path, index=False)
    stats = {}
    report, missing, _ = audit_sequence(path, stats)
    reason = missing_reason("gyro_magnitude", frame, 2, 2)
    assert "requests x/y/z" in reason
    assert report["overall_status"] == "WARNING"
    assert any(item["feature_name"] == "gyro_magnitude" for item in missing)


def test_feature_status_thresholds():
    assert feature_status(0.0, 0, False, False) == "PASS"
    assert feature_status(2.0, 0, False, False) == "WARNING"
    assert feature_status(100.0, 0, False, False) == "FAIL"
    assert feature_status(0.0, 1, False, False) == "FAIL"