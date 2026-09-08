from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3_features import (
    angular_delta,
    clean_and_engineer,
    detect_gap_threshold,
    detect_groups,
    vector_magnitude,
)


def test_vector_magnitude():
    frame = pd.DataFrame({"x": [3.0], "y": [4.0], "z": [12.0]})
    assert vector_magnitude(frame, {"x": "x", "y": "y", "z": "z"}).iloc[0] == pytest.approx(13.0)


def test_actual_gyro_headers_are_detected_and_magnitude_is_computed():
    frame = pd.DataFrame(
        {
            "S_time_seconds": [0.0, 0.1],
            "S_GYROSCOPE Yaw (rad/s)": [3.0, 3.0],
            "S_GYROSCOPE Pitch (rad/s)": [4.0, 4.0],
            "S_GYROSCOPE Roll (rad/s)": [12.0, 12.0],
        }
    )
    groups = detect_groups(frame)
    assert set(groups.gyroscope) == {"yaw", "pitch", "roll"}
    features, report = clean_and_engineer(frame, "sample", window=2)
    assert features["gyro_magnitude"].tolist() == pytest.approx([13.0, 13.0])
    assert report["gyro_derived_missing_percentages"]["gyro_magnitude"] == 0.0


def test_missing_gyro_source_columns_fail_with_clear_diagnostic():
    frame = pd.DataFrame({"S_time_seconds": [0.0, 0.1], "S_GYROSCOPE Yaw (rad/s)": [1.0, 1.0]})
    with pytest.raises(ValueError, match="Required gyroscope source columns"):
        clean_and_engineer(frame, "sample", window=2)


def test_angular_delta_wraps_at_360_degrees():
    result = angular_delta(pd.Series([359.0, 1.0, 359.0]))
    assert result.iloc[1] == pytest.approx(2.0)
    assert result.iloc[2] == pytest.approx(-2.0)


def test_gap_threshold_detects_large_gap():
    threshold = detect_gap_threshold(pd.Series([np.nan, 0.1, 0.1, 2.0]))
    assert threshold < 2.0


def test_feature_generation_preserves_rows_and_marks_gps_quality():
    frame = pd.DataFrame(
        {
            "sequence_id": ["sample"] * 4,
            "S_time_seconds": [0.0, 0.1, 0.2, 2.0],
            "V_time_seconds": [0.0, 0.1, 0.2, 2.0],
            "synchronization_valid": [True, True, True, True],
            "S_ACCELEROMETER X (m/s²)": [3.0, 0.0, np.inf, 0.0],
            "S_ACCELEROMETER Y (m/s²)": [4.0, 0.0, 0.0, 0.0],
            "S_ACCELEROMETER Z (m/s²)": [0.0, 0.0, 0.0, 0.0],
            "S_GYROSCOPE Yaw (rad/s)": [0.0, 0.0, 0.0, 0.0],
            "S_GYROSCOPE Pitch (rad/s)": [0.0, 0.0, 0.0, 0.0],
            "S_GYROSCOPE Roll (rad/s)": [0.0, 0.0, 0.0, 0.0],
            "S_MAGNETIC FIELD X (µT)": [1.0, 1.0, 1.0, 1.0],
            "S_MAGNETIC FIELD Y (µT)": [2.0, 2.0, 2.0, 2.0],
            "S_MAGNETIC FIELD Z (µT)": [2.0, 2.0, 2.0, 2.0],
            "S_GRAVITY X (m/s²)": [0.0, 0.0, 0.0, 0.0],
            "S_GRAVITY Y (m/s²)": [0.0, 0.0, 0.0, 0.0],
            "S_GRAVITY Z (m/s²)": [9.8, 9.8, 9.8, 9.8],
            "S_ORIENTATION (Yaw) (°)": [359.0, 1.0, 2.0, 3.0],
            "S_ORIENTATION (Pitch) (°)": [0.0, 0.0, 0.0, 0.0],
            "S_ORIENTATION (Roll ) (°)": [0.0, 0.0, 0.0, 0.0],
            "S_GPS LATITUDE (degrees)": [52.0, 52.0, 95.0, 52.0],
            "S_GPS LONGITUDE (degrees)": [-1.0, -1.0, -1.0, -1.0],
            "S_GPS SPEED (Kmh)": [10.0, 11.0, 12.0, 13.0],
            "S_GPS ACCURACY (m)": [3.0, 3.0, -1.0, 3.0],
            "GPS SATELLITES IN RANGE": [10, 10, 10, 10],
        }
    )
    features, report = clean_and_engineer(frame, "sample", window=2)
    assert len(features) == len(frame)
    assert features["accel_magnitude"].iloc[0] == pytest.approx(5.0)
    assert features["gps_valid"].tolist() == [True, True, False, True]
    assert features["time_gap_flag"].any()
    assert features["yaw_change"].iloc[1] == pytest.approx(2.0)
    assert report["rows_removed"] == 0


def test_rolling_features_use_current_and_past_rows_only():
    frame = pd.DataFrame(
        {
            "S_time_seconds": [0.0, 1.0, 2.0],
            "S_ACCELEROMETER X (m/s²)": [1.0, 2.0, 100.0],
            "S_ACCELEROMETER Y (m/s²)": [0.0, 0.0, 0.0],
            "S_ACCELEROMETER Z (m/s²)": [0.0, 0.0, 0.0],
            "S_GYROSCOPE Yaw (rad/s)": [0.0, 0.0, 0.0],
            "S_GYROSCOPE Pitch (rad/s)": [0.0, 0.0, 0.0],
            "S_GYROSCOPE Roll (rad/s)": [0.0, 0.0, 0.0],
        }
    )
    features, _ = clean_and_engineer(frame, "sample", window=2)
    assert features["accel_magnitude_rolling_mean"].iloc[0] == pytest.approx(1.0)
    assert features["accel_magnitude_rolling_mean"].iloc[1] == pytest.approx(1.5)
    assert features["accel_magnitude_rolling_mean"].iloc[2] == pytest.approx(51.0)