"""
data_validator.py
------------------
Stage 1 module for NAV-SHIELD (Member 5 - AI Drift Correction pipeline).

Performs data-quality validation on a single loaded DataFrame (S or V):
    - row/column counts, column names, dtypes
    - missing values (count + percentage per column)
    - duplicate rows / duplicate timestamps
    - infinite values
    - invalid numeric values (NaN-producing coercions)
    - min/max per numeric column
    - latitude / longitude range validity
    - time-column detection
    - sampling interval / frequency estimation

This module NEVER silently drops or fills data. It only inspects and reports.
All functions are pure (they do not mutate the input DataFrame).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

# Keywords used to *propose* time-related column candidates. This is a
# heuristic starting point only -- the actual selection of the synchronization
# field is made in synchronizer.py after inspecting real values (monotonicity,
# units, plausibility), never from the name alone.
TIME_KEYWORDS = ("time", "date", "sample period", "timestamp")


def basic_shape_report(df: pd.DataFrame, name: str) -> dict:
    """Row count, column count, exact column names, dtypes."""
    return {
        "name": name,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def missing_value_report(df: pd.DataFrame) -> dict:
    """Missing value count + percentage per column."""
    total_rows = len(df)
    missing_counts = df.isna().sum()
    report = {}
    for col in df.columns:
        count = int(missing_counts[col])
        pct = float(count / total_rows * 100) if total_rows else 0.0
        report[col] = {"missing_count": count, "missing_percent": round(pct, 4)}
    return report


def duplicate_rows_report(df: pd.DataFrame) -> dict:
    """Fully duplicated rows (all columns identical)."""
    dup_mask = df.duplicated(keep=False)
    return {
        "duplicate_row_count": int(dup_mask.sum()),
        "duplicate_row_indices_sample": df.index[dup_mask].tolist()[:20],
    }


def duplicate_timestamp_report(df: pd.DataFrame, time_col: Optional[str]) -> dict:
    """Duplicate values within the chosen time column (if any)."""
    if time_col is None or time_col not in df.columns:
        return {"time_col": time_col, "duplicate_timestamp_count": None,
                "note": "Time column not available for duplicate-timestamp check."}
    dup_mask = df[time_col].duplicated(keep=False)
    return {
        "time_col": time_col,
        "duplicate_timestamp_count": int(dup_mask.sum()),
    }


def infinite_value_report(df: pd.DataFrame) -> dict:
    """Count of +/-inf values per numeric column."""
    numeric_df = df.select_dtypes(include=[np.number])
    report = {}
    for col in numeric_df.columns:
        inf_count = int(np.isinf(numeric_df[col]).sum())
        if inf_count > 0:
            report[col] = inf_count
    return report


def invalid_numeric_report(df: pd.DataFrame) -> dict:
    """
    Attempts a non-mutating numeric coercion of every *object*-typed column
    to detect values that look like they should be numbers but fail to parse
    (e.g. corrupted entries). Purely diagnostic -- does not alter df.
    """
    report = {}
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        newly_invalid = coerced.isna() & df[col].notna()
        count = int(newly_invalid.sum())
        if count > 0:
            report[col] = {
                "unparseable_non_null_count": count,
                "sample_values": df.loc[newly_invalid, col].astype(str).unique().tolist()[:5],
            }
    return report


def min_max_report(df: pd.DataFrame) -> dict:
    """Min/max for every numeric column."""
    numeric_df = df.select_dtypes(include=[np.number])
    report = {}
    for col in numeric_df.columns:
        series = numeric_df[col]
        report[col] = {
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
        }
    return report


def latlon_validity_report(df: pd.DataFrame, lat_col: Optional[str], lon_col: Optional[str]) -> dict:
    """Validate latitude in [-90, 90] and longitude in [-180, 180]."""
    result = {"lat_col": lat_col, "lon_col": lon_col}

    if lat_col and lat_col in df.columns:
        lat = df[lat_col]
        invalid_lat = lat[(lat < LAT_MIN) | (lat > LAT_MAX)]
        result["lat_invalid_count"] = int(invalid_lat.shape[0])
        result["lat_min"] = float(lat.min()) if lat.notna().any() else None
        result["lat_max"] = float(lat.max()) if lat.notna().any() else None
    else:
        result["lat_invalid_count"] = None

    if lon_col and lon_col in df.columns:
        lon = df[lon_col]
        invalid_lon = lon[(lon < LON_MIN) | (lon > LON_MAX)]
        result["lon_invalid_count"] = int(invalid_lon.shape[0])
        result["lon_min"] = float(lon.min()) if lon.notna().any() else None
        result["lon_max"] = float(lon.max()) if lon.notna().any() else None
    else:
        result["lon_invalid_count"] = None

    return result


def detect_time_column_candidates(df: pd.DataFrame) -> list[dict]:
    """
    Propose candidate time-related columns based on column-name keywords.
    For each candidate, report dtype and (if numeric) whether it is
    monotonically increasing and its diff statistics, so the caller can
    make an evidence-based decision about which field truly represents
    time (this function never asserts which one to use).
    """
    candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in TIME_KEYWORDS):
            info = {"column": col, "dtype": str(df[col].dtype)}
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                info["is_monotonic_increasing"] = bool(series.is_monotonic_increasing)
                diffs = series.diff().dropna()
                if len(diffs) > 0:
                    info["diff_mean"] = float(diffs.mean())
                    info["diff_std"] = float(diffs.std())
                    info["diff_min"] = float(diffs.min())
                    info["diff_max"] = float(diffs.max())
            else:
                info["sample_values"] = series.dropna().astype(str).unique().tolist()[:3]
            candidates.append(info)
    return candidates


def compute_sampling_info(time_seconds: pd.Series) -> dict:
    """
    Given a numeric time series expressed in seconds (monotonic, one value
    per sample), compute the approximate sampling interval and frequency.
    """
    diffs = time_seconds.diff().dropna()
    diffs = diffs[diffs > 0]  # ignore non-positive diffs (out-of-order/duplicate)
    if len(diffs) == 0:
        return {
            "interval_mean_seconds": None,
            "interval_std_seconds": None,
            "interval_min_seconds": None,
            "interval_max_seconds": None,
            "frequency_hz": None,
            "non_positive_diff_count": int((time_seconds.diff().dropna() <= 0).sum()),
        }
    mean_interval = float(diffs.mean())
    return {
        "interval_mean_seconds": mean_interval,
        "interval_std_seconds": float(diffs.std()),
        "interval_min_seconds": float(diffs.min()),
        "interval_max_seconds": float(diffs.max()),
        "frequency_hz": float(1.0 / mean_interval) if mean_interval > 0 else None,
        "non_positive_diff_count": int((time_seconds.diff().dropna() <= 0).sum()),
    }


def time_range_report(time_seconds: pd.Series) -> dict:
    """Min/max of a numeric seconds-based time series."""
    valid = time_seconds.dropna()
    if len(valid) == 0:
        return {"start_seconds": None, "end_seconds": None, "duration_seconds": None}
    return {
        "start_seconds": float(valid.min()),
        "end_seconds": float(valid.max()),
        "duration_seconds": float(valid.max() - valid.min()),
    }


def full_validation_report(
    df: pd.DataFrame,
    name: str,
    time_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
) -> dict:
    """
    Convenience wrapper that assembles all checks above into one dict for a
    single DataFrame. `time_col` (if given) must already be a numeric
    seconds-based series present in df (callers typically pass a derived
    sync-time column here after time analysis in synchronizer.py).
    """
    report = basic_shape_report(df, name)
    report["missing_values"] = missing_value_report(df)
    report["duplicates"] = duplicate_rows_report(df)
    report["duplicate_timestamps"] = duplicate_timestamp_report(df, time_col)
    report["infinite_values"] = infinite_value_report(df)
    report["invalid_numeric_values"] = invalid_numeric_report(df)
    report["min_max"] = min_max_report(df)
    report["latlon_validity"] = latlon_validity_report(df, lat_col, lon_col)
    report["time_column_candidates"] = detect_time_column_candidates(df)

    if time_col is not None and time_col in df.columns:
        report["time_range"] = time_range_report(df[time_col])
        report["sampling_info"] = compute_sampling_info(df[time_col])
    else:
        report["time_range"] = None
        report["sampling_info"] = None

    return report
