"""
synchronizer.py
----------------
Stage 1 module for NAV-SHIELD (Member 5 - AI Drift Correction pipeline).

Responsible for:
    1. Deriving a common, comparable time base for the S (smartphone) and
       V (vehicle) DataFrames.
    2. Performing TIME-BASED (never row-based) nearest-time synchronization
       between S and V samples, within a configurable tolerance.
    3. Assembling one aligned DataFrame with clearly prefixed columns
       (S_<col>, V_<col>) and no fabricated values.

WHY TIME-BASED, NOT ROW-BASED
------------------------------
Inspection of the actual IO-VNBD Vfa01 sequence shows:
    - V-Vfa01.csv has 11535 rows, sampled at a constant 0.1s interval, in a
      column called "Time Since Start of Day (seconds)" (i.e. seconds
      elapsed since local midnight).
    - S-Vfa01.csv has 11486 rows, sampled at an approximately-100ms
      (but NOT perfectly constant) interval. It has two time-like columns:
        * "TIME SINCE START (ms)": milliseconds elapsed since the phone's
          logging app started -- an arbitrary, device-local epoch that has
          NO fixed relationship to V's "seconds since start of day".
        * "DATE (YYYY-MO-DD HH-MI-SS_SSS)": an absolute wall-clock
          timestamp string (observed format: "YYYY-MM-DD HH:MM:SS:mmm").
    - Only the DATE column is on the same absolute time base as V's
      "seconds since start of day" column (both ultimately reference the
      phone/vehicle-logger's real-world clock on the same day).

Therefore Stage 1 derives "seconds since midnight" from S's DATE column and
uses THAT as the synchronization key against V's native
"Time Since Start of Day (seconds)" column. Row counts differ (11486 vs
11535) precisely because the two loggers started/stopped at slightly
different wall-clock moments and sampled at slightly different real rates --
which is exactly why row-index alignment would be wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Regex for the observed S DATE format: "2019-11-08 09:20:26:236"
# (Note: the column HEADER text claims an underscore before milliseconds,
# but the actual data uses a colon. We parse what is ACTUALLY in the file.)
_DATE_PATTERN = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})[:_](\d{1,3})\s*$"
)


def derive_seconds_since_midnight_from_date(date_series: pd.Series) -> pd.Series:
    """
    Parse the S DATE column ("YYYY-MM-DD HH:MM:SS:mmm") into a float number
    of seconds elapsed since local midnight of that date, so it can be
    directly compared against V's native "Time Since Start of Day (seconds)"
    column. Unparseable entries become NaN (never fabricated).
    """

    def _parse_one(value) -> float:
        if not isinstance(value, str):
            return np.nan
        match = _DATE_PATTERN.match(value)
        if not match:
            return np.nan
        _, _, _, hh, mm, ss, ms = match.groups()
        ms_padded = ms.ljust(3, "0")  # tolerate 1-3 digit ms fragments
        total_seconds = (
            int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms_padded) / 1000.0
        )
        return float(total_seconds)

    return date_series.apply(_parse_one)


@dataclass
class SyncResult:
    aligned_df: pd.DataFrame
    matched_count: int
    unmatched_count: int
    match_percentage: float
    tolerance_seconds: float
    s_sync_col: str
    v_sync_col: str
    method: str


def synchronize_sv(
    df_s: pd.DataFrame,
    df_v: pd.DataFrame,
    s_sync_col: str,
    v_sync_col: str,
    tolerance_seconds: float = 0.05,
    s_prefix: str = "S_",
    v_prefix: str = "V_",
) -> SyncResult:
    """
    Time-based nearest-neighbour synchronization of S against V.

    For every S sample (driven by S_sync_col), find the nearest V sample
    (by V_sync_col) within +/- tolerance_seconds. If no V sample falls
    within tolerance, the S sample is kept but marked unmatched and all
    V_* columns for that row are left as NaN (never fabricated/interpolated
    in Stage 1).

    Parameters
    ----------
    df_s, df_v : pd.DataFrame
        Raw S and V DataFrames (not mutated).
    s_sync_col, v_sync_col : str
        Names of the *already-numeric, seconds-based* synchronization
        columns in df_s / df_v respectively.
    tolerance_seconds : float
        Maximum allowed |t_S - t_V| for a match. Configurable by caller/CLI.
    s_prefix, v_prefix : str
        Column prefixes applied to avoid ambiguous/duplicate column names
        in the aligned output.

    Returns
    -------
    SyncResult
    """
    if s_sync_col not in df_s.columns:
        raise KeyError(f"S synchronization column '{s_sync_col}' not found in S DataFrame.")
    if v_sync_col not in df_v.columns:
        raise KeyError(f"V synchronization column '{v_sync_col}' not found in V DataFrame.")

    # Work on copies; original DataFrames (and CSV files) are never modified.
    s = df_s.copy()
    v = df_v.copy()

    # merge_asof requires sorted-by-key input. We sort but retain the
    # original row identity via an explicit index column so ordering in the
    # final output can be restored to S's natural (time) order.
    s = s.sort_values(s_sync_col, kind="mergesort").reset_index(drop=True)
    v = v.sort_values(v_sync_col, kind="mergesort").reset_index(drop=True)

    # Drop S rows where the sync time itself could not be determined
    # (e.g. unparseable DATE string) -- reported, not silently discarded
    # from the user's view: these are counted as unmatched with a distinct
    # reason so the report can surface them.
    s_valid_mask = s[s_sync_col].notna()
    s_time_missing_count = int((~s_valid_mask).sum())
    s_usable = s[s_valid_mask].copy()

    v_valid_mask = v[v_sync_col].notna()
    v_usable = v[v_valid_mask].copy()

    # Rename columns with prefixes BEFORE merge_asof to avoid any ambiguity,
    # keeping the sync columns under distinct, traceable names too.
    s_renamed = s_usable.rename(columns={c: f"{s_prefix}{c}" for c in s_usable.columns})
    v_renamed = v_usable.rename(columns={c: f"{v_prefix}{c}" for c in v_usable.columns})

    s_key = f"{s_prefix}{s_sync_col}"
    v_key = f"{v_prefix}{v_sync_col}"

    merged = pd.merge_asof(
        s_renamed,
        v_renamed,
        left_on=s_key,
        right_on=v_key,
        direction="nearest",
        tolerance=tolerance_seconds,
    )

    merged["time_diff_seconds"] = (merged[s_key] - merged[v_key]).abs()
    merged["matched"] = merged[v_key].notna()

    # Canonical synchronization timestamp for the aligned dataset: the S
    # (smartphone) sync time, since S is the primary sensor stream that
    # future stages will feed into the LSTM; V is reference/ground truth.
    merged = merged.rename(columns={s_key: "timestamp_seconds"})
    # Keep a copy of V's own sync time too (already present as v_key/{V_...}).

    matched_count = int(merged["matched"].sum())
    total_s_rows = len(df_s)  # denominator includes rows dropped for missing sync time
    unmatched_count = total_s_rows - matched_count
    match_percentage = (matched_count / total_s_rows * 100.0) if total_s_rows else 0.0

    # Restore chronological ordering by the canonical timestamp.
    merged = merged.sort_values("timestamp_seconds", kind="mergesort").reset_index(drop=True)

    return SyncResult(
        aligned_df=merged,
        matched_count=matched_count,
        unmatched_count=unmatched_count,
        match_percentage=match_percentage,
        tolerance_seconds=tolerance_seconds,
        s_sync_col=s_sync_col,
        v_sync_col=v_sync_col,
        method=(
            f"pandas.merge_asof nearest-time matching, tolerance="
            f"{tolerance_seconds}s. {s_time_missing_count} S row(s) excluded "
            f"beforehand due to unparseable/missing synchronization time."
        ),
    )
