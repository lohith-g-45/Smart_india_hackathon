"""
tests/test_stage1.py
---------------------
Unit + integration tests for NAV-SHIELD Member 5 Stage 1.

IMPORTANT: All synthetic data here exists ONLY to exercise the pipeline
logic (loading, validation, synchronization, reporting). It is NEVER used
as a stand-in for the real IO-VNBD dataset in the actual pipeline run --
Stage 1 must always be run against the real, locally-extracted
S-Vfa01.csv / V-Vfa01.csv (or other real sequence) files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    CSVLoadError,
    DatasetPathError,
    SVFileNotFoundError,
    discover_sv_pair,
    load_csv_robust,
    load_sv_pair,
    resolve_dataset_root,
)
from src.data_validator import (
    compute_sampling_info,
    detect_time_column_candidates,
    duplicate_rows_report,
    full_validation_report,
    latlon_validity_report,
    time_range_report,
)
from src.synchronizer import derive_seconds_since_midnight_from_date, synchronize_sv

import main as stage1_main


# ---------------------------------------------------------------------------
# Synthetic S/V dataset builders (test-only; NOT the real IO-VNBD dataset).
# ---------------------------------------------------------------------------

V_TIME_COL = stage1_main.V_TIME_COL
V_LAT_COL = stage1_main.V_LAT_COL
V_LON_COL = stage1_main.V_LON_COL
S_DATE_COL = stage1_main.S_DATE_COL
S_LAT_COL = stage1_main.S_LAT_COL
S_LON_COL = stage1_main.S_LON_COL


def _seconds_to_date_string(seconds_since_midnight: float, date_prefix: str = "2024-01-01") -> str:
    """Build a DATE string in the observed IO-VNBD S format from a seconds value."""
    hh = int(seconds_since_midnight // 3600)
    mm = int((seconds_since_midnight % 3600) // 60)
    ss = int(seconds_since_midnight % 60)
    ms = int(round((seconds_since_midnight - int(seconds_since_midnight)) * 1000))
    return f"{date_prefix} {hh:02d}:{mm:02d}:{ss:02d}:{ms:03d}"


def build_synthetic_v(n=20, start=100.0, step=0.1) -> pd.DataFrame:
    times = [start + i * step for i in range(n)]
    return pd.DataFrame(
        {
            "No of GPS Satellites Available": [10] * n,
            V_TIME_COL: times,
            V_LAT_COL: [52.5 + 0.0001 * i for i in range(n)],
            V_LON_COL: [-1.48 - 0.0001 * i for i in range(n)],
            " Velocity (km/hr)": [40.0 + i for i in range(n)],
            " Sample period (seconds)": [step] * n,
        }
    )


def build_synthetic_s(n=18, start=100.02, step=0.1, offset=0.02, add_far_outlier=True) -> pd.DataFrame:
    times = [start + i * step for i in range(n)]
    dates = [_seconds_to_date_string(t) for t in times]
    df = pd.DataFrame(
        {
            S_LAT_COL: [52.5 + 0.0001 * i for i in range(n)],
            S_LON_COL: [-1.48 - 0.0001 * i for i in range(n)],
            " GPS ALTITUDE (m)": [100.0] * n,
            " TIME SINCE START (ms)": [i * 100 for i in range(n)],
            S_DATE_COL: dates,
            " ACCELEROMETER X (m/s\u00b2) ": [0.1 * i for i in range(n)],
        }
    )
    if add_far_outlier:
        far_row = df.iloc[[-1]].copy()
        far_row[S_DATE_COL] = _seconds_to_date_string(500.0)
        far_row[" TIME SINCE START (ms)"] = 999999
        df = pd.concat([df, far_row], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# 1 & 2. Loading real-shaped files succeeds
# ---------------------------------------------------------------------------

def test_v_file_loads_successfully(tmp_path):
    df_v = build_synthetic_v()
    path = tmp_path / "V-TestSeq.csv"
    df_v.to_csv(path, index=False)
    loaded, encoding = load_csv_robust(path)
    assert loaded.shape == df_v.shape
    assert encoding in ("utf-8", "cp1252", "latin-1")


def test_s_file_loads_successfully(tmp_path):
    df_s = build_synthetic_s()
    path = tmp_path / "S-TestSeq.csv"
    df_s.to_csv(path, index=False, encoding="cp1252")
    loaded, encoding = load_csv_robust(path)
    assert loaded.shape == df_s.shape


# ---------------------------------------------------------------------------
# 3. Missing files generate clear errors
# ---------------------------------------------------------------------------

def test_missing_dataset_path_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(DatasetPathError):
        resolve_dataset_root(str(missing))


def test_missing_sv_pair_raises_clear_error(tmp_path):
    # Directory exists but contains no matching S/V files.
    with pytest.raises(SVFileNotFoundError):
        discover_sv_pair(tmp_path, "NoSuchSequence")


def test_load_csv_robust_missing_file_raises(tmp_path):
    with pytest.raises(SVFileNotFoundError):
        load_csv_robust(tmp_path / "ghost.csv")


# ---------------------------------------------------------------------------
# 4. Time fields are detected
# ---------------------------------------------------------------------------

def test_time_columns_detected_in_v():
    df_v = build_synthetic_v()
    candidates = detect_time_column_candidates(df_v)
    candidate_names = [c["column"] for c in candidates]
    assert V_TIME_COL in candidate_names


def test_time_columns_detected_in_s():
    df_s = build_synthetic_s()
    candidates = detect_time_column_candidates(df_s)
    candidate_names = [c["column"] for c in candidates]
    assert S_DATE_COL in candidate_names


# ---------------------------------------------------------------------------
# 5. Sampling information is calculated
# ---------------------------------------------------------------------------

def test_sampling_info_calculation():
    times = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])
    info = compute_sampling_info(times)
    assert info["interval_mean_seconds"] == pytest.approx(0.1)
    assert info["frequency_hz"] == pytest.approx(10.0)


def test_date_derivation_produces_valid_seconds():
    df_s = build_synthetic_s(add_far_outlier=False)
    derived = derive_seconds_since_midnight_from_date(df_s[S_DATE_COL])
    assert derived.notna().all()
    # Should be monotonically increasing given how the synthetic data was built.
    assert derived.is_monotonic_increasing


# ---------------------------------------------------------------------------
# 6. S/V time ranges are compared
# ---------------------------------------------------------------------------

def test_time_range_report():
    df_v = build_synthetic_v()
    report = time_range_report(df_v[V_TIME_COL])
    assert report["start_seconds"] == pytest.approx(100.0)
    assert report["end_seconds"] == pytest.approx(100.0 + 0.1 * 19)


# ---------------------------------------------------------------------------
# 7. Time-based synchronization works
# ---------------------------------------------------------------------------

def test_synchronization_matches_close_samples():
    df_v = build_synthetic_v()
    df_s = build_synthetic_s(add_far_outlier=False)
    df_s = df_s.copy()
    df_s["sync_time_derived_seconds"] = derive_seconds_since_midnight_from_date(df_s[S_DATE_COL])
    df_v = df_v.copy()

    result = synchronize_sv(
        df_s, df_v, s_sync_col="sync_time_derived_seconds", v_sync_col=V_TIME_COL,
        tolerance_seconds=0.05,
    )
    # All 18 in-range S samples (offset 0.02s from V, within 0.05s tolerance) should match.
    assert result.matched_count == 18
    assert result.unmatched_count == 0
    assert result.match_percentage == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 8. Unmatched samples are handled correctly (never fabricated)
# ---------------------------------------------------------------------------

def test_unmatched_samples_marked_not_fabricated():
    df_v = build_synthetic_v()
    df_s = build_synthetic_s(add_far_outlier=True)  # adds a sample at t=500s, far outside V's range
    df_s = df_s.copy()
    df_s["sync_time_derived_seconds"] = derive_seconds_since_midnight_from_date(df_s[S_DATE_COL])

    result = synchronize_sv(
        df_s, df_v, s_sync_col="sync_time_derived_seconds", v_sync_col=V_TIME_COL,
        tolerance_seconds=0.05,
    )
    assert result.unmatched_count == 1
    aligned = result.aligned_df
    unmatched_rows = aligned.loc[~aligned["matched"]]
    assert len(unmatched_rows) == 1
    # The V_ columns for the unmatched row must be NaN, never a fabricated value.
    v_cols = [c for c in aligned.columns if c.startswith("V_")]
    assert unmatched_rows[v_cols].isna().all().all()


# ---------------------------------------------------------------------------
# 9. Output contains no duplicate column names
# ---------------------------------------------------------------------------

def test_no_duplicate_columns_in_aligned_output():
    df_v = build_synthetic_v()
    df_s = build_synthetic_s(add_far_outlier=False)
    df_s = df_s.copy()
    df_s["sync_time_derived_seconds"] = derive_seconds_since_midnight_from_date(df_s[S_DATE_COL])

    result = synchronize_sv(
        df_s, df_v, s_sync_col="sync_time_derived_seconds", v_sync_col=V_TIME_COL,
        tolerance_seconds=0.05,
    )
    cols = result.aligned_df.columns
    assert len(cols) == len(set(cols))


# ---------------------------------------------------------------------------
# 10. Output can be saved and loaded again
# ---------------------------------------------------------------------------

def test_aligned_output_round_trips_through_csv(tmp_path):
    df_v = build_synthetic_v()
    df_s = build_synthetic_s(add_far_outlier=False)
    df_s = df_s.copy()
    df_s["sync_time_derived_seconds"] = derive_seconds_since_midnight_from_date(df_s[S_DATE_COL])

    result = synchronize_sv(
        df_s, df_v, s_sync_col="sync_time_derived_seconds", v_sync_col=V_TIME_COL,
        tolerance_seconds=0.05,
    )
    out_path = tmp_path / "aligned.csv"
    result.aligned_df.to_csv(out_path, index=False)
    reloaded = pd.read_csv(out_path)
    assert reloaded.shape == result.aligned_df.shape
    assert list(reloaded.columns) == list(result.aligned_df.columns)


# ---------------------------------------------------------------------------
# 11 & 12. JSON and TXT reports are generated (via end-to-end CLI run)
# ---------------------------------------------------------------------------

def test_end_to_end_cli_generates_reports_and_plot(tmp_path, monkeypatch):
    # Build a tiny, real-shaped dataset on disk and point the CLI at it.
    dataset_root = tmp_path / "io_vnbd_synth"
    seq_dir = dataset_root / "Vf (Driver E)" / "V-TestSeq"
    seq_dir.mkdir(parents=True)

    df_v = build_synthetic_v()
    df_s = build_synthetic_s(add_far_outlier=True)

    df_v.to_csv(seq_dir / "V-TestSeq.csv", index=False)
    df_s.to_csv(seq_dir / "S-TestSeq.csv", index=False, encoding="cp1252")

    # Redirect the pipeline's output directories into tmp_path so we don't
    # touch the real project's data/results folders during testing.
    monkeypatch.setattr(stage1_main, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(stage1_main, "REPORTS_DIR", tmp_path / "reports")

    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--dataset", str(dataset_root), "--sequence", "TestSeq", "--tolerance", "0.05"],
    )

    exit_code = stage1_main.main()
    assert exit_code == 0

    json_path = tmp_path / "reports" / "TestSeq_report.json"
    txt_path = tmp_path / "reports" / "TestSeq_report.txt"
    plot_path = tmp_path / "reports" / "TestSeq_sync_diagnostic.png"
    aligned_path = tmp_path / "processed" / "TestSeq_aligned.csv"

    assert json_path.exists()
    assert txt_path.exists()
    assert plot_path.exists()
    assert aligned_path.exists()

    with open(json_path) as f:
        report = json.load(f)
    assert report["sequence"] == "TestSeq"
    assert report["matched_rows"] == 18
    assert report["unmatched_rows"] == 1

    assert "STAGE 1" in txt_path.read_text(encoding="utf-8")
