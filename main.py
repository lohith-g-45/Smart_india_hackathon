#!/usr/bin/env python3
"""
main.py
-------
NAV-SHIELD - Member 5 AI Drift Correction module
STAGE 1: IO-VNBD Data Loading, Validation & S/V Synchronization

Usage
-----
    python main.py --dataset "D:\\SIH\\IO-VNBD\\Synchronised V and S Dataset" --sequence Vfa01

This script performs ONLY Stage 1 of the NAV-SHIELD pipeline:
    IO-VNBD Dataset -> Data Loading -> Data Validation -> S/V Synchronization

It does NOT perform preprocessing/calibration, feature engineering, LSTM
modeling, dead reckoning, drift correction, or evaluation. Those are later
stages.

Outputs
-------
    data/processed/<sequence>_aligned.csv
    results/data_reports/<sequence>_report.json
    results/data_reports/<sequence>_report.txt
    results/data_reports/<sequence>_sync_diagnostic.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import (
    CSVLoadError,
    DatasetPathError,
    SVFileNotFoundError,
    discover_sv_pair,
    load_sv_pair,
    resolve_dataset_root,
)
from src.data_validator import full_validation_report
from src.synchronizer import derive_seconds_since_midnight_from_date, synchronize_sv

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "results" / "data_reports"

# --- Actual IO-VNBD column names (as discovered by inspecting the real
# S-Vfa01.csv / V-Vfa01.csv files). These are used ONLY as the default
# lookup keys; if a future sequence has different exact spacing/naming,
# the script will report a clear error rather than silently guessing.
V_TIME_COL = " Time Since Start of Day (seconds)"
V_LAT_COL = " Latitude (degrees)"
V_LON_COL = " Longitude (degrees)"

S_DATE_COL = " DATE (YYYY-MO-DD HH-MI-SS_SSS"
S_LAT_COL = "GPS LATITUDE (degrees)"
S_LON_COL = " GPS LONGITUDE (degrees)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NAV-SHIELD Stage 1: IO-VNBD loading, validation & S/V synchronization."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the local, extracted IO-VNBD 'Synchronised V and S Dataset' folder.",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Sequence name, e.g. Vfa01 (matches S-<seq>.csv / V-<seq>.csv).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Max allowed |S_time - V_time| in seconds for a synchronization "
        "match (default: 0.05s, i.e. half of V's 0.1s sample period).",
    )
    return parser.parse_args()


def find_column(df: pd.DataFrame, exact_name: str) -> str:
    """
    Resolve a column name against the DataFrame, tolerating differences in
    surrounding whitespace (the raw CSV headers have inconsistent leading
    spaces). Raises a clear KeyError if nothing matches.
    """
    if exact_name in df.columns:
        return exact_name
    stripped_target = exact_name.strip().lower()
    for col in df.columns:
        if col.strip().lower() == stripped_target:
            return col
    raise KeyError(
        f"Expected column resembling '{exact_name}' not found. "
        f"Actual columns: {list(df.columns)}"
    )


def make_json_safe(obj):
    """Recursively convert numpy/pandas scalar types into plain Python types for json.dump."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def write_txt_report(report: dict, path: Path) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("NAV-SHIELD STAGE 1 - DATA VALIDATION & SYNCHRONIZATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Dataset name: {report['dataset_name']}")
    lines.append(f"Sequence: {report['sequence']}")
    lines.append(f"S file: {report['s_file_path']}")
    lines.append(f"V file: {report['v_file_path']}")
    lines.append("")
    lines.append("-- SHAPE --")
    lines.append(f"S rows/cols: {report['s_row_count']} / {report['s_column_count']}")
    lines.append(f"V rows/cols: {report['v_row_count']} / {report['v_column_count']}")
    lines.append("")
    lines.append("-- TIME --")
    lines.append(f"S time range (s since midnight): {report['s_time_range']}")
    lines.append(f"V time range (s since midnight): {report['v_time_range']}")
    lines.append(f"S sampling info: {report['s_sampling_info']}")
    lines.append(f"V sampling info: {report['v_sampling_info']}")
    lines.append("")
    lines.append("-- SYNCHRONIZATION --")
    lines.append(f"Selected S sync field: {report['selected_s_sync_field']}")
    lines.append(f"Selected V sync field: {report['selected_v_sync_field']}")
    lines.append(f"Method: {report['synchronization_method']}")
    lines.append(f"Tolerance (s): {report['synchronization_tolerance_seconds']}")
    lines.append(f"Matched rows: {report['matched_rows']}")
    lines.append(f"Unmatched rows: {report['unmatched_rows']}")
    lines.append(f"Match percentage: {report['match_percentage']:.2f}%")
    lines.append("")
    lines.append("-- DATA QUALITY (S) --")
    lines.append(f"Duplicate rows: {report['s_validation']['duplicates']['duplicate_row_count']}")
    lines.append(f"Duplicate timestamps: {report['s_validation']['duplicate_timestamps']}")
    lines.append(f"Infinite values: {report['s_validation']['infinite_values']}")
    lines.append(f"Invalid numeric values: {report['s_validation']['invalid_numeric_values']}")
    lines.append(f"Lat/Lon validity: {report['s_validation']['latlon_validity']}")
    lines.append("")
    lines.append("-- DATA QUALITY (V) --")
    lines.append(f"Duplicate rows: {report['v_validation']['duplicates']['duplicate_row_count']}")
    lines.append(f"Duplicate timestamps: {report['v_validation']['duplicate_timestamps']}")
    lines.append(f"Infinite values: {report['v_validation']['infinite_values']}")
    lines.append(f"Invalid numeric values: {report['v_validation']['invalid_numeric_values']}")
    lines.append(f"Lat/Lon validity: {report['v_validation']['latlon_validity']}")
    lines.append("")
    lines.append("-- WARNINGS / PROBLEMS --")
    if report["warnings"]:
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    else:
        lines.append("  (none)")
    lines.append("=" * 70)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_diagnostic_plot(
    s_time: pd.Series, v_time: pd.Series, aligned_df: pd.DataFrame, out_path: Path
) -> None:
    """
    Basic Stage-1 diagnostic plot: S time coverage, V time coverage, and
    matched-vs-unmatched coverage along the synchronization timeline.
    This is NOT the final drift/trajectory visualization (that is a later stage).
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].eventplot([s_time.dropna().values], lineoffsets=1, colors="tab:blue")
    axes[0].set_yticks([])
    axes[0].set_title("S (smartphone) time coverage")

    axes[1].eventplot([v_time.dropna().values], lineoffsets=1, colors="tab:orange")
    axes[1].set_yticks([])
    axes[1].set_title("V (vehicle) time coverage")

    matched = aligned_df.loc[aligned_df["matched"], "timestamp_seconds"]
    unmatched = aligned_df.loc[~aligned_df["matched"], "timestamp_seconds"]
    if len(matched) > 0:
        axes[2].eventplot([matched.values], lineoffsets=1, colors="tab:green")
    if len(unmatched) > 0:
        axes[2].eventplot([unmatched.values], lineoffsets=0, colors="tab:red")
    axes[2].set_yticks([0, 1], ["unmatched", "matched"])
    axes[2].set_title("Synchronization match coverage (S samples)")
    axes[2].set_xlabel("Time since local midnight (seconds)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("NAV-SHIELD Stage 1: Data Loading, Validation & S/V Synchronization")
    print("=" * 70)

    # ---- 1-2. Locate + load ----
    try:
        dataset_root = resolve_dataset_root(args.dataset)
        pair = discover_sv_pair(dataset_root, args.sequence)
        print(f"[OK] S file located: {pair.s_path}")
        print(f"[OK] V file located: {pair.v_path}")
        df_s, s_encoding, df_v, v_encoding = load_sv_pair(pair)
        print(f"[OK] Loaded S: {df_s.shape[0]} rows x {df_s.shape[1]} cols (encoding={s_encoding})")
        print(f"[OK] Loaded V: {df_v.shape[0]} rows x {df_v.shape[1]} cols (encoding={v_encoding})")
    except (DatasetPathError, SVFileNotFoundError, CSVLoadError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    warnings = []

    # ---- 3-6. Resolve actual time/lat/lon columns ----
    try:
        v_time_col = find_column(df_v, V_TIME_COL)
        v_lat_col = find_column(df_v, V_LAT_COL)
        v_lon_col = find_column(df_v, V_LON_COL)
        s_date_col = find_column(df_s, S_DATE_COL)
        s_lat_col = find_column(df_s, S_LAT_COL)
        s_lon_col = find_column(df_s, S_LON_COL)
    except KeyError as exc:
        print(f"[ERROR] Expected column not found: {exc}", file=sys.stderr)
        return 1

    # Derive S's seconds-since-midnight sync field from its DATE column.
    # Named WITHOUT a leading underscore (and distinct from any real S column)
    # so that after S_/V_ prefixing in the aligned output it reads cleanly,
    # e.g. "S_sync_time_derived_seconds", with no duplicate/ambiguous columns.
    df_s = df_s.copy()
    df_s["sync_time_derived_seconds"] = derive_seconds_since_midnight_from_date(df_s[s_date_col])
    unparseable_dates = int(df_s["sync_time_derived_seconds"].isna().sum())
    if unparseable_dates > 0:
        warnings.append(
            f"{unparseable_dates} S row(s) had an unparseable DATE value and "
            "could not be assigned a synchronization time."
        )

    df_v = df_v.copy()
    # V's own "Time Since Start of Day (seconds)" column IS the sync field --
    # just coerce it to numeric in place rather than duplicating it under a
    # new column name.
    df_v[v_time_col] = pd.to_numeric(df_v[v_time_col], errors="coerce")

    s_sync_col = "sync_time_derived_seconds"
    v_sync_col = v_time_col

    print(f"[OK] S synchronization field: derived seconds-since-midnight from '{s_date_col}'")
    print(f"[OK] V synchronization field: '{v_time_col}'")

    # ---- 7-11. Validation ----
    print("[..] Running data validation on S and V ...")
    s_report = full_validation_report(
        df_s, name="S (smartphone)", time_col=s_sync_col, lat_col=s_lat_col, lon_col=s_lon_col
    )
    v_report = full_validation_report(
        df_v, name="V (vehicle)", time_col=v_sync_col, lat_col=v_lat_col, lon_col=v_lon_col
    )

    if s_report["duplicates"]["duplicate_row_count"] > 0:
        warnings.append(f"S contains {s_report['duplicates']['duplicate_row_count']} fully duplicated row(s).")
    if v_report["duplicates"]["duplicate_row_count"] > 0:
        warnings.append(f"V contains {v_report['duplicates']['duplicate_row_count']} fully duplicated row(s).")
    if s_report["latlon_validity"].get("lat_invalid_count"):
        warnings.append(f"S has {s_report['latlon_validity']['lat_invalid_count']} out-of-range latitude value(s).")
    if v_report["latlon_validity"].get("lat_invalid_count"):
        warnings.append(f"V has {v_report['latlon_validity']['lat_invalid_count']} out-of-range latitude value(s).")

    s_time_range = s_report["time_range"]
    v_time_range = v_report["time_range"]
    if s_time_range and v_time_range and s_time_range["start_seconds"] is not None and v_time_range["start_seconds"] is not None:
        overlap_start = max(s_time_range["start_seconds"], v_time_range["start_seconds"])
        overlap_end = min(s_time_range["end_seconds"], v_time_range["end_seconds"])
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            warnings.append("S and V time ranges do NOT overlap. Synchronization will fail for all rows.")
        else:
            print(f"[OK] S/V time range overlap: {overlap:.2f} seconds")

    # ---- 12-13. Synchronization ----
    print(f"[..] Synchronizing S/V by time (tolerance={args.tolerance}s) ...")
    sync_result = synchronize_sv(
        df_s, df_v, s_sync_col=s_sync_col, v_sync_col=v_sync_col, tolerance_seconds=args.tolerance
    )
    aligned_df = sync_result.aligned_df
    print(
        f"[OK] Matched {sync_result.matched_count} / {len(df_s)} S samples "
        f"({sync_result.match_percentage:.2f}%)"
    )

    # ---- 14. Save aligned dataset ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    aligned_csv_path = PROCESSED_DIR / f"{args.sequence}_aligned.csv"
    aligned_df.to_csv(aligned_csv_path, index=False)
    print(f"[OK] Aligned dataset saved: {aligned_csv_path}")

    # Sanity check: no duplicate/ambiguous column names in the aligned output.
    dup_cols = aligned_df.columns[aligned_df.columns.duplicated()].tolist()
    if dup_cols:
        warnings.append(f"Aligned output has duplicate column names: {dup_cols}")

    # ---- 15. Reports ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "IO-VNBD Synchronised V and S Dataset",
        "sequence": args.sequence,
        "s_file_path": str(pair.s_path),
        "v_file_path": str(pair.v_path),
        "s_encoding": s_encoding,
        "v_encoding": v_encoding,
        "s_row_count": s_report["row_count"],
        "v_row_count": v_report["row_count"],
        "s_column_count": s_report["column_count"],
        "v_column_count": v_report["column_count"],
        "s_columns": s_report["columns"],
        "v_columns": v_report["columns"],
        "s_time_range": s_time_range,
        "v_time_range": v_time_range,
        "s_sampling_info": s_report["sampling_info"],
        "v_sampling_info": v_report["sampling_info"],
        "selected_s_sync_field": f"derived seconds-since-midnight from '{s_date_col}'",
        "selected_v_sync_field": v_time_col,
        "synchronization_method": sync_result.method,
        "synchronization_tolerance_seconds": sync_result.tolerance_seconds,
        "matched_rows": sync_result.matched_count,
        "unmatched_rows": sync_result.unmatched_count,
        "match_percentage": sync_result.match_percentage,
        "s_validation": s_report,
        "v_validation": v_report,
        "warnings": warnings,
        "aligned_csv_path": str(aligned_csv_path),
    }

    json_path = REPORTS_DIR / f"{args.sequence}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2)
    print(f"[OK] JSON report saved: {json_path}")

    txt_path = REPORTS_DIR / f"{args.sequence}_report.txt"
    write_txt_report(report, txt_path)
    print(f"[OK] TXT report saved: {txt_path}")

    # ---- 16. Diagnostic plot ----
    plot_path = REPORTS_DIR / f"{args.sequence}_sync_diagnostic.png"
    generate_diagnostic_plot(df_s[s_sync_col], df_v[v_sync_col], aligned_df, plot_path)
    print(f"[OK] Diagnostic plot saved: {plot_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("STAGE 1 SUMMARY")
    print("=" * 70)
    print(f"Sequence:            {args.sequence}")
    print(f"S rows / V rows:     {s_report['row_count']} / {v_report['row_count']}")
    print(f"Matched / Unmatched: {sync_result.matched_count} / {sync_result.unmatched_count}")
    print(f"Match percentage:    {sync_result.match_percentage:.2f}%")
    print(f"Warnings raised:     {len(warnings)}")
    for w in warnings:
        print(f"  - {w}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
