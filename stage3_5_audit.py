#!/usr/bin/env python3
"""NAV-SHIELD Stage 3.5: read-only audit of Stage 3 feature outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "features" / "categorised"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "data_reports" / "stage3_5"
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
EXPECTED_SEQUENCES = 72
EXPECTED_FEATURE_COLUMNS = 146
ALLOWED_OBJECT_TOKENS = ("sequence_id", "date")


def load_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
            if len(columns) != len(set(columns)):
                raise ValueError("duplicate column names")
            frame.columns = columns
            return frame
        except (UnicodeError, pd.errors.ParserError, OSError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"Could not read '{path}': {last_error}")


def sequence_from_path(path: Path) -> str:
    return path.name[: -len("_features.csv")].lower()


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Feature input directory does not exist: {input_path}")
    return sorted(input_path.rglob("*_features.csv"), key=lambda path: path.name.lower())


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=[np.number]).columns.tolist()


def finite_values(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def missing_reason(feature: str, frame: pd.DataFrame, missing_count: int, total_rows: int) -> str:
    if not missing_count:
        return "None"
    name = feature.lower()
    if feature == "gyro_magnitude" or feature.startswith("gyro_magnitude_") or feature in {"gyro_change", "gyro_change_rate"} or feature.startswith("gyro_change_"):
        if {"gyro_yaw", "gyro_pitch", "gyro_roll"}.issubset(frame.columns):
            return "Other feature-engineering issue: Stage 3 gyro magnitude requests x/y/z keys although detected gyro axes are yaw/pitch/roll."
    if name.endswith("delta_time") or name == "delta_time" or name.endswith("_change") or name.endswith("_change_rate") or name in {"accel_change", "speed_change", "yaw_change", "pitch_change", "roll_change"}:
        return "B. Differencing / derivative initialization or unavailable preceding sample."
    if "rolling" in name:
        base = re.sub(r"_rolling_(mean|std|min|max|range)$", "", name)
        if base in frame.columns and frame[base].isna().sum() > 0:
            return "Inherited missingness from the source/underlying feature propagated through a trailing rolling calculation."
        return "A. Rolling-window initialization or unavailable underlying feature."
    if name.endswith("_rate"):
        return "F. Division by zero or unavailable delta time in rate calculation."
    if feature.startswith("V_") and "synchronization_valid" in frame:
        invalid = (~frame["synchronization_valid"].astype(bool)).sum()
        if invalid > 0:
            return "G. Synchronization issue: unmatched Stage 2 rows have no V-side values."
    if feature.startswith(("S_", "V_")):
        return "D. Sensor/source missingness inherited from the synchronized input."
    if feature in {"timestamp", "elapsed_time"}:
        return "H. Other feature-engineering issue: unavailable synchronization timestamp."
    return "Cause requires further inspection."


def range_flags(frame: pd.DataFrame) -> dict[str, int]:
    rules = {
        "latitude_invalid": ("gps_latitude", lambda values: ~values.between(-90, 90)),
        "longitude_invalid": ("gps_longitude", lambda values: ~values.between(-180, 180)),
        "gps_speed_invalid": ("gps_speed", lambda values: (values < 0) | (values > 300)),
        "gps_accuracy_invalid": ("gps_accuracy", lambda values: (values < 0) | (values > 1000)),
        "altitude_invalid": ("gps_altitude", lambda values: (values < -1000) | (values > 10000)),
        "accelerometer_outlier": ("accel_magnitude", lambda values: values > 100),
        "gyroscope_outlier": ("gyro_yaw", lambda values: values.abs() > 50),
        "magnetometer_outlier": ("mag_magnitude", lambda values: values > 1000),
        "orientation_outlier": ("orientation_yaw", lambda values: values.abs() > 720),
    }
    result = {}
    for label, (column, predicate) in rules.items():
        if column not in frame:
            result[label] = 0
            continue
        values = finite_values(frame[column])
        result[label] = int(predicate(values).fillna(False).sum())
    return result


def likely_numeric_feature(column: str, dtype: str) -> bool:
    return dtype.startswith(("int", "float", "bool")) or column in {"timestamp", "elapsed_time", "delta_time"}


def feature_status(missing_percentage: float, infinite_count: int, constant: bool, near_constant: bool) -> str:
    if infinite_count or missing_percentage >= 100.0:
        return "FAIL"
    if missing_percentage > 0 or constant or near_constant:
        return "WARNING"
    return "PASS"


def update_stats(stats: dict[str, dict], frame: pd.DataFrame) -> None:
    total_rows = len(frame)
    for column in frame.columns:
        entry = stats.setdefault(column, {
            "dtype_values": set(), "total_rows": 0, "missing_count": 0, "infinite_count": 0,
            "finite_count": 0, "unique_values": 0, "min": np.nan, "max": np.nan,
            "sum": 0.0, "sumsq": 0.0, "sample": [], "dominant_count": 0,
        })
        entry["dtype_values"].add(str(frame[column].dtype))
        entry["total_rows"] += total_rows
        entry["missing_count"] += int(frame[column].isna().sum())
        numeric = pd.api.types.is_numeric_dtype(frame[column])
        if not numeric:
            entry["unique_values"] += int(frame[column].nunique(dropna=True))
            continue
        raw_values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        infinite = np.isinf(raw_values)
        finite = np.isfinite(raw_values)
        entry["infinite_count"] += int(infinite.sum())
        values = raw_values[finite]
        entry["finite_count"] += int(values.size)
        entry["unique_values"] += int(pd.Series(values).nunique())
        if values.size:
            entry["min"] = float(values.min()) if pd.isna(entry["min"]) else min(entry["min"], float(values.min()))
            entry["max"] = float(values.max()) if pd.isna(entry["max"]) else max(entry["max"], float(values.max()))
            entry["sum"] += float(values.sum())
            entry["sumsq"] += float(np.square(values).sum())
            sample = values[:: max(1, values.size // 200)]
            entry["sample"].extend(sample[:200].tolist())
            counts = pd.Series(values).value_counts()
            entry["dominant_count"] += int(counts.iloc[0]) if not counts.empty else 0


def finalize_feature_stats(stats: dict[str, dict], total_sequences: int) -> pd.DataFrame:
    rows = []
    for feature, entry in sorted(stats.items()):
        finite_count = entry["finite_count"]
        total = entry["total_rows"]
        missing_percentage = entry["missing_count"] / total * 100.0 if total else 0.0
        values = np.asarray(entry["sample"], dtype=float)
        mean = entry["sum"] / finite_count if finite_count else np.nan
        variance = max(0.0, entry["sumsq"] / finite_count - mean * mean) if finite_count else np.nan
        unique = entry["unique_values"]
        constant = finite_count > 0 and unique <= 1
        dominant_percentage = entry["dominant_count"] / finite_count * 100.0 if finite_count else 0.0
        near_constant = finite_count > 0 and (dominant_percentage >= 99.0 or unique <= max(2, int(finite_count * 0.01)))
        reason = "None"
        if entry["missing_count"]:
            reason = "Cause requires further inspection."
            if feature == "gyro_magnitude" or feature.startswith("gyro_magnitude_") or feature in {"gyro_change", "gyro_change_rate"} or feature.startswith("gyro_change_"):
                reason = "Other feature-engineering issue: Stage 3 vector_magnitude expects x/y/z, but gyroscope axes are yaw/pitch/roll."
            elif feature == "delta_time" or feature.endswith("_change") or feature.endswith("_change_rate"):
                reason = "B/F. Differencing initialization and/or division by zero in rate calculation."
            elif "rolling" in feature:
                reason = "A. Trailing rolling feature initialization or propagation from its underlying feature."
            elif feature.startswith(("S_", "V_")):
                reason = "D/G. Source missingness or unmatched synchronized V row."
        rows.append({
            "feature_name": feature,
            "dtype": ";".join(sorted(entry["dtype_values"])),
            "total_rows": total,
            "missing_count": entry["missing_count"],
            "missing_percentage": missing_percentage,
            "infinite_count": entry["infinite_count"],
            "finite_count": finite_count,
            "unique_values": unique,
            "is_constant": constant,
            "is_near_constant": near_constant,
            "min": entry["min"],
            "max": entry["max"],
            "mean": mean,
            "std": float(np.sqrt(variance)) if np.isfinite(variance) else np.nan,
            "median": float(np.median(values)) if values.size else np.nan,
            "quality_status": feature_status(missing_percentage, entry["infinite_count"], constant, near_constant),
            "likely_missing_reason": reason,
        })
    return pd.DataFrame(rows)


def audit_sequence(path: Path, stats: dict[str, dict]) -> tuple[dict, list[dict], pd.DataFrame | None]:
    sequence = sequence_from_path(path)
    frame = load_csv(path)
    update_stats(stats, frame)
    total_rows = len(frame)
    missing_records = []
    for column in frame.columns:
        values = frame[column]
        infinite = int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum()) if pd.api.types.is_numeric_dtype(values) else 0
        missing = int(values.isna().sum())
        missing_records.append({
            "sequence_id": sequence, "feature_name": column, "total_rows": total_rows,
            "missing_count": missing, "missing_percentage": missing / total_rows * 100.0 if total_rows else 0.0,
            "infinite_count": infinite, "finite_count": int(np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).sum()) if pd.api.types.is_numeric_dtype(values) else np.nan,
            "likely_missing_reason": missing_reason(column, frame, missing, total_rows),
        })
    numeric = frame.select_dtypes(include=[np.number])
    constants = int(sum(numeric[column].nunique(dropna=True) <= 1 for column in numeric.columns))
    near_constants = 0
    for column in numeric.columns:
        valid = numeric[column].dropna()
        if len(valid) and (valid.nunique() <= max(2, int(len(valid) * 0.01)) or valid.value_counts(normalize=True).iloc[0] >= 0.99):
            near_constants += 1
    timestamp = finite_values(frame["timestamp"]) if "timestamp" in frame else pd.Series(dtype=float)
    timestamp_issues = []
    if timestamp.isna().any():
        timestamp_issues.append(f"missing_timestamp={int(timestamp.isna().sum())}")
    if not timestamp.dropna().is_monotonic_increasing:
        timestamp_issues.append("non_monotonic_timestamp")
    duplicate_timestamps = int(timestamp.duplicated().sum())
    if duplicate_timestamps:
        timestamp_issues.append(f"duplicate_timestamps={duplicate_timestamps}")
    gaps = int(frame["time_gap_flag"].sum()) if "time_gap_flag" in frame else 0
    if gaps:
        timestamp_issues.append(f"time_gaps={gaps}")
    flags = range_flags(frame)
    outlier_flags = {key: value for key, value in flags.items() if value}
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_column_names = [
        column for column in frame.columns
        if re.search(r"\.\d+$", column) and column.rsplit(".", 1)[0] in frame.columns
    ]
    non_numeric_columns = frame.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    unexpected_object_columns = [
        column for column in non_numeric_columns
        if not any(token in column.lower() for token in ALLOWED_OBJECT_TOKENS)
    ]
    infinite_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if not numeric.empty else 0
    missing_rate = float(frame.isna().mean().mean() * 100.0) if total_rows else 100.0
    major_engineered_bug = frame.get("gyro_magnitude", pd.Series(dtype=float)).isna().all() and {"gyro_yaw", "gyro_pitch", "gyro_roll"}.issubset(frame.columns)
    warnings = bool(missing_rate or duplicate_rows or duplicate_column_names or unexpected_object_columns or constants or near_constants or timestamp_issues or outlier_flags or infinite_count or major_engineered_bug)
    status = "WARNING" if warnings else "PASS"
    report = {
        "sequence_id": sequence, "rows": total_rows, "feature_count": len(frame.columns),
        "missing_rate": missing_rate, "infinite_count": infinite_count, "duplicate_rows": duplicate_rows,
        "constant_feature_count": constants, "near_constant_feature_count": near_constants,
        "timestamp_issues": "; ".join(timestamp_issues) or "None", "outlier_flags": json.dumps(outlier_flags, sort_keys=True),
        "duplicate_column_names": "; ".join(duplicate_column_names) or "None",
        "non_numeric_columns": "; ".join(non_numeric_columns) or "None",
        "unexpected_object_columns": "; ".join(unexpected_object_columns) or "None",
        "overall_status": status, "major_engineered_bug": bool(major_engineered_bug),
    }
    return report, missing_records, frame


def make_plots(feature_stats: pd.DataFrame, missing: pd.DataFrame, output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    top = feature_stats.sort_values("missing_percentage", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top["feature_name"].iloc[::-1], top["missing_percentage"].iloc[::-1])
    ax.set_xlabel("Missing percentage")
    ax.set_title("Stage 3.5 missing percentage by feature")
    fig.tight_layout()
    fig.savefig(plots / "missing_percentage_by_feature.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(feature_stats["missing_percentage"].fillna(0), bins=20)
    ax.set_xlabel("Feature missing percentage")
    ax.set_ylabel("Feature count")
    ax.set_title("Distribution of feature missingness")
    fig.tight_layout()
    fig.savefig(plots / "missing_percentage_distribution.png", dpi=130)
    plt.close(fig)

    matrix = missing.pivot_table(index="sequence_id", columns="feature_name", values="missing_percentage", aggfunc="first")
    top_names = feature_stats.sort_values("missing_percentage", ascending=False).head(25)["feature_name"]
    matrix = matrix.reindex(columns=[name for name in top_names if name in matrix.columns])
    fig, ax = plt.subplots(figsize=(14, 9))
    image = ax.imshow(matrix.fillna(0).to_numpy(), aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    ax.set_title("Missing percentage by sequence and feature (top 25)")
    fig.colorbar(image, ax=ax, label="Missing percentage")
    fig.tight_layout()
    fig.savefig(plots / "missing_percentage_by_sequence.png", dpi=130)
    plt.close(fig)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_summary(summary: dict, path: Path) -> None:
    lines = [
        "NAV-SHIELD STAGE 3.5: DATA QUALITY AUDIT", "=" * 60,
        f"Sequences audited: {summary['sequences_audited']}", f"Total rows: {summary['total_rows']}",
        f"Feature columns: {summary['feature_columns']}", f"Overall missing-value rate: {summary['overall_missing_rate']:.4f}%",
        f"Total infinite values: {summary['total_infinite_values']}", f"Duplicate rows: {summary['duplicate_rows']}",
        f"Constant features: {summary['constant_features']}", f"Near-constant features: {summary['near_constant_features']}", "",
        f"PASS sequences: {summary['pass_sequences']}", f"WARNING sequences: {summary['warning_sequences']}", f"FAIL sequences: {summary['fail_sequences']}", "",
        "Missing-value assessment:", summary["missing_value_assessment"], "", "Data quality status: " + summary["data_quality_status"], "",
        "Recommended action before Stage 4:",
        "- Correct the Stage 3 gyroscope magnitude axis-key bug before training; do not impute it as if it were sensor missingness.",
        "- Preserve legitimate first-row derivative NaNs and unmatched V-side NaNs, then define training-time imputation using training data only.",
        "- Do not remove features or outliers solely from this audit; review flagged ranges and near-constant fields first.",
        "- No normalization, splitting, imputation, feature removal, or model training was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_before_after_comparison(before_dir: Path, after_summary: dict, after_features: pd.DataFrame, path: Path) -> None:
    before_summary_path = before_dir / "stage3_5_summary.json"
    before_features_path = before_dir / "feature_statistics.csv"
    if not before_summary_path.exists() or not before_features_path.exists():
        return
    before_summary = json.loads(before_summary_path.read_text(encoding="utf-8"))
    before_features = pd.read_csv(before_features_path)
    metrics = [
        ("Total rows", before_summary.get("total_rows"), after_summary.get("total_rows")),
        ("Feature count", before_summary.get("feature_columns"), after_summary.get("feature_columns")),
        ("Overall missing-value rate", before_summary.get("overall_missing_rate"), after_summary.get("overall_missing_rate")),
        ("100%-missing features", before_summary.get("missing_value_buckets", {}).get("one_hundred"), after_summary.get("missing_value_buckets", {}).get("one_hundred")),
        ("Infinite values", before_summary.get("total_infinite_values"), after_summary.get("total_infinite_values")),
        ("Duplicate rows", before_summary.get("duplicate_rows"), after_summary.get("duplicate_rows")),
        ("Constant features", before_summary.get("constant_features"), after_summary.get("constant_features")),
        ("Near-constant features", before_summary.get("near_constant_features"), after_summary.get("near_constant_features")),
    ]
    derived_gyro_names = {
        "gyro_magnitude", "gyro_change", "gyro_change_rate",
        "gyro_magnitude_rolling_mean", "gyro_magnitude_rolling_std",
        "gyro_magnitude_rolling_min", "gyro_magnitude_rolling_max",
        "gyro_magnitude_rolling_range", "gyro_change_rolling_mean",
        "gyro_change_rolling_std", "gyro_change_rolling_min",
        "gyro_change_rolling_max", "gyro_change_rolling_range",
    }
    before_gyro = before_features[before_features["feature_name"].isin(derived_gyro_names)]
    after_gyro = after_features[after_features["feature_name"].isin(derived_gyro_names)]
    lines = ["NAV-SHIELD STAGE 3.5 BEFORE/AFTER GYRO FIX", "=" * 60, "", "Metric | BEFORE GYRO FIX | AFTER GYRO FIX | CHANGE", "--- | ---: | ---: | ---:"]
    for label, before, after in metrics:
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            change = after - before
        else:
            change = "n/a"
        lines.append(f"{label} | {before} | {after} | {change}")
    lines.extend(["", "Affected gyro-derived features", "", "Feature | Before missing % | After missing % | After status", "--- | ---: | ---: | ---"])
    for feature in sorted(set(before_gyro["feature_name"]) | set(after_gyro["feature_name"])):
        before_row = before_gyro[before_gyro["feature_name"] == feature]
        after_row = after_gyro[after_gyro["feature_name"] == feature]
        before_pct = before_row["missing_percentage"].iloc[0] if not before_row.empty else "n/a"
        after_pct = after_row["missing_percentage"].iloc[0] if not after_row.empty else "n/a"
        after_status = after_row["quality_status"].iloc[0] if not after_row.empty else "n/a"
        lines.append(f"{feature} | {before_pct} | {after_pct} | {after_status}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 3 feature-engineered CSV files without modifying them.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Stage 3 feature directory or one feature CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Stage 3.5 report directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        inputs = discover_inputs(input_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print("=" * 60)
    print("NAV-SHIELD STAGE 3.5: DATA QUALITY AUDIT")
    print("=" * 60)
    print(f"Feature files found: {len(inputs)}")
    print(f"Sequence IDs: {', '.join(sequence_from_path(path) for path in inputs)}")
    if not inputs:
        print("No feature files found.")
        return 1

    stats: dict[str, dict] = {}
    sequence_reports = []
    missing_records = []
    unreadable = []
    for path in inputs:
        try:
            report, records, _ = audit_sequence(path, stats)
            sequence_reports.append(report)
            missing_records.extend(records)
            print(f"[{report['overall_status']}] {report['sequence_id']}: {report['rows']} rows")
        except Exception as exc:
            sequence = sequence_from_path(path)
            unreadable.append(f"{sequence}: {exc}")
            sequence_reports.append({"sequence_id": sequence, "rows": 0, "feature_count": 0, "missing_rate": 100.0, "infinite_count": 0, "duplicate_rows": 0, "constant_feature_count": 0, "near_constant_feature_count": 0, "timestamp_issues": "unreadable", "outlier_flags": "{}", "duplicate_column_names": "None", "non_numeric_columns": "None", "unexpected_object_columns": "None", "overall_status": "FAIL", "major_engineered_bug": False})

    feature_stats = finalize_feature_stats(stats, len(sequence_reports))
    missing_frame = pd.DataFrame(missing_records)
    if missing_frame.empty:
        missing_frame = pd.DataFrame(columns=["sequence_id", "feature_name", "total_rows", "missing_count", "missing_percentage", "infinite_count", "finite_count", "likely_missing_reason"])
    feature_stats.to_csv(output / "feature_statistics.csv", index=False)
    feature_stats.to_csv(output / "feature_quality_report.csv", index=False)
    derived_gyro_names = {
        "gyro_magnitude", "gyro_change", "gyro_change_rate",
        "gyro_magnitude_rolling_mean", "gyro_magnitude_rolling_std",
        "gyro_magnitude_rolling_min", "gyro_magnitude_rolling_max",
        "gyro_magnitude_rolling_range", "gyro_change_rolling_mean",
        "gyro_change_rolling_std", "gyro_change_rolling_min",
        "gyro_change_rolling_max", "gyro_change_rolling_range",
    }
    feature_stats[feature_stats["feature_name"].isin(derived_gyro_names)].to_csv(output / "gyro_feature_report.csv", index=False)
    missing_frame.to_csv(output / "missing_value_report.csv", index=False)
    sequence_frame = pd.DataFrame(sequence_reports)
    sequence_frame.to_csv(output / "sequence_quality_report.csv", index=False)

    total_rows = int(sum(report["rows"] for report in sequence_reports))
    total_missing = int(missing_frame["missing_count"].sum()) if not missing_frame.empty else 0
    total_cells = int(missing_frame["total_rows"].sum()) if not missing_frame.empty else 0
    total_infinite = int(feature_stats["infinite_count"].sum()) if not feature_stats.empty else 0
    constant_features = int(feature_stats["is_constant"].sum()) if not feature_stats.empty else 0
    near_constant_features = int(feature_stats["is_near_constant"].sum()) if not feature_stats.empty else 0
    missing_100 = feature_stats.loc[feature_stats["missing_percentage"] >= 100, "feature_name"].tolist()
    major_bug_features = feature_stats.loc[feature_stats["feature_name"].isin(["gyro_magnitude", "gyro_change", "gyro_magnitude_rolling_mean", "gyro_change_rolling_mean"]), "feature_name"].tolist()
    bug_detected = bool(major_bug_features and all(feature_stats.loc[feature_stats["feature_name"] == feature, "missing_percentage"].iloc[0] >= 100 for feature in major_bug_features))
    pass_count = int((sequence_frame["overall_status"] == "PASS").sum())
    warning_count = int((sequence_frame["overall_status"] == "WARNING").sum())
    fail_count = int((sequence_frame["overall_status"] == "FAIL").sum())
    bug_assessment = (
        "The missingness is not entirely legitimate: gyro-derived features are 100% missing across the audit because of a deterministic Stage 3 axis-key mismatch. Other missingness is consistent with first-row derivatives, division by zero in rates, source missingness, or unmatched V rows."
        if bug_detected else
        "The 100%-missing gyro-derived feature defect is resolved. Remaining missingness is consistent with first-row derivatives, division by zero in rates, source missingness, or unmatched V rows and should be handled later using training-data-only policy."
    )
    recommendation = (
        "Correct the Stage 3 gyro magnitude implementation before Stage 4; do not impute those features as sensor missingness."
        if bug_detected else
        "Review remaining derivative/rate/source missingness and duplicate/near-constant findings before Stage 4; do not automatically impute or remove them in Stage 3.5."
    )
    summary = {
        "input_directory": str(input_path), "output_directory": str(output), "feature_files_found": len(inputs),
        "sequence_ids": [sequence_from_path(path) for path in inputs], "sequences_audited": len(sequence_reports),
        "total_rows": total_rows, "feature_columns": int(max((report["feature_count"] for report in sequence_reports), default=0)),
        "overall_missing_rate": total_missing / total_cells * 100.0 if total_cells else 0.0,
        "total_infinite_values": total_infinite, "duplicate_rows": int(sequence_frame["duplicate_rows"].sum()),
        "constant_features": constant_features, "near_constant_features": near_constant_features,
        "pass_sequences": pass_count, "warning_sequences": warning_count, "fail_sequences": fail_count,
        "missing_value_buckets": {
            "zero": int((feature_stats["missing_percentage"] == 0).sum()),
            "zero_to_one": int(((feature_stats["missing_percentage"] > 0) & (feature_stats["missing_percentage"] <= 1)).sum()),
            "one_to_five": int(((feature_stats["missing_percentage"] > 1) & (feature_stats["missing_percentage"] <= 5)).sum()),
            "five_to_ten": int(((feature_stats["missing_percentage"] > 5) & (feature_stats["missing_percentage"] <= 10)).sum()),
            "ten_to_twenty_five": int(((feature_stats["missing_percentage"] > 10) & (feature_stats["missing_percentage"] <= 25)).sum()),
            "over_twenty_five": int((feature_stats["missing_percentage"] > 25).sum()),
            "one_hundred": len(missing_100),
        },
        "features_with_100_percent_missing": missing_100,
        "engineered_bug_features": major_bug_features if bug_detected else [],
        "unreadable_files": unreadable,
        "schema_audit": {
            "feature_count_values": sorted({int(report["feature_count"]) for report in sequence_reports}),
            "non_numeric_columns": sorted({column for report in sequence_reports for column in str(report.get("non_numeric_columns", "None")).split("; ") if column and column != "None"}),
            "unexpected_object_columns": sorted({column for report in sequence_reports for column in str(report.get("unexpected_object_columns", "None")).split("; ") if column and column != "None"}),
            "duplicate_column_name_sequences": [report["sequence_id"] for report in sequence_reports if report.get("duplicate_column_names") not in (None, "None")],
        },
        "missing_value_assessment": bug_assessment,
        "data_quality_status": "REQUIRES CORRECTION" if bug_detected or total_infinite or fail_count else "READY FOR NEXT STAGE",
        "recommendation": recommendation,
    }
    (output / "stage3_5_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    write_summary(summary, output / "stage3_5_summary.txt")
    write_before_after_comparison(
        output.parent / "stage3_5",
        summary,
        feature_stats,
        output / "before_after_comparison.txt",
    )
    make_plots(feature_stats, missing_frame, output)

    print("\n" + "=" * 60)
    print("NAV-SHIELD STAGE 3.5: DATA QUALITY AUDIT")
    print("=" * 60)
    print(f"Sequences audited: {len(sequence_reports)}")
    print(f"Total rows: {total_rows}")
    print(f"Feature columns: {summary['feature_columns']}")
    print(f"Overall missing-value rate: {summary['overall_missing_rate']:.4f}%")
    print(f"Total infinite values: {total_infinite}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print(f"Constant features: {constant_features}")
    print(f"Near-constant features: {near_constant_features}")
    print(f"PASS sequences: {pass_count}")
    print(f"WARNING sequences: {warning_count}")
    print(f"FAIL sequences: {fail_count}")
    print("\nMissing-value assessment:")
    print(summary["missing_value_assessment"])
    print(f"\nData quality status: {summary['data_quality_status']}")
    print(f"Recommended action before Stage 4: {recommendation}")
    return 0 if not unreadable else 1


if __name__ == "__main__":
    raise SystemExit(main())