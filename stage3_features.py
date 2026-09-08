#!/usr/bin/env python3
"""NAV-SHIELD Stage 3: leakage-aware feature engineering for aligned S/V data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "data" / "aligned"
DEFAULT_OUTPUT = PROJECT_ROOT / "results"
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
DEFAULT_WINDOW = 10
KEY_FEATURES = (
    "accel_magnitude",
    "gyro_magnitude",
    "mag_magnitude",
    "gravity_magnitude",
    "gps_speed",
    "gps_accuracy",
)


@dataclass
class FeatureGroups:
    timestamp: str
    accelerometer: dict[str, str]
    gyroscope: dict[str, str]
    magnetometer: dict[str, str]
    gravity: dict[str, str]
    orientation: dict[str, str]
    gps: dict[str, str]
    explicit_labels: list[str]


def load_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            frame.columns = make_unique_columns([str(c).replace("\ufeff", "").strip() for c in frame.columns])
            return frame
        except (UnicodeError, pd.errors.ParserError, OSError) as exc:
            last_error = exc
    raise ValueError(f"Could not read '{path}': {last_error}")


def make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for column in columns:
        count = seen.get(column, 0)
        result.append(column if count == 0 else f"{column}_{count}")
        seen[column] = count + 1
    return result


def normalized(value: object) -> str:
    text = str(value).lower().replace("µ", "u").replace("²", "2")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def candidates(columns: Iterable[str], include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    result = []
    for column in columns:
        name = normalized(column)
        if all(token in name for token in include) and not any(token in name for token in exclude):
            result.append(column)
    return sorted(result, key=lambda column: (0 if column.upper().startswith("S_") else 1, len(column)))


def first_candidate(columns: Iterable[str], include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
    found = candidates(columns, include, exclude)
    return found[0] if found else None


def detect_groups(frame: pd.DataFrame) -> FeatureGroups:
    columns = list(frame.columns)
    timestamp = first_candidate(columns, ("s time seconds",)) or first_candidate(columns, ("v time seconds",)) or "S_time_seconds"
    accelerometer = {
        axis: first_candidate(columns, ("accelerometer", axis)) for axis in ("x", "y", "z")
    }
    gyroscope = {
        axis: first_candidate(columns, ("gyroscope", axis)) for axis in ("yaw", "pitch", "roll")
    }
    magnetometer = {
        axis: first_candidate(columns, ("magnetic field", axis)) for axis in ("x", "y", "z")
    }
    gravity = {axis: first_candidate(columns, ("gravity", axis)) for axis in ("x", "y", "z")}
    orientation = {
        axis: first_candidate(columns, ("orientation", axis)) for axis in ("yaw", "pitch", "roll")
    }
    gps = {
        "latitude": first_candidate(columns, ("gps", "latitude")) or first_candidate(columns, ("latitude",)),
        "longitude": first_candidate(columns, ("gps", "longitude")) or first_candidate(columns, ("longitude",)),
        "altitude": first_candidate(columns, ("gps", "altitude")) or first_candidate(columns, ("altitude",)) or first_candidate(columns, ("height",)),
        "speed": first_candidate(columns, ("gps", "speed")) or first_candidate(columns, ("velocity",)) or first_candidate(columns, ("speed",)),
        "accuracy": first_candidate(columns, ("gps", "accuracy")) or first_candidate(columns, ("accuracy",)),
        "satellites": first_candidate(columns, ("gps", "satellites")) or first_candidate(columns, ("satellites",)),
    }
    explicit_labels = [
        column for column in columns
        if any(token in normalized(column) for token in ("label", "ground truth", "groundtruth", "drift label"))
    ]
    return FeatureGroups(
        timestamp=timestamp,
        accelerometer={key: value for key, value in accelerometer.items() if value},
        gyroscope={key: value for key, value in gyroscope.items() if value},
        magnetometer={key: value for key, value in magnetometer.items() if value},
        gravity={key: value for key, value in gravity.items() if value},
        orientation={key: value for key, value in orientation.items() if value},
        gps={key: value for key, value in gps.items() if value},
        explicit_labels=explicit_labels,
    )


def numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[column]
    if "satellite" in normalized(column):
        extracted = values.astype("string").str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        values = extracted
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def angular_delta(values: pd.Series, period: float = 360.0) -> pd.Series:
    raw = values.diff()
    return ((raw + period / 2.0) % period) - period / 2.0


def vector_magnitude(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    values = [numeric_series(frame, column) for column in mapping.values()]
    return np.sqrt(sum(value.pow(2) for value in values))


def add_rate(frame: pd.DataFrame, source: str, dt: pd.Series, target: str) -> None:
    frame[target] = frame[source].diff() / dt.replace(0, np.nan)


def add_rolling_features(frame: pd.DataFrame, window: int) -> None:
    for source in ("accel_magnitude", "gyro_magnitude", "mag_magnitude", "gravity_magnitude", "gps_speed", "accel_change", "gyro_change", "speed_change"):
        if source not in frame:
            continue
        rolling = frame[source].rolling(window=window, min_periods=1)
        frame[f"{source}_rolling_mean"] = rolling.mean()
        frame[f"{source}_rolling_std"] = rolling.std(ddof=0)
        frame[f"{source}_rolling_min"] = rolling.min()
        frame[f"{source}_rolling_max"] = rolling.max()
        frame[f"{source}_rolling_range"] = frame[f"{source}_rolling_max"] - frame[f"{source}_rolling_min"]


def detect_gap_threshold(dt: pd.Series) -> float:
    valid = dt[(dt > 0) & dt.notna()]
    if valid.empty:
        return float("nan")
    median = float(valid.median())
    deviation = float((valid - median).abs().median())
    return max(median * 3.0, median + 6.0 * deviation, median + 0.05)


def clean_and_engineer(raw: pd.DataFrame, sequence: str, window: int) -> tuple[pd.DataFrame, dict]:
    if window < 1:
        raise ValueError("window must be at least 1")
    frame = raw.copy()
    groups = detect_groups(frame)
    if groups.timestamp not in frame:
        raise ValueError("No synchronized timestamp column was found")

    timestamp = numeric_series(frame, groups.timestamp)
    fallback = numeric_series(frame, "V_time_seconds")
    timestamp = timestamp.where(timestamp.notna(), fallback)
    frame["timestamp"] = timestamp
    frame = frame.sort_values("timestamp", kind="mergesort", na_position="last").reset_index(drop=True)
    frame["sequence_id"] = sequence
    frame["elapsed_time"] = frame["timestamp"] - frame["timestamp"].dropna().iloc[0] if frame["timestamp"].notna().any() else np.nan
    frame["delta_time"] = frame["elapsed_time"].diff()
    frame["delta_time"] = frame["delta_time"].clip(lower=0)
    threshold = detect_gap_threshold(frame["delta_time"])
    frame["time_gap_flag"] = frame["delta_time"].gt(threshold).fillna(False) if np.isfinite(threshold) else False

    source_maps = {
        "accel": (groups.accelerometer, "accel"),
        "gyro": (groups.gyroscope, "gyro"),
        "mag": (groups.magnetometer, "mag"),
        "gravity": (groups.gravity, "gravity"),
    }
    for prefix, (mapping, output_prefix) in source_maps.items():
        for axis, column in mapping.items():
            frame[f"{output_prefix}_{axis}"] = numeric_series(frame, column)
        if len(mapping) == 3:
            frame[f"{output_prefix}_magnitude"] = vector_magnitude(frame, mapping)

    if len(groups.gyroscope) != 3:
        missing_axes = sorted(set(("yaw", "pitch", "roll")) - set(groups.gyroscope))
        raise ValueError(
            "Required gyroscope source columns were not detected; "
            f"missing axes: {missing_axes}. Detected: {groups.gyroscope}"
        )

    for axis, column in groups.orientation.items():
        frame[f"orientation_{axis}"] = numeric_series(frame, column)
    for name, column in groups.gps.items():
        frame[f"gps_{name}"] = numeric_series(frame, column)

    if "gps_latitude" in frame and "gps_longitude" in frame:
        frame["gps_valid"] = (
            frame["gps_latitude"].between(-90, 90)
            & frame["gps_longitude"].between(-180, 180)
            & frame["gps_latitude"].notna()
            & frame["gps_longitude"].notna()
        )
    else:
        frame["gps_valid"] = False
    frame["gps_accuracy_valid"] = frame["gps_accuracy"].ge(0) if "gps_accuracy" in frame else False
    frame["satellite_count_valid"] = frame["gps_satellites"].ge(0) if "gps_satellites" in frame else False

    if "accel_magnitude" in frame:
        frame["accel_change"] = frame["accel_magnitude"].diff().abs()
        add_rate(frame, "accel_magnitude", frame["delta_time"], "accel_change_rate")
    if "gyro_magnitude" in frame:
        frame["gyro_change"] = frame["gyro_magnitude"].diff().abs()
        add_rate(frame, "gyro_magnitude", frame["delta_time"], "gyro_change_rate")
    if "mag_magnitude" in frame:
        frame["magnetic_field_change"] = frame["mag_magnitude"].diff().abs()
        add_rate(frame, "mag_magnitude", frame["delta_time"], "magnetic_field_change_rate")
    if "gps_speed" in frame:
        frame["speed_change"] = frame["gps_speed"].diff().abs()
        add_rate(frame, "gps_speed", frame["delta_time"], "speed_change_rate")
    for axis in ("yaw", "pitch", "roll"):
        column = f"orientation_{axis}"
        if column in frame:
            frame[f"{axis}_change"] = angular_delta(frame[column])
            add_rate(frame, f"{axis}_change", frame["delta_time"], f"{axis}_change_rate")

    sensor_columns = [column for column in ("accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll", "mag_x", "mag_y", "mag_z", "gravity_x", "gravity_y", "gravity_z") if column in frame]
    frame["missing_sensor_flag"] = frame[sensor_columns].isna().any(axis=1) if sensor_columns else True
    quality_columns = [column for column in ("gps_valid", "gps_accuracy_valid", "satellite_count_valid") if column in frame]
    quality = frame[quality_columns].astype(float)
    quality["sensor_complete"] = (~frame["missing_sensor_flag"]).astype(float)
    frame["sensor_quality_score"] = quality.mean(axis=1)
    add_rolling_features(frame, window)

    gyro_derived_features = [
        "gyro_magnitude", "gyro_change", "gyro_change_rate",
        "gyro_magnitude_rolling_mean", "gyro_magnitude_rolling_std",
        "gyro_magnitude_rolling_min", "gyro_magnitude_rolling_max",
        "gyro_magnitude_rolling_range", "gyro_change_rolling_mean",
        "gyro_change_rolling_std", "gyro_change_rolling_min",
        "gyro_change_rolling_max", "gyro_change_rolling_range",
    ]
    gyro_missing_percentages = {
        feature: float(frame[feature].isna().mean() * 100.0)
        for feature in gyro_derived_features if feature in frame
    }
    broken_gyro = [feature for feature, percentage in gyro_missing_percentages.items() if percentage >= 100.0]
    if broken_gyro:
        raise ValueError(
            "Gyro-derived feature(s) are 100% missing despite detected gyro sources: "
            f"{broken_gyro}; detected columns: {groups.gyroscope}"
        )

    numeric_features = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_features] = frame[numeric_features].replace([np.inf, -np.inf], np.nan)
    row_count = len(frame)
    missing_important = {
        column: {"count": int(frame[column].isna().sum()), "percentage": float(frame[column].isna().mean() * 100.0)}
        for column in KEY_FEATURES if column in frame
    }
    report = {
        "sequence_id": sequence,
        "input_rows": int(len(raw)),
        "output_rows": int(row_count),
        "rows_removed": int(len(raw) - row_count),
        "retention_percentage": float(row_count / len(raw) * 100.0) if len(raw) else 0.0,
        "feature_count": int(len(frame.columns)),
        "missing_value_percentage": float(frame.isna().mean().mean() * 100.0),
        "missing_important_features": missing_important,
        "mean_dt": float(frame["delta_time"].mean()) if frame["delta_time"].notna().any() else None,
        "median_dt": float(frame["delta_time"].median()) if frame["delta_time"].notna().any() else None,
        "std_dt": float(frame["delta_time"].std()) if frame["delta_time"].notna().any() else None,
        "min_dt": float(frame["delta_time"].min()) if frame["delta_time"].notna().any() else None,
        "max_dt": float(frame["delta_time"].max()) if frame["delta_time"].notna().any() else None,
        "time_gap_count": int(frame["time_gap_flag"].sum()),
        "duplicate_timestamp_count": int(frame["timestamp"].duplicated().sum()),
        "gap_threshold_seconds": threshold if np.isfinite(threshold) else None,
        "gps_valid_percentage": float(frame["gps_valid"].mean() * 100.0),
        "sensor_quality_mean": float(frame["sensor_quality_score"].mean()),
        "sensor_quality_min": float(frame["sensor_quality_score"].min()),
        "sensor_quality_max": float(frame["sensor_quality_score"].max()),
        "detected_groups": {
            "timestamp": groups.timestamp,
            "accelerometer": groups.accelerometer,
            "gyroscope": groups.gyroscope,
            "magnetometer": groups.magnetometer,
            "gravity": groups.gravity,
            "orientation": groups.orientation,
            "gps": groups.gps,
            "explicit_labels": groups.explicit_labels,
        },
        "detected_gyro_columns": groups.gyroscope,
        "gyro_derived_feature_count": len(gyro_derived_features),
        "gyro_derived_missing_percentages": gyro_missing_percentages,
        "label_statement": "Explicit labels were preserved." if groups.explicit_labels else "No explicit drift ground-truth label was found.",
        "rolling_window": window,
        "rolling_direction": "trailing; current row and prior rows only",
    }
    return frame, report


def validate_features(frame: pd.DataFrame, sequence: str, window: int) -> list[str]:
    errors = []
    if len(frame) and frame["sequence_id"].nunique(dropna=False) != 1:
        errors.append("sequence identity is not constant")
    if frame["sequence_id"].iloc[0] != sequence:
        errors.append("sequence identity does not match filename")
    timestamp = frame["timestamp"].dropna()
    if not timestamp.is_monotonic_increasing:
        errors.append("timestamps are not chronological")
    if (frame.select_dtypes(include=[np.number]) == np.inf).any().any() or (frame.select_dtypes(include=[np.number]) == -np.inf).any().any():
        errors.append("infinite numeric feature found")
    expected = ["accel_magnitude_rolling_mean", "gyro_magnitude_rolling_mean"]
    missing = [column for column in expected if column not in frame]
    if missing:
        errors.append(f"missing rolling features: {missing}")
    for column in frame.columns:
        if "rolling" in column and frame[column].dtype.kind not in "fc":
            errors.append(f"rolling feature is not numeric: {column}")
    if window < 1:
        errors.append("invalid rolling window")
    return errors


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


def write_sequence_report(report: dict, path: Path) -> None:
    lines = [
        "NAV-SHIELD STAGE 3 FEATURE REPORT", "=" * 56,
        f"Sequence: {report['sequence_id']}", f"Input rows: {report['input_rows']}",
        f"Output rows: {report['output_rows']}", f"Rows removed: {report['rows_removed']}",
        f"Retention: {report['retention_percentage']:.2f}%", f"Feature count: {report['feature_count']}",
        f"Missing-value rate: {report['missing_value_percentage']:.4f}%", f"Mean dt: {report['mean_dt']}",
        f"Median dt: {report['median_dt']}", f"Min dt: {report['min_dt']}", f"Max dt: {report['max_dt']}",
        f"Time gaps: {report['time_gap_count']}", f"GPS valid: {report['gps_valid_percentage']:.2f}%",
        f"Duplicate timestamps: {report['duplicate_timestamp_count']}",
        f"Sensor quality mean/min/max: {report['sensor_quality_mean']:.3f} / {report['sensor_quality_min']:.3f} / {report['sensor_quality_max']:.3f}",
        report["label_statement"], "", "Detected groups:", json.dumps(report["detected_groups"], indent=2),
        "", "Rolling features use a trailing window including the current row only.",
        "No global normalization or invented drift labels were applied.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_diagnostic_plot(frame: pd.DataFrame, sequence: str, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    plots = (
        ("accel_magnitude", "Accelerometer magnitude"),
        ("gyro_magnitude", "Gyroscope magnitude"),
        ("gps_speed", "GPS speed"),
        ("gps_accuracy", "GPS accuracy"),
        ("accel_magnitude_rolling_mean", "Rolling acceleration mean"),
        ("gyro_magnitude_rolling_std", "Rolling gyro standard deviation"),
    )
    for axis, (column, title) in zip(axes.flat, plots):
        if column in frame:
            axis.plot(frame["timestamp"], frame[column], linewidth=0.6)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Elapsed time (s)")
    axes[-1, 1].set_xlabel("Elapsed time (s)")
    fig.suptitle(f"{sequence}: Stage 3 feature diagnostics")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Stage 2 input directory does not exist: {input_path}")
    return sorted(input_path.rglob("*_aligned.csv"), key=lambda path: path.name.lower())


def sequence_from_path(path: Path) -> str:
    return path.name[: -len("_aligned.csv")].lower()


def process_file(path: Path, output_root: Path, window: int) -> tuple[dict, pd.DataFrame | None]:
    sequence = sequence_from_path(path)
    try:
        raw = load_csv(path)
        features, report = clean_and_engineer(raw, sequence, window)
        errors = validate_features(features, sequence, window)
        if errors:
            raise ValueError("; ".join(errors))
        output_dir = output_root / "features" / "categorised"
        report_dir = output_root / "feature_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{sequence}_features.csv"
        features.to_csv(output_path, index=False)
        report["status"] = "PASS"
        report["input_file"] = str(path)
        report["output_file"] = str(output_path)
        report["error"] = None
        (report_dir / f"{sequence}_feature_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
        write_sequence_report(report, report_dir / f"{sequence}_feature_report.txt")
        return report, features
    except Exception as exc:  # batch processing continues for one bad sequence
        return {"sequence_id": sequence, "status": "FAIL", "error": str(exc), "input_rows": 0, "output_rows": 0}, None


def write_summary(reports: list[dict], output_root: Path, selected_plots: list[tuple[str, pd.DataFrame]]) -> None:
    report_dir = output_root / "feature_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_columns = [
        "sequence_id", "input_rows", "output_rows", "rows_removed", "retention_percentage", "feature_count",
        "missing_value_percentage", "mean_dt", "median_dt", "min_dt", "max_dt", "time_gap_count",
        "duplicate_timestamp_count", "gps_valid_percentage", "sensor_quality_mean", "sensor_quality_min", "sensor_quality_max", "status", "error",
    ]
    summary = pd.DataFrame(reports).reindex(columns=summary_columns)
    summary.to_csv(output_root / "feature_summary.csv", index=False)
    (output_root / "feature_summary.json").write_text(json.dumps(json_safe(reports), indent=2), encoding="utf-8")
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for sequence, frame in selected_plots:
        make_diagnostic_plot(frame, sequence, plots_dir / f"{sequence}_feature_diagnostics.png")
    (report_dir / "plot_selection.txt").write_text(
        "Representative diagnostic sequences: " + ", ".join(sequence for sequence, _ in selected_plots) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean, causal Stage 3 features from Stage 2 aligned CSV files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Stage 2 aligned CSV directory or one CSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Stage 3 output root directory.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Trailing rolling window size in samples (default: 10).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.window < 1:
        print("ERROR: --window must be at least 1", file=sys.stderr)
        return 2
    try:
        inputs = discover_inputs(args.input.expanduser().resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    sequences = [sequence_from_path(path) for path in inputs]
    print("=" * 60)
    print("NAV-SHIELD STAGE 3: FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Synchronized files found: {len(inputs)}")
    print(f"Sequence IDs detected: {', '.join(sequences)}")
    print(f"Total files: {len(inputs)}")
    if len(inputs) != 72:
        print(f"Missing or unexpected synchronized file count: expected 72, found {len(inputs)}")
    else:
        print("Missing synchronized files: none detected")
    if not inputs:
        print("Missing or unreadable files: no *_aligned.csv files found", file=sys.stderr)
        return 1

    reports = []
    frames = []
    for path in inputs:
        report, frame = process_file(path, args.output, args.window)
        reports.append(report)
        if frame is not None:
            frames.append((report["sequence_id"], frame))
        print(f"[{report['status']}] {report['sequence_id']}: {report.get('error') or 'features generated'}")
    successful = [report for report in reports if report.get("status") == "PASS"]
    selected = [frames[0], frames[len(frames) // 2], frames[-1]] if frames else []
    selected = list(dict((sequence, frame) for sequence, frame in selected).items())
    write_summary(reports, args.output, selected)
    total_input = sum(int(report.get("input_rows", 0)) for report in reports)
    total_output = sum(int(report.get("output_rows", 0)) for report in reports)
    average_retention = float(np.mean([report["retention_percentage"] for report in successful])) if successful else 0.0
    feature_counts = sorted({int(report["feature_count"]) for report in successful})
    missing_rate = float(np.average([report["missing_value_percentage"] for report in successful], weights=[report["output_rows"] for report in successful])) if successful and total_output else 0.0
    print("\nFEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Sequences processed: {len(reports)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(reports) - len(successful)}")
    print(f"Total input samples: {total_input}")
    print(f"Total output samples: {total_output}")
    print(f"Average retention: {average_retention:.2f}%")
    print(f"Feature columns: {feature_counts[0] if len(feature_counts) == 1 else feature_counts}")
    print(f"Missing-value rate: {missing_rate:.4f}%")
    print("=" * 60)
    return 0 if len(successful) == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())