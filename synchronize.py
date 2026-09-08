#!/usr/bin/env python3
"""NAV-SHIELD Stage 2: robust S/V elapsed-time synchronization."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "results" / "data_reports"
ALIGNED_DIR = PROJECT_ROOT / "results" / "data" / "aligned"
DEFAULT_INVENTORY = REPORTS_DIR / "dataset_inventory.csv"
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def clean_column_name(value: object) -> str:
    """Normalize whitespace and common UTF-8-as-Latin-1 mojibake."""
    name = str(value).replace("\ufeff", "").strip()
    return name.replace("Â°", "°").replace("Î¼", "µ").replace("Âµ", "µ")


def unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for name in columns:
        count = counts.get(name, 0)
        result.append(name if count == 0 else f"{name}_{count}")
        counts[name] = count + 1
    return result


def load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            frame.columns = unique_columns([clean_column_name(c) for c in frame.columns])
            return frame, encoding
        except (UnicodeError, pd.errors.ParserError, OSError) as exc:
            last_error = exc
    raise ValueError(f"Could not load {path}: {last_error}")


def normalized_id(value: object) -> str:
    stem = Path(str(value)).stem.strip().lower()
    stem = re.sub(r"^[sv]-", "", stem, flags=re.IGNORECASE)
    return stem


def inventory_sequences(inventory_path: Path, selected: str | None) -> list[dict]:
    if inventory_path.suffix.lower() == ".json":
        rows = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    else:
        rows = pd.read_csv(inventory_path).to_dict(orient="records")
    rows = [row for row in rows if str(row.get("status", "")).upper() == "PASS"]
    if selected:
        rows = [row for row in rows if normalized_id(row.get("sequence_id")) == normalized_id(selected)]
    return sorted(rows, key=lambda row: normalized_id(row.get("sequence_id", "")))


def discover_file(dataset: Path, sequence: str, kind: str, inventory_path: str | None) -> Path:
    expected = normalized_id(sequence)
    if inventory_path:
        candidate = Path(inventory_path)
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = []
    for path in dataset.rglob("*.csv"):
        if path.resolve() == DEFAULT_INVENTORY.resolve():
            continue
        if path.stem.lower().startswith(f"{kind.lower()}-") and normalized_id(path.name) == expected:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No {kind} file found for sequence '{sequence}'")
    return sorted(matches, key=lambda path: str(path).lower())[0]


def find_s_time_column(frame: pd.DataFrame) -> str:
    candidates = [col for col in frame.columns if "time since start" in col.lower() and "ms" in col.lower()]
    if not candidates:
        raise ValueError("S TIME SINCE START (ms) column was not found")
    return candidates[0]


def find_v_time_column(frame: pd.DataFrame) -> str:
    candidates = [
        col for col in frame.columns
        if "time" in col.lower() and "second" in col.lower()
    ]
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return column
    raise ValueError("V elapsed-time column containing time and seconds was not found")


def numeric_time(frame: pd.DataFrame, column: str, divisor: float = 1.0) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce") / divisor
    if values.notna().sum() == 0:
        raise ValueError(f"Time column '{column}' contains no numeric values")
    first = values.dropna().iloc[0]
    return values - first


def safe_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def synchronize_frame(s_frame: pd.DataFrame, v_frame: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, dict]:
    s_col = find_s_time_column(s_frame)
    v_col = find_v_time_column(v_frame)
    s_time = numeric_time(s_frame, s_col, 1000.0)
    v_time = numeric_time(v_frame, v_col)
    s_valid = s_time.notna().to_numpy()
    v_valid = v_time.notna().to_numpy()
    v_indices = np.flatnonzero(v_valid)
    v_values = v_time.to_numpy(dtype=float)[v_valid]
    order = np.argsort(v_values, kind="mergesort")
    sorted_values = v_values[order]

    nearest_indices = np.full(len(s_frame), -1, dtype=int)
    differences = np.full(len(s_frame), np.nan, dtype=float)
    for s_index, value in enumerate(s_time.to_numpy(dtype=float)):
        if not s_valid[s_index] or not len(sorted_values):
            continue
        position = int(np.searchsorted(sorted_values, value, side="left"))
        options = []
        if position < len(sorted_values):
            options.append(position)
        if position > 0:
            options.append(position - 1)
        best = min(options, key=lambda item: abs(sorted_values[item] - value))
        difference = abs(sorted_values[best] - value)
        if difference <= tolerance:
            nearest_indices[s_index] = int(v_indices[order[best]])
            differences[s_index] = difference

    s_out = s_frame.copy()
    v_out = v_frame.copy()
    s_out.columns = [f"S_{column}" for column in s_out.columns]
    v_out.columns = [f"V_{column}" for column in v_out.columns]
    aligned = pd.DataFrame({
        "sequence_id": "",
        "S_time_seconds": s_time,
        "V_time_seconds": [v_time.iloc[index] if index >= 0 else np.nan for index in nearest_indices],
        "time_difference_seconds": differences,
        "synchronization_valid": nearest_indices >= 0,
    })
    aligned["sequence_id"] = ""
    for column in s_out.columns:
        aligned[column] = s_out[column].to_numpy()
    safe_indices = nearest_indices.copy()
    safe_indices[safe_indices < 0] = 0
    matched_v = v_out.iloc[safe_indices].reset_index(drop=True)
    matched_mask = nearest_indices >= 0
    for column in v_out.columns:
        aligned[column] = matched_v[column].where(matched_mask, np.nan).to_numpy()

    valid_diffs = differences[np.isfinite(differences)]
    s_numeric = s_time.dropna()
    v_numeric = v_time.dropna()
    overlap = max(0.0, min(float(s_numeric.max()), float(v_numeric.max())) - max(float(s_numeric.min()), float(v_numeric.min())))
    matched = int((nearest_indices >= 0).sum())
    report = {
        "S_rows": int(len(s_frame)),
        "V_rows": int(len(v_frame)),
        "matched_rows": matched,
        "unmatched_rows": int(len(s_frame) - matched),
        "match_percentage": matched / len(s_frame) * 100.0 if len(s_frame) else 0.0,
        "mean_time_difference": safe_number(np.mean(valid_diffs) if len(valid_diffs) else None),
        "median_time_difference": safe_number(np.median(valid_diffs) if len(valid_diffs) else None),
        "max_time_difference": safe_number(np.max(valid_diffs) if len(valid_diffs) else None),
        "min_time_difference": safe_number(np.min(valid_diffs) if len(valid_diffs) else None),
        "duplicate_timestamps": int(s_time.duplicated().sum()),
        "S_time_range": [safe_number(s_numeric.min()), safe_number(s_numeric.max())],
        "V_time_range": [safe_number(v_numeric.min()), safe_number(v_numeric.max())],
        "overlap_duration": overlap,
        "S_time_column": s_col,
        "V_time_column": v_col,
    }
    return aligned, report


def validate_output(path: Path, frame: pd.DataFrame) -> None:
    required = {"sequence_id", "S_time_seconds", "V_time_seconds", "time_difference_seconds", "synchronization_valid"}
    if not path.exists() or path.stat().st_size == 0 or not required.issubset(frame.columns):
        raise ValueError(f"Invalid aligned output: {path}")
    for column in ("S_time_seconds", "V_time_seconds", "time_difference_seconds"):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Aligned column is not numeric: {column}")
    if (frame["time_difference_seconds"].dropna() < 0).any():
        raise ValueError("Negative time difference found")


def make_plot(aligned: pd.DataFrame, sequence: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(aligned["S_time_seconds"], aligned["V_time_seconds"], ".", markersize=2)
    axes[0].set_ylabel("V time (s)")
    axes[0].set_title(f"{sequence}: S timestamps vs matched V timestamps")
    axes[1].plot(aligned["S_time_seconds"], aligned["time_difference_seconds"], ".", markersize=2)
    axes[1].set_xlabel("S elapsed time (s)")
    axes[1].set_ylabel("Absolute difference (s)")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def process_sequence(row: dict, dataset: Path, tolerance: float) -> tuple[dict, pd.DataFrame | None]:
    sequence = str(row["sequence_id"]).strip().lower()
    try:
        s_path = discover_file(dataset, sequence, "S", row.get("s_file"))
        v_path = discover_file(dataset, sequence, "V", row.get("v_file"))
        s_frame, s_encoding = load_csv(s_path)
        v_frame, v_encoding = load_csv(v_path)
        aligned, metrics = synchronize_frame(s_frame, v_frame, tolerance)
        aligned["sequence_id"] = sequence
        output = ALIGNED_DIR / f"{sequence}_aligned.csv"
        aligned.to_csv(output, index=False)
        validate_output(output, aligned)
        make_plot(aligned, sequence, REPORTS_DIR / "sync_plots" / f"{sequence}_sync.png")
        metrics.update({"sequence_id": sequence, "status": "PASS", "error": None, "aligned_file": str(output), "s_encoding": s_encoding, "v_encoding": v_encoding})
        return metrics, aligned
    except Exception as exc:  # individual failures must not stop the batch
        return {"sequence_id": sequence, "status": "FAIL", "error": str(exc), "aligned_file": None}, None


def write_reports(results: list[dict], tolerance: float) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_csv = REPORTS_DIR / "synchronization_report.csv"
    pd.DataFrame(results).to_csv(report_csv, index=False)
    (REPORTS_DIR / "synchronization_report.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    index_columns = ["sequence_id", "aligned_file", "S_rows", "V_rows", "matched_rows", "unmatched_rows", "match_percentage", "mean_time_difference", "median_time_difference", "max_time_difference", "status"]
    pd.DataFrame(results).reindex(columns=index_columns).to_csv(REPORTS_DIR / "aligned_dataset_index.csv", index=False)
    successful = [result for result in results if result.get("status") == "PASS"]
    total_s = sum(result.get("S_rows", 0) for result in successful)
    total_v = sum(result.get("V_rows", 0) for result in successful)
    total_matched = sum(result.get("matched_rows", 0) for result in successful)
    lines = [
        "NAV-SHIELD STAGE 2 SYNCHRONIZATION SUMMARY", "=" * 56,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}", f"Tolerance: {tolerance} seconds",
        f"Sequences processed: {len(results)}", f"Successful: {len(successful)}", f"Failed: {len(results) - len(successful)}",
        f"Total S samples: {total_s}", f"Total V samples: {total_v}", f"Total matched samples: {total_matched}",
        f"Overall match percentage: {total_matched / total_s * 100.0 if total_s else 0.0:.2f}%", "",
    ]
    failures = [result for result in results if result.get("status") != "PASS"]
    if failures:
        lines.append("Failures:")
        lines.extend(f"- {item['sequence_id']}: {item.get('error')}" for item in failures)
    (REPORTS_DIR / "synchronization_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize all valid NAV-SHIELD S/V pairs by elapsed time.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--sequence", help="Process one inventory sequence after the vta11 gate.")
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    return parser.parse_args()


def print_sequence(result: dict) -> None:
    print(f"Sequence: {result['sequence_id']}")
    print(f"S rows: {result.get('S_rows', 0)}")
    print(f"V rows: {result.get('V_rows', 0)}")
    print(f"Matched: {result.get('matched_rows', 0)}")
    print(f"Unmatched: {result.get('unmatched_rows', 0)}")
    print(f"Match percentage: {result.get('match_percentage', 0.0):.2f}%")
    print(f"Mean time difference: {result.get('mean_time_difference', 0.0) or 0.0:.4f} s")
    print(f"Maximum time difference: {result.get('max_time_difference', 0.0) or 0.0:.4f} s")
    print(f"Status: {result.get('status')}")


def main() -> int:
    args = parse_args()
    if args.tolerance < 0:
        print("ERROR: tolerance must be non-negative", file=sys.stderr)
        return 2
    inventory = Path(args.inventory)
    rows = inventory_sequences(inventory, args.sequence)
    if not rows:
        print(f"ERROR: no PASS sequences found in {inventory}", file=sys.stderr)
        return 1
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_dir():
        print(f"ERROR: dataset directory does not exist: {dataset}", file=sys.stderr)
        return 1
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "sync_plots").mkdir(parents=True, exist_ok=True)
    all_rows = inventory_sequences(inventory, None)
    gate_row = next((row for row in all_rows if normalized_id(row.get("sequence_id")) == "vta11"), all_rows[0])
    print("=" * 64)
    print("NAV-SHIELD STAGE 2: S/V SYNCHRONIZATION")
    print("=" * 64)
    gate_result, _ = process_sequence(gate_row, dataset, args.tolerance)
    print_sequence(gate_result)
    print("=" * 64)
    if gate_result.get("status") != "PASS" or gate_result.get("matched_rows", 0) == 0:
        print("Stage 2 gate failed; full processing was not started.", file=sys.stderr)
        return 1
    results = []
    for row in rows if args.sequence else all_rows:
        if normalized_id(row.get("sequence_id")) == normalized_id(gate_row.get("sequence_id")):
            results.append(gate_result)
        else:
            result, _ = process_sequence(row, dataset, args.tolerance)
            results.append(result)
    write_reports(results, args.tolerance)
    successful = [result for result in results if result.get("status") == "PASS"]
    total_s = sum(result.get("S_rows", 0) for result in successful)
    total_v = sum(result.get("V_rows", 0) for result in successful)
    total_matched = sum(result.get("matched_rows", 0) for result in successful)
    print("STAGE 2 SUMMARY")
    print("=" * 64)
    print(f"Sequences processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Total S samples: {total_s}")
    print(f"Total V samples: {total_v}")
    print(f"Total matched samples: {total_matched}")
    print(f"Overall match percentage: {total_matched / total_s * 100.0 if total_s else 0.0:.2f}%")
    print("=" * 64)
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())