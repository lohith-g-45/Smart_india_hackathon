"""
NAV-SHIELD STAGE 3.6
Uncategorised IOVNB Dataset Inspection

READ-ONLY INSPECTION ONLY.

This script:
- Finds all CSV files recursively
- Identifies S/V files
- Extracts sequence IDs
- Handles different filename conventions
- Reads CSV headers and basic statistics
- Detects timestamp columns
- Detects possible label/category columns
- Matches S/V pairs
- Generates inspection reports

It DOES NOT:
- modify raw data
- synchronize data
- engineer features
- impute values
- normalize data
- remove duplicates
- remove columns
- split data
- train models
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

S_KEYWORDS = {
    "s-dataset",
    "s_dataset",
    "s dataset",
}

V_KEYWORDS = {
    "v-dataset",
    "v_dataset",
    "v dataset",
}

LABEL_KEYWORDS = [
    "label",
    "class",
    "category",
    "driver",
    "event",
    "scenario",
    "type",
    "target",
    "ground truth",
    "ground_truth",
    "groundtruth",
    "anomaly",
    "attack",
    "drift",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_text(value):
    """Normalize text for comparisons."""
    if value is None:
        return ""

    value = str(value).strip().lower()

    value = value.replace("–", "-")
    value = value.replace("—", "-")
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)

    return value


def detect_file_type(path):
    """
    Determine whether a file belongs to S or V dataset.

    We inspect both filename and parent folders.
    """

    parts = [normalize_text(p) for p in path.parts]

    filename = normalize_text(path.name)

    # Strong folder-based detection
    for part in parts:
        if part in S_KEYWORDS:
            return "S"

        if part in V_KEYWORDS:
            return "V"

    # Filename-based detection
    if re.search(r"(^|[-_ ])s[-_ ]", filename):
        return "S"

    if filename.startswith("s-") or filename.startswith("s_"):
        return "S"

    if filename.startswith("v-") or filename.startswith("v_"):
        return "V"

    # Direct S/V prefix
    if filename.startswith("s"):
        return "S"

    if filename.startswith("v"):
        return "V"

    return "UNKNOWN"


def extract_sequence_id(path):
    """
    Extract sequence ID from filename/path.

    Examples:
        S-Vta11.csv       -> vta11
        V-Vta11.csv       -> vta11
        S-Vfa01.csv       -> vfa01
        V-Vfa01.csv       -> vfa01

    Also supports descriptive directory structures.
    """

    filename = path.stem

    # Remove leading S-/V-/S_/V_
    cleaned = re.sub(
        r"^[sSvV][-_ ]+",
        "",
        filename
    )

    cleaned = cleaned.strip()

    # Common sequence pattern:
    # Vfa01, Vta11, Vtb1, Vw14a, etc.
    match = re.search(
        r"(v(?:fa|ta|tb|w)\d+[a-z]?)",
        cleaned,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    # More general V + alphanumeric identifier
    match = re.search(
        r"\b(v[a-z]*\d+[a-z]?)\b",
        cleaned,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    # Special sequences such as M, S1, S2, Y1
    match = re.search(
        r"\b([msy]\d*[a-z]*)\b",
        cleaned,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    # If filename itself is meaningful, use it
    if cleaned:
        return normalize_text(cleaned).replace(" ", "_")

    return "unknown"


def detect_encoding(path):
    """
    Try common encodings.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]

    for encoding in encodings:
        try:
            with open(
                path,
                "r",
                encoding=encoding,
                errors="strict",
                newline=""
            ) as f:
                f.read(4096)

            return encoding

        except Exception:
            continue

    return "unknown"


def detect_timestamp_columns(columns):
    """
    Detect columns related to time/date.
    """

    result = []

    for col in columns:
        normalized = normalize_text(col)

        if (
            "time" in normalized
            or "date" in normalized
            or "timestamp" in normalized
        ):
            result.append(col)

    return result


def detect_label_columns(columns):
    """
    Detect possible label/category/ground-truth columns.
    """

    result = []

    for col in columns:

        normalized = normalize_text(col)

        for keyword in LABEL_KEYWORDS:

            if keyword in normalized:
                result.append(col)
                break

    return result


def safe_read_csv(path, encoding):
    """
    Read CSV using pandas.
    """

    try:
        df = pd.read_csv(
            path,
            encoding=encoding,
            low_memory=False
        )

        return df, None

    except Exception as exc:

        return None, str(exc)


def calculate_missing_percentage(df):
    """
    Calculate overall missing percentage.
    """

    if df.empty:
        return 0.0

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0.0

    missing_cells = int(df.isna().sum().sum())

    return (missing_cells / total_cells) * 100.0


def find_time_range(df, timestamp_columns):
    """
    Try to determine timestamp range.

    Does NOT modify the dataframe.
    """

    for column in timestamp_columns:

        series = df[column]

        # Numeric time columns
        if pd.api.types.is_numeric_dtype(series):

            numeric = pd.to_numeric(
                series,
                errors="coerce"
            ).dropna()

            if len(numeric) > 0:

                return {
                    "column": column,
                    "type": "numeric",
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                }

        # Date/time columns
        try:

            parsed = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed"
            )

            valid = parsed.dropna()

            if len(valid) > 0:

                return {
                    "column": column,
                    "type": "datetime",
                    "min": str(valid.min()),
                    "max": str(valid.max()),
                }

        except Exception:
            pass

    return None


# ============================================================
# FILE INSPECTION
# ============================================================

def inspect_file(path):
    """
    Inspect one CSV file.
    """

    file_type = detect_file_type(path)

    sequence_id = extract_sequence_id(path)

    encoding = detect_encoding(path)

    result = {
        "file": str(path),
        "filename": path.name,
        "type": file_type,
        "sequence_id": sequence_id,
        "encoding": encoding,
        "rows": None,
        "columns_count": None,
        "columns": [],
        "timestamp_columns": [],
        "time_since_start_columns": [],
        "label_columns": [],
        "missing_cells": None,
        "missing_percentage": None,
        "time_range": None,
        "error": None,
    }

    if encoding == "unknown":

        result["error"] = "Could not detect encoding"

        return result

    df, error = safe_read_csv(path, encoding)

    if error:

        result["error"] = error

        return result

    result["rows"] = int(df.shape[0])

    result["columns_count"] = int(df.shape[1])

    result["columns"] = [
        str(c)
        for c in df.columns
    ]

    result["timestamp_columns"] = detect_timestamp_columns(
        df.columns
    )

    result["time_since_start_columns"] = [
        str(c)
        for c in df.columns
        if (
            "time since start" in normalize_text(c)
            or "time_since_start" in normalize_text(c)
        )
    ]

    result["label_columns"] = detect_label_columns(
        df.columns
    )

    result["missing_cells"] = int(
        df.isna().sum().sum()
    )

    result["missing_percentage"] = round(
        calculate_missing_percentage(df),
        6
    )

    result["time_range"] = find_time_range(
        df,
        result["timestamp_columns"]
    )

    return result


# ============================================================
# PAIRING
# ============================================================

def build_pairing_report(file_records):

    grouped = defaultdict(
        lambda: {
            "S": [],
            "V": []
        }
    )

    for record in file_records:

        seq = record["sequence_id"]

        if record["type"] in ("S", "V"):

            grouped[seq][record["type"]].append(record)

    pairing = []

    for sequence_id in sorted(grouped.keys()):

        s_files = grouped[sequence_id]["S"]

        v_files = grouped[sequence_id]["V"]

        if len(s_files) == 1 and len(v_files) == 1:

            status = "COMPLETE"

        elif len(s_files) == 0 and len(v_files) > 0:

            status = "V_ONLY"

        elif len(v_files) == 0 and len(s_files) > 0:

            status = "S_ONLY"

        elif len(s_files) > 1 or len(v_files) > 1:

            status = "AMBIGUOUS"

        else:

            status = "UNKNOWN"

        pairing.append({
            "sequence_id": sequence_id,
            "status": status,
            "S_count": len(s_files),
            "V_count": len(v_files),
            "S_files": [
                x["file"] for x in s_files
            ],
            "V_files": [
                x["file"] for x in v_files
            ],
        })

    return pairing


# ============================================================
# STRUCTURE ANALYSIS
# ============================================================

def analyse_structure(file_records):

    s_records = [
        r for r in file_records
        if r["type"] == "S"
        and not r["error"]
    ]

    v_records = [
        r for r in file_records
        if r["type"] == "V"
        and not r["error"]
    ]

    def common_columns(records):

        if not records:
            return []

        sets = [
            set(r["columns"])
            for r in records
        ]

        common = sets[0]

        for s in sets[1:]:
            common = common.intersection(s)

        return sorted(common)

    s_common = common_columns(s_records)

    v_common = common_columns(v_records)

    s_all = set()

    for r in s_records:
        s_all.update(r["columns"])

    v_all = set()

    for r in v_records:
        v_all.update(r["columns"])

    return {
        "S_common_columns": s_common,
        "V_common_columns": v_common,
        "S_union_columns": sorted(s_all),
        "V_union_columns": sorted(v_all),
        "S_column_variations": sorted(
            s_all - set(s_common)
        ),
        "V_column_variations": sorted(
            v_all - set(v_common)
        ),
    }


# ============================================================
# REPORT WRITERS
# ============================================================

def write_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def write_inventory_csv(path, records):

    if not records:
        return

    rows = []

    for r in records:

        rows.append({
            "file": r["file"],
            "filename": r["filename"],
            "type": r["type"],
            "sequence_id": r["sequence_id"],
            "encoding": r["encoding"],
            "rows": r["rows"],
            "columns_count": r["columns_count"],
            "timestamp_columns": " | ".join(
                r["timestamp_columns"]
            ),
            "time_since_start_columns": " | ".join(
                r["time_since_start_columns"]
            ),
            "label_columns": " | ".join(
                r["label_columns"]
            ),
            "missing_cells": r["missing_cells"],
            "missing_percentage": r["missing_percentage"],
            "time_range": json.dumps(
                r["time_range"],
                ensure_ascii=False
            ),
            "error": r["error"],
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8"
    )


def write_pairing_csv(path, pairing):

    rows = []

    for p in pairing:

        rows.append({
            "sequence_id": p["sequence_id"],
            "status": p["status"],
            "S_count": p["S_count"],
            "V_count": p["V_count"],
            "S_files": " | ".join(
                p["S_files"]
            ),
            "V_files": " | ".join(
                p["V_files"]
            ),
        })

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8"
    )


def write_summary(
    path,
    dataset_root,
    records,
    pairing,
    structure
):

    total_files = len(records)

    s_records = [
        r for r in records
        if r["type"] == "S"
    ]

    v_records = [
        r for r in records
        if r["type"] == "V"
    ]

    errors = [
        r for r in records
        if r["error"]
    ]

    complete = [
        p for p in pairing
        if p["status"] == "COMPLETE"
    ]

    s_only = [
        p for p in pairing
        if p["status"] == "S_ONLY"
    ]

    v_only = [
        p for p in pairing
        if p["status"] == "V_ONLY"
    ]

    ambiguous = [
        p for p in pairing
        if p["status"] == "AMBIGUOUS"
    ]

    label_records = [
        r for r in records
        if r["label_columns"]
    ]

    total_s_rows = sum(
        r["rows"] or 0
        for r in s_records
    )

    total_v_rows = sum(
        r["rows"] or 0
        for r in v_records
    )

    status = "PASS"

    if errors or ambiguous:
        status = "WARNING"

    lines = []

    lines.append("=" * 72)
    lines.append(
        "NAV-SHIELD STAGE 3.6: "
        "UNCATEGORISED DATASET INSPECTION"
    )
    lines.append("=" * 72)

    lines.append(
        f"Dataset root: {dataset_root}"
    )

    lines.append("")

    lines.append(
        f"Total CSV files: {total_files}"
    )

    lines.append(
        f"S files: {len(s_records)}"
    )

    lines.append(
        f"V files: {len(v_records)}"
    )

    lines.append("")

    lines.append(
        f"Complete S/V pairs: {len(complete)}"
    )

    lines.append(
        f"S-only: {len(s_only)}"
    )

    lines.append(
        f"V-only: {len(v_only)}"
    )

    lines.append(
        f"Ambiguous: {len(ambiguous)}"
    )

    lines.append(
        f"Errors: {len(errors)}"
    )

    lines.append("")

    lines.append(
        f"Total S rows: {total_s_rows}"
    )

    lines.append(
        f"Total V rows: {total_v_rows}"
    )

    lines.append("")

    lines.append(
        "FILES CONTAINING POSSIBLE LABEL/CATEGORY COLUMNS"
    )

    if label_records:

        for r in label_records:

            lines.append(
                f"- {r['sequence_id']} | "
                f"{r['type']} | "
                f"{r['label_columns']}"
            )

    else:

        lines.append(
            "None detected"
        )

    lines.append("")

    lines.append(
        "COMMON S COLUMNS"
    )

    for c in structure["S_common_columns"]:
        lines.append(f"- {c}")

    lines.append("")

    lines.append(
        "COMMON V COLUMNS"
    )

    for c in structure["V_common_columns"]:
        lines.append(f"- {c}")

    lines.append("")

    lines.append(
        "S COLUMN VARIATIONS"
    )

    for c in structure["S_column_variations"]:
        lines.append(f"- {c}")

    lines.append("")

    lines.append(
        "V COLUMN VARIATIONS"
    )

    for c in structure["V_column_variations"]:
        lines.append(f"- {c}")

    lines.append("")

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "No synchronization, feature engineering, "
        "imputation, normalization, splitting, "
        "merging, or model training was performed."
    )

    lines.append("")

    lines.append(
        f"STAGE 3.6 STATUS: {status}"
    )

    lines.append("=" * 72)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "NAV-SHIELD Stage 3.6 "
            "Uncategorised Dataset Inspection"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to Uncategorised IOVNB Dataset"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset)

    print("=" * 72)

    print(
        "NAV-SHIELD STAGE 3.6: "
        "UNCATEGORISED DATASET INSPECTION"
    )

    print("=" * 72)

    print(
        f"Dataset root: {dataset_root}"
    )

    # --------------------------------------------------------
    # Validate path
    # --------------------------------------------------------

    if not dataset_root.exists():

        print(
            f"ERROR: Dataset path does not exist:\n"
            f"{dataset_root}"
        )

        return

    if not dataset_root.is_dir():

        print(
            f"ERROR: Dataset path is not a directory:\n"
            f"{dataset_root}"
        )

        return

    # --------------------------------------------------------
    # Find CSV files
    # --------------------------------------------------------

    csv_files = sorted(
        dataset_root.rglob("*.csv")
    )

    print(
        f"CSV files discovered: {len(csv_files)}"
    )

    print()

    if not csv_files:

        print(
            "ERROR: No CSV files found."
        )

        return

    # --------------------------------------------------------
    # Inspect files
    # --------------------------------------------------------

    records = []

    for index, path in enumerate(
        csv_files,
        start=1
    ):

        print(
            f"[{index}/{len(csv_files)}] "
            f"Inspecting {path.name} ..."
        )

        record = inspect_file(path)

        records.append(record)

    print()

    # --------------------------------------------------------
    # Pair S/V
    # --------------------------------------------------------

    pairing = build_pairing_report(
        records
    )

    structure = analyse_structure(
        records
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        "results"
    ) / "data_reports" / "uncategorised_inspection"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    write_inventory_csv(
        output_dir /
        "uncategorised_file_inventory.csv",
        records
    )

    write_json(
        output_dir /
        "uncategorised_file_inventory.json",
        records
    )

    write_pairing_csv(
        output_dir /
        "uncategorised_pairing_report.csv",
        pairing
    )

    write_json(
        output_dir /
        "uncategorised_pairing_report.json",
        pairing
    )

    write_json(
        output_dir /
        "uncategorised_structure_report.json",
        structure
    )

    write_summary(
        output_dir /
        "uncategorised_inspection_summary.txt",
        dataset_root,
        records,
        pairing,
        structure
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    complete = [
        p for p in pairing
        if p["status"] == "COMPLETE"
    ]

    s_only = [
        p for p in pairing
        if p["status"] == "S_ONLY"
    ]

    v_only = [
        p for p in pairing
        if p["status"] == "V_ONLY"
    ]

    ambiguous = [
        p for p in pairing
        if p["status"] == "AMBIGUOUS"
    ]

    errors = [
        r for r in records
        if r["error"]
    ]

    label_files = [
        r for r in records
        if r["label_columns"]
    ]

    print("=" * 72)

    print(
        "UNCATEGORISED DATASET INSPECTION SUMMARY"
    )

    print("=" * 72)

    print(
        f"Total CSV files: {len(csv_files)}"
    )

    print(
        f"S files: "
        f"{sum(r['type'] == 'S' for r in records)}"
    )

    print(
        f"V files: "
        f"{sum(r['type'] == 'V' for r in records)}"
    )

    print()

    print(
        f"Complete S/V pairs: {len(complete)}"
    )

    print(
        f"S-only: {len(s_only)}"
    )

    print(
        f"V-only: {len(v_only)}"
    )

    print(
        f"Ambiguous: {len(ambiguous)}"
    )

    print(
        f"Errors: {len(errors)}"
    )

    print()

    print(
        f"Files containing possible labels: "
        f"{len(label_files)}"
    )

    if label_files:

        for record in label_files:

            print(
                f"  {record['sequence_id']} "
                f"({record['type']}): "
                f"{record['label_columns']}"
            )

    print()

    print(
        "Reports saved to:"
    )

    print(
        output_dir.resolve()
    )

    print()

    if errors or ambiguous:

        status = "WARNING"

    else:

        status = "PASS"

    print(
        f"STAGE 3.6 STATUS: {status}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()