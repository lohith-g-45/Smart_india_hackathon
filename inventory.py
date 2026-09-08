from pathlib import Path
import pandas as pd
import json
import re


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\tuala\Downloads\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

OUTPUT_DIR = Path("results/data_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NORMALIZE FILE IDENTIFIER
# ============================================================

def normalize_id(filename):

    name = Path(filename).stem.strip()

    # Remove S- or V- prefix
    if name.lower().startswith("s-"):
        name = name[2:]

    elif name.lower().startswith("v-"):
        name = name[2:]

    return name.strip().lower()


# ============================================================
# FIND S AND V FILES
# ============================================================

def find_files():

    s_files = {}
    v_files = {}

    for path in DATASET_ROOT.rglob("*.csv"):

        filename = path.name.strip()

        if filename.lower().startswith("s-"):

            seq_id = normalize_id(filename)

            s_files[seq_id] = path

        elif filename.lower().startswith("v-"):

            seq_id = normalize_id(filename)

            v_files[seq_id] = path

    return s_files, v_files


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    encodings = [
        "utf-8",
        "cp1252",
        "latin1"
    ]

    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )

            df.columns = [
                str(c).strip()
                for c in df.columns
            ]

            return df, encoding

        except UnicodeDecodeError:

            continue

    raise ValueError(
        f"Could not read file: {path}"
    )


# ============================================================
# FIND TIME COLUMN
# ============================================================

def find_time_column(df, dataset_type):

    columns = list(df.columns)

    if dataset_type == "V":

        for column in columns:

            if (
                str(column).strip().lower()
                ==
                "time since start of day (seconds)"
            ):

                return column

    if dataset_type == "S":

        for column in columns:

            name = str(column).strip().lower()

            if "date" in name and "yyyy" in name:

                return column

            if "timestamp" in name:

                return column

            if "time" in name:

                return column

    return None


# ============================================================
# CONVERT TIME
# ============================================================

def convert_time(df, dataset_type):

    column = find_time_column(
        df,
        dataset_type
    )

    if column is None:
        raise ValueError(
            f"No time column found for {dataset_type}"
        )

    # ========================================================
    # V DATASET
    # ========================================================

    if dataset_type == "V":

        time = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if time.notna().sum() == 0:
            raise ValueError(
                f"V time column could not be parsed: {column}"
            )

        return time

    # ========================================================
    # S DATASET
    # ========================================================

    raw = (
        df[column]
        .astype("string")
        .str.strip()
    )

    raw = raw.str.strip('"').str.strip("'")

    # --------------------------------------------------------
    # CASE 1:
    # S file selected TIME SINCE START (ms)
    # --------------------------------------------------------

    if "TIME SINCE START" in column.upper():

        numeric_time = pd.to_numeric(
            raw,
            errors="coerce"
        )

        if numeric_time.notna().sum() > 0:

            # milliseconds -> seconds
            return numeric_time / 1000.0

    # --------------------------------------------------------
    # CASE 2:
    # S file selected DATE column
    # --------------------------------------------------------

    # Convert:
    #
    # 2019-11-06 12:04:32:393
    #
    # into:
    #
    # 2019-11-06 12:04:32.393

    raw = raw.str.replace(
        r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d{1,6})$",
        r"\1.\2",
        regex=True
    )

    dt = pd.to_datetime(
        raw,
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce"
    )

    if dt.notna().sum() == 0:

        dt = pd.to_datetime(
            raw,
            errors="coerce"
        )

    if dt.notna().sum() == 0:

        examples = raw.dropna().head(3).tolist()

        raise ValueError(
            "S timestamp could not be parsed. "
            f"Column='{column}', "
            f"Examples={examples}"
        )

    # Convert datetime to elapsed seconds
    first = dt.dropna().iloc[0]

    seconds = (
        dt - first
    ).dt.total_seconds()

    return seconds

    # --------------------------------------------------------
    # S timestamp
    # --------------------------------------------------------

    raw = (
        df[column]
        .astype(str)
        .str.strip()
    )

    # Try exact IO-VNBD-style datetime parsing
    # IO-VNBD S-Dataset uses timestamps like:
# 2019-11-06 12:04:32:393
# where the final :393 represents milliseconds.
#
# Convert:
# 2019-11-06 12:04:32:393
#          ->
# 2019-11-06 12:04:32.393

    raw = raw.str.replace(
    r"(\d{2}:\d{2}:\d{2}):(\d{1,3})$",
    r"\1.\2",
    regex=True
)

    dt = pd.to_datetime(
    raw,
    format="%Y-%m-%d %H:%M:%S.%f",
    errors="coerce"
)

    if dt.notna().sum() == 0:

        raise ValueError(
            "S timestamp could not be parsed"
        )

    # Convert relative to first valid timestamp
    first = dt.dropna().iloc[0]

    seconds = (
        dt - first
    ).dt.total_seconds()

    return seconds


# ============================================================
# ANALYZE PAIR
# ============================================================

def analyze_pair(seq_id, s_path, v_path):

    result = {
        "sequence_id": seq_id,
        "s_file": str(s_path),
        "v_file": str(v_path),
        "status": "ERROR"
    }

    try:

        s_df, s_encoding = load_csv(s_path)
        v_df, v_encoding = load_csv(v_path)

        s_time = convert_time(
            s_df,
            "S"
        )

        v_time = convert_time(
            v_df,
            "V"
        )

        s_valid = s_time.dropna()
        v_valid = v_time.dropna()

        if len(s_valid) == 0:

            raise ValueError(
                "No valid S timestamps"
            )

        if len(v_valid) == 0:

            raise ValueError(
                "No valid V timestamps"
            )

        s_start = float(s_valid.min())
        s_end = float(s_valid.max())

        v_start = float(v_valid.min())
        v_end = float(v_valid.max())

        overlap = max(
            0,
            min(s_end, v_end)
            -
            max(s_start, v_start)
        )

        result.update({

            "s_rows": len(s_df),

            "v_rows": len(v_df),

            "s_columns":
                len(s_df.columns),

            "v_columns":
                len(v_df.columns),

            "s_time_column":
                find_time_column(
                    s_df,
                    "S"
                ),

            "v_time_column":
                find_time_column(
                    v_df,
                    "V"
                ),

            "s_encoding":
                s_encoding,

            "v_encoding":
                v_encoding,

            "s_duration_seconds":
                s_end - s_start,

            "v_duration_seconds":
                v_end - v_start,

            "overlap_seconds":
                overlap,

            "s_missing_percentage":
                float(
                    s_df.isna()
                    .mean()
                    .mean()
                    * 100
                ),

            "v_missing_percentage":
                float(
                    v_df.isna()
                    .mean()
                    .mean()
                    * 100
                ),

            "status":
                "PASS"
        })

    except Exception as e:

        result["error"] = str(e)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "NAV-SHIELD Stage 1.5: "
        "IO-VNBD Dataset Inventory"
    )
    print("=" * 72)

    print(
        f"Dataset root: {DATASET_ROOT}"
    )

    if not DATASET_ROOT.exists():

        print(
            "\nERROR: Dataset root does not exist:"
        )

        print(DATASET_ROOT)

        return

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    s_files, v_files = find_files()

    print(
        f"S files found: {len(s_files)}"
    )

    print(
        f"V files found: {len(v_files)}"
    )

    # --------------------------------------------------------
    # Pair using normalized lowercase ID
    # --------------------------------------------------------

    all_ids = sorted(
        set(s_files.keys())
        |
        set(v_files.keys())
    )

    print(
        f"Sequence IDs found: {len(all_ids)}"
    )

    print()

    results = []

    missing_s = []
    missing_v = []

    complete_ids = []

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    for index, seq_id in enumerate(
        all_ids,
        start=1
    ):

        has_s = seq_id in s_files
        has_v = seq_id in v_files

        print(
            f"[{index}/{len(all_ids)}] "
            f"Analyzing {seq_id} ..."
        )

        if not has_s:

            missing_s.append(seq_id)

            results.append({

                "sequence_id":
                    seq_id,

                "status":
                    "MISSING_S",

                "s_file":
                    None,

                "v_file":
                    str(v_files[seq_id])
            })

            continue

        if not has_v:

            missing_v.append(seq_id)

            results.append({

                "sequence_id":
                    seq_id,

                "status":
                    "MISSING_V",

                "s_file":
                    str(s_files[seq_id]),

                "v_file":
                    None
            })

            continue

        result = analyze_pair(
            seq_id,
            s_files[seq_id],
            v_files[seq_id]
        )

        results.append(result)

        if result["status"] == "PASS":

            complete_ids.append(seq_id)

    # ========================================================
    # SAVE REPORT
    # ========================================================

    df = pd.DataFrame(results)

    csv_path = (
        OUTPUT_DIR
        / "dataset_inventory.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    json_path = (
        OUTPUT_DIR
        / "dataset_inventory.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    errors = [
        r
        for r in results
        if r["status"] == "ERROR"
    ]

    print()
    print("=" * 72)
    print("DATASET INVENTORY SUMMARY")
    print("=" * 72)

    print(
        f"Complete S/V pairs : "
        f"{len(complete_ids)}"
    )

    print(
        f"Missing S          : "
        f"{len(missing_s)}"
    )

    print(
        f"Missing V          : "
        f"{len(missing_v)}"
    )

    print(
        f"Errors             : "
        f"{len(errors)}"
    )

    print()

    print(
        "Complete sequences:"
    )

    print(
        complete_ids
    )

    if missing_s:

        print()
        print(
            "Missing S:"
        )

        print(
            missing_s
        )

    if missing_v:

        print()
        print(
            "Missing V:"
        )

        print(
            missing_v
        )

    if errors:

        print()
        print(
            "ERROR DETAILS:"
        )

        for error in errors:

            print(
                f"{error['sequence_id']}: "
                f"{error.get('error')}"
            )

    print()

    print(
        f"Inventory CSV: {csv_path}"
    )

    print(
        f"Inventory JSON: {json_path}"
    )

    print("=" * 72)


if __name__ == "__main__":

    main()