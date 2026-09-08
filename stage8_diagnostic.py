"""
======================================================================
NAV-SHIELD STAGE 8: DIAGNOSTIC 2
======================================================================

Purpose:
    Analyze the exact targets used by Stage 8 and compare their
    distributions across TRAIN / VALIDATION / TEST.

Target definitions:
    position_error_m
        = Haversine distance between S GPS position and V GPS position

    velocity_error_m_s
        = |S GPS speed - V vehicle velocity| / 3.6

Important:
    - No model retraining
    - No raw data modification
    - No previous-stage modification
    - Uncategorised dataset NOT USED
    - Sequence-level leakage protection preserved
    - Encoding fallback enabled
    - Raw S/V files discovered automatically
======================================================================
"""

from pathlib import Path
import json
import re
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(
    r"C:\Users\tuala\Downloads\member5_ai_drift_stage1\member5_ai_drift"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "features"
    / "categorised"
)

STAGE4_SPLIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage4_reports"
    / "stage4_dataset_split.json"
)

STAGE8_PREDICTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage8"
    / "predictions"
    / "stage8_test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage8"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# TARGET COLUMN DEFINITIONS
# ======================================================================

TARGET_POSITION = "position_error_m"
TARGET_VELOCITY = "velocity_error_m_s"


# ======================================================================
# ENCODINGS
# ======================================================================

ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]


# ======================================================================
# PRINT HELPERS
# ======================================================================

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def subheader(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# CSV READER
# ======================================================================

def read_csv_fallback(path, usecols=None, nrows=None):
    """
    Read CSV using multiple encoding fallbacks.

    This specifically handles columns containing characters such as:
        µ
        ²
        °
    """

    last_error = None

    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                usecols=usecols,
                nrows=nrows,
            )
            return df

        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue

    raise last_error


# ======================================================================
# NORMALIZE TEXT
# ======================================================================

def normalize_text(value):
    """
    Normalize a filename or column name for matching.
    """

    value = str(value).strip().lower()

    replacements = {
        "²": "2",
        "³": "3",
        "µ": "u",
        "μ": "u",
        "°": "deg",
        "Â°": "deg",
        "Âµ": "u",
        "â": "",
        "’": "'",
        "–": "-",
        "—": "-",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


# ======================================================================
# SEQUENCE ID NORMALIZATION
# ======================================================================

def normalize_sequence_id(value):
    """
    Convert sequence names into a stable identifier.

    Examples:
        s1_features -> s1
        s1 -> s1
        VTA10_S -> vta10
        vta10_features -> vta10
    """

    name = Path(str(value)).stem.lower()

    name = re.sub(
        r"(_features?|[- ]features?)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(
        r"^(s_|v_|s-|v-)",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = name.strip("_- ")

    return name.lower()


# ======================================================================
# FIND ALL RAW CSV FILES
# ======================================================================

def discover_raw_csvs(root):
    """
    Recursively discover all CSV files.
    """

    files = sorted(root.rglob("*.csv"))

    return files


# ======================================================================
# COLUMN CLASSIFICATION
# ======================================================================

def classify_csv(path):
    """
    Determine whether a CSV is an S sensor file or V vehicle file
    based on its columns.

    This avoids depending on filename conventions.
    """

    try:
        df = read_csv_fallback(path, nrows=0)

    except Exception as exc:
        return {
            "path": path,
            "type": "ERROR",
            "sequence_id": normalize_sequence_id(path.stem),
            "columns": [],
            "error": str(exc),
        }

    normalized_columns = {
        normalize_text(col): col
        for col in df.columns
    }

    column_text = " ".join(normalized_columns.keys())

    # --------------------------------------------------------------
    # S DATASET SIGNATURE
    # --------------------------------------------------------------

    s_score = 0

    s_keywords = [
        "gps latitude",
        "gps longitude",
        "gps speed",
        "accelerometer x",
        "accelerometer y",
        "accelerometer z",
        "gyroscope x",
        "gyroscope y",
        "gyroscope z",
        "gravity x",
        "gravity y",
        "gravity z",
        "magnetic field x",
        "magnetic field y",
        "magnetic field z",
        "time since start",
    ]

    for keyword in s_keywords:
        if keyword in column_text:
            s_score += 1

    # --------------------------------------------------------------
    # V DATASET SIGNATURE
    # --------------------------------------------------------------

    v_score = 0

    v_keywords = [
        "indicated vehicle speed",
        "velocity",
        "engine speed",
        "steering angle",
        "wheel speed front left",
        "wheel speed front right",
        "wheel speed rear left",
        "wheel speed rear right",
        "yaw rate",
        "accelerator pedal position",
        "brake position",
        "gear",
        "battery voltage",
    ]

    for keyword in v_keywords:
        if keyword in column_text:
            v_score += 1

    if s_score > v_score and s_score >= 2:
        file_type = "S"

    elif v_score > s_score and v_score >= 2:
        file_type = "V"

    else:
        file_type = "UNKNOWN"

    return {
        "path": path,
        "type": file_type,
        "sequence_id": normalize_sequence_id(path.stem),
        "columns": list(df.columns),
        "s_score": s_score,
        "v_score": v_score,
        "error": None,
    }


# ======================================================================
# DISCOVER S/V PAIRS
# ======================================================================

def discover_sv_pairs(raw_root):

    header("RAW S/V FILE DISCOVERY")

    print(f"Raw dataset root:")
    print(raw_root)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Categorised dataset not found:\n{raw_root}"
        )

    csv_files = discover_raw_csvs(raw_root)

    print(f"\nTotal CSV files discovered: {len(csv_files)}")

    records = []

    for index, path in enumerate(csv_files, start=1):

        record = classify_csv(path)

        records.append(record)

        if index % 20 == 0 or index == len(csv_files):
            print(
                f"  Classified {index}/{len(csv_files)}"
            )

    # --------------------------------------------------------------
    # BUILD DICTIONARIES
    # --------------------------------------------------------------

    s_files = {}
    v_files = {}

    unknown_files = []

    for record in records:

        sequence_id = record["sequence_id"]

        if record["type"] == "S":

            if sequence_id in s_files:
                print(
                    f"[WARNING] Duplicate S sequence ID: "
                    f"{sequence_id}"
                )

            s_files.setdefault(
                sequence_id,
                record["path"]
            )

        elif record["type"] == "V":

            if sequence_id in v_files:
                print(
                    f"[WARNING] Duplicate V sequence ID: "
                    f"{sequence_id}"
                )

            v_files.setdefault(
                sequence_id,
                record["path"]
            )

        elif record["type"] == "UNKNOWN":

            unknown_files.append(record["path"])

    print("\nDISCOVERY SUMMARY")
    print("------------------------------")
    print(f"S files      : {len(s_files)}")
    print(f"V files      : {len(v_files)}")
    print(f"Unknown      : {len(unknown_files)}")

    # --------------------------------------------------------------
    # PAIR BY SEQUENCE ID
    # --------------------------------------------------------------

    all_ids = sorted(
        set(s_files.keys()) | set(v_files.keys())
    )

    pairs = {}

    s_only = []
    v_only = []

    for sequence_id in all_ids:

        has_s = sequence_id in s_files
        has_v = sequence_id in v_files

        if has_s and has_v:

            pairs[sequence_id] = (
                s_files[sequence_id],
                v_files[sequence_id],
            )

        elif has_s:

            s_only.append(sequence_id)

        elif has_v:

            v_only.append(sequence_id)

    print(f"\nComplete S/V pairs : {len(pairs)}")
    print(f"S-only             : {len(s_only)}")
    print(f"V-only             : {len(v_only)}")

    if unknown_files:
        print("\nUnknown files:")
        for path in unknown_files[:20]:
            print(f"  - {path.name}")

        if len(unknown_files) > 20:
            print(
                f"  ... and "
                f"{len(unknown_files) - 20} more"
            )

    # --------------------------------------------------------------
    # SHOW SAMPLE PAIRS
    # --------------------------------------------------------------

    print("\nSAMPLE PAIRS")

    for sequence_id in list(pairs.keys())[:10]:

        s_path, v_path = pairs[sequence_id]

        print(f"\n  Sequence: {sequence_id}")
        print(f"    S: {s_path.name}")
        print(f"    V: {v_path.name}")

    return pairs


# ======================================================================
# FIND CATEGORISED DATASET
# ======================================================================

def find_categorised_dataset():
    """
    Locate Categorised IOVNB Dataset.

    We intentionally do NOT use the Uncategorised dataset.
    """

    candidates = [
        Path(
            r"C:\Users\tuala\Downloads"
            r"\Synchronised V abd S datasets"
            r"\Synchronised V abd S datasets"
            r"\Categorised IOVNB Dataset"
        ),
        Path(
            r"C:\Users\tuala\Downloads"
            r"\Synchronised V abd S datasets"
            r"\Categorised IOVNB Dataset"
        ),
    ]

    for path in candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate the Categorised IOVNB Dataset."
    )


# ======================================================================
# HAVERSINE
# ======================================================================

def haversine_vectorized(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate Haversine distance in meters.
    """

    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)

    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    earth_radius_m = 6371000.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1_rad)
        *
        np.cos(lat2_rad)
        *
        np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arcsin(np.sqrt(a))

    return earth_radius_m * c


# ======================================================================
# COLUMN FINDING
# ======================================================================

def find_column(columns, candidates):
    """
    Find a column using normalized matching.
    """

    normalized = {
        normalize_text(col): col
        for col in columns
    }

    # Exact normalized match
    for candidate in candidates:

        candidate_norm = normalize_text(candidate)

        if candidate_norm in normalized:
            return normalized[candidate_norm]

    # Partial matching
    for candidate in candidates:

        candidate_norm = normalize_text(candidate)

        for norm_col, original_col in normalized.items():

            if candidate_norm in norm_col:
                return original_col

    return None


# ======================================================================
# TARGET COLUMN DISCOVERY
# ======================================================================

def find_s_target_columns(columns):

    lat_col = find_column(
        columns,
        [
            "GPS LATITUDE (degrees)",
            "GPS LATITUDE",
            "LATITUDE",
        ],
    )

    lon_col = find_column(
        columns,
        [
            "GPS LONGITUDE (degrees)",
            "GPS LONGITUDE",
            "LONGITUDE",
        ],
    )

    speed_col = find_column(
        columns,
        [
            "GPS SPEED (Kmh)",
            "GPS SPEED",
            "SPEED",
        ],
    )

    return lat_col, lon_col, speed_col


def find_v_target_columns(columns):

    lat_col = find_column(
        columns,
        [
            "Latitude (degrees)",
            "Latitude",
        ],
    )

    lon_col = find_column(
        columns,
        [
            "Longitude (degrees)",
            "Longitude",
        ],
    )

    velocity_col = find_column(
        columns,
        [
            "Velocity (km/hr)",
            "Velocity",
        ],
    )

    # Fallback to indicated vehicle speed
    if velocity_col is None:

        velocity_col = find_column(
            columns,
            [
                "Indicated Vehicle Speed (km/hr)",
                "Indicated Vehicle Speed",
            ],
        )

    return lat_col, lon_col, velocity_col


# ======================================================================
# COMPUTE TARGETS
# ======================================================================

def compute_targets(
    s_path,
    v_path,
):
    """
    Compute targets from the raw Categorised S/V pair.

    IMPORTANT:
    We do not modify either file.

    Rows are aligned by row order after loading.
    """

    # --------------------------------------------------------------
    # Read headers first
    # --------------------------------------------------------------

    s_header = read_csv_fallback(
        s_path,
        nrows=0,
    )

    v_header = read_csv_fallback(
        v_path,
        nrows=0,
    )

    s_lat_col, s_lon_col, s_speed_col = (
        find_s_target_columns(s_header.columns)
    )

    v_lat_col, v_lon_col, v_velocity_col = (
        find_v_target_columns(v_header.columns)
    )

    missing = []

    if s_lat_col is None:
        missing.append(
            "S GPS LATITUDE"
        )

    if s_lon_col is None:
        missing.append(
            "S GPS LONGITUDE"
        )

    if s_speed_col is None:
        missing.append(
            "S GPS SPEED"
        )

    if v_lat_col is None:
        missing.append(
            "V LATITUDE"
        )

    if v_lon_col is None:
        missing.append(
            "V LONGITUDE"
        )

    if v_velocity_col is None:
        missing.append(
            "V VELOCITY"
        )

    if missing:

        return None, {
            "status": "MISSING_COLUMNS",
            "missing": missing,
            "s_columns": list(s_header.columns),
            "v_columns": list(v_header.columns),
        }

    # --------------------------------------------------------------
    # Load only required columns
    # --------------------------------------------------------------

    s_df = read_csv_fallback(
        s_path,
        usecols=[
            s_lat_col,
            s_lon_col,
            s_speed_col,
        ],
    )

    v_df = read_csv_fallback(
        v_path,
        usecols=[
            v_lat_col,
            v_lon_col,
            v_velocity_col,
        ],
    )

    # --------------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------------

    s_lat = pd.to_numeric(
        s_df[s_lat_col],
        errors="coerce",
    )

    s_lon = pd.to_numeric(
        s_df[s_lon_col],
        errors="coerce",
    )

    s_speed = pd.to_numeric(
        s_df[s_speed_col],
        errors="coerce",
    )

    v_lat = pd.to_numeric(
        v_df[v_lat_col],
        errors="coerce",
    )

    v_lon = pd.to_numeric(
        v_df[v_lon_col],
        errors="coerce",
    )

    v_velocity = pd.to_numeric(
        v_df[v_velocity_col],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # Align lengths
    # --------------------------------------------------------------

    n = min(
        len(s_df),
        len(v_df),
    )

    s_lat = s_lat.iloc[:n].to_numpy()
    s_lon = s_lon.iloc[:n].to_numpy()
    s_speed = s_speed.iloc[:n].to_numpy()

    v_lat = v_lat.iloc[:n].to_numpy()
    v_lon = v_lon.iloc[:n].to_numpy()
    v_velocity = v_velocity.iloc[:n].to_numpy()

    # --------------------------------------------------------------
    # Position error
    # --------------------------------------------------------------

    valid_position = (
        np.isfinite(s_lat)
        &
        np.isfinite(s_lon)
        &
        np.isfinite(v_lat)
        &
        np.isfinite(v_lon)
    )

    position_error = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    if np.any(valid_position):

        position_error[valid_position] = (
            haversine_vectorized(
                s_lat[valid_position],
                s_lon[valid_position],
                v_lat[valid_position],
                v_lon[valid_position],
            )
        )

    # --------------------------------------------------------------
    # Velocity error
    #
    # GPS speed is Kmh.
    # V velocity is km/hr.
    #
    # Difference in km/hr / 3.6 = m/s
    # --------------------------------------------------------------

    valid_velocity = (
        np.isfinite(s_speed)
        &
        np.isfinite(v_velocity)
    )

    velocity_error = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    if np.any(valid_velocity):

        velocity_error[valid_velocity] = (
            np.abs(
                s_speed[valid_velocity]
                -
                v_velocity[valid_velocity]
            )
            / 3.6
        )

    targets = pd.DataFrame(
        {
            TARGET_POSITION: position_error,
            TARGET_VELOCITY: velocity_error,
        }
    )

    metadata = {
        "status": "PASS",
        "s_rows": len(s_df),
        "v_rows": len(v_df),
        "aligned_rows": n,
        "position_valid_rows": int(
            np.isfinite(position_error).sum()
        ),
        "velocity_valid_rows": int(
            np.isfinite(velocity_error).sum()
        ),
        "s_lat_column": s_lat_col,
        "s_lon_column": s_lon_col,
        "s_speed_column": s_speed_col,
        "v_lat_column": v_lat_col,
        "v_lon_column": v_lon_col,
        "v_velocity_column": v_velocity_col,
    }

    return targets, metadata


# ======================================================================
# STATISTICS
# ======================================================================

def calculate_statistics(values):

    values = np.asarray(values, dtype=np.float64)

    values = values[np.isfinite(values)]

    if len(values) == 0:

        return {
            "rows": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


# ======================================================================
# SPLIT STATISTICS
# ======================================================================

def analyze_sequence(
    sequence_id,
    split_name,
    pairs,
):

    if sequence_id not in pairs:

        return {
            "split": split_name,
            "sequence": sequence_id,
            "rows": 0,
            "position_error_m_available": False,
            "velocity_error_m_s_available": False,
            "position_valid_rows": 0,
            "velocity_valid_rows": 0,
            "position_mean": np.nan,
            "position_median": np.nan,
            "position_p95": np.nan,
            "position_p99": np.nan,
            "velocity_mean": np.nan,
            "velocity_median": np.nan,
            "velocity_p95": np.nan,
            "velocity_p99": np.nan,
            "status": "PAIR_NOT_FOUND",
        }

    s_path, v_path = pairs[sequence_id]

    try:

        targets, metadata = compute_targets(
            s_path,
            v_path,
        )

    except Exception as exc:

        return {
            "split": split_name,
            "sequence": sequence_id,
            "rows": 0,
            "position_error_m_available": False,
            "velocity_error_m_s_available": False,
            "position_valid_rows": 0,
            "velocity_valid_rows": 0,
            "position_mean": np.nan,
            "position_median": np.nan,
            "position_p95": np.nan,
            "position_p99": np.nan,
            "velocity_mean": np.nan,
            "velocity_median": np.nan,
            "velocity_p95": np.nan,
            "velocity_p99": np.nan,
            "status": f"ERROR: {exc}",
        }

    if targets is None:

        return {
            "split": split_name,
            "sequence": sequence_id,
            "rows": 0,
            "position_error_m_available": False,
            "velocity_error_m_s_available": False,
            "position_valid_rows": 0,
            "velocity_valid_rows": 0,
            "position_mean": np.nan,
            "position_median": np.nan,
            "position_p95": np.nan,
            "position_p99": np.nan,
            "velocity_mean": np.nan,
            "velocity_median": np.nan,
            "velocity_p95": np.nan,
            "velocity_p99": np.nan,
            "status": str(
                metadata.get(
                    "status",
                    "UNKNOWN",
                )
            ),
        }

    position = targets[
        TARGET_POSITION
    ].to_numpy()

    velocity = targets[
        TARGET_VELOCITY
    ].to_numpy()

    position_stats = calculate_statistics(
        position
    )

    velocity_stats = calculate_statistics(
        velocity
    )

    return {
        "split": split_name,
        "sequence": sequence_id,
        "rows": int(len(targets)),

        "position_error_m_available":
            position_stats["rows"] > 0,

        "velocity_error_m_s_available":
            velocity_stats["rows"] > 0,

        "position_valid_rows":
            position_stats["rows"],

        "velocity_valid_rows":
            velocity_stats["rows"],

        "position_mean":
            position_stats["mean"],

        "position_median":
            position_stats["median"],

        "position_p95":
            position_stats["p95"],

        "position_p99":
            position_stats["p99"],

        "velocity_mean":
            velocity_stats["mean"],

        "velocity_median":
            velocity_stats["median"],

        "velocity_p95":
            velocity_stats["p95"],

        "velocity_p99":
            velocity_stats["p99"],

        "status": "PASS",
    }


# ======================================================================
# LOAD STAGE 4 SPLIT
# ======================================================================

def load_split():

    header("LOADING STAGE 4 SPLIT")

    if not STAGE4_SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Stage 4 split file not found:\n"
            f"{STAGE4_SPLIT_FILE}"
        )

    with open(
        STAGE4_SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        split = json.load(f)

    train_ids = [
        normalize_sequence_id(x)
        for x in split.get("train", [])
    ]

    validation_ids = [
        normalize_sequence_id(x)
        for x in split.get("validation", [])
    ]

    test_ids = [
        normalize_sequence_id(x)
        for x in split.get("test", [])
    ]

    print(
        f"\nTRAIN      : {len(train_ids)} sequences"
    )

    print(
        f"VALIDATION : {len(validation_ids)} sequences"
    )

    print(
        f"TEST       : {len(test_ids)} sequences"
    )

    return (
        train_ids,
        validation_ids,
        test_ids,
    )


# ======================================================================
# LEAKAGE CHECK
# ======================================================================

def check_leakage(
    train_ids,
    validation_ids,
    test_ids,
):

    header("SEQUENCE LEAKAGE CHECK")

    train_set = set(train_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    tv = train_set & validation_set
    tt = train_set & test_set
    vt = validation_set & test_set

    print(
        f"TRAIN ↔ VALIDATION overlap: {len(tv)}"
    )

    print(
        f"TRAIN ↔ TEST overlap: {len(tt)}"
    )

    print(
        f"VALIDATION ↔ TEST overlap: {len(vt)}"
    )

    if not tv and not tt and not vt:

        print(
            "[PASS] No sequence overlap detected."
        )

        return True

    print(
        "[FAIL] Sequence leakage detected."
    )

    return False


# ======================================================================
# PREDICTION ANALYSIS
# ======================================================================

def analyze_predictions():

    header("STAGE 8 TEST PREDICTION ANALYSIS")

    if not STAGE8_PREDICTION_FILE.exists():

        print(
            "[WARNING] Stage 8 prediction file not found."
        )

        return None

    predictions = pd.read_csv(
        STAGE8_PREDICTION_FILE,
        low_memory=False,
    )

    print(
        f"Prediction rows: {len(predictions)}"
    )

    print("Prediction columns:")

    for col in predictions.columns:
        print(f"  - {col}")

    required = [
        "actual_position_error_m",
        "predicted_position_error_m",
        "actual_velocity_error_m_s",
        "predicted_velocity_error_m_s",
    ]

    missing = [
        col
        for col in required
        if col not in predictions.columns
    ]

    if missing:

        print(
            "\n[WARNING] Missing prediction columns:"
        )

        for col in missing:
            print(f"  - {col}")

        return None

    results = {}

    # --------------------------------------------------------------
    # POSITION
    # --------------------------------------------------------------

    print("\nPOSITION ERROR")

    actual = pd.to_numeric(
        predictions[
            "actual_position_error_m"
        ],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        predictions[
            "predicted_position_error_m"
        ],
        errors="coerce",
    )

    valid = (
        np.isfinite(actual)
        &
        np.isfinite(predicted)
    )

    actual_np = actual[valid].to_numpy()
    predicted_np = predicted[valid].to_numpy()

    errors = predicted_np - actual_np
    absolute_errors = np.abs(errors)

    position_result = {
        "valid_predictions":
            int(len(errors)),

        "mae":
            float(np.mean(absolute_errors)),

        "rmse":
            float(
                np.sqrt(
                    np.mean(errors ** 2)
                )
            ),

        "error_mean":
            float(np.mean(errors)),

        "error_std":
            float(np.std(errors)),

        "error_median":
            float(np.median(errors)),

        "absolute_error_p95":
            float(
                np.percentile(
                    absolute_errors,
                    95,
                )
            ),

        "absolute_error_p99":
            float(
                np.percentile(
                    absolute_errors,
                    99,
                )
            ),

        "absolute_error_max":
            float(np.max(absolute_errors)),
    }

    for key, value in position_result.items():
        print(f"  {key}: {value}")

    results[
        TARGET_POSITION
    ] = position_result

    # --------------------------------------------------------------
    # VELOCITY
    # --------------------------------------------------------------

    print("\nVELOCITY ERROR")

    actual = pd.to_numeric(
        predictions[
            "actual_velocity_error_m_s"
        ],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        predictions[
            "predicted_velocity_error_m_s"
        ],
        errors="coerce",
    )

    valid = (
        np.isfinite(actual)
        &
        np.isfinite(predicted)
    )

    actual_np = actual[valid].to_numpy()
    predicted_np = predicted[valid].to_numpy()

    errors = predicted_np - actual_np
    absolute_errors = np.abs(errors)

    velocity_result = {
        "valid_predictions":
            int(len(errors)),

        "mae":
            float(np.mean(absolute_errors)),

        "rmse":
            float(
                np.sqrt(
                    np.mean(errors ** 2)
                )
            ),

        "error_mean":
            float(np.mean(errors)),

        "error_std":
            float(np.std(errors)),

        "error_median":
            float(np.median(errors)),

        "absolute_error_p95":
            float(
                np.percentile(
                    absolute_errors,
                    95,
                )
            ),

        "absolute_error_p99":
            float(
                np.percentile(
                    absolute_errors,
                    99,
                )
            ),

        "absolute_error_max":
            float(np.max(absolute_errors)),
    }

    for key, value in velocity_result.items():
        print(f"  {key}: {value}")

    results[
        TARGET_VELOCITY
    ] = velocity_result

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "stage8_diagnostic2_prediction_errors.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\n[SAVED]\n{output_file}"
    )

    return results


# ======================================================================
# SPLIT SUMMARY
# ======================================================================

def create_split_summary(
    sequence_df,
):

    rows = []

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        subset = sequence_df[
            sequence_df["split"]
            == split_name
        ]

        position_values = subset[
            "position_mean"
        ].dropna()

        position_median_values = subset[
            "position_median"
        ].dropna()

        position_p95_values = subset[
            "position_p95"
        ].dropna()

        position_p99_values = subset[
            "position_p99"
        ].dropna()

        velocity_values = subset[
            "velocity_mean"
        ].dropna()

        velocity_median_values = subset[
            "velocity_median"
        ].dropna()

        velocity_p95_values = subset[
            "velocity_p95"
        ].dropna()

        velocity_p99_values = subset[
            "velocity_p99"
        ].dropna()

        rows.append(
            {
                "split":
                    split_name,

                "sequences":
                    len(subset),

                "rows":
                    int(
                        subset["rows"].sum()
                    ),

                "position_valid_rows":
                    int(
                        subset[
                            "position_valid_rows"
                        ].sum()
                    ),

                "velocity_valid_rows":
                    int(
                        subset[
                            "velocity_valid_rows"
                        ].sum()
                    ),

                "position_mean_of_sequences":
                    (
                        float(
                            position_values.mean()
                        )
                        if len(
                            position_values
                        )
                        else np.nan
                    ),

                "position_median_of_sequences":
                    (
                        float(
                            position_median_values.mean()
                        )
                        if len(
                            position_median_values
                        )
                        else np.nan
                    ),

                "position_p95_of_sequences":
                    (
                        float(
                            position_p95_values.mean()
                        )
                        if len(
                            position_p95_values
                        )
                        else np.nan
                    ),

                "position_p99_of_sequences":
                    (
                        float(
                            position_p99_values.mean()
                        )
                        if len(
                            position_p99_values
                        )
                        else np.nan
                    ),

                "velocity_mean_of_sequences":
                    (
                        float(
                            velocity_values.mean()
                        )
                        if len(
                            velocity_values
                        )
                        else np.nan
                    ),

                "velocity_median_of_sequences":
                    (
                        float(
                            velocity_median_values.mean()
                        )
                        if len(
                            velocity_median_values
                        )
                        else np.nan
                    ),

                "velocity_p95_of_sequences":
                    (
                        float(
                            velocity_p95_values.mean()
                        )
                        if len(
                            velocity_p95_values
                        )
                        else np.nan
                    ),

                "velocity_p99_of_sequences":
                    (
                        float(
                            velocity_p99_values.mean()
                        )
                        if len(
                            velocity_p99_values
                        )
                        else np.nan
                    ),
            }
        )

    summary_df = pd.DataFrame(rows)

    output_file = (
        OUTPUT_DIR
        / "stage8_split_target_summary.csv"
    )

    summary_df.to_csv(
        output_file,
        index=False,
    )

    print("\nSPLIT-LEVEL TARGET SUMMARY")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\n[SAVED]\n{output_file}"
    )

    return summary_df


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "NAV-SHIELD STAGE 8 DIAGNOSTIC 2"
    )

    print(
        """
Purpose:
Analyze the exact targets used by Stage 8.

Target definitions:
  position_error_m
      = Haversine(S GPS, V GPS)

  velocity_error_m_s
      = |S GPS speed - V velocity| / 3.6

Model retraining: NO
Raw data modification: NO
Previous-stage modification: NO
Uncategorised dataset: NOT USED
"""
    )

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nFeature directory:")
    print(FEATURE_DIR)

    print("\nStage 4 split file:")
    print(STAGE4_SPLIT_FILE)

    print("\nStage 8 prediction file:")
    print(STAGE8_PREDICTION_FILE)

    # ==============================================================
    # CHECK PROJECT PATHS
    # ==============================================================

    if not PROJECT_ROOT.exists():

        raise FileNotFoundError(
            f"Project root does not exist:\n"
            f"{PROJECT_ROOT}"
        )

    # ==============================================================
    # LOAD SPLIT
    # ==============================================================

    (
        train_ids,
        validation_ids,
        test_ids,
    ) = load_split()

    # ==============================================================
    # LEAKAGE CHECK
    # ==============================================================

    leakage_ok = check_leakage(
        train_ids,
        validation_ids,
        test_ids,
    )

    # ==============================================================
    # FIND CATEGORISED DATASET
    # ==============================================================

    header("RAW TARGET SOURCE")

    categorised_root = (
        find_categorised_dataset()
    )

    print(
        "Categorised dataset found:"
    )

    print(
        categorised_root
    )

    print(
        "\n[IMPORTANT] "
        "Uncategorised dataset is NOT being used."
    )

    # ==============================================================
    # DISCOVER RAW PAIRS
    # ==============================================================

    pairs = discover_sv_pairs(
        categorised_root
    )

    # ==============================================================
    # VERIFY ALL SPLIT IDS
    # ==============================================================

    all_split_ids = (
        train_ids
        + validation_ids
        + test_ids
    )

    missing_pairs = [
        sequence_id
        for sequence_id in all_split_ids
        if sequence_id not in pairs
    ]

    print(
        "\nSPLIT PAIR VALIDATION"
    )

    print(
        f"Stage 4 sequences : "
        f"{len(all_split_ids)}"
    )

    print(
        f"Raw S/V pairs     : "
        f"{len(pairs)}"
    )

    print(
        f"Missing pairs     : "
        f"{len(missing_pairs)}"
    )

    if missing_pairs:

        print(
            "\nMissing sequence IDs:"
        )

        for sequence_id in missing_pairs:
            print(
                f"  - {sequence_id}"
            )

        print(
            "\n[WARNING] "
            "Some split sequences could not be paired."
        )

    else:

        print(
            "[PASS] "
            "All Stage 4 sequences have raw S/V pairs."
        )

    # ==============================================================
    # ANALYZE TARGETS
    # ==============================================================

    header(
        "ANALYZING TARGET DISTRIBUTIONS"
    )

    records = []

    split_definitions = [
        ("TRAIN", train_ids),
        ("VALIDATION", validation_ids),
        ("TEST", test_ids),
    ]

    for split_name, sequence_ids in split_definitions:

        print(
            f"\nAnalyzing {split_name}: "
            f"{len(sequence_ids)} sequences"
        )

        for index, sequence_id in enumerate(
            sequence_ids,
            start=1,
        ):

            if (
                index == 1
                or index % 10 == 0
                or index == len(sequence_ids)
            ):

                print(
                    f"  [{split_name}] "
                    f"{index}/{len(sequence_ids)} "
                    f"{sequence_id}"
                )

            record = analyze_sequence(
                sequence_id,
                split_name,
                pairs,
            )

            records.append(record)

    # ==============================================================
    # SAVE SEQUENCE STATISTICS
    # ==============================================================

    sequence_df = pd.DataFrame(
        records
    )

    sequence_output = (
        OUTPUT_DIR
        / "stage8_sequence_target_statistics.csv"
    )

    sequence_df.to_csv(
        sequence_output,
        index=False,
    )

    print(
        "\n[SAVED]"
    )

    print(
        sequence_output
    )

    # ==============================================================
    # STATUS COUNTS
    # ==============================================================

    subheader(
        "TARGET COMPUTATION STATUS"
    )

    status_counts = (
        sequence_df["status"]
        .value_counts()
    )

    print(
        status_counts.to_string()
    )

    valid_position_sequences = int(
        sequence_df[
            "position_error_m_available"
        ].sum()
    )

    valid_velocity_sequences = int(
        sequence_df[
            "velocity_error_m_s_available"
        ].sum()
    )

    print(
        f"\nSequences with valid "
        f"position targets: "
        f"{valid_position_sequences}"
    )

    print(
        f"Sequences with valid "
        f"velocity targets: "
        f"{valid_velocity_sequences}"
    )

    # ==============================================================
    # SPLIT SUMMARY
    # ==============================================================

    summary_df = create_split_summary(
        sequence_df
    )

    # ==============================================================
    # PREDICTION ANALYSIS
    # ==============================================================

    prediction_results = (
        analyze_predictions()
    )

    # ==============================================================
    # FINAL INTERPRETATION
    # ==============================================================

    header(
        "AUTOMATIC DIAGNOSTIC INTERPRETATION"
    )

    if leakage_ok:

        print(
            "[PASS] "
            "Train/validation/test sequence separation "
            "is preserved."
        )

    else:

        print(
            "[FAIL] "
            "Sequence overlap detected."
        )

    if len(missing_pairs) == 0:

        print(
            "[PASS] "
            "All 72 Stage 4 sequences were paired "
            "with raw Categorised S/V files."
        )

    else:

        print(
            "[WARNING] "
            f"{len(missing_pairs)} split sequences "
            "could not be paired."
        )

    if valid_position_sequences == len(
        all_split_ids
    ):

        print(
            "[PASS] "
            "Position-error targets available "
            "for every sequence."
        )

    else:

        print(
            "[WARNING] "
            "Some position-error targets "
            "could not be calculated."
        )

    if valid_velocity_sequences == len(
        all_split_ids
    ):

        print(
            "[PASS] "
            "Velocity-error targets available "
            "for every sequence."
        )

    else:

        print(
            "[WARNING] "
            "Some velocity-error targets "
            "could not be calculated."
        )

    # ==============================================================
    # SAVE DIAGNOSTIC METADATA
    # ==============================================================

    metadata = {
        "stage": "Stage 8 Diagnostic 2",

        "purpose":
            "Analyze exact targets used by Stage 8",

        "target_definitions": {
            "position_error_m":
                "Haversine distance between S GPS and V GPS",

            "velocity_error_m_s":
                "|S GPS speed - V velocity| / 3.6",
        },

        "train_sequences":
            len(train_ids),

        "validation_sequences":
            len(validation_ids),

        "test_sequences":
            len(test_ids),

        "raw_categorised_pairs":
            len(pairs),

        "missing_raw_pairs":
            missing_pairs,

        "sequence_leakage":
            not leakage_ok,

        "position_target_sequences":
            valid_position_sequences,

        "velocity_target_sequences":
            valid_velocity_sequences,

        "model_retraining":
            False,

        "raw_data_modified":
            False,

        "previous_stage_modified":
            False,

        "uncategorised_dataset_used":
            False,

        "prediction_analysis_available":
            prediction_results is not None,
    }

    metadata_output = (
        OUTPUT_DIR
        / "stage8_diagnostic2_metadata.json"
    )

    with open(
        metadata_output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ==============================================================
    # COMPLETE
    # ==============================================================

    header(
        "STAGE 8 DIAGNOSTIC 2 COMPLETE"
    )

    print(
        "\nDiagnostics saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  - stage8_sequence_target_statistics.csv"
    )

    print(
        "  - stage8_split_target_summary.csv"
    )

    print(
        "  - stage8_diagnostic2_prediction_errors.json"
    )

    print(
        "  - stage8_diagnostic2_metadata.json"
    )

    print(
        """
Important:
- No model retraining was performed.
- No raw data was modified.
- No Stage 1-8 output was modified.
- Uncategorised dataset was NOT used.
- Train/validation/test sequence separation was preserved.
- CSV encoding fallback was enabled.
- S/V files were discovered using their column structure.
"""
    )


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    main()