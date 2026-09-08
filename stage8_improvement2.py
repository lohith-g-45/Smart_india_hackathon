# ================================================================
# NAV-SHIELD STAGE 8 IMPROVEMENT — EXPERIMENT 8B
# ROBUST TARGET-LOSS EXPERIMENT
# ================================================================
#
# Purpose:
#   Improve Stage 8 temporal regression using a robust loss.
#
# IMPORTANT:
#   position_error_m and velocity_error_m_s are CALCULATED targets.
#   They are NOT expected to exist in the feature CSV files.
#
# Target definitions:
#   position_error_m
#       = Haversine(S GPS, V GPS)
#
#   velocity_error_m_s
#       = abs(S GPS speed - V velocity) / 3.6
#
# Experiment 8B:
#   - Robust SmoothL1 / Huber-style loss
#   - Target-aware weighting
#   - Same basic LSTM family
#
# DATA POLICY:
#   Raw data modified: NO
#   Stage 1-8 outputs modified: NO
#   Uncategorised dataset: NOT USED
#   Sequence leakage protection: ENABLED
#   Feature preprocessing: TRAIN ONLY
#   Target scaling: TRAIN ONLY
#
# ================================================================

import os
import re
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

SEQUENCE_LENGTH = 30
BATCH_SIZE = 256
EPOCHS = 20

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.20

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5

# Robust loss parameter in STANDARDIZED target space.
HUBER_BETA = 1.0

# Relative importance of the two targets.
POSITION_WEIGHT = 1.0
VELOCITY_WEIGHT = 1.0

# Target-aware weighting.
# Keep this mild. We do not want extreme samples to dominate.
TARGET_WEIGHT_STRENGTH = 0.20

# Number of rows used to fit feature preprocessing.
PREPROCESS_ROWS = 300_000

# Number of rows used to estimate target statistics.
TARGET_STATS_MAX_ROWS = 300_000

# Number of samples per sequence used for DataLoader iteration.
WINDOW_CHUNK = 20_000

NUM_WORKERS = 0


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(
    r"C:\Users\tuala\Downloads\member5_ai_drift_stage1\member5_ai_drift"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "features"
    / "categorised"
)

STAGE4_SPLIT = (
    PROJECT_ROOT
    / "results"
    / "stage4_reports"
    / "stage4_dataset_split.json"
)

RAW_CATEGORISED_ROOT = Path(
    r"C:\Users\tuala\Downloads\Synchronised V abd S datasets"
    r"\\Synchronised V abd S datasets"
    r"\\Categorised IOVNB Dataset"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage8_improvement_8b"
)

MODEL_DIR = OUTPUT_ROOT / "models"
PRED_DIR = OUTPUT_ROOT / "predictions"
REPORT_DIR = OUTPUT_ROOT / "reports"


MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


BEST_MODEL_PATH = MODEL_DIR / "nav_shield_lstm_8b_best.pt"
PREDICTION_PATH = (
    PRED_DIR
    / "stage8_improvement_8b_test_predictions.csv"
)
METRICS_PATH = (
    REPORT_DIR
    / "stage8_improvement_8b_metrics.json"
)
HISTORY_PATH = (
    REPORT_DIR
    / "stage8_improvement_8b_training_history.csv"
)
SUMMARY_PATH = (
    REPORT_DIR
    / "stage8_improvement_8b_summary.txt"
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# DISPLAY
# ================================================================

print("=" * 70)
print("NAV-SHIELD STAGE 8 IMPROVEMENT — EXPERIMENT 8B")
print("=" * 70)

print("""
Experiment:
Robust temporal regression using SmoothL1/Huber-style loss
with mild target-aware weighting.

IMPORTANT:
Targets are CALCULATED from the Categorised S/V dataset.
They are NOT required to exist inside the Stage 3 feature CSV.

Original Stage 8 model will NOT be modified.
Experiment 8A model will NOT be modified.

Raw data will NOT be modified.
Previous stage outputs will NOT be modified.
Uncategorised dataset will NOT be used.
""")

print(f"Device         : {DEVICE}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Epochs         : {EPOCHS}")
print(f"Hidden size    : {HIDDEN_SIZE}")
print(f"LSTM layers    : {NUM_LAYERS}")
print(f"Dropout        : {DROPOUT}")
print(f"Learning rate  : {LEARNING_RATE}")
print(f"Weight decay   : {WEIGHT_DECAY}")
print(f"Huber beta     : {HUBER_BETA}")
print(f"Position weight: {POSITION_WEIGHT}")
print(f"Velocity weight: {VELOCITY_WEIGHT}")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def normalize_name(value):
    """
    Convert sequence/file names to a common representation.
    """
    value = str(value).strip().lower()
    value = value.replace(".csv", "")
    value = value.replace("_features", "")
    value = value.replace("s-", "")
    value = value.replace("v-", "")
    value = value.replace("s_", "")
    value = value.replace("v_", "")
    return value


def clean_columns(df):
    """
    Clean only column names in memory.
    Raw files are never modified.
    """
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]
    return df


def read_csv_safe(path, nrows=None):
    """
    Read CSV with encoding fallback.
    This handles degree / micro symbols present in the raw data.
    """
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                low_memory=False,
                nrows=nrows
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def find_column(columns, candidates):
    """
    Flexible column matching.
    """
    normalized = {}

    for c in columns:
        key = re.sub(
            r"[^a-z0-9]",
            "",
            str(c).lower()
        )
        normalized[key] = c

    for candidate in candidates:
        key = re.sub(
            r"[^a-z0-9]",
            "",
            candidate.lower()
        )

        if key in normalized:
            return normalized[key]

    # Partial matching
    for c in columns:
        c_norm = re.sub(
            r"[^a-z0-9]",
            "",
            str(c).lower()
        )

        for candidate in candidates:
            cand_norm = re.sub(
                r"[^a-z0-9]",
                "",
                candidate.lower()
            )

            if (
                cand_norm in c_norm
                or c_norm in cand_norm
            ):
                return c

    return None


# ================================================================
# RAW S/V FILE DISCOVERY
# ================================================================

def discover_raw_pairs():
    print()
    print("=" * 70)
    print("RAW CATEGORISED S/V FILE DISCOVERY")
    print("=" * 70)

    csv_files = list(
        RAW_CATEGORISED_ROOT.rglob("*.csv")
    )

    print(f"Total CSV files discovered: {len(csv_files)}")

    s_files = {}
    v_files = {}

    for path in csv_files:

        name = path.stem.strip()

        if name.lower().startswith("s-"):
            seq = normalize_name(name[2:])
            s_files[seq] = path

        elif name.lower().startswith("v-"):
            seq = normalize_name(name[2:])
            v_files[seq] = path

    pairs = {}

    for seq in sorted(
        set(s_files.keys()) & set(v_files.keys())
    ):
        pairs[seq] = (
            s_files[seq],
            v_files[seq]
        )

    print()
    print(f"S files           : {len(s_files)}")
    print(f"V files           : {len(v_files)}")
    print(f"Complete S/V pairs: {len(pairs)}")

    return pairs


RAW_PAIRS = discover_raw_pairs()


# ================================================================
# STAGE 4 SPLIT
# ================================================================

print()
print("=" * 70)
print("LOADING STAGE 4 SPLIT")
print("=" * 70)

if not STAGE4_SPLIT.exists():
    raise FileNotFoundError(
        f"Stage 4 split file not found:\n{STAGE4_SPLIT}"
    )

with open(
    STAGE4_SPLIT,
    "r",
    encoding="utf-8"
) as f:
    split_data = json.load(f)


TRAIN_IDS = [
    normalize_name(x)
    for x in split_data["train"]
]

VAL_IDS = [
    normalize_name(x)
    for x in split_data["validation"]
]

TEST_IDS = [
    normalize_name(x)
    for x in split_data["test"]
]


print()
print("SEQUENCE SPLIT")
print("-" * 70)

print(f"TRAIN      : {len(TRAIN_IDS)} sequences")
print(f"VALIDATION : {len(VAL_IDS)} sequences")
print(f"TEST       : {len(TEST_IDS)} sequences")


# ================================================================
# LEAKAGE CHECK
# ================================================================

train_set = set(TRAIN_IDS)
val_set = set(VAL_IDS)
test_set = set(TEST_IDS)

print()
print("=" * 70)
print("SEQUENCE LEAKAGE CHECK")
print("=" * 70)

print(
    f"TRAIN ↔ VALIDATION overlap: "
    f"{len(train_set & val_set)}"
)

print(
    f"TRAIN ↔ TEST overlap: "
    f"{len(train_set & test_set)}"
)

print(
    f"VALIDATION ↔ TEST overlap: "
    f"{len(val_set & test_set)}"
)

if (
    train_set & val_set
    or train_set & test_set
    or val_set & test_set
):
    raise RuntimeError(
        "SEQUENCE LEAKAGE DETECTED."
    )

print("[PASS] No sequence overlap detected.")


# ================================================================
# FEATURE FILE DISCOVERY
# ================================================================

print()
print("=" * 70)
print("FEATURE FILE DISCOVERY")
print("=" * 70)

feature_files = {}

for path in FEATURE_DIR.glob("*.csv"):
    seq = normalize_name(path.stem)
    feature_files[seq] = path

print(
    f"Feature files found: "
    f"{len(feature_files)}"
)

all_split_ids = (
    TRAIN_IDS +
    VAL_IDS +
    TEST_IDS
)

missing_features = [
    x for x in all_split_ids
    if x not in feature_files
]

if missing_features:
    print(
        "[WARNING] Missing feature files:"
    )

    for x in missing_features:
        print(f"  - {x}")

    raise RuntimeError(
        "Some Stage 4 sequences do not have "
        "Stage 3 feature files."
    )

print(
    "[PASS] All split sequences have "
    "feature files."
)


# ================================================================
# RAW PAIR VALIDATION
# ================================================================

missing_pairs = [
    x for x in all_split_ids
    if x not in RAW_PAIRS
]

if missing_pairs:
    print()
    print("[WARNING] Missing raw S/V pairs:")

    for x in missing_pairs:
        print(f"  - {x}")

    raise RuntimeError(
        "Some Stage 4 sequences do not have "
        "Categorised S/V pairs."
    )

print(
    "[PASS] All Stage 4 sequences have "
    "Categorised S/V pairs."
)


# ================================================================
# TARGET COLUMN IDENTIFICATION
# ================================================================

def identify_s_columns(df):
    """
    Identify S GPS columns using the actual
    Categorised dataset naming scheme.
    """

    lat = find_column(
        df.columns,
        [
            "GPS LATITUDE (degrees)",
            "GPS LATITUDE",
            "LATITUDE",
            "LAT"
        ]
    )

    lon = find_column(
        df.columns,
        [
            "GPS LONGITUDE (degrees)",
            "GPS LONGITUDE",
            "LONGITUDE",
            "LON",
            "LONG"
        ]
    )

    speed = find_column(
        df.columns,
        [
            "GPS SPEED (Kmh)",
            "GPS SPEED (kmh)",
            "GPS SPEED",
            "SPEED (Kmh)",
            "SPEED"
        ]
    )

    time_col = find_column(
        df.columns,
        [
            "TIME SINCE START (ms)",
            "TIME SINCE START",
            "TIME"
        ]
    )

    return lat, lon, speed, time_col


def identify_v_columns(df):
    """
    Identify V velocity/time columns.
    """

    velocity = find_column(
        df.columns,
        [
            "VELOCITY (km/h)",
            "VELOCITY",
            "Velocity (km/h)",
            "V VELOCITY",
            "SPEED (Kmh)",
            "SPEED"
        ]
    )

    lat = find_column(
        df.columns,
        [
            "GPS LATITUDE (degrees)",
            "GPS LATITUDE",
            "LATITUDE",
            "LAT"
        ]
    )

    lon = find_column(
        df.columns,
        [
            "GPS LONGITUDE (degrees)",
            "GPS LONGITUDE",
            "LONGITUDE",
            "LON",
            "LONG"
        ]
    )

    time_col = find_column(
        df.columns,
        [
            "TIME SINCE START (ms)",
            "TIME SINCE START",
            "TIME"
        ]
    )

    return lat, lon, velocity, time_col


# ================================================================
# HAVERSINE
# ================================================================

def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Vectorized Haversine distance in meters.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2.0
        *
        6371000.0
        *
        np.arcsin(np.sqrt(a))
    )


# ================================================================
# TARGET CALCULATION
# ================================================================

TARGET_CACHE = {}


def compute_targets(sequence_id):
    """
    Calculate the two Stage 8 targets from raw S/V files.

    IMPORTANT:
    The target columns do NOT need to exist in the
    Stage 3 feature CSV.
    """

    sequence_id = normalize_name(sequence_id)

    if sequence_id in TARGET_CACHE:
        return TARGET_CACHE[sequence_id]

    s_path, v_path = RAW_PAIRS[sequence_id]

    s_df = clean_columns(
        read_csv_safe(s_path)
    )

    v_df = clean_columns(
        read_csv_safe(v_path)
    )

    (
        s_lat_col,
        s_lon_col,
        s_speed_col,
        s_time_col
    ) = identify_s_columns(s_df)

    (
        v_lat_col,
        v_lon_col,
        v_velocity_col,
        v_time_col
    ) = identify_v_columns(v_df)

    if (
        s_lat_col is None
        or s_lon_col is None
    ):
        raise RuntimeError(
            f"Could not identify S GPS columns "
            f"for {sequence_id}.\n"
            f"S columns:\n"
            f"{list(s_df.columns)}"
        )

    if (
        v_lat_col is None
        or v_lon_col is None
    ):
        raise RuntimeError(
            f"Could not identify V GPS columns "
            f"for {sequence_id}.\n"
            f"V columns:\n"
            f"{list(v_df.columns)}"
        )

    if s_speed_col is None:
        raise RuntimeError(
            f"Could not identify S GPS speed "
            f"column for {sequence_id}."
        )

    if v_velocity_col is None:
        raise RuntimeError(
            f"Could not identify V velocity "
            f"column for {sequence_id}."
        )

    s_lat = pd.to_numeric(
        s_df[s_lat_col],
        errors="coerce"
    ).to_numpy()

    s_lon = pd.to_numeric(
        s_df[s_lon_col],
        errors="coerce"
    ).to_numpy()

    s_speed = pd.to_numeric(
        s_df[s_speed_col],
        errors="coerce"
    ).to_numpy()

    v_lat = pd.to_numeric(
        v_df[v_lat_col],
        errors="coerce"
    ).to_numpy()

    v_lon = pd.to_numeric(
        v_df[v_lon_col],
        errors="coerce"
    ).to_numpy()

    v_velocity = pd.to_numeric(
        v_df[v_velocity_col],
        errors="coerce"
    ).to_numpy()

    n = min(
        len(s_df),
        len(v_df)
    )

    s_lat = s_lat[:n]
    s_lon = s_lon[:n]
    s_speed = s_speed[:n]

    v_lat = v_lat[:n]
    v_lon = v_lon[:n]
    v_velocity = v_velocity[:n]

    position_error = haversine_m(
        s_lat,
        s_lon,
        v_lat,
        v_lon
    )

    velocity_error = (
        np.abs(
            s_speed - v_velocity
        )
        / 3.6
    )

    position_error[
        ~np.isfinite(position_error)
    ] = np.nan

    velocity_error[
        ~np.isfinite(velocity_error)
    ] = np.nan

    targets = pd.DataFrame(
        {
            "position_error_m":
                position_error,
            "velocity_error_m_s":
                velocity_error
        }
    )

    TARGET_CACHE[sequence_id] = targets

    return targets


# ================================================================
# FEATURE COLUMN SELECTION
# ================================================================

print()
print("=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

sample_id = TRAIN_IDS[0]

sample_df = clean_columns(
    read_csv_safe(
        feature_files[sample_id]
    )
)

numeric_columns = []

for c in sample_df.columns:

    if pd.api.types.is_numeric_dtype(
        sample_df[c]
    ):
        numeric_columns.append(c)

# Targets are calculated separately.
# Therefore we only exclude obvious target columns if present.
exclude_columns = {
    "position_error_m",
    "velocity_error_m_s"
}

numeric_columns = [
    c for c in numeric_columns
    if c not in exclude_columns
]

print(
    f"Sample sequence: "
    f"{sample_id}_features"
)

print(
    f"Total columns: "
    f"{len(sample_df.columns)}"
)

print(
    f"Numeric feature candidates: "
    f"{len(numeric_columns)}"
)


# ================================================================
# FIND COMMON FEATURES ACROSS ALL SEQUENCES
# ================================================================

common_features = set(
    numeric_columns
)

for sequence_id in all_split_ids:

    df = clean_columns(
        read_csv_safe(
            feature_files[sequence_id],
            nrows=5
        )
    )

    numeric = set(
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(
            df[c]
        )
    )

    numeric -= exclude_columns

    common_features &= numeric


FEATURE_COLUMNS = sorted(
    common_features
)

if not FEATURE_COLUMNS:
    raise RuntimeError(
        "No common numeric features found."
    )

print(
    f"Common numeric features: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Input features used: "
    f"{len(FEATURE_COLUMNS)}"
)


# ================================================================
# TRAIN-ONLY FEATURE PREPROCESSING
# ================================================================

print()
print("=" * 70)
print("TRAIN-ONLY FEATURE PREPROCESSING")
print("=" * 70)

feature_scaler = StandardScaler()

rows_seen = 0

for idx, sequence_id in enumerate(
    TRAIN_IDS,
    start=1
):

    df = clean_columns(
        read_csv_safe(
            feature_files[sequence_id]
        )
    )

    X = df[
        FEATURE_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    remaining = (
        PREPROCESS_ROWS - rows_seen
    )

    if remaining <= 0:
        break

    X = X.iloc[
        :remaining
    ]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    # If an entire column is NaN,
    # use zero for fitting purposes.
    X = X.fillna(0.0)

    feature_scaler.partial_fit(
        X.to_numpy(
            dtype=np.float64
        )
    )

    rows_seen += len(X)

    if idx % 10 == 0:
        print(
            f"  TRAIN sequences processed: "
            f"{idx}/{len(TRAIN_IDS)}"
        )

print(
    f"Rows used for preprocessing: "
    f"{rows_seen:,}"
)

print(
    "[PASS] Feature preprocessing fitted "
    "using TRAIN only."
)


# ================================================================
# TRAIN-ONLY TARGET SCALING
# ================================================================

print()
print("=" * 70)
print("TRAIN-ONLY TARGET SCALING")
print("=" * 70)

target_scaler = StandardScaler()

target_rows = []

for idx, sequence_id in enumerate(
    TRAIN_IDS,
    start=1
):

    targets = compute_targets(
        sequence_id
    )

    valid = targets[
        [
            "position_error_m",
            "velocity_error_m_s"
        ]
    ].dropna()

    if len(valid) > 0:
        target_rows.append(
            valid.to_numpy()
        )

    if idx % 10 == 0:
        print(
            f"  TRAIN targets processed: "
            f"{idx}/{len(TRAIN_IDS)}"
        )


if not target_rows:
    raise RuntimeError(
        "No valid training targets found."
    )


target_array = np.vstack(
    target_rows
)

if len(target_array) > TARGET_STATS_MAX_ROWS:
    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(target_array),
        TARGET_STATS_MAX_ROWS,
        replace=False
    )

    target_array = target_array[
        indices
    ]


target_scaler.fit(
    target_array
)

print()
print(
    f"Position target mean: "
    f"{target_scaler.mean_[0]:.6f}"
)

print(
    f"Position target std : "
    f"{target_scaler.scale_[0]:.6f}"
)

print(
    f"Velocity target mean: "
    f"{target_scaler.mean_[1]:.6f}"
)

print(
    f"Velocity target std : "
    f"{target_scaler.scale_[1]:.6f}"
)

print(
    "[PASS] Target scaling fitted "
    "using TRAIN only."
)


# ================================================================
# FEATURE + TARGET PREPARATION
# ================================================================

def load_sequence_data(sequence_id):
    """
    Load one sequence.

    Returns:
        X_scaled
        y_scaled
        y_original
    """

    feature_df = clean_columns(
        read_csv_safe(
            feature_files[sequence_id]
        )
    )

    targets = compute_targets(
        sequence_id
    )

    n = min(
        len(feature_df),
        len(targets)
    )

    feature_df = feature_df.iloc[
        :n
    ]

    targets = targets.iloc[
        :n
    ].reset_index(drop=True)

    X = feature_df[
        FEATURE_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # TRAIN-derived statistics only.
    # StandardScaler cannot directly impute.
    # Use TRAIN medians estimated from the same
    # training data.
    #
    # For simplicity and consistency with Stage 8,
    # use zero after scaling for remaining NaNs.
    #
    # This is safe because scaler statistics are
    # fitted from TRAIN only.
    X = X.fillna(0.0)

    X_scaled = feature_scaler.transform(
        X.to_numpy(
            dtype=np.float32
        )
    ).astype(
        np.float32
    )

    y_original = targets[
        [
            "position_error_m",
            "velocity_error_m_s"
        ]
    ].to_numpy(
        dtype=np.float32
    )

    valid_mask = np.isfinite(
        y_original
    ).all(axis=1)

    X_scaled = X_scaled[
        valid_mask
    ]

    y_original = y_original[
        valid_mask
    ]

    y_scaled = target_scaler.transform(
        y_original
    ).astype(
        np.float32
    )

    return (
        X_scaled,
        y_scaled,
        y_original
    )


# ================================================================
# LAZY WINDOW DATASET
# ================================================================

class SequenceWindowDataset(
    IterableDataset
):

    def __init__(
        self,
        sequence_ids,
        return_original=False
    ):
        super().__init__()

        self.sequence_ids = sequence_ids
        self.return_original = (
            return_original
        )

    def __iter__(self):

        for sequence_id in self.sequence_ids:

            X, y_scaled, y_original = (
                load_sequence_data(
                    sequence_id
                )
            )

            n = len(X)

            if n <= SEQUENCE_LENGTH:
                continue

            for start in range(
                0,
                n - SEQUENCE_LENGTH
            ):

                end = (
                    start
                    + SEQUENCE_LENGTH
                )

                x_window = X[
                    start:end
                ]

                # Predict target at the final
                # time point of the window.
                target = y_scaled[
                    end - 1
                ]

                original_target = (
                    y_original[
                        end - 1
                    ]
                )

                if self.return_original:

                    yield (
                        torch.tensor(
                            x_window,
                            dtype=torch.float32
                        ),
                        torch.tensor(
                            target,
                            dtype=torch.float32
                        ),
                        torch.tensor(
                            original_target,
                            dtype=torch.float32
                        )
                    )

                else:

                    yield (
                        torch.tensor(
                            x_window,
                            dtype=torch.float32
                        ),
                        torch.tensor(
                            target,
                            dtype=torch.float32
                        )
                    )


# ================================================================
# WINDOW COUNT
# ================================================================

def count_windows(sequence_ids):

    total = 0

    for sequence_id in sequence_ids:

        targets = compute_targets(
            sequence_id
        )

        valid = targets[
            [
                "position_error_m",
                "velocity_error_m_s"
            ]
        ].notna().all(axis=1)

        n = int(valid.sum())

        if n > SEQUENCE_LENGTH:
            total += (
                n - SEQUENCE_LENGTH
            )

    return total


print()
print("=" * 70)
print("WINDOW COUNT")
print("=" * 70)

train_windows = count_windows(
    TRAIN_IDS
)

val_windows = count_windows(
    VAL_IDS
)

test_windows = count_windows(
    TEST_IDS
)

print(
    f"Train windows      : "
    f"{train_windows:,}"
)

print(
    f"Validation windows : "
    f"{val_windows:,}"
)

print(
    f"Test windows       : "
    f"{test_windows:,}"
)


# ================================================================
# MODEL
# ================================================================

class RobustLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.20
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.fc1 = nn.Linear(
            hidden_size,
            64
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            64,
            2
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last = output[:, -1, :]

        last = self.norm(last)

        x = self.fc1(last)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


model = RobustLSTM(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print(model)


# ================================================================
# ROBUST TARGET-AWARE LOSS
# ================================================================

class RobustWeightedLoss(
    nn.Module
):

    def __init__(
        self,
        beta=1.0,
        position_weight=1.0,
        velocity_weight=1.0,
        target_weight_strength=0.20
    ):
        super().__init__()

        self.beta = beta
        self.position_weight = (
            position_weight
        )
        self.velocity_weight = (
            velocity_weight
        )
        self.target_weight_strength = (
            target_weight_strength
        )

    def forward(
        self,
        prediction,
        target
    ):

        # Element-wise SmoothL1
        loss = nn.functional.smooth_l1_loss(
            prediction,
            target,
            beta=self.beta,
            reduction="none"
        )

        # --------------------------------------------------------
        # Target-aware weighting
        #
        # Higher standardized target magnitude receives
        # slightly higher weight.
        #
        # Weight is clipped to prevent large errors
        # dominating the entire training process.
        # --------------------------------------------------------

        magnitude = torch.abs(target)

        weight = (
            1.0
            +
            self.target_weight_strength
            *
            torch.clamp(
                magnitude,
                0.0,
                3.0
            )
        )

        loss = loss * weight

        # Separate target importance.
        loss_position = (
            loss[:, 0]
            * self.position_weight
        )

        loss_velocity = (
            loss[:, 1]
            * self.velocity_weight
        )

        return (
            loss_position.mean()
            +
            loss_velocity.mean()
        ) / 2.0


criterion = RobustWeightedLoss(
    beta=HUBER_BETA,
    position_weight=POSITION_WEIGHT,
    velocity_weight=VELOCITY_WEIGHT,
    target_weight_strength=(
        TARGET_WEIGHT_STRENGTH
    )
)


# ================================================================
# OPTIMIZER
# ================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ================================================================
# DATALOADERS
# ================================================================

train_dataset = SequenceWindowDataset(
    TRAIN_IDS
)

val_dataset = SequenceWindowDataset(
    VAL_IDS
)

test_dataset = SequenceWindowDataset(
    TEST_IDS,
    return_original=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)


# ================================================================
# EVALUATION FUNCTION
# ================================================================

def evaluate_loss(loader):

    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            pred = model(X)

            loss = criterion(
                pred,
                y
            )

            batch_size = X.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_count += batch_size

    if total_count == 0:
        return float("inf")

    return (
        total_loss
        /
        total_count
    )


# ================================================================
# TRAINING
# ================================================================

print()
print("=" * 70)
print("TRAINING EXPERIMENT 8B")
print("=" * 70)

history = []

best_val_loss = float("inf")

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    sample_count = 0

    for X, y in train_loader:

        X = X.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(X)

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        # Gradient clipping improves
        # recurrent training stability.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = X.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        sample_count += batch_size

    train_loss = (
        running_loss
        /
        max(sample_count, 1)
    )

    val_loss = evaluate_loss(
        val_loader
    )

    scheduler.step(
        val_loss
    )

    current_lr = optimizer.param_groups[0][
        "lr"
    ]

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"LR: {current_lr:.7f}"
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": current_lr
        }
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "feature_columns":
                    FEATURE_COLUMNS,

                "sequence_length":
                    SEQUENCE_LENGTH,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT,

                "target_scaler_mean":
                    target_scaler.mean_.tolist(),

                "target_scaler_scale":
                    target_scaler.scale_.tolist(),

                "experiment":
                    "8B_ROBUST_TARGET_LOSS"
            },
            BEST_MODEL_PATH
        )

        print(
            "  [SAVED] New best validation model."
        )


# ================================================================
# SAVE TRAINING HISTORY
# ================================================================

pd.DataFrame(
    history
).to_csv(
    HISTORY_PATH,
    index=False
)


# ================================================================
# LOAD BEST MODEL
# ================================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "[PASS] Best Experiment 8B model loaded."
)


# ================================================================
# TEST PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

all_actual = []
all_predicted = []

with torch.no_grad():

    for X, y_scaled, y_original in test_loader:

        X = X.to(
            DEVICE,
            non_blocking=True
        )

        pred_scaled = model(X)

        pred_scaled = (
            pred_scaled
            .cpu()
            .numpy()
        )

        y_original_np = (
            y_original
            .numpy()
        )

        predicted_original = (
            target_scaler.inverse_transform(
                pred_scaled
            )
        )

        all_actual.append(
            y_original_np
        )

        all_predicted.append(
            predicted_original
        )


actual = np.vstack(
    all_actual
)

predicted = np.vstack(
    all_predicted
)


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    actual,
    predicted
):

    error = (
        predicted - actual
    )

    abs_error = np.abs(
        error
    )

    mae = float(
        np.mean(abs_error)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    return mae, rmse


position_mae, position_rmse = (
    calculate_metrics(
        actual[:, 0],
        predicted[:, 0]
    )
)

velocity_mae, velocity_rmse = (
    calculate_metrics(
        actual[:, 1],
        predicted[:, 1]
    )
)


print()
print(
    "EXPERIMENT 8B TEST METRICS"
)

print("-" * 70)

print(
    f"Position Error MAE  : "
    f"{position_mae:.6f} m"
)

print(
    f"Position Error RMSE : "
    f"{position_rmse:.6f} m"
)

print(
    f"Velocity Error MAE  : "
    f"{velocity_mae:.6f} m/s"
)

print(
    f"Velocity Error RMSE : "
    f"{velocity_rmse:.6f} m/s"
)


# ================================================================
# SAVE PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame(
    {
        "actual_position_error_m":
            actual[:, 0],

        "predicted_position_error_m":
            predicted[:, 0],

        "actual_velocity_error_m_s":
            actual[:, 1],

        "predicted_velocity_error_m_s":
            predicted[:, 1]
    }
)

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


# ================================================================
# METRICS JSON
# ================================================================

metrics = {

    "stage":
        "Stage 8 Improvement 8B",

    "experiment":
        "ROBUST_TARGET_LOSS",

    "model":
        "LSTM",

    "learning_type":
        "SUPERVISED_TEMPORAL_REGRESSION",

    "sequence_length":
        SEQUENCE_LENGTH,

    "features":
        len(FEATURE_COLUMNS),

    "train_sequences":
        len(TRAIN_IDS),

    "validation_sequences":
        len(VAL_IDS),

    "test_sequences":
        len(TEST_IDS),

    "train_windows":
        train_windows,

    "validation_windows":
        val_windows,

    "test_windows":
        test_windows,

    "best_validation_loss":
        float(best_val_loss),

    "test_metrics":
        {
            "position_error_m":
                {
                    "MAE":
                        position_mae,

                    "RMSE":
                        position_rmse
                },

            "velocity_error_m_s":
                {
                    "MAE":
                        velocity_mae,

                    "RMSE":
                        velocity_rmse
                }
        },

    "experiment_parameters":
        {
            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "huber_beta":
                HUBER_BETA,

            "position_weight":
                POSITION_WEIGHT,

            "velocity_weight":
                VELOCITY_WEIGHT,

            "target_weight_strength":
                TARGET_WEIGHT_STRENGTH
        },

    "target_definition":
        {
            "position_error_m":
                "Haversine(S GPS, V GPS)",

            "velocity_error_m_s":
                "abs(S GPS speed - V velocity) / 3.6"
        },

    "target_source":
        "Categorised S/V dataset",

    "preprocessing":
        "TRAIN ONLY",

    "target_scaling":
        "TRAIN ONLY",

    "sequence_leakage_protection":
        True,

    "lazy_window_generation":
        True,

    "raw_data_modified":
        False,

    "previous_stages_modified":
        False,

    "uncategorised_dataset_used":
        False,

    "model_path":
        str(BEST_MODEL_PATH),

    "prediction_path":
        str(PREDICTION_PATH)
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ================================================================
# SUMMARY
# ================================================================

summary = f"""
NAV-SHIELD STAGE 8 IMPROVEMENT — EXPERIMENT 8B
============================================================

Experiment:
Robust target-loss LSTM

Model:
LSTM + LayerNorm + Dense head

Loss:
SmoothL1 / Huber-style loss

Target-aware weighting:
{TARGET_WEIGHT_STRENGTH}

Target definitions:

position_error_m:
    Haversine(S GPS, V GPS)

velocity_error_m_s:
    abs(S GPS speed - V velocity) / 3.6


DATA
------------------------------------------------------------
Train sequences       : {len(TRAIN_IDS)}
Validation sequences  : {len(VAL_IDS)}
Test sequences        : {len(TEST_IDS)}

Features              : {len(FEATURE_COLUMNS)}

Train windows         : {train_windows}
Validation windows    : {val_windows}
Test windows          : {test_windows}


TEST RESULTS
------------------------------------------------------------

Position Error:
    MAE  : {position_mae:.6f} m
    RMSE : {position_rmse:.6f} m

Velocity Error:
    MAE  : {velocity_mae:.6f} m/s
    RMSE : {velocity_rmse:.6f} m/s


TRAINING
------------------------------------------------------------
Best validation loss  : {best_val_loss:.6f}

Hidden size           : {HIDDEN_SIZE}
LSTM layers           : {NUM_LAYERS}
Dropout               : {DROPOUT}
Learning rate         : {LEARNING_RATE}
Weight decay          : {WEIGHT_DECAY}
Huber beta            : {HUBER_BETA}


DATA POLICY
------------------------------------------------------------
Raw data modified             : NO
Previous stages modified     : NO
Uncategorised dataset used   : NO
Sequence leakage protection  : ENABLED
Feature preprocessing        : TRAIN ONLY
Target scaling               : TRAIN ONLY
Targets calculated from S/V  : YES


MODEL
------------------------------------------------------------
{BEST_MODEL_PATH}

PREDICTIONS
------------------------------------------------------------
{PREDICTION_PATH}

METRICS
------------------------------------------------------------
{METRICS_PATH}

TRAINING HISTORY
------------------------------------------------------------
{HISTORY_PATH}
"""


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        summary.strip()
        + "\n"
    )


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 70)
print("STAGE 8 IMPROVEMENT — EXPERIMENT 8B COMPLETE")
print("=" * 70)

print(
    f"Position Error Test MAE : "
    f"{position_mae:.6f} m"
)

print(
    f"Position Error Test RMSE: "
    f"{position_rmse:.6f} m"
)

print(
    f"Velocity Error Test MAE : "
    f"{velocity_mae:.6f} m/s"
)

print(
    f"Velocity Error Test RMSE: "
    f"{velocity_rmse:.6f} m/s"
)

print()
print("Model:")
print(BEST_MODEL_PATH)

print()
print("Predictions:")
print(PREDICTION_PATH)

print()
print("Metrics:")
print(METRICS_PATH)

print()
print("Training history:")
print(HISTORY_PATH)

print()
print("Summary:")
print(SUMMARY_PATH)

print()
print("Uncategorised dataset: NOT USED")
print("Raw data modified: NO")
print("Previous stages modified: NO")
print("Sequence leakage protection: ENABLED")
print("Targets calculated from Categorised S/V: YES")

print("=" * 70)