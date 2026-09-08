# ======================================================================
# NAV-SHIELD STAGE 8 IMPROVEMENT — EXPERIMENT 8A
# ======================================================================
#
# Improved LSTM architecture + regularization
#
# IMPORTANT:
# - Original Stage 8 model is NOT modified.
# - Raw data is NOT modified.
# - Stage 1-8 outputs are NOT modified.
# - Uncategorised dataset is NOT used.
# - Train/validation/test sequence separation is preserved.
# - Feature preprocessing is TRAIN ONLY.
# - Target scaling is TRAIN ONLY.
#
# Targets:
#   position_error_m
#       = Haversine(S GPS, V GPS)
#
#   velocity_error_m_s
#       = |S GPS speed - V velocity| / 3.6
#
# ======================================================================

import os
import re
import json
import math
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

SEQ_LEN = 30
BATCH_SIZE = 256
EPOCHS = 20

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.25

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5

MAX_TRAIN_ROWS_FOR_PREPROCESSING = 300000

PROJECT_ROOT = r"C:\Users\tuala\Downloads\member5_ai_drift_stage1\member5_ai_drift"

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "features",
    "categorised"
)

STAGE4_SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage4_reports",
    "stage4_dataset_split.json"
)

RAW_CATEGORISED_ROOT = (
    r"C:\Users\tuala\Downloads\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage8_improvement"
)

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

warnings.filterwarnings("ignore")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# PRINT HEADER
# ======================================================================

print("=" * 70)
print("NAV-SHIELD STAGE 8 IMPROVEMENT — EXPERIMENT 8A")
print("=" * 70)

print()
print("Experiment:")
print("Improved LSTM architecture + regularization")

print()
print("Original Stage 8 model will NOT be modified.")
print("Raw data will NOT be modified.")
print("Stage 1-8 outputs will NOT be modified.")
print("Uncategorised dataset will NOT be used.")

print()
print(f"Device         : {DEVICE}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Epochs         : {EPOCHS}")
print(f"Hidden size    : {HIDDEN_SIZE}")
print(f"LSTM layers    : {NUM_LAYERS}")
print(f"Dropout        : {DROPOUT}")
print(f"Learning rate  : {LEARNING_RATE}")
print(f"Weight decay   : {WEIGHT_DECAY}")


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def normalize_column_name(name):
    """
    Normalize messy CSV column names.

    Handles:
      - leading/trailing spaces
      - UTF-8/Latin-1 mojibake
      - degree / squared / micro symbols
      - repeated whitespace
    """

    if name is None:
        return ""

    s = str(name)

    # Common encoding artifacts
    replacements = {
        "Â°": "deg",
        "Â²": "2",
        "Â³": "3",
        "Âµ": "u",
        "Î¼": "u",
        "Ã‚Â°": "deg",
        "Ã‚Â²": "2",
        "Ã‚Âµ": "u",
        "μ": "u",
        "µ": "u",
        "°": "deg",
        "²": "2",
        "³": "3",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = s.strip()
    s = re.sub(r"\s+", " ", s)

    return s.lower()


def read_csv_robust(path):
    """
    Read CSV using encoding fallbacks.
    Does NOT modify the source file.
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
                low_memory=False
            )
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def find_column(df, patterns):
    """
    Find a column based on normalized names.
    """

    normalized = {
        normalize_column_name(c): c
        for c in df.columns
    }

    # Exact matches first
    for pattern in patterns:
        p = normalize_column_name(pattern)

        if p in normalized:
            return normalized[p]

    # Partial matches
    for norm_name, original_name in normalized.items():

        for pattern in patterns:

            p = normalize_column_name(pattern)

            if p in norm_name:
                return original_name

    return None


def haversine(lat1, lon1, lat2, lon2):
    """
    Haversine distance in meters.
    """

    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    r = 6371000.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)

    a = (
        np.sin(dp / 2.0) ** 2
        + np.cos(p1)
        * np.cos(p2)
        * np.sin(dl / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return 2.0 * r * np.arcsin(np.sqrt(a))


# ======================================================================
# DISCOVER RAW S/V FILES
# ======================================================================

def discover_sv_files():

    print()
    print("=" * 70)
    print("RAW CATEGORISED S/V FILE DISCOVERY")
    print("=" * 70)

    csv_files = []

    for root, _, files in os.walk(RAW_CATEGORISED_ROOT):

        for f in files:

            if f.lower().endswith(".csv"):

                csv_files.append(
                    os.path.join(root, f)
                )

    print(f"Total CSV files discovered: {len(csv_files)}")

    s_files = {}
    v_files = {}

    for path in csv_files:

        base = os.path.basename(path)

        name = os.path.splitext(base)[0]

        norm = normalize_column_name(name)

        if norm.startswith("s-"):

            sequence = norm[2:].replace("-", "")

            s_files[sequence] = path

        elif norm.startswith("v-"):

            sequence = norm[2:].replace("-", "")

            v_files[sequence] = path

    pairs = {}

    for key in s_files:

        if key in v_files:

            pairs[key] = (
                s_files[key],
                v_files[key]
            )

    print()
    print(f"S files: {len(s_files)}")
    print(f"V files: {len(v_files)}")
    print(f"Complete S/V pairs: {len(pairs)}")

    return pairs


# ======================================================================
# LOAD SPLIT
# ======================================================================

print()
print("=" * 70)
print("LOADING STAGE 4 SPLIT")
print("=" * 70)

with open(
    STAGE4_SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


TRAIN_IDS = split_data["train"]
VAL_IDS = split_data["validation"]
TEST_IDS = split_data["test"]

print()
print(f"TRAIN      : {len(TRAIN_IDS)} sequences")
print(f"VALIDATION : {len(VAL_IDS)} sequences")
print(f"TEST       : {len(TEST_IDS)} sequences")


# ======================================================================
# LEAKAGE CHECK
# ======================================================================

print()
print("=" * 70)
print("SEQUENCE LEAKAGE CHECK")
print("=" * 70)

train_set = set(TRAIN_IDS)
val_set = set(VAL_IDS)
test_set = set(TEST_IDS)

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
        "Sequence leakage detected."
    )

print("[PASS] No sequence overlap detected.")


# ======================================================================
# FEATURE FILE DISCOVERY
# ======================================================================

print()
print("=" * 70)
print("FEATURE FILE DISCOVERY")
print("=" * 70)

feature_files = {}

for file in os.listdir(FEATURE_DIR):

    if not file.lower().endswith(".csv"):
        continue

    path = os.path.join(
        FEATURE_DIR,
        file
    )

    sequence_id = os.path.splitext(file)[0]

    if sequence_id.endswith("_features"):

        sequence_id = sequence_id[:-9]

    feature_files[
        normalize_column_name(sequence_id)
    ] = path


print(
    f"Feature files found: "
    f"{len(feature_files)}"
)


def get_feature_path(sequence_id):

    key = normalize_column_name(sequence_id)

    if key in feature_files:
        return feature_files[key]

    key2 = key.replace("_features", "")

    for k, v in feature_files.items():

        if k.replace("_features", "") == key2:
            return v

    return None


all_sequences = (
    TRAIN_IDS
    + VAL_IDS
    + TEST_IDS
)

missing_features = []

for sid in all_sequences:

    if get_feature_path(sid) is None:

        missing_features.append(sid)


if missing_features:

    print()
    print("[ERROR] Missing feature files:")

    for sid in missing_features:
        print(f"  - {sid}")

    raise RuntimeError(
        "Some Stage 4 sequences do not have "
        "feature files."
    )

print(
    "[PASS] All split sequences have feature files."
)


# ======================================================================
# RAW S/V DISCOVERY
# ======================================================================

SV_PAIRS = discover_sv_files()


# ======================================================================
# ROBUST S/V PAIR LOOKUP
# ======================================================================

def get_sv_pair(sequence_id):

    target = normalize_column_name(sequence_id)

    target = target.replace(
        "_features",
        ""
    )

    target = target.replace(
        "-",
        ""
    )

    for key, pair in SV_PAIRS.items():

        clean_key = normalize_column_name(key)

        clean_key = clean_key.replace(
            "-",
            ""
        )

        if clean_key == target:

            return pair

    return None


missing_pairs = []

for sid in all_sequences:

    if get_sv_pair(sid) is None:

        missing_pairs.append(sid)


print()
print("=" * 70)
print("S/V PAIR VALIDATION")
print("=" * 70)

print(
    f"Stage 4 sequences : {len(all_sequences)}"
)

print(
    f"Raw S/V pairs     : {len(SV_PAIRS)}"
)

print(
    f"Missing pairs     : {len(missing_pairs)}"
)

if missing_pairs:

    for sid in missing_pairs:
        print(f"  - {sid}")

    raise RuntimeError(
        "Missing S/V pairs for Stage 4 sequences."
    )

print(
    "[PASS] All Stage 4 sequences have "
    "Categorised S/V pairs."
)


# ======================================================================
# TARGET COMPUTATION
# ======================================================================

TARGET_1 = "position_error_m"
TARGET_2 = "velocity_error_m_s"


def compute_targets(sequence_id):

    pair = get_sv_pair(sequence_id)

    if pair is None:

        raise RuntimeError(
            f"S/V pair not found for {sequence_id}"
        )

    s_path, v_path = pair

    s_df = read_csv_robust(s_path)
    v_df = read_csv_robust(v_path)

    # --------------------------------------------------------------
    # Normalize column names WITHOUT changing original data files.
    # --------------------------------------------------------------

    s_norm = {
        normalize_column_name(c): c
        for c in s_df.columns
    }

    v_norm = {
        normalize_column_name(c): c
        for c in v_df.columns
    }

    # --------------------------------------------------------------
    # S GPS latitude
    # --------------------------------------------------------------

    s_lat = find_column(
        s_df,
        [
            "GPS LATITUDE (degrees)",
            "GPS LATITUDE",
            "LATITUDE"
        ]
    )

    # --------------------------------------------------------------
    # S GPS longitude
    # --------------------------------------------------------------

    s_lon = find_column(
        s_df,
        [
            "GPS LONGITUDE (degrees)",
            "GPS LONGITUDE",
            "LONGITUDE"
        ]
    )

    # --------------------------------------------------------------
    # V GPS latitude
    # --------------------------------------------------------------

    v_lat = find_column(
        v_df,
        [
            "GPS LATITUDE (degrees)",
            "GPS LATITUDE",
            "LATITUDE"
        ]
    )

    # --------------------------------------------------------------
    # V GPS longitude
    # --------------------------------------------------------------

    v_lon = find_column(
        v_df,
        [
            "GPS LONGITUDE (degrees)",
            "GPS LONGITUDE",
            "LONGITUDE"
        ]
    )

    # --------------------------------------------------------------
    # S GPS speed
    # --------------------------------------------------------------

    s_speed = find_column(
        s_df,
        [
            "GPS SPEED (Kmh)",
            "GPS SPEED (kmh)",
            "GPS SPEED",
            "SPEED"
        ]
    )

    # --------------------------------------------------------------
    # V velocity
    #
    # We deliberately search several likely names because
    # the source CSVs can use slightly different labels.
    # --------------------------------------------------------------

    v_velocity = find_column(
        v_df,
        [
            "VELOCITY (Kmh)",
            "VELOCITY (kmh)",
            "VELOCITY",
            "SPEED (Kmh)",
            "SPEED (kmh)",
            "SPEED",
            "GPS SPEED (Kmh)",
            "GPS SPEED"
        ]
    )

    # --------------------------------------------------------------
    # DEBUG INFORMATION IF SOMETHING IS MISSING
    # --------------------------------------------------------------

    missing = []

    if s_lat is None:
        missing.append("S GPS latitude")

    if s_lon is None:
        missing.append("S GPS longitude")

    if v_lat is None:
        missing.append("V GPS latitude")

    if v_lon is None:
        missing.append("V GPS longitude")

    if s_speed is None:
        missing.append("S GPS speed")

    if v_velocity is None:
        missing.append("V velocity")

    if missing:

        print()
        print(
            f"[ERROR] Could not identify target columns "
            f"for {sequence_id}"
        )

        print()
        print("S columns:")

        for c in s_df.columns:
            print(f"  {repr(c)}")

        print()
        print("V columns:")

        for c in v_df.columns:
            print(f"  {repr(c)}")

        print()
        print("Missing:")

        for m in missing:
            print(f"  - {m}")

        raise RuntimeError(
            f"Required target columns could not be "
            f"identified for {sequence_id}."
        )

    # --------------------------------------------------------------
    # Extract numeric arrays
    # --------------------------------------------------------------

    s_lat_values = pd.to_numeric(
        s_df[s_lat],
        errors="coerce"
    ).to_numpy()

    s_lon_values = pd.to_numeric(
        s_df[s_lon],
        errors="coerce"
    ).to_numpy()

    v_lat_values = pd.to_numeric(
        v_df[v_lat],
        errors="coerce"
    ).to_numpy()

    v_lon_values = pd.to_numeric(
        v_df[v_lon],
        errors="coerce"
    ).to_numpy()

    s_speed_values = pd.to_numeric(
        s_df[s_speed],
        errors="coerce"
    ).to_numpy()

    v_velocity_values = pd.to_numeric(
        v_df[v_velocity],
        errors="coerce"
    ).to_numpy()

    # --------------------------------------------------------------
    # Align lengths.
    #
    # Stage 3 synchronization already gives aligned S/V samples,
    # but we safely use the common length.
    # --------------------------------------------------------------

    n = min(
        len(s_lat_values),
        len(s_lon_values),
        len(v_lat_values),
        len(v_lon_values),
        len(s_speed_values),
        len(v_velocity_values)
    )

    if n <= 0:

        raise RuntimeError(
            f"No usable rows for {sequence_id}"
        )

    s_lat_values = s_lat_values[:n]
    s_lon_values = s_lon_values[:n]

    v_lat_values = v_lat_values[:n]
    v_lon_values = v_lon_values[:n]

    s_speed_values = s_speed_values[:n]
    v_velocity_values = v_velocity_values[:n]

    # --------------------------------------------------------------
    # Position error
    # --------------------------------------------------------------

    position_error = haversine(
        s_lat_values,
        s_lon_values,
        v_lat_values,
        v_lon_values
    )

    # --------------------------------------------------------------
    # Velocity error
    #
    # Source velocity is assumed to be km/h.
    # Convert difference to m/s.
    # --------------------------------------------------------------

    velocity_error = (
        np.abs(
            s_speed_values
            - v_velocity_values
        )
        / 3.6
    )

    # --------------------------------------------------------------
    # Clean invalid values
    # --------------------------------------------------------------

    position_error[
        ~np.isfinite(position_error)
    ] = np.nan

    velocity_error[
        ~np.isfinite(velocity_error)
    ] = np.nan

    return (
        position_error.astype(np.float32),
        velocity_error.astype(np.float32)
    )


# ======================================================================
# TARGET CACHE
# ======================================================================

TARGET_CACHE = {}


def get_targets(sequence_id):

    if sequence_id not in TARGET_CACHE:

        TARGET_CACHE[sequence_id] = compute_targets(
            sequence_id
        )

    return TARGET_CACHE[sequence_id]


# ======================================================================
# FEATURE COLUMN SELECTION
# ======================================================================

print()
print("=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

sample_id = TRAIN_IDS[0]

sample_path = get_feature_path(sample_id)

sample_df = read_csv_robust(
    sample_path
)

numeric_columns = []

for c in sample_df.columns:

    if pd.api.types.is_numeric_dtype(
        sample_df[c]
    ):

        numeric_columns.append(c)


print(
    f"Sample sequence: {sample_id}"
)

print(
    f"Total columns: {len(sample_df.columns)}"
)

print(
    f"Numeric feature candidates: "
    f"{len(numeric_columns)}"
)


# ======================================================================
# COMMON NUMERIC FEATURES
# ======================================================================

common_features = set(numeric_columns)

for sid in all_sequences[1:]:

    path = get_feature_path(sid)

    df = read_csv_robust(path)

    cols = set()

    for c in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[c]
        ):

            cols.add(c)

    common_features &= cols


FEATURE_COLUMNS = sorted(
    list(common_features)
)


if not FEATURE_COLUMNS:

    raise RuntimeError(
        "No common numeric features found."
    )


print(
    f"Input features used: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Common numeric features: "
    f"{len(FEATURE_COLUMNS)}"
)


# ======================================================================
# TRAIN-ONLY FEATURE PREPROCESSING
# ======================================================================

print()
print("=" * 70)
print("TRAIN-ONLY FEATURE PREPROCESSING")
print("=" * 70)

train_chunks = []

rows_collected = 0

for i, sid in enumerate(TRAIN_IDS):

    path = get_feature_path(sid)

    df = read_csv_robust(path)

    X = df[FEATURE_COLUMNS].copy()

    remaining = (
        MAX_TRAIN_ROWS_FOR_PREPROCESSING
        - rows_collected
    )

    if remaining <= 0:
        break

    if len(X) > remaining:

        X = X.iloc[:remaining]

    train_chunks.append(X)

    rows_collected += len(X)

    if (
        (i + 1) % 10 == 0
        or i == len(TRAIN_IDS) - 1
    ):

        print(
            f"  TRAIN sequences processed: "
            f"{i + 1}/{len(TRAIN_IDS)}"
        )


train_matrix = pd.concat(
    train_chunks,
    axis=0,
    ignore_index=True
)

print(
    f"Rows used for preprocessing: "
    f"{len(train_matrix):,}"
)

feature_imputer = SimpleImputer(
    strategy="median"
)

feature_scaler = StandardScaler()

X_imputed = feature_imputer.fit_transform(
    train_matrix
)

feature_scaler.fit(
    X_imputed
)

del train_chunks
del train_matrix
del X_imputed

print(
    "[PASS] Feature preprocessing fitted "
    "using TRAIN only."
)


# ======================================================================
# TRAIN-ONLY TARGET SCALING
# ======================================================================

print()
print("=" * 70)
print("TRAIN-ONLY TARGET SCALING")
print("=" * 70)

train_position_values = []
train_velocity_values = []

for i, sid in enumerate(TRAIN_IDS):

    pos, vel = get_targets(sid)

    p = pos[np.isfinite(pos)]
    v = vel[np.isfinite(vel)]

    if len(p):
        train_position_values.append(p)

    if len(v):
        train_velocity_values.append(v)

    if (
        (i + 1) % 10 == 0
        or i == len(TRAIN_IDS) - 1
    ):

        print(
            f"  TRAIN targets processed: "
            f"{i + 1}/{len(TRAIN_IDS)}"
        )


train_position_values = np.concatenate(
    train_position_values
)

train_velocity_values = np.concatenate(
    train_velocity_values
)


position_mean = float(
    np.mean(train_position_values)
)

position_std = float(
    np.std(train_position_values)
)

velocity_mean = float(
    np.mean(train_velocity_values)
)

velocity_std = float(
    np.std(train_velocity_values)
)

if position_std < 1e-8:
    position_std = 1.0

if velocity_std < 1e-8:
    velocity_std = 1.0


print()
print(
    f"Position target mean: "
    f"{position_mean:.6f}"
)

print(
    f"Position target std : "
    f"{position_std:.6f}"
)

print(
    f"Velocity target mean: "
    f"{velocity_mean:.6f}"
)

print(
    f"Velocity target std : "
    f"{velocity_std:.6f}"
)

print()
print(
    "[PASS] Target scaling fitted using "
    "TRAIN only."
)


# ======================================================================
# LAZY WINDOW DATASET
# ======================================================================

class LSTMWindowDataset(IterableDataset):

    def __init__(
        self,
        sequence_ids
    ):

        super().__init__()

        self.sequence_ids = sequence_ids

    def __iter__(self):

        for sid in self.sequence_ids:

            feature_path = get_feature_path(sid)

            df = read_csv_robust(
                feature_path
            )

            X = df[
                FEATURE_COLUMNS
            ].copy()

            X = X.replace(
                [np.inf, -np.inf],
                np.nan
            )

            X = feature_imputer.transform(
                X
            )

            X = feature_scaler.transform(
                X
            )

            X = X.astype(
                np.float32
            )

            position, velocity = get_targets(
                sid
            )

            n = min(
                len(X),
                len(position),
                len(velocity)
            )

            X = X[:n]
            position = position[:n]
            velocity = velocity[:n]

            for i in range(
                SEQ_LEN,
                n + 1
            ):

                x_window = X[
                    i - SEQ_LEN:i
                ]

                p = position[i - 1]
                v = velocity[i - 1]

                if not np.isfinite(p):
                    continue

                if not np.isfinite(v):
                    continue

                p_scaled = (
                    p - position_mean
                ) / position_std

                v_scaled = (
                    v - velocity_mean
                ) / velocity_std

                y = np.array(
                    [
                        p_scaled,
                        v_scaled
                    ],
                    dtype=np.float32
                )

                yield (
                    torch.from_numpy(
                        x_window
                    ),
                    torch.from_numpy(y)
                )


# ======================================================================
# COUNT WINDOWS
# ======================================================================

def count_windows(sequence_ids):

    total = 0

    for sid in sequence_ids:

        path = get_feature_path(sid)

        df = read_csv_robust(
            path,
            )

        n_features = len(df)

        pos, vel = get_targets(
            sid
        )

        n = min(
            n_features,
            len(pos),
            len(vel)
        )

        if n > SEQ_LEN:

            total += n - SEQ_LEN

    return total


print()
print("=" * 70)
print("WINDOW COUNT")
print("=" * 70)

TRAIN_WINDOWS = count_windows(
    TRAIN_IDS
)

VAL_WINDOWS = count_windows(
    VAL_IDS
)

TEST_WINDOWS = count_windows(
    TEST_IDS
)

print(
    f"Train windows      : "
    f"{TRAIN_WINDOWS:,}"
)

print(
    f"Validation windows : "
    f"{VAL_WINDOWS:,}"
)

print(
    f"Test windows       : "
    f"{TEST_WINDOWS:,}"
)


# ======================================================================
# DATA LOADERS
# ======================================================================

train_dataset = LSTMWindowDataset(
    TRAIN_IDS
)

val_dataset = LSTMWindowDataset(
    VAL_IDS
)

test_dataset = LSTMWindowDataset(
    TEST_IDS
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)


# ======================================================================
# IMPROVED LSTM MODEL
# ======================================================================

class ImprovedLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
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
            hidden_size // 2
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            hidden_size // 2,
            2
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.norm(out)

        out = self.fc1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out


MODEL = ImprovedLSTM(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print(MODEL)


# ======================================================================
# LOSS / OPTIMIZER
# ======================================================================

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    MODEL.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# ======================================================================
# TRAINING
# ======================================================================

best_val_loss = float("inf")

best_model_path = os.path.join(
    MODEL_DIR,
    "nav_shield_lstm_8a_best.pt"
)

history = []


print()
print("=" * 70)
print("TRAINING EXPERIMENT 8A")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    MODEL.train()

    train_loss = 0.0
    train_count = 0

    for X, y in train_loader:

        X = X.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        pred = MODEL(X)

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            MODEL.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = X.size(0)

        train_loss += (
            loss.item()
            * batch_size
        )

        train_count += batch_size

    train_loss /= max(
        train_count,
        1
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    MODEL.eval()

    val_loss = 0.0
    val_count = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = MODEL(X)

            loss = criterion(
                pred,
                y
            )

            batch_size = X.size(0)

            val_loss += (
                loss.item()
                * batch_size
            )

            val_count += batch_size

    val_loss /= max(
        val_count,
        1
    )

    scheduler.step(
        val_loss
    )

    current_lr = optimizer.param_groups[0]["lr"]

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
            MODEL.state_dict(),
            best_model_path
        )

        print(
            "  [SAVED] New best validation model."
        )


# ======================================================================
# LOAD BEST MODEL
# ======================================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

MODEL.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

MODEL.eval()

print(
    "[PASS] Best Experiment 8A model loaded."
)


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate(loader):

    all_actual = []
    all_predicted = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)

            pred = MODEL(X)

            pred = pred.cpu().numpy()
            actual = y.numpy()

            all_predicted.append(
                pred
            )

            all_actual.append(
                actual
            )

    actual = np.concatenate(
        all_actual,
        axis=0
    )

    predicted = np.concatenate(
        all_predicted,
        axis=0
    )

    # --------------------------------------------------------------
    # Convert target values back to original units
    # --------------------------------------------------------------

    actual_position = (
        actual[:, 0]
        * position_std
        + position_mean
    )

    predicted_position = (
        predicted[:, 0]
        * position_std
        + position_mean
    )

    actual_velocity = (
        actual[:, 1]
        * velocity_std
        + velocity_mean
    )

    predicted_velocity = (
        predicted[:, 1]
        * velocity_std
        + velocity_mean
    )

    return (
        actual_position,
        predicted_position,
        actual_velocity,
        predicted_velocity
    )


print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    actual_position,
    predicted_position,
    actual_velocity,
    predicted_velocity
) = evaluate(
    test_loader
)


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    actual,
    predicted
):

    error = (
        predicted
        - actual
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    return (
        float(mae),
        float(rmse)
    )


position_mae, position_rmse = calculate_metrics(
    actual_position,
    predicted_position
)

velocity_mae, velocity_rmse = calculate_metrics(
    actual_velocity,
    predicted_velocity
)


print()
print("EXPERIMENT 8A TEST METRICS")
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


# ======================================================================
# SAVE PREDICTIONS
# ======================================================================

prediction_file = os.path.join(
    PRED_DIR,
    "stage8_improvement_8a_test_predictions.csv"
)

prediction_df = pd.DataFrame(
    {
        "actual_position_error_m":
            actual_position,

        "predicted_position_error_m":
            predicted_position,

        "actual_velocity_error_m_s":
            actual_velocity,

        "predicted_velocity_error_m_s":
            predicted_velocity
    }
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print()
print(
    f"[SAVED] Predictions:\n"
    f"{prediction_file}"
)


# ======================================================================
# SAVE METRICS
# ======================================================================

metrics = {

    "experiment": "Stage 8 Improvement 8A",

    "model":
        "Improved LSTM + LayerNorm + Dropout + AdamW",

    "learning_type":
        "SUPERVISED_TEMPORAL_REGRESSION",

    "sequence_length":
        SEQ_LEN,

    "features":
        len(FEATURE_COLUMNS),

    "hidden_size":
        HIDDEN_SIZE,

    "lstm_layers":
        NUM_LAYERS,

    "dropout":
        DROPOUT,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "train_sequences":
        len(TRAIN_IDS),

    "validation_sequences":
        len(VAL_IDS),

    "test_sequences":
        len(TEST_IDS),

    "train_windows":
        TRAIN_WINDOWS,

    "validation_windows":
        VAL_WINDOWS,

    "test_windows":
        TEST_WINDOWS,

    "best_validation_loss":
        float(best_val_loss),

    "test_metrics": {

        "position_error_m": {

            "MAE":
                position_mae,

            "RMSE":
                position_rmse
        },

        "velocity_error_m_s": {

            "MAE":
                velocity_mae,

            "RMSE":
                velocity_rmse
        }
    },

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

    "target_source":
        "Categorised S/V dataset",

    "model_path":
        best_model_path,

    "prediction_path":
        prediction_file
}


metrics_file = os.path.join(
    REPORT_DIR,
    "stage8_improvement_8a_metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ======================================================================
# SAVE TRAINING HISTORY
# ======================================================================

history_file = os.path.join(
    REPORT_DIR,
    "stage8_improvement_8a_training_history.csv"
)

pd.DataFrame(
    history
).to_csv(
    history_file,
    index=False
)


# ======================================================================
# SAVE SUMMARY
# ======================================================================

summary_file = os.path.join(
    REPORT_DIR,
    "stage8_improvement_8a_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NAV-SHIELD STAGE 8 IMPROVEMENT — EXPERIMENT 8A\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Improved LSTM architecture + regularization\n\n"
    )

    f.write(
        f"Device: {DEVICE}\n"
    )

    f.write(
        f"Sequence length: {SEQ_LEN}\n"
    )

    f.write(
        f"Features: {len(FEATURE_COLUMNS)}\n"
    )

    f.write(
        f"Hidden size: {HIDDEN_SIZE}\n"
    )

    f.write(
        f"LSTM layers: {NUM_LAYERS}\n"
    )

    f.write(
        f"Dropout: {DROPOUT}\n"
    )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Weight decay: {WEIGHT_DECAY}\n\n"
    )

    f.write(
        f"Train sequences: {len(TRAIN_IDS)}\n"
    )

    f.write(
        f"Validation sequences: {len(VAL_IDS)}\n"
    )

    f.write(
        f"Test sequences: {len(TEST_IDS)}\n\n"
    )

    f.write(
        f"Train windows: {TRAIN_WINDOWS:,}\n"
    )

    f.write(
        f"Validation windows: {VAL_WINDOWS:,}\n"
    )

    f.write(
        f"Test windows: {TEST_WINDOWS:,}\n\n"
    )

    f.write(
        f"Best validation loss: "
        f"{best_val_loss:.6f}\n\n"
    )

    f.write(
        "TEST METRICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Position MAE: "
        f"{position_mae:.6f} m\n"
    )

    f.write(
        f"Position RMSE: "
        f"{position_rmse:.6f} m\n"
    )

    f.write(
        f"Velocity MAE: "
        f"{velocity_mae:.6f} m/s\n"
    )

    f.write(
        f"Velocity RMSE: "
        f"{velocity_rmse:.6f} m/s\n\n"
    )

    f.write(
        "DATASET POLICY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Raw data modified: NO\n"
    )

    f.write(
        "Previous stages modified: NO\n"
    )

    f.write(
        "Uncategorised dataset used: NO\n"
    )

    f.write(
        "Sequence leakage protection: ENABLED\n"
    )

    f.write(
        "Feature preprocessing: TRAIN ONLY\n"
    )

    f.write(
        "Target scaling: TRAIN ONLY\n"
    )


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print()
print("=" * 70)
print("STAGE 8 IMPROVEMENT — EXPERIMENT 8A COMPLETE")
print("=" * 70)

print()
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
print(
    f"Model:\n{best_model_path}"
)

print()
print(
    f"Predictions:\n{prediction_file}"
)

print()
print(
    f"Metrics:\n{metrics_file}"
)

print()
print(
    f"Training history:\n{history_file}"
)

print()
print(
    f"Summary:\n{summary_file}"
)

print()
print("Uncategorised dataset: NOT USED")
print("Raw data modified: NO")
print("Previous stages modified: NO")
print("Sequence leakage protection: ENABLED")

print("=" * 70)