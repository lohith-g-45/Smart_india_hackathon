"""
======================================================================
NAV-SHIELD STAGE 8
LSTM TEMPORAL NAVIGATION DRIFT MODEL
======================================================================

Purpose:
    Train an LSTM using temporal windows of navigation features.

Targets:
    1. position_error_m
    2. velocity_error_m_s

IMPORTANT:
    - Stage 4 sequence split is used.
    - Stage 3 engineered features are used as inputs.
    - Targets are calculated READ-ONLY from Categorised S/V data.
    - Raw data is NEVER modified.
    - Stage 1-7 outputs are NEVER modified.
    - Imputation is fitted on TRAIN only.
    - Scaling is fitted on TRAIN only.
    - Temporal windows NEVER cross sequence boundaries.
    - Windows are generated LAZILY.
    - No huge X_windows NumPy array is created.
    - Uncategorised dataset is NOT used.
======================================================================
"""

from pathlib import Path
import json
import random
import warnings
import gc

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_DIR = (
    BASE_DIR
    / "results"
    / "features"
    / "categorised"
)

STAGE4_SPLIT = (
    BASE_DIR
    / "results"
    / "stage4_reports"
    / "stage4_dataset_split.json"
)

CATEGORISED_ROOT = Path(
    r"C:\Users\tuala\Downloads"
    r"\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "stage8"
)

MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"

for directory in [
    MODEL_DIR,
    REPORT_DIR,
    PRED_DIR,
    PLOT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ======================================================================
# LSTM PARAMETERS
# ======================================================================

SEQUENCE_LENGTH = 30

# Reduced from 256 to avoid unnecessary RAM pressure.
BATCH_SIZE = 64

EPOCHS = 20

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.2

LEARNING_RATE = 0.001

RANDOM_SEED = 42

# Number of batches printed during training.
PRINT_EVERY = 100

TARGET_COLUMNS = [
    "position_error_m",
    "velocity_error_m_s",
]

# The 107 smartphone-deployable features for Member 5
DEPLOYABLE_SMARTPHONE_FEATURES = [
    "S_time_seconds",
    "time_difference_seconds",
    "S_GPS LATITUDE (degrees)",
    "S_GPS LONGITUDE (degrees)",
    "S_GPS ALTITUDE (m)",
    "S_GPS SPEED (Kmh)",
    "S_GPS ACCURACY (m)",
    "S_GPS ORIENTATION (°)",
    "S_TIME SINCE START (ms)",
    "S_ACCELEROMETER X (m/s²)",
    "S_ACCELEROMETER Y (m/s²)",
    "S_ACCELEROMETER Z (m/s²)",
    "S_GRAVITY X (m/s²)",
    "S_GRAVITY Y (m/s²)",
    "S_GRAVITY Z (m/s²)",
    "S_GYROSCOPE Yaw (rad/s)",
    "S_GYROSCOPE Pitch (rad/s)",
    "S_GYROSCOPE Roll (rad/s)",
    "S_MAGNETIC FIELD X (µT)",
    "S_MAGNETIC FIELD Y (µT)",
    "S_MAGNETIC FIELD Z (µT)",
    "S_ORIENTATION (Yaw) (°)",
    "S_ORIENTATION (Pitch) (°)",
    "S_ORIENTATION (Roll ) (°)",
    "timestamp",
    "elapsed_time",
    "delta_time",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_magnitude",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "gyro_magnitude",
    "mag_x",
    "mag_y",
    "mag_z",
    "mag_magnitude",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "gravity_magnitude",
    "orientation_yaw",
    "orientation_pitch",
    "orientation_roll",
    "gps_latitude",
    "gps_longitude",
    "gps_altitude",
    "gps_speed",
    "gps_accuracy",
    "gps_satellites",
    "accel_change",
    "accel_change_rate",
    "gyro_change",
    "gyro_change_rate",
    "magnetic_field_change",
    "magnetic_field_change_rate",
    "speed_change",
    "speed_change_rate",
    "yaw_change",
    "yaw_change_rate",
    "pitch_change",
    "pitch_change_rate",
    "roll_change",
    "roll_change_rate",
    "sensor_quality_score",
    "accel_magnitude_rolling_mean",
    "accel_magnitude_rolling_std",
    "accel_magnitude_rolling_min",
    "accel_magnitude_rolling_max",
    "accel_magnitude_rolling_range",
    "gyro_magnitude_rolling_mean",
    "gyro_magnitude_rolling_std",
    "gyro_magnitude_rolling_min",
    "gyro_magnitude_rolling_max",
    "gyro_magnitude_rolling_range",
    "mag_magnitude_rolling_mean",
    "mag_magnitude_rolling_std",
    "mag_magnitude_rolling_min",
    "mag_magnitude_rolling_max",
    "mag_magnitude_rolling_range",
    "gravity_magnitude_rolling_mean",
    "gravity_magnitude_rolling_std",
    "gravity_magnitude_rolling_min",
    "gravity_magnitude_rolling_max",
    "gravity_magnitude_rolling_range",
    "gps_speed_rolling_mean",
    "gps_speed_rolling_std",
    "gps_speed_rolling_min",
    "gps_speed_rolling_max",
    "gps_speed_rolling_range",
    "accel_change_rolling_mean",
    "accel_change_rolling_std",
    "accel_change_rolling_min",
    "accel_change_rolling_max",
    "accel_change_rolling_range",
    "gyro_change_rolling_mean",
    "gyro_change_rolling_std",
    "gyro_change_rolling_min",
    "gyro_change_rolling_max",
    "gyro_change_rolling_range",
    "speed_change_rolling_mean",
    "speed_change_rolling_std",
    "speed_change_rolling_min",
    "speed_change_rolling_max",
    "speed_change_rolling_range"
]


# ======================================================================
# RANDOM SEED
# ======================================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("NAV-SHIELD STAGE 8: LSTM TEMPORAL DRIFT MODEL")
print("=" * 70)

print()
print("Purpose:")
print("Train an LSTM to learn temporal navigation-error patterns.")

print()
print("Targets:")
print("- position_error_m")
print("- velocity_error_m_s")

print()
print("Device:", DEVICE)
print("Sequence length:", SEQUENCE_LENGTH)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

print()
print("Raw data modified: NO")
print("Stage 1-7 outputs modified: NO")
print("Uncategorised dataset: NOT USED")
print("Sequence leakage protection: ENABLED")
print("Preprocessing policy: TRAIN ONLY")
print("Window generation: LAZY / MEMORY SAFE")


# ======================================================================
# HELPERS
# ======================================================================

def normalize_name(value):
    """
    Normalize names for reliable matching.
    """

    text = str(value).strip().lower()

    replacements = [
        (" ", "_"),
        ("-", "_"),
        (".csv", ""),
        ("(", ""),
        (")", ""),
        ("/", "_"),
        ("°", ""),
        ("²", "2"),
        ("µ", "u"),
        ("μ", "u"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    while "__" in text:
        text = text.replace("__", "_")

    return text


def normalize_sequence_id(value):
    """
    Convert different names into the same sequence ID.
    """

    name = normalize_name(value)

    if name.endswith("_features"):
        name = name[:-9]

    if name.startswith("features_"):
        name = name[9:]

    return name


def read_csv_safely(path):
    """
    Read CSV using several encodings.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:

        try:

            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        f"Could not read CSV:\n{path}\n{last_error}"
    )


def find_column(df, candidates):
    """
    Find dataframe column using normalized names.
    """

    lookup = {
        normalize_name(col): col
        for col in df.columns
    }

    for candidate in candidates:

        key = normalize_name(candidate)

        if key in lookup:
            return lookup[key]

    return None


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate horizontal distance in metres.
    """

    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)

    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    radius = 6371000.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    dlat = np.radians(
        lat2 - lat1
    )

    dlon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1_rad)
        *
        np.cos(lat2_rad)
        *
        np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0
    )

    return (
        2.0
        * radius
        * np.arcsin(
            np.sqrt(a)
        )
    )


# ======================================================================
# FIND STAGE 4 SPLIT
# ======================================================================

if not STAGE4_SPLIT.exists():

    raise FileNotFoundError(
        "Stage 4 split file not found:\n"
        f"{STAGE4_SPLIT}"
    )


print()
print("=" * 70)
print("LOADING STAGE 4 SEQUENCE SPLIT")
print("=" * 70)


with open(
    STAGE4_SPLIT,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


def extract_split(name):

    value = split_data.get(name)

    if value is None:
        value = split_data.get(name.upper())

    if value is None:

        raise KeyError(
            f"Split '{name}' not found."
        )

    if isinstance(value, list):

        return [
            normalize_sequence_id(x)
            for x in value
        ]

    if isinstance(value, dict):

        for key in [
            "sequences",
            "sequence_ids",
            "ids"
        ]:

            if key in value:

                return [
                    normalize_sequence_id(x)
                    for x in value[key]
                ]

    raise ValueError(
        f"Unsupported split format: {name}"
    )


TRAIN_IDS = extract_split("train")

VAL_IDS = extract_split("validation")

TEST_IDS = extract_split("test")


print()
print("SEQUENCE SPLIT")
print("-" * 70)

print(
    "TRAIN      :",
    len(TRAIN_IDS),
    "sequences"
)

print(
    "VALIDATION :",
    len(VAL_IDS),
    "sequences"
)

print(
    "TEST       :",
    len(TEST_IDS),
    "sequences"
)


if (
    len(TRAIN_IDS) == 0
    or len(VAL_IDS) == 0
    or len(TEST_IDS) == 0
):

    raise RuntimeError(
        "Stage 4 split contains an empty split."
    )


# ======================================================================
# VERIFY NO SEQUENCE LEAKAGE
# ======================================================================

all_train = set(TRAIN_IDS)
all_val = set(VAL_IDS)
all_test = set(TEST_IDS)

if all_train & all_val:
    raise RuntimeError(
        "Sequence leakage detected between TRAIN and VALIDATION."
    )

if all_train & all_test:
    raise RuntimeError(
        "Sequence leakage detected between TRAIN and TEST."
    )

if all_val & all_test:
    raise RuntimeError(
        "Sequence leakage detected between VALIDATION and TEST."
    )

print(
    "[PASS] Sequence leakage protection verified."
)


# ======================================================================
# DISCOVER FEATURE FILES
# ======================================================================

print()
print("=" * 70)
print("FEATURE FILE DISCOVERY")
print("=" * 70)


if not FEATURE_DIR.exists():

    raise FileNotFoundError(
        "Feature directory not found:\n"
        f"{FEATURE_DIR}"
    )


feature_files = {}


for path in FEATURE_DIR.glob("*.csv"):

    seq_id = normalize_sequence_id(
        path.stem
    )

    feature_files[seq_id] = path


print(
    "Feature files found:",
    len(feature_files)
)


all_ids = (
    TRAIN_IDS
    + VAL_IDS
    + TEST_IDS
)


missing_features = [
    seq
    for seq in all_ids
    if seq not in feature_files
]


if missing_features:

    print()
    print("Missing feature files:")

    for seq in missing_features:
        print(" -", seq)

    raise RuntimeError(
        "Some Stage 4 sequences do not have Stage 3 feature files."
    )


print(
    "[PASS] All split sequences have feature files."
)


# ======================================================================
# DISCOVER ORIGINAL CATEGORISED S/V FILES
# ======================================================================

print()
print("=" * 70)
print("CATEGORISED S/V FILE DISCOVERY")
print("=" * 70)


if not CATEGORISED_ROOT.exists():

    raise FileNotFoundError(
        "Categorised dataset directory does not exist:\n"
        f"{CATEGORISED_ROOT}"
    )


s_files = {}
v_files = {}


for path in CATEGORISED_ROOT.rglob("*.csv"):

    normalized_stem = normalize_name(
        path.stem
    )

    if normalized_stem.startswith("s-"):

        seq_id = normalize_sequence_id(
            normalized_stem[2:]
        )

        s_files[seq_id] = path

    elif normalized_stem.startswith("s_"):

        seq_id = normalize_sequence_id(
            normalized_stem[2:]
        )

        s_files[seq_id] = path

    elif normalized_stem.startswith("v-"):

        seq_id = normalize_sequence_id(
            normalized_stem[2:]
        )

        v_files[seq_id] = path

    elif normalized_stem.startswith("v_"):

        seq_id = normalize_sequence_id(
            normalized_stem[2:]
        )

        v_files[seq_id] = path


print(
    "S files found:",
    len(s_files)
)

print(
    "V files found:",
    len(v_files)
)


missing_s = [
    seq
    for seq in all_ids
    if seq not in s_files
]

missing_v = [
    seq
    for seq in all_ids
    if seq not in v_files
]


if missing_s:

    raise RuntimeError(
        "Missing Categorised S files:\n"
        + "\n".join(missing_s)
    )


if missing_v:

    raise RuntimeError(
        "Missing Categorised V files:\n"
        + "\n".join(missing_v)
    )


print(
    "[PASS] All required Categorised S/V pairs found."
)


# ======================================================================
# TARGET CALCULATION
# ======================================================================

def calculate_targets(
    s_df,
    v_df
):
    """
    Calculate navigation-error targets.

    position_error_m:
        Horizontal GPS distance between S and V.

    velocity_error_m_s:
        Absolute S/V speed difference converted
        from km/h to m/s.

    Original dataframes are NOT modified.
    """

    # --------------------------------------------------------------
    # S columns
    # --------------------------------------------------------------

    s_lat_col = find_column(
        s_df,
        [
            "GPS LATITUDE (degrees)",
            "GPS LATITUDE",
            "latitude",
        ]
    )

    s_lon_col = find_column(
        s_df,
        [
            "GPS LONGITUDE (degrees)",
            "GPS LONGITUDE",
            "longitude",
        ]
    )

    s_speed_col = find_column(
        s_df,
        [
            "GPS SPEED (Kmh)",
            "GPS SPEED",
            "speed",
        ]
    )

    # --------------------------------------------------------------
    # V columns
    # --------------------------------------------------------------

    v_lat_col = find_column(
        v_df,
        [
            "Latitude (degrees)",
            "Latitude",
        ]
    )

    v_lon_col = find_column(
        v_df,
        [
            "Longitude (degrees)",
            "Longitude",
        ]
    )

    v_speed_col = find_column(
        v_df,
        [
            "Velocity (km/hr)",
            "Indicated Vehicle Speed (km/hr)",
            "Velocity",
            "Indicated Vehicle Speed",
        ]
    )

    required = [
        s_lat_col,
        s_lon_col,
        s_speed_col,
        v_lat_col,
        v_lon_col,
        v_speed_col,
    ]

    if any(
        x is None
        for x in required
    ):

        raise RuntimeError(
            "Required S/V navigation columns "
            "could not be found."
        )

    # --------------------------------------------------------------
    # Row alignment
    # --------------------------------------------------------------

    n = min(
        len(s_df),
        len(v_df)
    )

    s = s_df.iloc[:n]

    v = v_df.iloc[:n]

    # --------------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------------

    s_lat = pd.to_numeric(
        s[s_lat_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    s_lon = pd.to_numeric(
        s[s_lon_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    v_lat = pd.to_numeric(
        v[v_lat_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    v_lon = pd.to_numeric(
        v[v_lon_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    s_speed = pd.to_numeric(
        s[s_speed_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    v_speed = pd.to_numeric(
        v[v_speed_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------------
    # Position error
    # --------------------------------------------------------------

    position_error = haversine_distance(
        s_lat,
        s_lon,
        v_lat,
        v_lon
    )

    # --------------------------------------------------------------
    # Velocity error
    # --------------------------------------------------------------

    velocity_error = (
        np.abs(
            s_speed
            -
            v_speed
        )
        / 3.6
    )

    return np.column_stack(
        [
            position_error,
            velocity_error,
        ]
    ).astype(
        np.float32
    )


# ======================================================================
# LOAD ONE SEQUENCE
# ======================================================================

def load_sequence(sequence_id):

    feature_path = feature_files[
        sequence_id
    ]

    s_path = s_files[
        sequence_id
    ]

    v_path = v_files[
        sequence_id
    ]

    # --------------------------------------------------------------
    # Feature data
    # --------------------------------------------------------------

    feature_df = read_csv_safely(
        feature_path
    )

    # --------------------------------------------------------------
    # Original S/V
    # --------------------------------------------------------------

    s_df = read_csv_safely(
        s_path
    )

    v_df = read_csv_safely(
        v_path
    )

    # --------------------------------------------------------------
    # Calculate targets
    # --------------------------------------------------------------

    targets = calculate_targets(
        s_df,
        v_df
    )

    n = min(
        len(feature_df),
        len(targets)
    )

    feature_df = feature_df.iloc[
        :n
    ].copy()

    targets = targets[
        :n
    ]

    # --------------------------------------------------------------
    # Numeric features only (Filtered to deployable set)
    # --------------------------------------------------------------

    X = feature_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    return X, targets


# ======================================================================
# LOAD ALL DATA
#
# IMPORTANT:
# We keep individual sequence arrays.
# We NEVER concatenate temporal windows into one giant array.
# ======================================================================

def load_sequences(
    sequence_ids,
    split_name
):

    data = {}

    print()
    print(
        f"Loading {split_name} sequences..."
    )

    for index, sequence_id in enumerate(
        sequence_ids,
        start=1
    ):

        print(
            f"[{index}/{len(sequence_ids)}] "
            f"{sequence_id}"
        )

        X, y = load_sequence(
            sequence_id
        )

        data[sequence_id] = {
            "X": X,
            "y": y,
        }

    return data

sample_df = read_csv_safely(
    feature_files[
        TRAIN_IDS[0]
    ]
)

feature_columns = [
    f for f in DEPLOYABLE_SMARTPHONE_FEATURES
    if f in sample_df.columns
]


train_data = load_sequences(
    TRAIN_IDS,
    "TRAIN"
)

val_data = load_sequences(
    VAL_IDS,
    "VALIDATION"
)

test_data = load_sequences(
    TEST_IDS,
    "TEST"
)


# ======================================================================
# FEATURE CONSISTENCY
# ======================================================================

print()
print("=" * 70)
print("FEATURE CONSISTENCY")
print("=" * 70)


# Get numeric columns from the first feature file, filtering for deployable features.


print(
    "Numeric deployable feature count:",
    len(feature_columns)
)

if len(feature_columns) != 107:
    print(f"[WARNING] Expected 107 features, but found {len(feature_columns)}")
    missing = set(DEPLOYABLE_SMARTPHONE_FEATURES) - set(sample_df.columns)
    if missing:
        print("Missing features from Stage 3 output:")
        for m in sorted(missing):
            print(f" - {m}")

# Make sure every sequence has the same number of numeric features.
expected_feature_count = len(
    feature_columns
)


for sequence_id, item in {
    **train_data,
    **val_data,
    **test_data,
}.items():

    if (
        item["X"].shape[1]
        != expected_feature_count
    ):

        raise RuntimeError(
            "Feature count mismatch in "
            f"{sequence_id}: "
            f"{item['X'].shape[1]} vs "
            f"{expected_feature_count}"
        )


print(
    "[PASS] Feature consistency verified."
)


# ======================================================================
# TARGET VALIDATION
# ======================================================================

print()
print("=" * 70)
print("TARGET VALIDATION")
print("=" * 70)


for target_index, target_name in enumerate(
    TARGET_COLUMNS
):

    train_values = np.concatenate(
        [
            item["y"][:, target_index]
            for item in train_data.values()
        ]
    )

    valid = np.isfinite(
        train_values
    )

    print(
        f"{target_name}: "
        f"{valid.sum():,} valid / "
        f"{len(train_values):,}"
    )

    if valid.sum() == 0:

        raise RuntimeError(
            f"No valid training targets for "
            f"{target_name}"
        )


# ======================================================================
# TRAIN-ONLY FEATURE PREPROCESSING
# ======================================================================

print()
print("=" * 70)
print("TRAIN-ONLY FEATURE PREPROCESSING")
print("=" * 70)


# Instead of creating temporal windows, we only combine
# the ordinary training feature rows here.
#
# This is approximately:
# 594,546 x 142
#
# which is manageable compared with:
# 313,752 x 30 x 137

train_feature_matrix = np.concatenate(
    [
        item["X"]
        for item in train_data.values()
    ],
    axis=0
)


print(
    "Training preprocessing rows:",
    f"{len(train_feature_matrix):,}"
)

print(
    "Features:",
    train_feature_matrix.shape[1]
)


# --------------------------------------------------------------
# Imputation
# --------------------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

imputer.fit(
    train_feature_matrix
)


# --------------------------------------------------------------
# Scaling
# --------------------------------------------------------------

train_imputed = imputer.transform(
    train_feature_matrix
)


scaler = StandardScaler()

scaler.fit(
    train_imputed
)


print(
    "[PASS] Feature imputer fitted on TRAIN only."
)

print(
    "[PASS] Feature scaler fitted on TRAIN only."
)


# --------------------------------------------------------------
# Apply preprocessing sequence-by-sequence
# --------------------------------------------------------------

def preprocess_sequences(
    data,
    name
):

    print()
    print(
        f"Applying preprocessing to {name}..."
    )

    for sequence_id, item in data.items():

        X = item["X"]

        X = imputer.transform(
            X
        )

        X = scaler.transform(
            X
        )

        item["X"] = np.asarray(
            X,
            dtype=np.float32
        )

        print(
            f"[PASS] {sequence_id}"
        )

    return data


train_data = preprocess_sequences(
    train_data,
    "TRAIN"
)

val_data = preprocess_sequences(
    val_data,
    "VALIDATION"
)

test_data = preprocess_sequences(
    test_data,
    "TEST"
)


# Save preprocessing objects.

joblib.dump(
    imputer,
    REPORT_DIR / "stage8_feature_imputer.joblib"
)

joblib.dump(
    scaler,
    REPORT_DIR / "stage8_feature_scaler.joblib"
)


# Free temporary preprocessing matrix.

del train_feature_matrix
del train_imputed

gc.collect()


# ======================================================================
# TRAIN-ONLY TARGET SCALING
# ======================================================================

print()
print("=" * 70)
print("TRAIN-ONLY TARGET PREPROCESSING")
print("=" * 70)


train_targets_matrix = np.concatenate(
    [
        item["y"]
        for item in train_data.values()
    ],
    axis=0
)


# Only use rows where both targets are finite.

valid_target_rows = np.all(
    np.isfinite(
        train_targets_matrix
    ),
    axis=1
)


train_targets_valid = (
    train_targets_matrix[
        valid_target_rows
    ]
)


if len(train_targets_valid) == 0:

    raise RuntimeError(
        "No valid training target rows."
    )


target_scaler = StandardScaler()

target_scaler.fit(
    train_targets_valid
)


print(
    "[PASS] Target scaler fitted on TRAIN only."
)


joblib.dump(
    target_scaler,
    REPORT_DIR / "stage8_target_scaler.joblib"
)


del train_targets_matrix
del train_targets_valid

gc.collect()


# ======================================================================
# LAZY LSTM DATASET
# ======================================================================

class LazyTemporalDataset(Dataset):
    """
    Memory-safe temporal dataset.

    IMPORTANT:
        It does NOT create:

            X_windows = np.array(...)

        Instead, every window is created only when
        DataLoader asks for a sample.
    """

    def __init__(
        self,
        sequence_data,
        sequence_length,
        target_scaler
    ):

        self.sequence_data = sequence_data

        self.sequence_length = (
            sequence_length
        )

        self.target_scaler = (
            target_scaler
        )

        # Each index is:
        # (sequence_id, end_index)

        self.indices = []

        for sequence_id, item in (
            sequence_data.items()
        ):

            X = item["X"]

            y = item["y"]

            n = min(
                len(X),
                len(y)
            )

            if n < sequence_length:
                continue

            # A window ending at index i uses:
            #
            # X[i-29 : i+1]
            #
            # target = y[i]

            for end_index in range(
                sequence_length - 1,
                n
            ):

                target = y[
                    end_index
                ]

                if np.all(
                    np.isfinite(target)
                ):

                    self.indices.append(
                        (
                            sequence_id,
                            end_index
                        )
                    )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index
    ):

        sequence_id, end_index = (
            self.indices[index]
        )

        item = self.sequence_data[
            sequence_id
        ]

        X = item["X"]

        y = item["y"]

        start_index = (
            end_index
            -
            self.sequence_length
            +
            1
        )

        window = X[
            start_index:
            end_index + 1
        ]

        target = y[
            end_index
        ]

        # Scale target using TRAIN-fitted scaler.

        target = self.target_scaler.transform(
            target.reshape(1, -1)
        )[0]

        return (
            torch.from_numpy(
                np.asarray(
                    window,
                    dtype=np.float32
                )
            ),
            torch.from_numpy(
                np.asarray(
                    target,
                    dtype=np.float32
                )
            )
        )


# ======================================================================
# CREATE DATASETS
# ======================================================================

print()
print("=" * 70)
print("CREATING LAZY TEMPORAL DATASETS")
print("=" * 70)


train_dataset = LazyTemporalDataset(
    train_data,
    SEQUENCE_LENGTH,
    target_scaler
)

val_dataset = LazyTemporalDataset(
    val_data,
    SEQUENCE_LENGTH,
    target_scaler
)

test_dataset = LazyTemporalDataset(
    test_data,
    SEQUENCE_LENGTH,
    target_scaler
)


print(
    "TRAIN windows:",
    f"{len(train_dataset):,}"
)

print(
    "VALIDATION windows:",
    f"{len(val_dataset):,}"
)

print(
    "TEST windows:",
    f"{len(test_dataset):,}"
)


if len(train_dataset) == 0:

    raise RuntimeError(
        "Training temporal dataset contains zero windows."
    )


if len(val_dataset) == 0:

    raise RuntimeError(
        "Validation temporal dataset contains zero windows."
    )


if len(test_dataset) == 0:

    raise RuntimeError(
        "Test temporal dataset contains zero windows."
    )


print(
    "[PASS] Lazy temporal datasets created."
)

print(
    "[PASS] No giant X_windows array created."
)


# ======================================================================
# DATALOADERS
# ======================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ======================================================================
# LSTM MODEL
# ======================================================================

class NavigationLSTM(nn.Module):

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
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                2
            )
        )

    def forward(
        self,
        x
    ):

        output, _ = self.lstm(
            x
        )

        # Last timestep.

        last_output = output[
            :, -1, :
        ]

        return self.fc(
            last_output
        )


INPUT_SIZE = expected_feature_count


model = NavigationLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)


print()
print("=" * 70)
print("LSTM MODEL")
print("=" * 70)

print(
    "Input features:",
    INPUT_SIZE
)

print(
    "Hidden size:",
    HIDDEN_SIZE
)

print(
    "Layers:",
    NUM_LAYERS
)

print(
    "Output targets:",
    2
)


# ======================================================================
# LOSS / OPTIMIZER
# ======================================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ======================================================================
# TRAINING FUNCTION
# ======================================================================

def train_one_epoch(
    model,
    loader
):

    model.train()

    running_loss = 0.0

    total_samples = 0

    for batch_index, (
        X_batch,
        y_batch
    ) in enumerate(loader):

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        # Prevent exploding gradients.

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = (
            X_batch.size(0)
        )

        running_loss += (
            loss.item()
            *
            batch_size
        )

        total_samples += (
            batch_size
        )

        if (
            batch_index + 1
        ) % PRINT_EVERY == 0:

            print(
                f"  Batch "
                f"{batch_index + 1:,}/"
                f"{len(loader):,} "
                f"| Loss: "
                f"{loss.item():.6f}"
            )

    return (
        running_loss
        /
        max(total_samples, 1)
    )


# ======================================================================
# VALIDATION FUNCTION
# ======================================================================

def evaluate_loss(
    model,
    loader
):

    model.eval()

    running_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size = (
                X_batch.size(0)
            )

            running_loss += (
                loss.item()
                *
                batch_size
            )

            total_samples += (
                batch_size
            )

    return (
        running_loss
        /
        max(total_samples, 1)
    )


# ======================================================================
# TRAIN LSTM
# ======================================================================

print()
print("=" * 70)
print("TRAINING LSTM")
print("=" * 70)


best_val_loss = float(
    "inf"
)

best_model_path = (
    MODEL_DIR
    / "nav_shield_lstm_android_107.pt"
)

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    train_loss = train_one_epoch(
        model,
        train_loader
    )

    val_loss = evaluate_loss(
        model,
        val_loader
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        }
    )

    print(
        f"Train Loss      : "
        f"{train_loss:.6f}"
    )

    print(
        f"Validation Loss : "
        f"{val_loss:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    INPUT_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT,

                "sequence_length":
                    SEQUENCE_LENGTH,

                "feature_columns":
                    feature_columns,

                "target_columns":
                    TARGET_COLUMNS,
            },
            best_model_path
        )

        print(
            "[PASS] Best model saved."
        )


# ======================================================================
# SAVE TRAINING HISTORY
# ======================================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    REPORT_DIR
    / "stage8_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ======================================================================
# LOAD BEST MODEL
# ======================================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print(
    "[PASS] Best validation model loaded."
)


# ======================================================================
# PREDICTION FUNCTION
# ======================================================================

def predict_dataset(
    model,
    loader
):

    model.eval()

    all_predictions = []

    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            predictions = model(
                X_batch
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                y_batch.numpy()
            )

    predictions = np.concatenate(
        all_predictions,
        axis=0
    )

    targets = np.concatenate(
        all_targets,
        axis=0
    )

    # Inverse target scaling.

    predictions = target_scaler.inverse_transform(
        predictions
    )

    targets = target_scaler.inverse_transform(
        targets
    )

    return (
        predictions,
        targets
    )


# ======================================================================
# VALIDATION PREDICTIONS
# ======================================================================

print()
print("=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)


val_predictions, val_targets = (
    predict_dataset(
        model,
        val_loader
    )
)


# ======================================================================
# TEST PREDICTIONS
# ======================================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_predictions, test_targets = (
    predict_dataset(
        model,
        test_loader
    )
)


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    predictions,
    targets
):

    result = {}

    for index, target_name in enumerate(
        TARGET_COLUMNS
    ):

        y_true = targets[
            :, index
        ]

        y_pred = predictions[
            :, index
        ]

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        result[target_name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
        }

    return result


validation_metrics = calculate_metrics(
    val_predictions,
    val_targets
)

test_metrics = calculate_metrics(
    test_predictions,
    test_targets
)


# ======================================================================
# PRINT METRICS
# ======================================================================

print()
print(
    "VALIDATION METRICS"
)

for target_name, metrics in (
    validation_metrics.items()
):

    print(
        f"{target_name}: "
        f"MAE={metrics['MAE']:.6f}, "
        f"RMSE={metrics['RMSE']:.6f}"
    )


print()
print(
    "TEST METRICS"
)

for target_name, metrics in (
    test_metrics.items()
):

    print(
        f"{target_name}: "
        f"MAE={metrics['MAE']:.6f}, "
        f"RMSE={metrics['RMSE']:.6f}"
    )


# ======================================================================
# SAVE TEST PREDICTIONS
# ======================================================================

prediction_df = pd.DataFrame(
    {
        "actual_position_error_m":
            test_targets[:, 0],

        "predicted_position_error_m":
            test_predictions[:, 0],

        "actual_velocity_error_m_s":
            test_targets[:, 1],

        "predicted_velocity_error_m_s":
            test_predictions[:, 1],
    }
)


prediction_path = (
    PRED_DIR
    / "stage8_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


print()
print(
    "[SAVED] Test predictions:"
)

print(
    prediction_path
)


# ======================================================================
# SAVE METRICS JSON
# ======================================================================

metrics_json = {

    "stage":
        "Stage 8",

    "model":
        "LSTM",

    "learning_type":
        "SUPERVISED_TEMPORAL_REGRESSION",

    "sequence_length":
        SEQUENCE_LENGTH,

    "features":
        INPUT_SIZE,

    "train_sequences":
        len(TRAIN_IDS),

    "validation_sequences":
        len(VAL_IDS),

    "test_sequences":
        len(TEST_IDS),

    "train_windows":
        len(train_dataset),

    "validation_windows":
        len(val_dataset),

    "test_windows":
        len(test_dataset),

    "best_validation_loss":
        float(best_val_loss),

    "validation_metrics":
        validation_metrics,

    "test_metrics":
        test_metrics,

    "device":
        str(DEVICE),

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
        str(best_model_path),

    "prediction_path":
        str(prediction_path),
}


metrics_path = (
    REPORT_DIR
    / "stage8_metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_json,
        f,
        indent=2
    )


# ======================================================================
# SAVE SUMMARY
# ======================================================================

summary = [

    "NAV-SHIELD STAGE 8: LSTM TEMPORAL DRIFT MODEL",

    "=" * 70,

    "",

    "MODEL",
    "LSTM",

    "",

    "LEARNING TYPE",
    "SUPERVISED TEMPORAL REGRESSION",

    "",

    f"Sequence length: "
    f"{SEQUENCE_LENGTH}",

    f"Features: "
    f"{INPUT_SIZE}",

    "",

    f"TRAIN sequences: "
    f"{len(TRAIN_IDS)}",

    f"VALIDATION sequences: "
    f"{len(VAL_IDS)}",

    f"TEST sequences: "
    f"{len(TEST_IDS)}",

    "",

    f"TRAIN windows: "
    f"{len(train_dataset):,}",

    f"VALIDATION windows: "
    f"{len(val_dataset):,}",

    f"TEST windows: "
    f"{len(test_dataset):,}",

    "",

    "TARGETS",

    "- position_error_m",

    "- velocity_error_m_s",

    "",

    "PREPROCESSING",

    "FEATURE IMPUTATION: TRAIN ONLY",

    "FEATURE SCALING: TRAIN ONLY",

    "TARGET SCALING: TRAIN ONLY",

    "",

    "SEQUENCE LEAKAGE PROTECTION",

    "ENABLED",

    "",

    "LAZY WINDOW GENERATION",

    "ENABLED",

    "",

    "RAW DATA MODIFIED",

    "NO",

    "",

    "STAGE 1-7 OUTPUTS MODIFIED",

    "NO",

    "",

    "UNCATEGORISED DATASET USED",

    "NO",

    "",

    "FINAL TEST METRICS",

    f"Position Error MAE: "
    f"{test_metrics['position_error_m']['MAE']:.6f} m",

    f"Position Error RMSE: "
    f"{test_metrics['position_error_m']['RMSE']:.6f} m",

    f"Velocity Error MAE: "
    f"{test_metrics['velocity_error_m_s']['MAE']:.6f} m/s",

    f"Velocity Error RMSE: "
    f"{test_metrics['velocity_error_m_s']['RMSE']:.6f} m/s",

    "",

    f"Model: "
    f"{best_model_path}",

    f"Predictions: "
    f"{prediction_path}",

    f"Metrics: "
    f"{metrics_path}",
]


summary_path = (
    REPORT_DIR
    / "stage8_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print()
print("=" * 70)
print("STAGE 8 COMPLETE")
print("=" * 70)

print()
print(
    "LSTM training completed successfully."
)

print(
    "Train sequences:",
    len(TRAIN_IDS)
)

print(
    "Validation sequences:",
    len(VAL_IDS)
)

print(
    "Test sequences:",
    len(TEST_IDS)
)

print(
    "Train windows:",
    f"{len(train_dataset):,}"
)

print(
    "Validation windows:",
    f"{len(val_dataset):,}"
)

print(
    "Test windows:",
    f"{len(test_dataset):,}"
)

print()
print(
    "Position Error Test MAE:",
    f"{test_metrics['position_error_m']['MAE']:.6f} m"
)

print(
    "Position Error Test RMSE:",
    f"{test_metrics['position_error_m']['RMSE']:.6f} m"
)

print(
    "Velocity Error Test MAE:",
    f"{test_metrics['velocity_error_m_s']['MAE']:.6f} m/s"
)

print(
    "Velocity Error Test RMSE:",
    f"{test_metrics['velocity_error_m_s']['RMSE']:.6f} m/s"
)

print()
print(
    "Model:",
    best_model_path
)

print(
    "Predictions:",
    prediction_path
)

print(
    "Metrics:",
    metrics_path
)

print(
    "Summary:",
    summary_path
)

print()
print(
    "Uncategorised dataset: NOT USED"
)

print(
    "Raw data modified: NO"
)

print(
    "Stage 1-7 outputs modified: NO"
)

print("=" * 70)