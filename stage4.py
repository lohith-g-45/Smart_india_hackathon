"""
NAV-SHIELD STAGE 4
ML Dataset Preparation

Purpose:
1. Read Stage 3.5 feature-engineered files
2. Keep complete sequence boundaries
3. Split sequences into TRAIN / VALIDATION / TEST
4. Fit preprocessing ONLY on TRAIN
5. Apply the same preprocessing to validation/test
6. Save ML-ready datasets and preprocessing metadata

IMPORTANT:
- Raw data is never modified.
- Stage 1/2/3/3.5 outputs are never overwritten.
- Uncategorised dataset is NOT used for training.
- No model training happens here.
"""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Columns which should never be used as ML features
NON_FEATURE_COLUMNS = {
    "sequence_id",
    "timestamp",
    "time",
    "datetime",
    "date",
}

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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_DIR = BASE_DIR / "results" / "features" / "categorised"
OUTPUT_DIR = BASE_DIR / "data" / "stage4"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "validation"
TEST_DIR = OUTPUT_DIR / "test"

REPORT_DIR = BASE_DIR / "results" / "stage4_reports"

for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(name):
    """
    Normalize column names for comparison only.
    Original column names are retained in saved datasets.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def is_numeric_feature(column):
    """
    Check whether a column can be treated as a numeric ML feature.
    """
    return pd.api.types.is_numeric_dtype(column)


def find_sequence_column(df):
    """
    Find sequence identifier column if Stage 3 created one.
    """
    candidates = [
        "sequence_id",
        "sequence",
        "seq_id",
        "sequenceid",
    ]

    normalized = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def load_feature_files():
    """
    Load all Stage 3.5 feature files.
    """

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory does not exist:\n{FEATURE_DIR}\n\n"
            "Make sure Stage 3.5 has generated the feature files."
        )

    files = sorted(FEATURE_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV feature files found in:\n{FEATURE_DIR}"
        )

    print("=" * 70)
    print("NAV-SHIELD STAGE 4: ML DATASET PREPARATION")
    print("=" * 70)
    print(f"Feature directory: {FEATURE_DIR}")
    print(f"Feature files found: {len(files)}")
    print()

    datasets = {}

    for file in files:

        sequence_id = file.stem.lower()

        try:
            df = pd.read_csv(file)

            if df.empty:
                print(f"[WARNING] {sequence_id}: empty file")
                continue

            datasets[sequence_id] = df

        except Exception as exc:
            print(
                f"[ERROR] Could not read {file.name}: {exc}"
            )

    return datasets


# ============================================================
# SEQUENCE SPLIT
# ============================================================

def split_sequences(sequence_ids):
    """
    Split by complete sequences rather than individual rows.

    This is critical because randomly splitting rows from the same
    drive/sequence can cause severe data leakage.
    """

    sequence_ids = sorted(sequence_ids)

    rng = np.random.default_rng(RANDOM_SEED)

    shuffled = list(sequence_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_count = int(round(total * TRAIN_RATIO))
    validation_count = int(round(total * VALIDATION_RATIO))

    # Make sure every sequence belongs to exactly one split.
    if train_count + validation_count >= total:
        validation_count = max(
            1,
            total - train_count - 1
        )

    test_count = total - train_count - validation_count

    train_sequences = sorted(
        shuffled[:train_count]
    )

    validation_sequences = sorted(
        shuffled[
            train_count:
            train_count + validation_count
        ]
    )

    test_sequences = sorted(
        shuffled[
            train_count + validation_count:
        ]
    )

    return (
        train_sequences,
        validation_sequences,
        test_sequences,
    )


# ============================================================
# FEATURE IDENTIFICATION
# ============================================================

def identify_features(datasets):
    """
    Determine common numeric feature columns across sequences.
    Restricted to the 107 smartphone-deployable features for Member 5.
    """

    if not datasets:
        raise ValueError("No datasets available.")

    # We start with the 107 features we want for Android
    target_features = set(DEPLOYABLE_SMARTPHONE_FEATURES)
    common_columns = None

    for df in datasets.values():
        numeric_columns = set()
        for column in df.columns:
            if column in target_features:
                if is_numeric_feature(df[column]):
                    numeric_columns.add(column)

        if common_columns is None:
            common_columns = numeric_columns
        else:
            common_columns &= numeric_columns

    if not common_columns:
        raise ValueError(
            "No common smartphone numeric feature columns were found."
        )

    # Maintain the exact order from DEPLOYABLE_SMARTPHONE_FEATURES
    final_ordered = [
        f for f in DEPLOYABLE_SMARTPHONE_FEATURES
        if f in common_columns
    ]

    return final_ordered


# ============================================================
# PREPROCESSING
# ============================================================

def build_training_matrix(
    datasets,
    sequences,
    feature_columns
):

    frames = []

    for sequence in sequences:

        df = datasets[sequence]

        available = [
            column
            for column in feature_columns
            if column in df.columns
        ]

        frame = df[available].copy()

        # Ensure exact column order.
        frame = frame.reindex(
            columns=feature_columns
        )

        frames.append(frame)

    if not frames:
        raise ValueError(
            "No data available for selected sequences."
        )

    return pd.concat(
        frames,
        axis=0,
        ignore_index=True
    )


def fit_preprocessor(training_matrix):
    """
    Fit imputer and scaler ONLY on training data.
    """

    print()
    print("Fitting preprocessing on TRAINING data only...")

    # Replace infinities with NaN before imputation.
    training_matrix = training_matrix.replace(
        [np.inf, -np.inf],
        np.nan
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    imputed = imputer.fit_transform(
        training_matrix
    )

    scaler = StandardScaler()

    scaler.fit(imputed)

    return imputer, scaler


def transform_matrix(
    matrix,
    imputer,
    scaler
):

    matrix = matrix.replace(
        [np.inf, -np.inf],
        np.nan
    )

    imputed = imputer.transform(matrix)

    scaled = scaler.transform(imputed)

    return scaled


# ============================================================
# SAVE SPLIT DATA
# ============================================================

def save_split(
    split_name,
    datasets,
    sequences,
    feature_columns,
    imputer,
    scaler
):

    if split_name == "train":
        output_dir = TRAIN_DIR

    elif split_name == "validation":
        output_dir = VAL_DIR

    elif split_name == "test":
        output_dir = TEST_DIR

    else:
        raise ValueError(
            f"Unknown split: {split_name}"
        )

    rows = 0

    split_summary = []

    for sequence in sequences:

        df = datasets[sequence]

        matrix = df[
            feature_columns
        ].copy()

        transformed = transform_matrix(
            matrix,
            imputer,
            scaler
        )

        output = pd.DataFrame(
            transformed,
            columns=feature_columns
        )

        output.insert(
            0,
            "sequence_id",
            sequence
        )

        output_file = (
            output_dir /
            f"{sequence}_ml_ready.csv"
        )

        output.to_csv(
            output_file,
            index=False
        )

        rows += len(output)

        split_summary.append({
            "sequence_id": sequence,
            "rows": len(output),
            "file": str(output_file)
        })

    return rows, split_summary


# ============================================================
# MAIN
# ============================================================

def main():

    datasets = load_feature_files()

    if not datasets:
        raise RuntimeError(
            "No valid Stage 3.5 feature datasets found."
        )

    print(
        f"Valid sequences loaded: {len(datasets)}"
    )

    # --------------------------------------------------------
    # Split sequences
    # --------------------------------------------------------

    (
        train_sequences,
        validation_sequences,
        test_sequences,
    ) = split_sequences(
        list(datasets.keys())
    )

    print()
    print("SEQUENCE SPLIT")
    print("-" * 70)

    print(
        f"TRAIN      : {len(train_sequences)} sequences"
    )

    print(
        f"VALIDATION : {len(validation_sequences)} sequences"
    )

    print(
        f"TEST       : {len(test_sequences)} sequences"
    )

    # --------------------------------------------------------
    # Feature identification
    # --------------------------------------------------------

    feature_columns = identify_features(
        datasets
    )

    print()
    print(
        f"Common numeric features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Build TRAIN matrix
    # --------------------------------------------------------

    training_matrix = build_training_matrix(
        datasets,
        train_sequences,
        feature_columns
    )

    print(
        f"Training rows: "
        f"{len(training_matrix):,}"
    )

    # --------------------------------------------------------
    # Fit preprocessing ONLY on training
    # --------------------------------------------------------

    imputer, scaler = fit_preprocessor(
        training_matrix
    )

    # --------------------------------------------------------
    # Transform and save datasets
    # --------------------------------------------------------

    train_rows, train_summary = save_split(
        "train",
        datasets,
        train_sequences,
        feature_columns,
        imputer,
        scaler
    )

    validation_rows, validation_summary = save_split(
        "validation",
        datasets,
        validation_sequences,
        feature_columns,
        imputer,
        scaler
    )

    test_rows, test_summary = save_split(
        "test",
        datasets,
        test_sequences,
        feature_columns,
        imputer,
        scaler
    )

    # --------------------------------------------------------
    # Save preprocessing metadata
    # --------------------------------------------------------

    preprocessing = {
        "stage": "4",
        "random_seed": RANDOM_SEED,

        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },

        "feature_count": len(feature_columns),

        "features": feature_columns,

        "imputation": {
            "method": "median",
            "fit_on": "training_data_only",
        },

        "scaling": {
            "method": "StandardScaler",
            "fit_on": "training_data_only",
        },

        "training_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "test_sequences": test_sequences,

        "training_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
    }

    metadata_file = (
        REPORT_DIR /
        "stage4_preprocessing.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            preprocessing,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save split information
    # --------------------------------------------------------

    split_info = {
        "train": train_sequences,
        "validation": validation_sequences,
        "test": test_sequences,
    }

    split_file = (
        REPORT_DIR /
        "stage4_dataset_split.json"
    )

    with open(
        split_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_info,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save human-readable report
    # --------------------------------------------------------

    report_file = (
        REPORT_DIR /
        "stage4_summary.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "NAV-SHIELD STAGE 4: ML DATASET PREPARATION\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Total sequences: {len(datasets)}\n"
        )

        f.write(
            f"Training sequences: "
            f"{len(train_sequences)}\n"
        )

        f.write(
            f"Validation sequences: "
            f"{len(validation_sequences)}\n"
        )

        f.write(
            f"Test sequences: "
            f"{len(test_sequences)}\n\n"
        )

        f.write(
            f"Training rows: {train_rows:,}\n"
        )

        f.write(
            f"Validation rows: {validation_rows:,}\n"
        )

        f.write(
            f"Test rows: {test_rows:,}\n\n"
        )

        f.write(
            f"Feature count: {len(feature_columns)}\n"
        )

        f.write(
            "Imputation: median, fitted on training only\n"
        )

        f.write(
            "Scaling: StandardScaler, fitted on training only\n"
        )

        f.write(
            "Sequence leakage protection: ENABLED\n"
        )

        f.write(
            "Model training: NOT PERFORMED\n"
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 4 SUMMARY")
    print("=" * 70)

    print(
        f"Sequences processed : {len(datasets)}"
    )

    print(
        f"TRAIN sequences     : {len(train_sequences)}"
    )

    print(
        f"VALIDATION sequences: {len(validation_sequences)}"
    )

    print(
        f"TEST sequences      : {len(test_sequences)}"
    )

    print(
        f"TRAIN rows          : {train_rows:,}"
    )

    print(
        f"VALIDATION rows     : {validation_rows:,}"
    )

    print(
        f"TEST rows           : {test_rows:,}"
    )

    print(
        f"Features            : {len(feature_columns)}"
    )

    print(
        "Imputation          : TRAIN ONLY"
    )

    print(
        "Scaling             : TRAIN ONLY"
    )

    print(
        "Sequence leakage    : PROTECTED"
    )

    print(
        "Model training      : NOT PERFORMED"
    )

    print()
    print(
        f"Reports saved to:\n{REPORT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()