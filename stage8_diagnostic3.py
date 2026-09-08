"""
======================================================================
NAV-SHIELD STAGE 8 DIAGNOSTIC 3
======================================================================

Purpose:
    Compare the trained Stage 8 LSTM against simple baseline models
    and analyze performance separately for every TEST sequence.

IMPORTANT:
    - NO model retraining
    - NO raw data modification
    - NO Stage 1-8 modification
    - NO use of Uncategorised dataset
    - TRAIN / VALIDATION / TEST separation preserved

Baselines:
    1. Training-median baseline
    2. Persistence / previous-value baseline

LSTM:
    Uses the already generated Stage 8 TEST predictions.

Outputs:
    results/stage8/diagnostics/
        stage8_diagnostic3_overall_comparison.csv
        stage8_diagnostic3_per_sequence.csv
        stage8_diagnostic3_summary.json
        stage8_diagnostic3_summary.txt
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = r"C:\Users\tuala\Downloads\member5_ai_drift_stage1\member5_ai_drift"

STAGE4_SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage4_reports",
    "stage4_dataset_split.json"
)

PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage8",
    "predictions",
    "stage8_test_predictions.csv"
)

DIAGNOSTIC_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage8",
    "diagnostics"
)

os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)


# ======================================================================
# TARGET DEFINITIONS
# ======================================================================

POSITION_ACTUAL = "actual_position_error_m"
POSITION_PREDICTED = "predicted_position_error_m"

VELOCITY_ACTUAL = "actual_velocity_error_m_s"
VELOCITY_PREDICTED = "predicted_velocity_error_m_s"


# ======================================================================
# DISPLAY
# ======================================================================

print("=" * 70)
print("NAV-SHIELD STAGE 8 DIAGNOSTIC 3")
print("=" * 70)

print("""
Purpose:
Compare the trained Stage 8 LSTM against simple baselines
and analyze TEST performance by sequence.

Baselines:
  1. Training-median baseline
  2. Persistence / previous-value baseline

Model retraining: NO
Raw data modification: NO
Previous-stage modification: NO
Uncategorised dataset: NOT USED
Sequence leakage protection: ENABLED
""")


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def rmse(y_true, y_pred):
    """Calculate RMSE safely."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                (y_true[mask] - y_pred[mask]) ** 2
            )
        )
    )


def mae(y_true, y_pred):
    """Calculate MAE safely."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(
        np.mean(
            np.abs(
                y_true[mask] - y_pred[mask]
            )
        )
    )


def percentile_error(y_true, y_pred, percentile):
    """Absolute-error percentile."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan

    errors = np.abs(
        y_true[mask] - y_pred[mask]
    )

    return float(
        np.percentile(errors, percentile)
    )


def percentage_improvement(baseline, model):
    """
    Positive percentage means model is better than baseline.
    """
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan

    return float(
        ((baseline - model) / baseline) * 100.0
    )


def safe_mean(values):
    values = pd.Series(values, dtype="float64")
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(values.mean())


# ======================================================================
# LOAD STAGE 4 SPLIT
# ======================================================================

print("=" * 70)
print("LOADING STAGE 4 SPLIT")
print("=" * 70)

if not os.path.exists(STAGE4_SPLIT_FILE):
    raise FileNotFoundError(
        f"Stage 4 split file not found:\n{STAGE4_SPLIT_FILE}"
    )

with open(
    STAGE4_SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:
    split_data = json.load(f)


TRAIN_IDS = split_data.get("train", [])
VALIDATION_IDS = split_data.get("validation", [])
TEST_IDS = split_data.get("test", [])


print()
print(f"TRAIN      : {len(TRAIN_IDS)} sequences")
print(f"VALIDATION : {len(VALIDATION_IDS)} sequences")
print(f"TEST       : {len(TEST_IDS)} sequences")


if (
    len(TRAIN_IDS) == 0
    or len(VALIDATION_IDS) == 0
    or len(TEST_IDS) == 0
):
    raise RuntimeError(
        "Stage 4 split file contains an empty split."
    )


# ======================================================================
# SEQUENCE LEAKAGE CHECK
# ======================================================================

print()
print("=" * 70)
print("SEQUENCE LEAKAGE CHECK")
print("=" * 70)

train_set = set(TRAIN_IDS)
validation_set = set(VALIDATION_IDS)
test_set = set(TEST_IDS)

tv = train_set.intersection(validation_set)
tt = train_set.intersection(test_set)
vt = validation_set.intersection(test_set)

print(f"TRAIN ↔ VALIDATION overlap: {len(tv)}")
print(f"TRAIN ↔ TEST overlap: {len(tt)}")
print(f"VALIDATION ↔ TEST overlap: {len(vt)}")

if tv or tt or vt:
    raise RuntimeError(
        "SEQUENCE LEAKAGE DETECTED. Diagnostic stopped."
    )

print("[PASS] No sequence overlap detected.")


# ======================================================================
# LOAD STAGE 8 PREDICTIONS
# ======================================================================

print()
print("=" * 70)
print("LOADING STAGE 8 TEST PREDICTIONS")
print("=" * 70)

if not os.path.exists(PREDICTION_FILE):
    raise FileNotFoundError(
        f"Stage 8 prediction file not found:\n{PREDICTION_FILE}"
    )

pred_df = pd.read_csv(
    PREDICTION_FILE,
    low_memory=False
)

print(f"Prediction rows: {len(pred_df):,}")

required_columns = [
    POSITION_ACTUAL,
    POSITION_PREDICTED,
    VELOCITY_ACTUAL,
    VELOCITY_PREDICTED
]

missing_columns = [
    c for c in required_columns
    if c not in pred_df.columns
]

if missing_columns:
    raise RuntimeError(
        "Required prediction columns are missing:\n"
        + "\n".join(missing_columns)
    )

print("[PASS] Required Stage 8 prediction columns found.")


# ======================================================================
# CHECK FOR SEQUENCE COLUMN
# ======================================================================

sequence_column_candidates = [
    "sequence",
    "sequence_id",
    "Sequence",
    "sequence_name",
    "file"
]

sequence_column = None

for col in sequence_column_candidates:
    if col in pred_df.columns:
        sequence_column = col
        break


if sequence_column is None:

    print()
    print("[WARNING] No sequence column found in prediction file.")

    print("""
Stage 8 predictions contain only target/prediction columns.

Therefore:
    - Overall LSTM analysis is possible.
    - Exact per-sequence analysis is NOT possible from predictions alone.
    - Persistence baseline requires sequence boundaries.

The diagnostic will continue with overall LSTM analysis.
""")

else:

    print(
        f"[PASS] Sequence column detected: {sequence_column}"
    )


# ======================================================================
# NUMERIC CONVERSION
# ======================================================================

for col in required_columns:

    pred_df[col] = pd.to_numeric(
        pred_df[col],
        errors="coerce"
    )


# ======================================================================
# BASIC VALIDITY
# ======================================================================

print()
print("=" * 70)
print("PREDICTION VALIDITY CHECK")
print("=" * 70)

position_mask = (
    np.isfinite(pred_df[POSITION_ACTUAL])
    &
    np.isfinite(pred_df[POSITION_PREDICTED])
)

velocity_mask = (
    np.isfinite(pred_df[VELOCITY_ACTUAL])
    &
    np.isfinite(pred_df[VELOCITY_PREDICTED])
)

print(
    f"Valid position predictions : "
    f"{position_mask.sum():,}"
)

print(
    f"Valid velocity predictions : "
    f"{velocity_mask.sum():,}"
)


# ======================================================================
# IMPORTANT NOTE ABOUT BASELINES
# ======================================================================

print()
print("=" * 70)
print("BASELINE POLICY")
print("=" * 70)

print("""
Training-median baseline:
    Uses only TRAIN target statistics.

Persistence baseline:
    Predicts the previous target value within the same sequence.

No TEST target statistics are used to create the
training-median baseline.
""")


# ======================================================================
# LOAD TRAIN TARGETS FROM PREDICTION FILE IF AVAILABLE
# ======================================================================

# Stage 8 prediction files normally contain TEST data only.
# Therefore, we cannot safely calculate a TRAIN median from them.
#
# Instead, look for Stage 8 metadata/statistics produced by Diagnostic 2.

TRAIN_MEDIAN_POSITION = np.nan
TRAIN_MEDIAN_VELOCITY = np.nan


# ----------------------------------------------------------------------
# Attempt 1: Diagnostic 2 sequence statistics
# ----------------------------------------------------------------------

SEQ_STATS_FILE = os.path.join(
    DIAGNOSTIC_DIR,
    "stage8_sequence_target_statistics.csv"
)

if os.path.exists(SEQ_STATS_FILE):

    print(
        f"[FOUND] Sequence target statistics:\n"
        f"{SEQ_STATS_FILE}"
    )

    try:

        stats_df = pd.read_csv(
            SEQ_STATS_FILE,
            low_memory=False
        )

        print(
            f"Statistics rows: {len(stats_df):,}"
        )

        # Try to identify median columns.
        position_median_candidates = [
            "position_median",
            "position_error_median",
            "position_median_m",
            "position_error_m_median"
        ]

        velocity_median_candidates = [
            "velocity_median",
            "velocity_error_m_s_median",
            "velocity_median_m_s",
            "velocity_error_median"
        ]

        position_median_col = None
        velocity_median_col = None

        for c in position_median_candidates:
            if c in stats_df.columns:
                position_median_col = c
                break

        for c in velocity_median_candidates:
            if c in stats_df.columns:
                velocity_median_col = c
                break

        # Restrict to TRAIN sequences if split column exists.
        train_stats = stats_df.copy()

        if "split" in train_stats.columns:
            train_stats = train_stats[
                train_stats["split"].astype(str).str.upper()
                == "TRAIN"
            ]

        if position_median_col is not None:

            vals = pd.to_numeric(
                train_stats[position_median_col],
                errors="coerce"
            )

            vals = vals[np.isfinite(vals)]

            if len(vals) > 0:
                TRAIN_MEDIAN_POSITION = float(
                    np.median(vals)
                )

        if velocity_median_col is not None:

            vals = pd.to_numeric(
                train_stats[velocity_median_col],
                errors="coerce"
            )

            vals = vals[np.isfinite(vals)]

            if len(vals) > 0:
                TRAIN_MEDIAN_VELOCITY = float(
                    np.median(vals)
                )

    except Exception as e:

        print(
            "[WARNING] Could not read diagnostic statistics:"
        )
        print(str(e))


# ======================================================================
# FALLBACK: USE SUMMARY JSON
# ======================================================================

if (
    not np.isfinite(TRAIN_MEDIAN_POSITION)
    or
    not np.isfinite(TRAIN_MEDIAN_VELOCITY)
):

    SUMMARY_FILE = os.path.join(
        DIAGNOSTIC_DIR,
        "stage8_split_target_summary.csv"
    )

    if os.path.exists(SUMMARY_FILE):

        print(
            f"[FOUND] Split target summary:\n"
            f"{SUMMARY_FILE}"
        )

        try:

            summary_df = pd.read_csv(
                SUMMARY_FILE,
                low_memory=False
            )

            train_summary = summary_df[
                summary_df["split"]
                .astype(str)
                .str.upper()
                == "TRAIN"
            ]

            # Diagnostic 2 stores:
            # position_median_of_sequences
            # velocity_median_of_sequences

            if (
                "position_median_of_sequences"
                in train_summary.columns
            ):

                val = pd.to_numeric(
                    train_summary[
                        "position_median_of_sequences"
                    ],
                    errors="coerce"
                ).iloc[0]

                if np.isfinite(val):
                    TRAIN_MEDIAN_POSITION = float(val)

            if (
                "velocity_median_of_sequences"
                in train_summary.columns
            ):

                val = pd.to_numeric(
                    train_summary[
                        "velocity_median_of_sequences"
                    ],
                    errors="coerce"
                ).iloc[0]

                if np.isfinite(val):
                    TRAIN_MEDIAN_VELOCITY = float(val)

        except Exception as e:

            print(
                "[WARNING] Could not read split target summary:"
            )
            print(str(e))


# ======================================================================
# IMPORTANT: MEDIAN OF SEQUENCE MEDIANS IS AN APPROXIMATION
# ======================================================================

if np.isfinite(TRAIN_MEDIAN_POSITION):
    print(
        f"\nTraining position baseline: "
        f"{TRAIN_MEDIAN_POSITION:.6f} m"
    )
else:
    print(
        "\n[WARNING] Training position median unavailable."
    )


if np.isfinite(TRAIN_MEDIAN_VELOCITY):
    print(
        f"Training velocity baseline: "
        f"{TRAIN_MEDIAN_VELOCITY:.6f} m/s"
    )
else:
    print(
        "[WARNING] Training velocity median unavailable."
    )


# ======================================================================
# OVERALL LSTM PERFORMANCE
# ======================================================================

print()
print("=" * 70)
print("OVERALL LSTM PERFORMANCE")
print("=" * 70)

lstm_position_mae = mae(
    pred_df[POSITION_ACTUAL],
    pred_df[POSITION_PREDICTED]
)

lstm_position_rmse = rmse(
    pred_df[POSITION_ACTUAL],
    pred_df[POSITION_PREDICTED]
)

lstm_velocity_mae = mae(
    pred_df[VELOCITY_ACTUAL],
    pred_df[VELOCITY_PREDICTED]
)

lstm_velocity_rmse = rmse(
    pred_df[VELOCITY_ACTUAL],
    pred_df[VELOCITY_PREDICTED]
)


print()
print("POSITION")
print(
    f"  LSTM MAE  : {lstm_position_mae:.6f} m"
)
print(
    f"  LSTM RMSE : {lstm_position_rmse:.6f} m"
)

print()
print("VELOCITY")
print(
    f"  LSTM MAE  : {lstm_velocity_mae:.6f} m/s"
)
print(
    f"  LSTM RMSE : {lstm_velocity_rmse:.6f} m/s"
)


# ======================================================================
# MEDIAN BASELINE
# ======================================================================

print()
print("=" * 70)
print("TRAINING-MEDIAN BASELINE")
print("=" * 70)

median_position_mae = np.nan
median_position_rmse = np.nan

median_velocity_mae = np.nan
median_velocity_rmse = np.nan


if np.isfinite(TRAIN_MEDIAN_POSITION):

    median_position_predictions = np.full(
        len(pred_df),
        TRAIN_MEDIAN_POSITION,
        dtype=np.float64
    )

    median_position_mae = mae(
        pred_df[POSITION_ACTUAL],
        median_position_predictions
    )

    median_position_rmse = rmse(
        pred_df[POSITION_ACTUAL],
        median_position_predictions
    )

    print(
        f"Position median MAE  : "
        f"{median_position_mae:.6f} m"
    )

    print(
        f"Position median RMSE : "
        f"{median_position_rmse:.6f} m"
    )


if np.isfinite(TRAIN_MEDIAN_VELOCITY):

    median_velocity_predictions = np.full(
        len(pred_df),
        TRAIN_MEDIAN_VELOCITY,
        dtype=np.float64
    )

    median_velocity_mae = mae(
        pred_df[VELOCITY_ACTUAL],
        median_velocity_predictions
    )

    median_velocity_rmse = rmse(
        pred_df[VELOCITY_ACTUAL],
        median_velocity_predictions
    )

    print(
        f"Velocity median MAE  : "
        f"{median_velocity_mae:.6f} m/s"
    )

    print(
        f"Velocity median RMSE : "
        f"{median_velocity_rmse:.6f} m/s"
    )


# ======================================================================
# PERSISTENCE BASELINE
# ======================================================================

print()
print("=" * 70)
print("PERSISTENCE BASELINE")
print("=" * 70)

persistence_position_mae = np.nan
persistence_position_rmse = np.nan

persistence_velocity_mae = np.nan
persistence_velocity_rmse = np.nan

if sequence_column is not None:

    print(
        "Sequence boundaries detected."
    )

    persistence_position = np.full(
        len(pred_df),
        np.nan,
        dtype=np.float64
    )

    persistence_velocity = np.full(
        len(pred_df),
        np.nan,
        dtype=np.float64
    )

    # We need previous actual target inside each sequence.
    for sequence_id, group in pred_df.groupby(
        sequence_column,
        sort=False
    ):

        indices = group.index.to_numpy()

        position_actual_values = (
            group[POSITION_ACTUAL]
            .to_numpy(dtype=np.float64)
        )

        velocity_actual_values = (
            group[VELOCITY_ACTUAL]
            .to_numpy(dtype=np.float64)
        )

        if len(indices) > 1:

            persistence_position[
                indices[1:]
            ] = position_actual_values[:-1]

            persistence_velocity[
                indices[1:]
            ] = velocity_actual_values[:-1]

    persistence_position_mae = mae(
        pred_df[POSITION_ACTUAL],
        persistence_position
    )

    persistence_position_rmse = rmse(
        pred_df[POSITION_ACTUAL],
        persistence_position
    )

    persistence_velocity_mae = mae(
        pred_df[VELOCITY_ACTUAL],
        persistence_velocity
    )

    persistence_velocity_rmse = rmse(
        pred_df[VELOCITY_ACTUAL],
        persistence_velocity
    )

    print(
        f"Position persistence MAE  : "
        f"{persistence_position_mae:.6f} m"
    )

    print(
        f"Position persistence RMSE : "
        f"{persistence_position_rmse:.6f} m"
    )

    print(
        f"Velocity persistence MAE  : "
        f"{persistence_velocity_mae:.6f} m/s"
    )

    print(
        f"Velocity persistence RMSE : "
        f"{persistence_velocity_rmse:.6f} m/s"
    )

else:

    print("""
[WARNING]
The Stage 8 prediction file does not contain sequence IDs.

Persistence baseline cannot be safely calculated because
sequence boundaries are required.

This is NOT treated as a model failure.
""")


# ======================================================================
# OVERALL COMPARISON TABLE
# ======================================================================

comparison_records = []


comparison_records.append({
    "target": "position_error_m",
    "model": "LSTM",
    "MAE": lstm_position_mae,
    "RMSE": lstm_position_rmse
})

if np.isfinite(median_position_mae):

    comparison_records.append({
        "target": "position_error_m",
        "model": "TRAIN_MEDIAN",
        "MAE": median_position_mae,
        "RMSE": median_position_rmse
    })


if np.isfinite(persistence_position_mae):

    comparison_records.append({
        "target": "position_error_m",
        "model": "PERSISTENCE",
        "MAE": persistence_position_mae,
        "RMSE": persistence_position_rmse
    })


comparison_records.append({
    "target": "velocity_error_m_s",
    "model": "LSTM",
    "MAE": lstm_velocity_mae,
    "RMSE": lstm_velocity_rmse
})


if np.isfinite(median_velocity_mae):

    comparison_records.append({
        "target": "velocity_error_m_s",
        "model": "TRAIN_MEDIAN",
        "MAE": median_velocity_mae,
        "RMSE": median_velocity_rmse
    })


if np.isfinite(persistence_velocity_mae):

    comparison_records.append({
        "target": "velocity_error_m_s",
        "model": "PERSISTENCE",
        "MAE": persistence_velocity_mae,
        "RMSE": persistence_velocity_rmse
    })


comparison_df = pd.DataFrame(
    comparison_records
)


comparison_path = os.path.join(
    DIAGNOSTIC_DIR,
    "stage8_diagnostic3_overall_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print()
print("[SAVED]")
print(comparison_path)


# ======================================================================
# IMPROVEMENT ANALYSIS
# ======================================================================

print()
print("=" * 70)
print("LSTM IMPROVEMENT ANALYSIS")
print("=" * 70)


position_median_improvement = (
    percentage_improvement(
        median_position_mae,
        lstm_position_mae
    )
)

velocity_median_improvement = (
    percentage_improvement(
        median_velocity_mae,
        lstm_velocity_mae
    )
)

position_persistence_improvement = (
    percentage_improvement(
        persistence_position_mae,
        lstm_position_mae
    )
)

velocity_persistence_improvement = (
    percentage_improvement(
        persistence_velocity_mae,
        lstm_velocity_mae
    )
)


if np.isfinite(position_median_improvement):

    print(
        f"Position vs TRAIN MEDIAN: "
        f"{position_median_improvement:.2f}%"
    )

if np.isfinite(velocity_median_improvement):

    print(
        f"Velocity vs TRAIN MEDIAN: "
        f"{velocity_median_improvement:.2f}%"
    )

if np.isfinite(position_persistence_improvement):

    print(
        f"Position vs PERSISTENCE: "
        f"{position_persistence_improvement:.2f}%"
    )

if np.isfinite(velocity_persistence_improvement):

    print(
        f"Velocity vs PERSISTENCE: "
        f"{velocity_persistence_improvement:.2f}%"
    )


# ======================================================================
# PER-SEQUENCE ANALYSIS
# ======================================================================

print()
print("=" * 70)
print("PER-SEQUENCE TEST ANALYSIS")
print("=" * 70)


per_sequence_records = []


if sequence_column is not None:

    grouped = pred_df.groupby(
        sequence_column,
        sort=False
    )

    print(
        f"Test sequences detected: "
        f"{len(grouped)}"
    )

    for counter, (sequence_id, group) in enumerate(
        grouped,
        start=1
    ):

        y_pos = group[
            POSITION_ACTUAL
        ].to_numpy(dtype=np.float64)

        p_pos = group[
            POSITION_PREDICTED
        ].to_numpy(dtype=np.float64)

        y_vel = group[
            VELOCITY_ACTUAL
        ].to_numpy(dtype=np.float64)

        p_vel = group[
            VELOCITY_PREDICTED
        ].to_numpy(dtype=np.float64)


        # --------------------------------------------------------------
        # LSTM
        # --------------------------------------------------------------

        seq_position_mae = mae(
            y_pos,
            p_pos
        )

        seq_position_rmse = rmse(
            y_pos,
            p_pos
        )

        seq_velocity_mae = mae(
            y_vel,
            p_vel
        )

        seq_velocity_rmse = rmse(
            y_vel,
            p_vel
        )


        # --------------------------------------------------------------
        # Error percentiles
        # --------------------------------------------------------------

        seq_position_p95 = percentile_error(
            y_pos,
            p_pos,
            95
        )

        seq_position_p99 = percentile_error(
            y_pos,
            p_pos,
            99
        )

        seq_velocity_p95 = percentile_error(
            y_vel,
            p_vel,
            95
        )

        seq_velocity_p99 = percentile_error(
            y_vel,
            p_vel,
            99
        )


        # --------------------------------------------------------------
        # Large-error rates
        # --------------------------------------------------------------

        pos_mask = (
            np.isfinite(y_pos)
            &
            np.isfinite(p_pos)
        )

        vel_mask = (
            np.isfinite(y_vel)
            &
            np.isfinite(p_vel)
        )


        if pos_mask.sum() > 0:

            pos_abs_error = np.abs(
                y_pos[pos_mask]
                -
                p_pos[pos_mask]
            )

            position_over_100 = (
                np.mean(
                    pos_abs_error > 100.0
                )
                * 100.0
            )

            position_over_200 = (
                np.mean(
                    pos_abs_error > 200.0
                )
                * 100.0
            )

        else:

            position_over_100 = np.nan
            position_over_200 = np.nan


        if vel_mask.sum() > 0:

            vel_abs_error = np.abs(
                y_vel[vel_mask]
                -
                p_vel[vel_mask]
            )

            velocity_over_1 = (
                np.mean(
                    vel_abs_error > 1.0
                )
                * 100.0
            )

            velocity_over_2 = (
                np.mean(
                    vel_abs_error > 2.0
                )
                * 100.0
            )

        else:

            velocity_over_1 = np.nan
            velocity_over_2 = np.nan


        # --------------------------------------------------------------
        # Target statistics
        # --------------------------------------------------------------

        position_actual_mean = safe_mean(
            y_pos
        )

        position_actual_median = (
            float(
                np.nanmedian(y_pos)
            )
            if np.isfinite(y_pos).any()
            else np.nan
        )

        velocity_actual_mean = safe_mean(
            y_vel
        )

        velocity_actual_median = (
            float(
                np.nanmedian(y_vel)
            )
            if np.isfinite(y_vel).any()
            else np.nan
        )


        per_sequence_records.append({

            "sequence": str(sequence_id),

            "rows": len(group),

            "position_actual_mean_m":
                position_actual_mean,

            "position_actual_median_m":
                position_actual_median,

            "position_LSTM_MAE_m":
                seq_position_mae,

            "position_LSTM_RMSE_m":
                seq_position_rmse,

            "position_LSTM_P95_error_m":
                seq_position_p95,

            "position_LSTM_P99_error_m":
                seq_position_p99,

            "position_error_over_100m_percent":
                position_over_100,

            "position_error_over_200m_percent":
                position_over_200,

            "velocity_actual_mean_m_s":
                velocity_actual_mean,

            "velocity_actual_median_m_s":
                velocity_actual_median,

            "velocity_LSTM_MAE_m_s":
                seq_velocity_mae,

            "velocity_LSTM_RMSE_m_s":
                seq_velocity_rmse,

            "velocity_LSTM_P95_error_m_s":
                seq_velocity_p95,

            "velocity_LSTM_P99_error_m_s":
                seq_velocity_p99,

            "velocity_error_over_1m_s_percent":
                velocity_over_1,

            "velocity_error_over_2m_s_percent":
                velocity_over_2
        })


        if (
            counter == 1
            or counter % 5 == 0
            or counter == len(grouped)
        ):

            print(
                f"  {counter}/{len(grouped)} "
                f"{sequence_id}"
            )


else:

    print(
        "[SKIPPED] Sequence column unavailable."
    )


# ======================================================================
# SAVE PER-SEQUENCE RESULTS
# ======================================================================

per_sequence_df = pd.DataFrame(
    per_sequence_records
)

per_sequence_path = os.path.join(
    DIAGNOSTIC_DIR,
    "stage8_diagnostic3_per_sequence.csv"
)

per_sequence_df.to_csv(
    per_sequence_path,
    index=False
)

print()
print("[SAVED]")
print(per_sequence_path)


# ======================================================================
# WORST SEQUENCES
# ======================================================================

print()
print("=" * 70)
print("WORST TEST SEQUENCES")
print("=" * 70)


worst_position = []

worst_velocity = []


if len(per_sequence_df) > 0:

    worst_position_df = (
        per_sequence_df
        .sort_values(
            "position_LSTM_MAE_m",
            ascending=False
        )
        .head(5)
    )

    worst_velocity_df = (
        per_sequence_df
        .sort_values(
            "velocity_LSTM_MAE_m_s",
            ascending=False
        )
        .head(5)
    )


    print()
    print("TOP 5 WORST POSITION SEQUENCES")

    for _, row in worst_position_df.iterrows():

        print(
            f"  {row['sequence']}: "
            f"{row['position_LSTM_MAE_m']:.3f} m MAE"
        )

        worst_position.append({
            "sequence":
                str(row["sequence"]),
            "MAE":
                float(row["position_LSTM_MAE_m"])
        })


    print()
    print("TOP 5 WORST VELOCITY SEQUENCES")

    for _, row in worst_velocity_df.iterrows():

        print(
            f"  {row['sequence']}: "
            f"{row['velocity_LSTM_MAE_m_s']:.3f} m/s MAE"
        )

        worst_velocity.append({
            "sequence":
                str(row["sequence"]),
            "MAE":
                float(row["velocity_LSTM_MAE_m_s"])
        })


# ======================================================================
# LARGE ERROR ANALYSIS
# ======================================================================

print()
print("=" * 70)
print("OVERALL LARGE-ERROR ANALYSIS")
print("=" * 70)


position_abs_error = np.abs(
    pred_df[POSITION_ACTUAL]
    -
    pred_df[POSITION_PREDICTED]
)

velocity_abs_error = np.abs(
    pred_df[VELOCITY_ACTUAL]
    -
    pred_df[VELOCITY_PREDICTED]
)


position_valid_errors = (
    position_abs_error[
        np.isfinite(position_abs_error)
    ]
)

velocity_valid_errors = (
    velocity_abs_error[
        np.isfinite(velocity_abs_error)
    ]
)


position_over_100 = (
    float(
        np.mean(
            position_valid_errors > 100
        ) * 100
    )
    if len(position_valid_errors) > 0
    else np.nan
)


position_over_200 = (
    float(
        np.mean(
            position_valid_errors > 200
        ) * 100
    )
    if len(position_valid_errors) > 0
    else np.nan
)


position_over_500 = (
    float(
        np.mean(
            position_valid_errors > 500
        ) * 100
    )
    if len(position_valid_errors) > 0
    else np.nan
)


velocity_over_1 = (
    float(
        np.mean(
            velocity_valid_errors > 1
        ) * 100
    )
    if len(velocity_valid_errors) > 0
    else np.nan
)


velocity_over_2 = (
    float(
        np.mean(
            velocity_valid_errors > 2
        ) * 100
    )
    if len(velocity_valid_errors) > 0
    else np.nan
)


velocity_over_5 = (
    float(
        np.mean(
            velocity_valid_errors > 5
        ) * 100
    )
    if len(velocity_valid_errors) > 0
    else np.nan
)


print(
    f"Position error >100 m : "
    f"{position_over_100:.4f}%"
)

print(
    f"Position error >200 m : "
    f"{position_over_200:.4f}%"
)

print(
    f"Position error >500 m : "
    f"{position_over_500:.4f}%"
)

print(
    f"Velocity error >1 m/s : "
    f"{velocity_over_1:.4f}%"
)

print(
    f"Velocity error >2 m/s : "
    f"{velocity_over_2:.4f}%"
)

print(
    f"Velocity error >5 m/s : "
    f"{velocity_over_5:.4f}%"
)


# ======================================================================
# AUTOMATIC INTERPRETATION
# ======================================================================

print()
print("=" * 70)
print("AUTOMATIC DIAGNOSTIC INTERPRETATION")
print("=" * 70)


findings = []


# ----------------------------------------------------------------------
# Position baseline
# ----------------------------------------------------------------------

if np.isfinite(position_median_improvement):

    if position_median_improvement > 0:

        findings.append(
            "LSTM improves over the TRAIN-MEDIAN baseline "
            "for position error."
        )

        print(
            "[PASS] LSTM beats training-median "
            "position baseline."
        )

    else:

        findings.append(
            "LSTM does not improve over the TRAIN-MEDIAN "
            "baseline for position error."
        )

        print(
            "[WARNING] LSTM does not beat training-median "
            "position baseline."
        )


# ----------------------------------------------------------------------
# Velocity baseline
# ----------------------------------------------------------------------

if np.isfinite(velocity_median_improvement):

    if velocity_median_improvement > 0:

        findings.append(
            "LSTM improves over the TRAIN-MEDIAN baseline "
            "for velocity error."
        )

        print(
            "[PASS] LSTM beats training-median "
            "velocity baseline."
        )

    else:

        findings.append(
            "LSTM does not improve over the TRAIN-MEDIAN "
            "baseline for velocity error."
        )

        print(
            "[WARNING] LSTM does not beat training-median "
            "velocity baseline."
        )


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

if np.isfinite(position_persistence_improvement):

    if position_persistence_improvement > 0:

        print(
            "[PASS] LSTM beats persistence "
            "for position error."
        )

        findings.append(
            "LSTM improves over persistence "
            "for position error."
        )

    else:

        print(
            "[WARNING] LSTM does not beat persistence "
            "for position error."
        )

        findings.append(
            "LSTM does not improve over persistence "
            "for position error."
        )


if np.isfinite(velocity_persistence_improvement):

    if velocity_persistence_improvement > 0:

        print(
            "[PASS] LSTM beats persistence "
            "for velocity error."
        )

        findings.append(
            "LSTM improves over persistence "
            "for velocity error."
        )

    else:

        print(
            "[WARNING] LSTM does not beat persistence "
            "for velocity error."
        )

        findings.append(
            "LSTM does not improve over persistence "
            "for velocity error."
        )


# ----------------------------------------------------------------------
# Position error magnitude
# ----------------------------------------------------------------------

if lstm_position_mae < 50:

    print(
        "[PASS] Position MAE is below 50 m."
    )

elif lstm_position_mae < 100:

    print(
        "[INFO] Position MAE is between 50 m and 100 m."
    )

    findings.append(
        "Position MAE is moderate and requires "
        "navigation-specific interpretation."
    )

else:

    print(
        "[WARNING] Position MAE exceeds 100 m."
    )

    findings.append(
        "Position MAE exceeds 100 m and should be investigated."
    )


# ----------------------------------------------------------------------
# Velocity error magnitude
# ----------------------------------------------------------------------

if lstm_velocity_mae < 1:

    print(
        "[PASS] Velocity MAE is below 1 m/s."
    )

elif lstm_velocity_mae < 2:

    print(
        "[INFO] Velocity MAE is between 1 and 2 m/s."
    )

else:

    print(
        "[WARNING] Velocity MAE exceeds 2 m/s."
    )


# ======================================================================
# FINAL SUMMARY OBJECT
# ======================================================================

summary = {

    "stage": "Stage 8 Diagnostic 3",

    "purpose":
        "LSTM versus baseline and per-sequence analysis",

    "model_retraining":
        False,

    "raw_data_modified":
        False,

    "previous_stages_modified":
        False,

    "uncategorised_dataset_used":
        False,

    "sequence_leakage_protection":
        True,

    "train_sequences":
        len(TRAIN_IDS),

    "validation_sequences":
        len(VALIDATION_IDS),

    "test_sequences":
        len(TEST_IDS),

    "test_prediction_rows":
        int(len(pred_df)),

    "lstm": {

        "position_MAE_m":
            lstm_position_mae,

        "position_RMSE_m":
            lstm_position_rmse,

        "velocity_MAE_m_s":
            lstm_velocity_mae,

        "velocity_RMSE_m_s":
            lstm_velocity_rmse
    },

    "train_median_baseline": {

        "position_MAE_m":
            median_position_mae,

        "position_RMSE_m":
            median_position_rmse,

        "velocity_MAE_m_s":
            median_velocity_mae,

        "velocity_RMSE_m_s":
            median_velocity_rmse
    },

    "persistence_baseline": {

        "position_MAE_m":
            persistence_position_mae,

        "position_RMSE_m":
            persistence_position_rmse,

        "velocity_MAE_m_s":
            persistence_velocity_mae,

        "velocity_RMSE_m_s":
            persistence_velocity_rmse
    },

    "improvement_percent": {

        "position_vs_train_median":
            position_median_improvement,

        "velocity_vs_train_median":
            velocity_median_improvement,

        "position_vs_persistence":
            position_persistence_improvement,

        "velocity_vs_persistence":
            velocity_persistence_improvement
    },

    "large_error_percent": {

        "position_over_100m":
            position_over_100,

        "position_over_200m":
            position_over_200,

        "position_over_500m":
            position_over_500,

        "velocity_over_1m_s":
            velocity_over_1,

        "velocity_over_2m_s":
            velocity_over_2,

        "velocity_over_5m_s":
            velocity_over_5
    },

    "worst_position_sequences":
        worst_position,

    "worst_velocity_sequences":
        worst_velocity,

    "findings":
        findings
}


# ======================================================================
# SAVE JSON
# ======================================================================

json_path = os.path.join(
    DIAGNOSTIC_DIR,
    "stage8_diagnostic3_summary.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=True
    )


print()
print("[SAVED]")
print(json_path)


# ======================================================================
# SAVE HUMAN-READABLE SUMMARY
# ======================================================================

txt_path = os.path.join(
    DIAGNOSTIC_DIR,
    "stage8_diagnostic3_summary.txt"
)


with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NAV-SHIELD STAGE 8 DIAGNOSTIC 3\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "PURPOSE\n"
    )

    f.write(
        "Compare Stage 8 LSTM against simple baselines "
        "and analyze test-sequence performance.\n\n"
    )

    f.write(
        "MODEL RETRAINING: NO\n"
    )

    f.write(
        "RAW DATA MODIFIED: NO\n"
    )

    f.write(
        "PREVIOUS STAGES MODIFIED: NO\n"
    )

    f.write(
        "UNCATEGORISED DATASET USED: NO\n\n"
    )

    f.write(
        "SEQUENCE SPLIT\n"
    )

    f.write(
        f"TRAIN: {len(TRAIN_IDS)}\n"
    )

    f.write(
        f"VALIDATION: {len(VALIDATION_IDS)}\n"
    )

    f.write(
        f"TEST: {len(TEST_IDS)}\n\n"
    )

    f.write(
        "LSTM PERFORMANCE\n"
    )

    f.write(
        f"Position MAE : "
        f"{lstm_position_mae:.6f} m\n"
    )

    f.write(
        f"Position RMSE: "
        f"{lstm_position_rmse:.6f} m\n"
    )

    f.write(
        f"Velocity MAE : "
        f"{lstm_velocity_mae:.6f} m/s\n"
    )

    f.write(
        f"Velocity RMSE: "
        f"{lstm_velocity_rmse:.6f} m/s\n\n"
    )

    f.write(
        "TRAIN-MEDIAN BASELINE\n"
    )

    f.write(
        f"Position MAE : "
        f"{median_position_mae:.6f} m\n"
    )

    f.write(
        f"Position RMSE: "
        f"{median_position_rmse:.6f} m\n"
    )

    f.write(
        f"Velocity MAE : "
        f"{median_velocity_mae:.6f} m/s\n"
    )

    f.write(
        f"Velocity RMSE: "
        f"{median_velocity_rmse:.6f} m/s\n\n"
    )

    f.write(
        "PERSISTENCE BASELINE\n"
    )

    f.write(
        f"Position MAE : "
        f"{persistence_position_mae:.6f} m\n"
    )

    f.write(
        f"Position RMSE: "
        f"{persistence_position_rmse:.6f} m\n"
    )

    f.write(
        f"Velocity MAE : "
        f"{persistence_velocity_mae:.6f} m/s\n"
    )

    f.write(
        f"Velocity RMSE: "
        f"{persistence_velocity_rmse:.6f} m/s\n\n"
    )

    f.write(
        "LSTM IMPROVEMENT\n"
    )

    f.write(
        f"Position vs TRAIN MEDIAN: "
        f"{position_median_improvement:.4f}%\n"
    )

    f.write(
        f"Velocity vs TRAIN MEDIAN: "
        f"{velocity_median_improvement:.4f}%\n"
    )

    f.write(
        f"Position vs PERSISTENCE: "
        f"{position_persistence_improvement:.4f}%\n"
    )

    f.write(
        f"Velocity vs PERSISTENCE: "
        f"{velocity_persistence_improvement:.4f}%\n\n"
    )

    f.write(
        "LARGE ERROR RATES\n"
    )

    f.write(
        f"Position >100m: "
        f"{position_over_100:.4f}%\n"
    )

    f.write(
        f"Position >200m: "
        f"{position_over_200:.4f}%\n"
    )

    f.write(
        f"Position >500m: "
        f"{position_over_500:.4f}%\n"
    )

    f.write(
        f"Velocity >1m/s: "
        f"{velocity_over_1:.4f}%\n"
    )

    f.write(
        f"Velocity >2m/s: "
        f"{velocity_over_2:.4f}%\n"
    )

    f.write(
        f"Velocity >5m/s: "
        f"{velocity_over_5:.4f}%\n\n"
    )

    f.write(
        "AUTOMATIC FINDINGS\n"
    )

    for finding in findings:

        f.write(
            f"- {finding}\n"
        )

    f.write(
        "\n"
        "IMPORTANT:\n"
        "- No model retraining was performed.\n"
        "- No raw data was modified.\n"
        "- No Stage 1-8 output was modified.\n"
        "- Uncategorised dataset was not used.\n"
        "- Sequence separation was preserved.\n"
    )


print()
print("[SAVED]")
print(txt_path)


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print()
print("=" * 70)
print("STAGE 8 DIAGNOSTIC 3 COMPLETE")
print("=" * 70)

print()
print("Generated files:")

print(
    "  - stage8_diagnostic3_overall_comparison.csv"
)

print(
    "  - stage8_diagnostic3_per_sequence.csv"
)

print(
    "  - stage8_diagnostic3_summary.json"
)

print(
    "  - stage8_diagnostic3_summary.txt"
)

print()
print(
    "No model retraining was performed."
)

print(
    "No raw data was modified."
)

print(
    "No previous-stage output was modified."
)

print(
    "Uncategorised dataset was NOT used."
)

print(
    "Sequence leakage protection was preserved."
)

print("=" * 70)