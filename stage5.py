"""
======================================================================
NAV-SHIELD STAGE 5: AI DRIFT DETECTION
======================================================================

Purpose
-------
Train an unsupervised AI model to identify navigation states that
deviate from normal training behaviour.

Model
-----
Isolation Forest

Data policy
-----------
TRAIN:
    Used to fit the AI model.

VALIDATION:
    Used only for threshold selection.

TEST:
    Used only for final evaluation.

UNCATEGORISED DATASET:
    NOT used for model training or threshold selection.

Important
---------
There are currently no verified row-level "drift / no-drift" labels.
Therefore this stage performs UNSUPERVISED anomaly/drift detection.
It must not report classification accuracy, precision, recall, or F1
as if ground-truth labels existed.

No raw data is modified.
No Stage 1-4 outputs are overwritten.
======================================================================
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_SEED = 42

# Percentage of validation observations considered anomalous
# for threshold selection.
VALIDATION_ANOMALY_PERCENTILE = 99.0

# Isolation Forest configuration.
#
# 100,000 training samples are enough for a first robust model while
# avoiding unnecessarily large memory/runtime requirements.
MAX_TRAINING_SAMPLES = 100_000

N_ESTIMATORS = 200

CONTAMINATION = "auto"

N_JOBS = -1


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

STAGE4_DIR = BASE_DIR / "data" / "stage4"

TRAIN_DIR = STAGE4_DIR / "train"
VALIDATION_DIR = STAGE4_DIR / "validation"
TEST_DIR = STAGE4_DIR / "test"

OUTPUT_DIR = BASE_DIR / "results" / "stage5"

MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"

for directory in [
    MODEL_DIR,
    PREDICTION_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ======================================================================
# LOAD DATA
# ======================================================================

def load_split(directory, split_name):
    """
    Load all ML-ready CSV files belonging to a split.

    sequence_id is retained separately and is NOT used as a feature.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"{split_name} directory does not exist:\n"
            f"{directory}"
        )

    files = sorted(
        directory.glob("*_ml_ready.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No ML-ready CSV files found in:\n"
            f"{directory}"
        )

    print(
        f"\nLoading {split_name}: "
        f"{len(files)} sequence files"
    )

    frames = []

    for file in files:

        df = pd.read_csv(file)

        if df.empty:
            print(
                f"[WARNING] Empty file: {file.name}"
            )
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No valid data found for {split_name}."
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        f"{split_name} rows: "
        f"{len(combined):,}"
    )

    return combined


# ======================================================================
# FEATURE PREPARATION
# ======================================================================

def prepare_features(df, split_name):
    """
    Extract numeric ML features.

    sequence_id is metadata only.
    """

    excluded_columns = {
        "sequence_id"
    }

    feature_columns = []

    for column in df.columns:

        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            feature_columns.append(column)

    if not feature_columns:
        raise RuntimeError(
            f"No numeric ML features found in {split_name}."
        )

    X = df[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Stage 4 should already have removed missing values through
    # training-only imputation. This check ensures that Stage 5
    # doesn't silently accept unexpected missing values.
    missing_count = int(
        X.isna().sum().sum()
    )

    if missing_count > 0:
        raise RuntimeError(
            f"{split_name} still contains "
            f"{missing_count} missing feature values."
        )

    infinite_count = int(
        np.isinf(X.to_numpy()).sum()
    )

    if infinite_count > 0:
        raise RuntimeError(
            f"{split_name} contains "
            f"{infinite_count} infinite values."
        )

    return X, feature_columns


# ======================================================================
# TRAINING SAMPLE
# ======================================================================

def sample_training_data(X):
    """
    Randomly sample training observations if the training dataset
    is very large.

    Sampling happens ONLY from TRAIN.
    """

    if len(X) <= MAX_TRAINING_SAMPLES:
        return X

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = rng.choice(
        len(X),
        size=MAX_TRAINING_SAMPLES,
        replace=False
    )

    sampled = X.iloc[
        indices
    ].copy()

    return sampled


# ======================================================================
# MODEL TRAINING
# ======================================================================

def train_model(X_train):

    print()
    print("=" * 70)
    print("TRAINING ISOLATION FOREST")
    print("=" * 70)

    training_sample = sample_training_data(
        X_train
    )

    print(
        f"Full training rows : "
        f"{len(X_train):,}"
    )

    print(
        f"Rows used for model : "
        f"{len(training_sample):,}"
    )

    print(
        f"Features            : "
        f"{training_sample.shape[1]}"
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        max_samples="auto"
    )

    model.fit(
        training_sample
    )

    print(
        "[PASS] Isolation Forest trained."
    )

    return model


# ======================================================================
# SCORE GENERATION
# ======================================================================

def get_scores(model, X):
    """
    IsolationForest's decision_function:

        larger value = more normal
        smaller value = more anomalous

    We invert it so that:

        larger drift_score = more anomalous
    """

    normality_score = model.decision_function(
        X
    )

    drift_score = -normality_score

    return drift_score


# ======================================================================
# THRESHOLD SELECTION
# ======================================================================

def select_threshold(validation_scores):

    threshold = np.percentile(
        validation_scores,
        VALIDATION_ANOMALY_PERCENTILE
    )

    return float(threshold)


# ======================================================================
# PREDICTION GENERATION
# ======================================================================

def create_predictions(
    df,
    scores,
    threshold
):

    result = pd.DataFrame()

    if "sequence_id" in df.columns:
        result["sequence_id"] = (
            df["sequence_id"].values
        )

    result["drift_score"] = scores

    result["drift_detected"] = (
        scores >= threshold
    )

    return result


# ======================================================================
# SUMMARY STATISTICS
# ======================================================================

def summarize_scores(
    split_name,
    scores,
    predictions,
    threshold
):

    drift_count = int(
        predictions["drift_detected"].sum()
    )

    total = len(scores)

    drift_percentage = (
        100.0 * drift_count / total
        if total > 0
        else 0.0
    )

    summary = {
        "split": split_name,
        "rows": int(total),
        "drift_detected": drift_count,
        "drift_percentage": float(
            drift_percentage
        ),
        "score_min": float(
            np.min(scores)
        ),
        "score_max": float(
            np.max(scores)
        ),
        "score_mean": float(
            np.mean(scores)
        ),
        "score_median": float(
            np.median(scores)
        ),
        "threshold": float(threshold),
    }

    return summary


# ======================================================================
# SEQUENCE SUMMARY
# ======================================================================

def create_sequence_summary(
    predictions
):

    if "sequence_id" not in predictions.columns:
        return pd.DataFrame()

    grouped = (
        predictions
        .groupby("sequence_id")
        .agg(
            rows=(
                "drift_score",
                "count"
            ),
            mean_drift_score=(
                "drift_score",
                "mean"
            ),
            max_drift_score=(
                "drift_score",
                "max"
            ),
            drift_count=(
                "drift_detected",
                "sum"
            )
        )
        .reset_index()
    )

    grouped["drift_percentage"] = (
        grouped["drift_count"]
        / grouped["rows"]
        * 100.0
    )

    return grouped


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("NAV-SHIELD STAGE 5: AI DRIFT DETECTION")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print(
        "This is UNSUPERVISED drift/anomaly detection."
    )
    print(
        "No ground-truth drift labels are assumed."
    )
    print()

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    train_df = load_split(
        TRAIN_DIR,
        "TRAIN"
    )

    validation_df = load_split(
        VALIDATION_DIR,
        "VALIDATION"
    )

    test_df = load_split(
        TEST_DIR,
        "TEST"
    )

    # ------------------------------------------------------------------
    # PREPARE FEATURES
    # ------------------------------------------------------------------

    X_train, feature_columns = prepare_features(
        train_df,
        "TRAIN"
    )

    X_validation, validation_features = (
        prepare_features(
            validation_df,
            "VALIDATION"
        )
    )

    X_test, test_features = prepare_features(
        test_df,
        "TEST"
    )

    # ------------------------------------------------------------------
    # FEATURE CONSISTENCY CHECK
    # ------------------------------------------------------------------

    if feature_columns != validation_features:
        raise RuntimeError(
            "TRAIN and VALIDATION feature columns do not match."
        )

    if feature_columns != test_features:
        raise RuntimeError(
            "TRAIN and TEST feature columns do not match."
        )

    print()
    print(
        f"Feature consistency check: PASS"
    )

    print(
        f"Features used: {len(feature_columns)}"
    )

    # ------------------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------------------

    model = train_model(
        X_train
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validation_scores = get_scores(
        model,
        X_validation
    )

    threshold = select_threshold(
        validation_scores
    )

    validation_predictions = create_predictions(
        validation_df,
        validation_scores,
        threshold
    )

    validation_summary = summarize_scores(
        "validation",
        validation_scores,
        validation_predictions,
        threshold
    )

    print(
        f"Validation threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation drift detections: "
        f"{validation_summary['drift_detected']:,}"
    )

    print(
        f"Validation drift percentage: "
        f"{validation_summary['drift_percentage']:.4f}%"
    )

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    test_scores = get_scores(
        model,
        X_test
    )

    test_predictions = create_predictions(
        test_df,
        test_scores,
        threshold
    )

    test_summary = summarize_scores(
        "test",
        test_scores,
        test_predictions,
        threshold
    )

    print(
        f"Test rows: "
        f"{test_summary['rows']:,}"
    )

    print(
        f"Test drift detections: "
        f"{test_summary['drift_detected']:,}"
    )

    print(
        f"Test drift percentage: "
        f"{test_summary['drift_percentage']:.4f}%"
    )

    print(
        f"Test score mean: "
        f"{test_summary['score_mean']:.6f}"
    )

    print(
        f"Test score median: "
        f"{test_summary['score_median']:.6f}"
    )

    # ------------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------------

    model_file = (
        MODEL_DIR /
        "nav_shield_isolation_forest.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print()
    print(
        f"[SAVED] Model:\n{model_file}"
    )

    # ------------------------------------------------------------------
    # SAVE FEATURE LIST
    # ------------------------------------------------------------------

    feature_file = (
        MODEL_DIR /
        "feature_columns.json"
    )

    with open(
        feature_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2
        )

    # ------------------------------------------------------------------
    # SAVE VALIDATION PREDICTIONS
    # ------------------------------------------------------------------

    validation_prediction_file = (
        PREDICTION_DIR /
        "validation_drift_predictions.csv"
    )

    validation_predictions.to_csv(
        validation_prediction_file,
        index=False
    )

    # ------------------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # ------------------------------------------------------------------

    test_prediction_file = (
        PREDICTION_DIR /
        "test_drift_predictions.csv"
    )

    test_predictions.to_csv(
        test_prediction_file,
        index=False
    )

    # ------------------------------------------------------------------
    # SEQUENCE SUMMARIES
    # ------------------------------------------------------------------

    validation_sequence_summary = (
        create_sequence_summary(
            validation_predictions
        )
    )

    test_sequence_summary = (
        create_sequence_summary(
            test_predictions
        )
    )

    validation_sequence_file = (
        PREDICTION_DIR /
        "validation_sequence_summary.csv"
    )

    test_sequence_file = (
        PREDICTION_DIR /
        "test_sequence_summary.csv"
    )

    validation_sequence_summary.to_csv(
        validation_sequence_file,
        index=False
    )

    test_sequence_summary.to_csv(
        test_sequence_file,
        index=False
    )

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    report = {
        "stage": "5",
        "stage_name": "AI Drift Detection",

        "model": {
            "type": "IsolationForest",
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "random_seed": RANDOM_SEED,
        },

        "data_policy": {
            "training": "used for model fitting",
            "validation": "used for threshold selection",
            "test": "used for final evaluation",
            "uncategorised": "not used",
        },

        "learning_type": "unsupervised",

        "ground_truth_labels_available": False,

        "feature_count": len(
            feature_columns
        ),

        "features": feature_columns,

        "training_rows": int(
            len(train_df)
        ),

        "validation_rows": int(
            len(validation_df)
        ),

        "test_rows": int(
            len(test_df)
        ),

        "training_sample_used": int(
            min(
                len(X_train),
                MAX_TRAINING_SAMPLES
            )
        ),

        "validation": validation_summary,

        "test": test_summary,

        "threshold_policy": {
            "method": (
                "validation score "
                f"{VALIDATION_ANOMALY_PERCENTILE}th percentile"
            ),
            "threshold": threshold,
        },
    }

    report_file = (
        REPORT_DIR /
        "stage5_summary.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ------------------------------------------------------------------
    # HUMAN-READABLE REPORT
    # ------------------------------------------------------------------

    text_report_file = (
        REPORT_DIR /
        "stage5_summary.txt"
    )

    with open(
        text_report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "NAV-SHIELD STAGE 5: AI DRIFT DETECTION\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Learning type: UNSUPERVISED\n"
        )

        f.write(
            "Model: Isolation Forest\n"
        )

        f.write(
            f"Features: {len(feature_columns)}\n"
        )

        f.write(
            f"Training rows: {len(train_df):,}\n"
        )

        f.write(
            f"Validation rows: {len(validation_df):,}\n"
        )

        f.write(
            f"Test rows: {len(test_df):,}\n\n"
        )

        f.write(
            f"Validation threshold: "
            f"{threshold:.6f}\n\n"
        )

        f.write(
            "VALIDATION\n"
        )

        f.write(
            f"Drift detections: "
            f"{validation_summary['drift_detected']:,}\n"
        )

        f.write(
            f"Drift percentage: "
            f"{validation_summary['drift_percentage']:.4f}%\n\n"
        )

        f.write(
            "TEST\n"
        )

        f.write(
            f"Drift detections: "
            f"{test_summary['drift_detected']:,}\n"
        )

        f.write(
            f"Drift percentage: "
            f"{test_summary['drift_percentage']:.4f}%\n"
        )

        f.write(
            f"Mean drift score: "
            f"{test_summary['score_mean']:.6f}\n"
        )

        f.write(
            f"Median drift score: "
            f"{test_summary['score_median']:.6f}\n\n"
        )

        f.write(
            "DATASET POLICY\n"
        )

        f.write(
            "Uncategorised dataset: NOT USED\n"
        )

        f.write(
            "Raw data modified: NO\n"
        )

        f.write(
            "Stage 1-4 outputs modified: NO\n"
        )

        f.write(
            "Ground-truth labels assumed: NO\n"
        )

    # ------------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 5 SUMMARY")
    print("=" * 70)

    print(
        "Model                 : Isolation Forest"
    )

    print(
        "Learning type         : UNSUPERVISED"
    )

    print(
        f"Features              : "
        f"{len(feature_columns)}"
    )

    print(
        f"Training rows         : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows       : "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows             : "
        f"{len(test_df):,}"
    )

    print(
        f"Validation threshold  : "
        f"{threshold:.6f}"
    )

    print(
        f"Test drift detections : "
        f"{test_summary['drift_detected']:,}"
    )

    print(
        f"Test drift percentage  : "
        f"{test_summary['drift_percentage']:.4f}%"
    )

    print()
    print(
        "Uncategorised dataset : NOT USED"
    )

    print(
        "Ground-truth labels   : NOT ASSUMED"
    )

    print()
    print(
        f"Model saved to:\n{model_file}"
    )

    print(
        f"Reports saved to:\n{REPORT_DIR}"
    )

    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()