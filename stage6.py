"""
======================================================================
NAV-SHIELD STAGE 6: DRIFT ANALYSIS & VALIDATION
======================================================================

Purpose
-------
Analyze the anomalies detected by the Stage 5 Isolation Forest.

Stage 6 DOES NOT:
    - retrain the model
    - modify Stage 1-5 outputs
    - perform imputation
    - perform scaling
    - change the Stage 5 threshold
    - use the Uncategorised dataset for training

Stage 6 DOES:
    - load the Stage 5 model
    - load the TEST ML-ready data
    - reproduce Stage 5 drift scores
    - identify anomalous samples
    - summarize anomalies by sequence
    - analyze temporal concentration
    - analyze feature abnormality
    - generate reports
    - generate diagnostic plots

Important
---------
Because Stage 5 is UNSUPERVISED, a detected anomaly is NOT automatically
a confirmed real-world navigation failure.

The purpose of this stage is to determine whether anomalies show
structured sensor/navigation behaviour rather than random noise.
======================================================================
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_SEED = 42

# Number of top features to report.
TOP_FEATURES = 20

# Number of individual anomaly rows to save.
MAX_ANOMALY_ROWS = 10000


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

STAGE4_DIR = BASE_DIR / "data" / "stage4"

TEST_DIR = STAGE4_DIR / "test"

STAGE5_DIR = BASE_DIR / "results" / "stage5"

MODEL_FILE = (
    STAGE5_DIR
    / "models"
    / "nav_shield_isolation_forest.joblib"
)

STAGE5_REPORT = (
    STAGE5_DIR
    / "reports"
    / "stage5_summary.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "stage6"
)

REPORT_DIR = OUTPUT_DIR / "reports"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

PLOT_DIR = OUTPUT_DIR / "plots"

for directory in [
    REPORT_DIR,
    PREDICTION_DIR,
    PLOT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ======================================================================
# LOAD STAGE 5 REPORT
# ======================================================================

def load_stage5_report():

    if not STAGE5_REPORT.exists():
        raise FileNotFoundError(
            "Stage 5 report not found:\n"
            f"{STAGE5_REPORT}"
        )

    with open(
        STAGE5_REPORT,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ======================================================================
# LOAD TEST DATA
# ======================================================================

def load_test_data():

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            "Stage 4 TEST directory not found:\n"
            f"{TEST_DIR}"
        )

    files = sorted(
        TEST_DIR.glob("*_ml_ready.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No Stage 4 TEST ML-ready files found."
        )

    print(
        f"Test sequence files found: {len(files)}"
    )

    frames = []

    for file in files:

        df = pd.read_csv(file)

        if df.empty:
            print(
                f"[WARNING] Empty file: {file.name}"
            )
            continue

        # If sequence_id does not exist, recover it from filename.
        if "sequence_id" not in df.columns:

            sequence_id = file.stem.replace(
                "_ml_ready",
                ""
            )

            df["sequence_id"] = sequence_id

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No usable TEST data found."
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    return combined


# ======================================================================
# FEATURE PREPARATION
# ======================================================================

def prepare_features(
    df,
    expected_features
):

    missing_features = [
        feature
        for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:

        raise RuntimeError(
            "TEST data is missing features expected by Stage 5:\n"
            + "\n".join(missing_features)
        )

    X = df[
        expected_features
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_count = int(
        X.isna().sum().sum()
    )

    if missing_count > 0:

        raise RuntimeError(
            f"TEST features contain "
            f"{missing_count:,} missing values."
        )

    return X


# ======================================================================
# REPRODUCE STAGE 5 SCORES
# ======================================================================

def calculate_scores(
    model,
    X
):

    normality_score = (
        model.decision_function(X)
    )

    drift_score = -normality_score

    return drift_score


# ======================================================================
# BUILD SAMPLE-LEVEL RESULTS
# ======================================================================

def build_results(
    df,
    drift_scores,
    threshold
):

    result = pd.DataFrame()

    result["sequence_id"] = (
        df["sequence_id"].astype(str).values
    )

    result["drift_score"] = drift_scores

    result["drift_detected"] = (
        drift_scores >= threshold
    )

    # --------------------------------------------------------------
    # Preserve useful metadata columns if available.
    # --------------------------------------------------------------

    preferred_columns = [

        # Time
        "TIME SINCE START (ms)",
        "time_since_start_ms",
        "Time Since Start of Day (seconds)",

        # GPS
        "GPS LATITUDE (degrees)",
        "GPS LONGITUDE (degrees)",
        "GPS SPEED (Kmh)",
        "GPS ACCURACY (m)",

        "Latitude (degrees)",
        "Longitude (degrees)",
        "Velocity (km/hr)",
        "Indicated Vehicle Speed (km/hr)",

        # Vehicle
        "Steering Angle (degrees)",
        "Yaw Rate (deg/sec)",
        "Brake Position (0 or 1)",
        "Accelerator Pedal Position (0 or 1)",
        "Gear",

        # Sensor
        "ACCELEROMETER X (m/s²)",
        "ACCELEROMETER Y (m/s²)",
        "ACCELEROMETER Z (m/s²)",

        "GYROSCOPE X (rad/s)",
        "GYROSCOPE Y (rad/s)",
        "GYROSCOPE Z (rad/s)",
    ]

    for column in preferred_columns:

        if column in df.columns:

            result[column] = (
                df[column].values
            )

    return result


# ======================================================================
# SEQUENCE ANALYSIS
# ======================================================================

def analyze_sequences(
    results
):

    sequence_summary = (

        results
        .groupby("sequence_id")
        .agg(
            total_samples=(
                "drift_score",
                "count"
            ),

            drift_samples=(
                "drift_detected",
                "sum"
            ),

            mean_drift_score=(
                "drift_score",
                "mean"
            ),

            max_drift_score=(
                "drift_score",
                "max"
            ),

            median_drift_score=(
                "drift_score",
                "median"
            )
        )
        .reset_index()
    )

    sequence_summary[
        "drift_percentage"
    ] = (

        sequence_summary["drift_samples"]
        /
        sequence_summary["total_samples"]
        *
        100.0
    )

    sequence_summary = (
        sequence_summary
        .sort_values(
            "drift_percentage",
            ascending=False
        )
    )

    return sequence_summary


# ======================================================================
# FEATURE ABNORMALITY ANALYSIS
# ======================================================================

def analyze_features(
    df,
    feature_columns,
    anomaly_mask
):

    normal_df = df.loc[
        ~anomaly_mask,
        feature_columns
    ]

    anomaly_df = df.loc[
        anomaly_mask,
        feature_columns
    ]

    rows = []

    for feature in feature_columns:

        normal_values = pd.to_numeric(
            normal_df[feature],
            errors="coerce"
        )

        anomaly_values = pd.to_numeric(
            anomaly_df[feature],
            errors="coerce"
        )

        normal_values = (
            normal_values
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        anomaly_values = (
            anomaly_values
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(normal_values) == 0:
            continue

        if len(anomaly_values) == 0:
            continue

        normal_mean = float(
            normal_values.mean()
        )

        normal_std = float(
            normal_values.std()
        )

        anomaly_mean = float(
            anomaly_values.mean()
        )

        anomaly_std = float(
            anomaly_values.std()
        )

        if normal_std > 1e-12:

            standardized_difference = (
                abs(
                    anomaly_mean
                    -
                    normal_mean
                )
                /
                normal_std
            )

        else:

            standardized_difference = 0.0

        rows.append(
            {
                "feature": feature,

                "normal_mean":
                    normal_mean,

                "normal_std":
                    normal_std,

                "anomaly_mean":
                    anomaly_mean,

                "anomaly_std":
                    anomaly_std,

                "standardized_difference":
                    float(
                        standardized_difference
                    )
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "feature",
                "normal_mean",
                "normal_std",
                "anomaly_mean",
                "anomaly_std",
                "standardized_difference"
            ]
        )

    feature_summary = pd.DataFrame(
        rows
    )

    feature_summary = (
        feature_summary
        .sort_values(
            "standardized_difference",
            ascending=False
        )
    )

    return feature_summary


# ======================================================================
# TEMPORAL ANALYSIS
# ======================================================================

def analyze_temporal_pattern(
    results
):

    time_column = None

    possible_time_columns = [

        "TIME SINCE START (ms)",

        "time_since_start_ms",

        "Time Since Start of Day (seconds)"
    ]

    for column in possible_time_columns:

        if column in results.columns:

            time_column = column
            break

    if time_column is None:

        return None

    temporal = results[
        [
            "sequence_id",
            time_column,
            "drift_score",
            "drift_detected"
        ]
    ].copy()

    temporal[
        time_column
    ] = pd.to_numeric(
        temporal[time_column],
        errors="coerce"
    )

    temporal = temporal.dropna(
        subset=[time_column]
    )

    return temporal


# ======================================================================
# PLOT 1: DRIFT SCORE DISTRIBUTION
# ======================================================================

def plot_score_distribution(
    results,
    threshold
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        results["drift_score"],
        bins=100
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.6f}"
    )

    plt.xlabel(
        "Drift Score"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "NAV-SHIELD Stage 6: Test Drift Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        PLOT_DIR
        / "test_drift_score_distribution.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    return output


# ======================================================================
# PLOT 2: DRIFT BY SEQUENCE
# ======================================================================

def plot_sequence_drift(
    sequence_summary
):

    if sequence_summary.empty:
        return None

    plot_df = (
        sequence_summary
        .sort_values(
            "drift_percentage",
            ascending=False
        )
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        plot_df["sequence_id"],
        plot_df["drift_percentage"]
    )

    plt.xlabel(
        "Sequence"
    )

    plt.ylabel(
        "Detected Drift (%)"
    )

    plt.title(
        "NAV-SHIELD Stage 6: Drift Percentage by Test Sequence"
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    output = (
        PLOT_DIR
        / "drift_percentage_by_sequence.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    return output


# ======================================================================
# PLOT 3: TOP FEATURE DIFFERENCES
# ======================================================================

def plot_feature_differences(
    feature_summary
):

    if feature_summary.empty:
        return None

    plot_df = (
        feature_summary
        .head(TOP_FEATURES)
        .sort_values(
            "standardized_difference"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        plot_df["feature"],
        plot_df[
            "standardized_difference"
        ]
    )

    plt.xlabel(
        "Absolute Standardized Difference"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Features Associated With Detected Anomalies"
    )

    plt.tight_layout()

    output = (
        PLOT_DIR
        / "top_anomaly_features.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    return output


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print(
        "NAV-SHIELD STAGE 6: DRIFT ANALYSIS & VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "Stage 5 model will NOT be retrained."
    )

    print(
        "Uncategorised dataset will NOT be used."
    )

    print(
        "Ground-truth drift labels are NOT assumed."
    )

    # ------------------------------------------------------------------
    # LOAD STAGE 5 REPORT
    # ------------------------------------------------------------------

    stage5_report = (
        load_stage5_report()
    )

    threshold = float(
        stage5_report[
            "threshold_policy"
        ][
            "threshold"
        ]
    )

    feature_columns = (
        stage5_report[
            "features"
        ]
    )

    print()
    print(
        f"Stage 5 threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Stage 5 features: "
        f"{len(feature_columns)}"
    )

    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Stage 5 model not found:\n"
            f"{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "[PASS] Stage 5 model loaded."
    )

    # ------------------------------------------------------------------
    # LOAD TEST
    # ------------------------------------------------------------------

    test_df = load_test_data()

    print(
        f"Test rows loaded: "
        f"{len(test_df):,}"
    )

    # ------------------------------------------------------------------
    # PREPARE FEATURES
    # ------------------------------------------------------------------

    X_test = prepare_features(
        test_df,
        feature_columns
    )

    print(
        "[PASS] Test feature consistency verified."
    )

    # ------------------------------------------------------------------
    # CALCULATE SCORES
    # ------------------------------------------------------------------

    drift_scores = calculate_scores(
        model,
        X_test
    )

    # ------------------------------------------------------------------
    # BUILD RESULTS
    # ------------------------------------------------------------------

    results = build_results(
        test_df,
        drift_scores,
        threshold
    )

    # ------------------------------------------------------------------
    # ANOMALY MASK
    # ------------------------------------------------------------------

    anomaly_mask = (
        results[
            "drift_detected"
        ].to_numpy()
    )

    anomaly_count = int(
        anomaly_mask.sum()
    )

    normal_count = (
        len(results)
        -
        anomaly_count
    )

    anomaly_percentage = (
        anomaly_count
        /
        len(results)
        *
        100.0
    )

    # ------------------------------------------------------------------
    # SEQUENCE ANALYSIS
    # ------------------------------------------------------------------

    sequence_summary = (
        analyze_sequences(
            results
        )
    )

    # ------------------------------------------------------------------
    # FEATURE ANALYSIS
    # ------------------------------------------------------------------

    feature_summary = (
        analyze_features(
            test_df,
            feature_columns,
            anomaly_mask
        )
    )

    # ------------------------------------------------------------------
    # TEMPORAL ANALYSIS
    # ------------------------------------------------------------------

    temporal = (
        analyze_temporal_pattern(
            results
        )
    )

    # ------------------------------------------------------------------
    # SAVE ALL ANOMALIES
    # ------------------------------------------------------------------

    anomaly_results = (
        results[
            results["drift_detected"]
        ]
        .sort_values(
            "drift_score",
            ascending=False
        )
    )

    anomaly_file = (
        PREDICTION_DIR
        / "test_detected_drift_samples.csv"
    )

    anomaly_results.to_csv(
        anomaly_file,
        index=False
    )

    # ------------------------------------------------------------------
    # SAVE FULL SCORES
    # ------------------------------------------------------------------

    full_prediction_file = (
        PREDICTION_DIR
        / "test_all_drift_scores.csv"
    )

    results.to_csv(
        full_prediction_file,
        index=False
    )

    # ------------------------------------------------------------------
    # SAVE SEQUENCE SUMMARY
    # ------------------------------------------------------------------

    sequence_file = (
        REPORT_DIR
        / "drift_by_sequence.csv"
    )

    sequence_summary.to_csv(
        sequence_file,
        index=False
    )

    # ------------------------------------------------------------------
    # SAVE FEATURE SUMMARY
    # ------------------------------------------------------------------

    feature_file = (
        REPORT_DIR
        / "anomaly_feature_analysis.csv"
    )

    feature_summary.to_csv(
        feature_file,
        index=False
    )

    # ------------------------------------------------------------------
    # TEMPORAL OUTPUT
    # ------------------------------------------------------------------

    temporal_file = None

    if temporal is not None:

        temporal_file = (
            REPORT_DIR
            / "temporal_drift_analysis.csv"
        )

        temporal.to_csv(
            temporal_file,
            index=False
        )

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    score_plot = (
        plot_score_distribution(
            results,
            threshold
        )
    )

    sequence_plot = (
        plot_sequence_drift(
            sequence_summary
        )
    )

    feature_plot = (
        plot_feature_differences(
            feature_summary
        )
    )

    # ------------------------------------------------------------------
    # TOP SEQUENCES
    # ------------------------------------------------------------------

    top_sequences = (
        sequence_summary
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    # ------------------------------------------------------------------
    # TOP FEATURES
    # ------------------------------------------------------------------

    top_features = (
        feature_summary
        .head(TOP_FEATURES)
        .to_dict(
            orient="records"
        )
    )

    # ------------------------------------------------------------------
    # JSON REPORT
    # ------------------------------------------------------------------

    report = {

        "stage": "6",

        "stage_name":
            "Drift Analysis & Validation",

        "analysis_type":
            "post-hoc unsupervised anomaly analysis",

        "model_retrained":
            False,

        "uncategorised_dataset_used":
            False,

        "ground_truth_labels_assumed":
            False,

        "test_rows":
            int(len(results)),

        "normal_samples":
            int(normal_count),

        "detected_drift_samples":
            int(anomaly_count),

        "detected_drift_percentage":
            float(anomaly_percentage),

        "stage5_threshold":
            float(threshold),

        "feature_count":
            int(len(feature_columns)),

        "sequence_count":
            int(
                results[
                    "sequence_id"
                ].nunique()
            ),

        "top_sequences":
            top_sequences,

        "top_features":
            top_features,

        "output_files": {

            "all_scores":
                str(
                    full_prediction_file
                ),

            "detected_anomalies":
                str(
                    anomaly_file
                ),

            "sequence_summary":
                str(
                    sequence_file
                ),

            "feature_analysis":
                str(
                    feature_file
                ),

            "temporal_analysis":
                (
                    str(temporal_file)
                    if temporal_file
                    else None
                ),

            "score_plot":
                str(score_plot)
                if score_plot
                else None,

            "sequence_plot":
                str(sequence_plot)
                if sequence_plot
                else None,

            "feature_plot":
                str(feature_plot)
                if feature_plot
                else None
        }
    }

    json_file = (
        REPORT_DIR
        / "stage6_summary.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ------------------------------------------------------------------
    # HUMAN READABLE REPORT
    # ------------------------------------------------------------------

    text_file = (
        REPORT_DIR
        / "stage6_summary.txt"
    )

    with open(
        text_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "NAV-SHIELD STAGE 6: "
            "DRIFT ANALYSIS & VALIDATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "MODEL\n"
        )

        f.write(
            "Stage 5 model retrained: NO\n"
        )

        f.write(
            "Learning type: UNSUPERVISED\n"
        )

        f.write(
            "Ground-truth labels assumed: NO\n"
        )

        f.write(
            "Uncategorised dataset used: NO\n\n"
        )

        f.write(
            "TEST SUMMARY\n"
        )

        f.write(
            f"Test rows: "
            f"{len(results):,}\n"
        )

        f.write(
            f"Normal samples: "
            f"{normal_count:,}\n"
        )

        f.write(
            f"Detected drift samples: "
            f"{anomaly_count:,}\n"
        )

        f.write(
            f"Detected drift percentage: "
            f"{anomaly_percentage:.4f}%\n"
        )

        f.write(
            f"Stage 5 threshold: "
            f"{threshold:.6f}\n\n"
        )

        f.write(
            "TOP SEQUENCES BY DRIFT PERCENTAGE\n"
        )

        for _, row in (
            sequence_summary
            .head(10)
            .iterrows()
        ):

            f.write(
                f"{row['sequence_id']}: "
                f"{int(row['drift_samples']):,} / "
                f"{int(row['total_samples']):,} "
                f"("
                f"{row['drift_percentage']:.4f}%"
                f")\n"
            )

        f.write(
            "\nTOP FEATURES ASSOCIATED WITH ANOMALIES\n"
        )

        for _, row in (
            feature_summary
            .head(TOP_FEATURES)
            .iterrows()
        ):

            f.write(
                f"{row['feature']}: "
                f"standardized difference = "
                f"{row['standardized_difference']:.4f}\n"
            )

        f.write(
            "\nINTERPRETATION\n"
        )

        f.write(
            "Detected samples are anomalous relative "
            "to the Stage 5 training distribution.\n"
        )

        f.write(
            "They are not automatically confirmed "
            "navigation failures because verified "
            "ground-truth drift labels are unavailable.\n"
        )

        f.write(
            "Further controlled validation is required "
            "before treating anomaly detections as "
            "confirmed GNSS/navigation drift events.\n"
        )

    # ------------------------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STAGE 6 SUMMARY"
    )
    print("=" * 70)

    print(
        f"Test samples              : "
        f"{len(results):,}"
    )

    print(
        f"Normal samples            : "
        f"{normal_count:,}"
    )

    print(
        f"Detected drift samples    : "
        f"{anomaly_count:,}"
    )

    print(
        f"Detected drift percentage : "
        f"{anomaly_percentage:.4f}%"
    )

    print(
        f"Stage 5 threshold         : "
        f"{threshold:.6f}"
    )

    print(
        f"Sequences analyzed        : "
        f"{results['sequence_id'].nunique()}"
    )

    print()
    print(
        "Model retrained           : NO"
    )

    print(
        "Uncategorised dataset     : NOT USED"
    )

    print(
        "Ground-truth labels       : NOT ASSUMED"
    )

    print()
    print(
        f"Reports:\n{REPORT_DIR}"
    )

    print(
        f"Predictions:\n{PREDICTION_DIR}"
    )

    print(
        f"Plots:\n{PLOT_DIR}"
    )

    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
    