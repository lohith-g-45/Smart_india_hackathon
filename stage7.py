"""
NAV-SHIELD STAGE 7
GROUND-TRUTH ERROR LABEL GENERATION

Purpose
-------
Generate supervised drift-error targets from the synchronized S/V data.

For each sequence:
    1. Load Stage 2 synchronized data.
    2. Detect accelerometer, gravity, GPS and reference vehicle fields.
    3. Estimate horizontal IMU acceleration.
    4. Perform local dead reckoning from the beginning of the sequence.
    5. Compare IMU dead-reckoned position with reference GPS position.
    6. Compare IMU dead-reckoned velocity with reference velocity.
    7. Save:
          position_error_m
          velocity_error_m_s

IMPORTANT
---------
- Raw data is NEVER modified.
- Stage 1-6 outputs are NEVER modified.
- No ML model is trained here.
- No train/validation/test statistics are fitted here.
- Labels are generated independently for each sequence.
- The output is intended for Stage 8 LSTM training.

Reference:
NAV-SHIELD Member 5 Phase E:
Ground Truth Error Labels.
"""

from pathlib import Path
import json
import math
import re
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Stage 2 synchronized data
ALIGNED_DIR = BASE_DIR / "results" / "data" / "aligned"

# Stage 7 output
OUTPUT_DIR = BASE_DIR / "results" / "stage7"
LABEL_DIR = OUTPUT_DIR / "labels"
REPORT_DIR = OUTPUT_DIR / "reports"

LABEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum reasonable integration timestep.
# If a timestamp gap is larger than this, the integration state
# is reset to prevent one large gap from creating unrealistic drift.
MAX_DT = 0.5

# Small epsilon
EPS = 1e-12


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def clean_column_name(name):
    """
    Normalize column names so that different spaces/encoding
    variations are easier to detect.
    """
    name = str(name)

    replacements = {
        "\ufeff": "",
        "Â": "",
        "Î": "",
        "µ": "u",
        "²": "2",
        "°": "deg",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    name = name.strip().lower()
    name = re.sub(r"\s+", " ", name)

    return name


def build_column_map(df):
    return {
        clean_column_name(col): col
        for col in df.columns
    }


def find_column(df, candidates, required=False):
    """
    Find the first matching column from candidate names.
    Supports exact normalized names and substring matching.
    """

    col_map = build_column_map(df)

    normalized_candidates = [
        clean_column_name(x) for x in candidates
    ]

    # Exact match
    for candidate in normalized_candidates:
        if candidate in col_map:
            return col_map[candidate]

    # Substring match
    for candidate in normalized_candidates:
        for normalized, original in col_map.items():
            if candidate in normalized:
                return original

    if required:
        raise ValueError(
            f"Required column not found. Tried: {candidates}"
        )

    return None


# ============================================================
# SENSOR COLUMN DETECTION
# ============================================================

def detect_columns(df):
    """
    Detect both categorised and uncategorised dataset naming styles.
    """

    columns = {}

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    columns["time_ms"] = find_column(
        df,
        [
            "S_time_ms",
            "S TIME SINCE START (ms)",
            "TIME SINCE START (ms)",
            "time_ms",
        ],
    )

    columns["time_seconds"] = find_column(
        df,
        [
            "S_time_seconds",
            "time_seconds",
        ],
    )

    columns["date"] = find_column(
        df,
        [
            "S_DATE",
            "DATE (YYYY-MO-DD HH-MI-SS_SSS",
            "DATE",
        ],
    )

    # --------------------------------------------------------
    # Accelerometer
    # --------------------------------------------------------

    columns["accel_x"] = find_column(
        df,
        [
            "S_ACCELEROMETER X (m/s²)",
            "ACCELEROMETER X (m/s²)",
            "accelerometer_x",
        ],
        required=True,
    )

    columns["accel_y"] = find_column(
        df,
        [
            "S_ACCELEROMETER Y (m/s²)",
            "ACCELEROMETER Y (m/s²)",
            "accelerometer_y",
        ],
        required=True,
    )

    columns["accel_z"] = find_column(
        df,
        [
            "S_ACCELEROMETER Z (m/s²)",
            "ACCELEROMETER Z (m/s²)",
            "accelerometer_z",
        ],
        required=True,
    )

    # --------------------------------------------------------
    # Gravity
    # --------------------------------------------------------

    columns["gravity_x"] = find_column(
        df,
        [
            "S_GRAVITY X (m/s²)",
            "GRAVITY X (m/s²)",
            "gravity_x",
        ]
    )

    columns["gravity_y"] = find_column(
        df,
        [
            "S_GRAVITY Y (m/s²)",
            "GRAVITY Y (m/s²)",
            "gravity_y",
        ]
    )

    columns["gravity_z"] = find_column(
        df,
        [
            "S_GRAVITY Z (m/s²)",
            "GRAVITY Z (m/s²)",
            "gravity_z",
        ]
    )

    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    columns["gyro_x"] = find_column(
        df,
        [
            "S_GYROSCOPE X (rad/s)",
            "GYROSCOPE X (rad/s)",
            "gyro_x",
        ]
    )

    columns["gyro_y"] = find_column(
        df,
        [
            "S_GYROSCOPE Y (rad/s)",
            "GYROSCOPE Y (rad/s)",
            "gyro_y",
        ]
    )

    columns["gyro_z"] = find_column(
        df,
        [
            "S_GYROSCOPE Z (rad/s)",
            "GYROSCOPE Z (rad/s)",
            "gyro_z",
        ]
    )

    # Categorised files use Yaw/Pitch/Roll
    columns["gyro_yaw"] = find_column(
        df,
        [
            "S_GYROSCOPE Yaw (rad/s)",
            "GYROSCOPE Yaw (rad/s)",
            "gyro_yaw",
        ]
    )

    columns["gyro_pitch"] = find_column(
        df,
        [
            "S_GYROSCOPE Pitch (rad/s)",
            "GYROSCOPE Pitch (rad/s)",
            "gyro_pitch",
        ]
    )

    columns["gyro_roll"] = find_column(
        df,
        [
            "S_GYROSCOPE Roll (rad/s)",
            "GYROSCOPE Roll (rad/s)",
            "gyro_roll",
        ]
    )

    # --------------------------------------------------------
    # Orientation / heading
    # --------------------------------------------------------

    columns["orientation_yaw"] = find_column(
        df,
        [
            "S_ORIENTATION (Yaw) (deg)",
            "ORIENTATION (Yaw) (deg)",
            "S_ORIENTATION (Azimuth) (deg)",
            "ORIENTATION (Azimuth) (deg)",
            "orientation_yaw",
            "azimuth",
            "heading",
        ]
    )

    # --------------------------------------------------------
    # GPS reference
    # --------------------------------------------------------

    columns["gps_lat"] = find_column(
        df,
        [
            "S_GPS LATITUDE (degrees)",
            "GPS LATITUDE (degrees)",
            "gps_latitude",
            "gps_lat",
        ],
        required=True,
    )

    columns["gps_lon"] = find_column(
        df,
        [
            "S_GPS LONGITUDE (degrees)",
            "GPS LONGITUDE (degrees)",
            "gps_longitude",
            "gps_lon",
        ],
        required=True,
    )

    columns["gps_alt"] = find_column(
        df,
        [
            "S_GPS ALTITUDE (m)",
            "GPS ALTITUDE (m)",
            "gps_altitude",
        ]
    )

    columns["gps_speed"] = find_column(
        df,
        [
            "S_GPS SPEED (Kmh)",
            "GPS SPEED (Kmh)",
            "gps_speed",
        ]
    )

    columns["gps_accuracy"] = find_column(
        df,
        [
            "S_GPS ACCURACY (m)",
            "GPS ACCURACY (m)",
            "gps_accuracy",
        ]
    )

    # --------------------------------------------------------
    # Vehicle reference
    # --------------------------------------------------------

    columns["v_lat"] = find_column(
        df,
        [
            "V_Latitude (degrees)",
            "Latitude (degrees)",
            "V latitude",
            "vehicle_latitude",
        ]
    )

    columns["v_lon"] = find_column(
        df,
        [
            "V_Longitude (degrees)",
            "Longitude (degrees)",
            "V longitude",
            "vehicle_longitude",
        ]
    )

    columns["v_speed"] = find_column(
        df,
        [
            "V_Indicated Vehicle Speed (km/hr)",
            "Indicated Vehicle Speed (km/hr)",
            "V_Velocity (km/hr)",
            "Velocity (km/hr)",
            "V_speed",
            "vehicle_speed",
        ]
    )

    columns["v_heading"] = find_column(
        df,
        [
            "V_Heading (degrees)",
            "Heading (degrees)",
            "V_heading",
            "vehicle_heading",
        ]
    )

    return columns


# ============================================================
# TIME CONVERSION
# ============================================================

def create_time_seconds(df, columns):
    """
    Prefer synchronized numeric time.
    """

    if columns["time_seconds"] is not None:
        t = pd.to_numeric(
            df[columns["time_seconds"]],
            errors="coerce"
        ).to_numpy(dtype=float)

        return t

    if columns["time_ms"] is not None:
        t = pd.to_numeric(
            df[columns["time_ms"]],
            errors="coerce"
        ).to_numpy(dtype=float)

        return t / 1000.0

    if columns["date"] is not None:
        dt = pd.to_datetime(
            df[columns["date"]],
            errors="coerce"
        )

        return (
            (dt - dt.iloc[0]).dt.total_seconds()
            .to_numpy(dtype=float)
        )

    raise ValueError(
        "No usable time column found."
    )


# ============================================================
# GEO UTILITIES
# ============================================================

EARTH_RADIUS_M = 6371000.0


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convert latitude/longitude to local tangent-plane
    East/North coordinates.

    x = East
    y = North
    """

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    lat0_rad = math.radians(float(lat0))
    lon0_rad = math.radians(float(lon0))

    x = (
        (lon_rad - lon0_rad)
        * math.cos(lat0_rad)
        * EARTH_RADIUS_M
    )

    y = (
        (lat_rad - lat0_rad)
        * EARTH_RADIUS_M
    )

    return x, y


# ============================================================
# REFERENCE VELOCITY
# ============================================================

def compute_reference_velocity(
    ref_x,
    ref_y,
    time_seconds
):
    """
    Compute reference velocity components from the reference
    trajectory.

    Central differences are used where possible.
    """

    vx = np.full(len(ref_x), np.nan)
    vy = np.full(len(ref_y), np.nan)

    valid = (
        np.isfinite(ref_x)
        & np.isfinite(ref_y)
        & np.isfinite(time_seconds)
    )

    valid_indices = np.where(valid)[0]

    if len(valid_indices) < 2:
        return vx, vy

    for pos, i in enumerate(valid_indices):

        if pos == 0:
            j = valid_indices[pos + 1]

            dt = time_seconds[j] - time_seconds[i]

            if dt > EPS:
                vx[i] = (ref_x[j] - ref_x[i]) / dt
                vy[i] = (ref_y[j] - ref_y[i]) / dt

        elif pos == len(valid_indices) - 1:
            j = valid_indices[pos - 1]

            dt = time_seconds[i] - time_seconds[j]

            if dt > EPS:
                vx[i] = (ref_x[i] - ref_x[j]) / dt
                vy[i] = (ref_y[i] - ref_y[j]) / dt

        else:
            j1 = valid_indices[pos - 1]
            j2 = valid_indices[pos + 1]

            dt = time_seconds[j2] - time_seconds[j1]

            if dt > EPS:
                vx[i] = (
                    ref_x[j2] - ref_x[j1]
                ) / dt

                vy[i] = (
                    ref_y[j2] - ref_y[j1]
                ) / dt

    return vx, vy


# ============================================================
# IMU DEAD RECKONING
# ============================================================

def compute_linear_acceleration(df, columns):
    """
    Remove gravity from accelerometer measurements.

    If gravity channels are available:
        linear_accel = accelerometer - gravity

    Otherwise:
        raw acceleration is used with a warning.
    """

    ax = pd.to_numeric(
        df[columns["accel_x"]],
        errors="coerce"
    ).to_numpy(dtype=float)

    ay = pd.to_numeric(
        df[columns["accel_y"]],
        errors="coerce"
    ).to_numpy(dtype=float)

    az = pd.to_numeric(
        df[columns["accel_z"]],
        errors="coerce"
    ).to_numpy(dtype=float)

    if (
        columns["gravity_x"] is not None
        and columns["gravity_y"] is not None
        and columns["gravity_z"] is not None
    ):

        gx = pd.to_numeric(
            df[columns["gravity_x"]],
            errors="coerce"
        ).to_numpy(dtype=float)

        gy = pd.to_numeric(
            df[columns["gravity_y"]],
            errors="coerce"
        ).to_numpy(dtype=float)

        gz = pd.to_numeric(
            df[columns["gravity_z"]],
            errors="coerce"
        ).to_numpy(dtype=float)

        ax = ax - gx
        ay = ay - gy
        az = az - gz

        gravity_removed = True

    else:
        warnings.warn(
            "Gravity columns unavailable. "
            "Raw accelerometer values are being used."
        )

        gravity_removed = False

    return ax, ay, az, gravity_removed


def get_heading_radians(df, columns):
    """
    Obtain heading/orientation yaw if available.

    If unavailable, use zero heading.

    NOTE:
    The exact phone-device-to-vehicle axis convention is dataset
    dependent. This stage uses the available orientation yaw as
    the horizontal rotation angle.
    """

    if columns["orientation_yaw"] is None:

        return np.zeros(len(df), dtype=float), False

    yaw = pd.to_numeric(
        df[columns["orientation_yaw"]],
        errors="coerce"
    ).to_numpy(dtype=float)

    # Fill short missing sections
    yaw_series = pd.Series(yaw)

    yaw_series = (
        yaw_series
        .interpolate(limit_direction="both")
        .fillna(0.0)
    )

    yaw = np.radians(
        yaw_series.to_numpy(dtype=float)
    )

    return yaw, True


def dead_reckon(
    ax,
    ay,
    az,
    yaw,
    time_seconds
):
    """
    Integrate horizontal acceleration.

    Assumption:
        device X/Y horizontal axes are rotated by yaw
        into local East/North coordinates.

    x = East
    y = North

    z is not integrated for horizontal position.

    Integration:
        v(t+dt) = v(t) + a(t)dt
        p(t+dt) = p(t) + v(t)dt + 0.5a(t)dt²
    """

    n = len(time_seconds)

    imu_x = np.zeros(n)
    imu_y = np.zeros(n)

    imu_vx = np.zeros(n)
    imu_vy = np.zeros(n)

    for i in range(1, n):

        dt = (
            time_seconds[i]
            - time_seconds[i - 1]
        )

        if (
            not np.isfinite(dt)
            or dt <= 0
            or dt > MAX_DT
        ):
            # Reset integration at invalid/large gaps
            imu_x[i] = imu_x[i - 1]
            imu_y[i] = imu_y[i - 1]

            imu_vx[i] = 0.0
            imu_vy[i] = 0.0

            continue

        a_device_x = ax[i]
        a_device_y = ay[i]

        if not (
            np.isfinite(a_device_x)
            and np.isfinite(a_device_y)
        ):
            imu_x[i] = imu_x[i - 1]
            imu_y[i] = imu_y[i - 1]

            imu_vx[i] = imu_vx[i - 1]
            imu_vy[i] = imu_vy[i - 1]

            continue

        # Device -> local horizontal frame
        c = math.cos(yaw[i])
        s = math.sin(yaw[i])

        a_east = (
            a_device_x * c
            - a_device_y * s
        )

        a_north = (
            a_device_x * s
            + a_device_y * c
        )

        # Position integration
        imu_x[i] = (
            imu_x[i - 1]
            + imu_vx[i - 1] * dt
            + 0.5 * a_east * dt * dt
        )

        imu_y[i] = (
            imu_y[i - 1]
            + imu_vy[i - 1] * dt
            + 0.5 * a_north * dt * dt
        )

        # Velocity integration
        imu_vx[i] = (
            imu_vx[i - 1]
            + a_east * dt
        )

        imu_vy[i] = (
            imu_vy[i - 1]
            + a_north * dt
        )

    return (
        imu_x,
        imu_y,
        imu_vx,
        imu_vy
    )


# ============================================================
# SINGLE SEQUENCE PROCESSING
# ============================================================

def process_sequence(csv_path):

    sequence_id = csv_path.stem

    print(
        f"\n[{sequence_id}] "
        f"Loading synchronized data..."
    )

    df = pd.read_csv(
        csv_path,
        low_memory=False
    )

    if df.empty:
        raise ValueError("CSV is empty.")

    columns = detect_columns(df)

    time_seconds = create_time_seconds(
        df,
        columns
    )

    # --------------------------------------------------------
    # Reference GPS
    # --------------------------------------------------------

    gps_lat = pd.to_numeric(
        df[columns["gps_lat"]],
        errors="coerce"
    ).to_numpy(dtype=float)

    gps_lon = pd.to_numeric(
        df[columns["gps_lon"]],
        errors="coerce"
    ).to_numpy(dtype=float)

    valid_gps = (
        np.isfinite(gps_lat)
        & np.isfinite(gps_lon)
    )

    if valid_gps.sum() < 2:
        raise ValueError(
            "Not enough valid GPS reference samples."
        )

    first_valid = np.where(valid_gps)[0][0]

    lat0 = gps_lat[first_valid]
    lon0 = gps_lon[first_valid]

    ref_x, ref_y = latlon_to_local_xy(
        gps_lat,
        gps_lon,
        lat0,
        lon0
    )

    # Interpolate missing reference positions
    ref_x_series = pd.Series(ref_x)
    ref_y_series = pd.Series(ref_y)

    ref_x = (
        ref_x_series
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    ref_y = (
        ref_y_series
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    # --------------------------------------------------------
    # Reference velocity
    # --------------------------------------------------------

    ref_vx, ref_vy = compute_reference_velocity(
        ref_x,
        ref_y,
        time_seconds
    )

    ref_speed = np.sqrt(
        ref_vx ** 2
        + ref_vy ** 2
    )

    # --------------------------------------------------------
    # IMU acceleration
    # --------------------------------------------------------

    ax, ay, az, gravity_removed = (
        compute_linear_acceleration(
            df,
            columns
        )
    )

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    yaw, heading_available = (
        get_heading_radians(
            df,
            columns
        )
    )

    # --------------------------------------------------------
    # Dead reckoning
    # --------------------------------------------------------

    (
        imu_x,
        imu_y,
        imu_vx,
        imu_vy
    ) = dead_reckon(
        ax,
        ay,
        az,
        yaw,
        time_seconds
    )

    # --------------------------------------------------------
    # Ground-truth errors
    # --------------------------------------------------------

    position_error = np.sqrt(
        (imu_x - ref_x) ** 2
        + (imu_y - ref_y) ** 2
    )

    imu_speed = np.sqrt(
        imu_vx ** 2
        + imu_vy ** 2
    )

    velocity_error = np.abs(
        imu_speed - ref_speed
    )

    # --------------------------------------------------------
    # Quality / validity mask
    # --------------------------------------------------------

    label_valid = (
        np.isfinite(position_error)
        & np.isfinite(velocity_error)
        & np.isfinite(ref_x)
        & np.isfinite(ref_y)
        & np.isfinite(time_seconds)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = pd.DataFrame({
        "sequence_id": sequence_id,

        "time_seconds": time_seconds,

        # Reference position
        "reference_x_m": ref_x,
        "reference_y_m": ref_y,

        # IMU dead-reckoned position
        "imu_dr_x_m": imu_x,
        "imu_dr_y_m": imu_y,

        # Reference velocity
        "reference_vx_m_s": ref_vx,
        "reference_vy_m_s": ref_vy,
        "reference_speed_m_s": ref_speed,

        # IMU velocity
        "imu_dr_vx_m_s": imu_vx,
        "imu_dr_vy_m_s": imu_vy,
        "imu_dr_speed_m_s": imu_speed,

        # TARGETS
        "position_error_m": position_error,
        "velocity_error_m_s": velocity_error,

        # Useful metadata
        "gravity_removed": gravity_removed,
        "heading_available": heading_available,
        "label_valid": label_valid,
    })

    output_path = (
        LABEL_DIR
        / f"{sequence_id}_labels.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "sequence_id": sequence_id,
        "input_file": str(csv_path),
        "output_file": str(output_path),

        "input_rows": int(len(df)),
        "output_rows": int(len(output)),

        "valid_label_rows": int(
            label_valid.sum()
        ),

        "invalid_label_rows": int(
            (~label_valid).sum()
        ),

        "position_error_mean_m": float(
            np.nanmean(position_error)
        ),

        "position_error_median_m": float(
            np.nanmedian(position_error)
        ),

        "position_error_max_m": float(
            np.nanmax(position_error)
        ),

        "velocity_error_mean_m_s": float(
            np.nanmean(velocity_error)
        ),

        "velocity_error_median_m_s": float(
            np.nanmedian(velocity_error)
        ),

        "velocity_error_max_m_s": float(
            np.nanmax(velocity_error)
        ),

        "gravity_removed": bool(
            gravity_removed
        ),

        "heading_available": bool(
            heading_available
        ),

        "status": "PASS",
    }

    report_path = (
        REPORT_DIR
        / f"{sequence_id}_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NAV-SHIELD STAGE 7: "
        "GROUND-TRUTH ERROR LABEL GENERATION"
    )
    print("=" * 70)

    print()
    print("Purpose:")
    print(
        "Generate IMU dead-reckoning error labels "
        "for supervised LSTM training."
    )

    print()
    print("Policy:")
    print("- Raw data will NOT be modified.")
    print("- Stage 1-6 outputs will NOT be modified.")
    print("- No imputation is fitted here.")
    print("- No model training is performed.")
    print("- Labels are generated independently per sequence.")

    print()
    print(f"Input directory:")
    print(ALIGNED_DIR)

    print()
    print(f"Output directory:")
    print(LABEL_DIR)

    # --------------------------------------------------------
    # Find synchronized files
    # --------------------------------------------------------

    csv_files = sorted(
        ALIGNED_DIR.glob("*_aligned.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "No synchronized *_aligned.csv files found."
        )

    print()
    print(
        f"Synchronized files found: "
        f"{len(csv_files)}"
    )

    successful = []
    failed = []

    total_rows = 0
    total_valid_labels = 0

    for index, csv_path in enumerate(
        csv_files,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(csv_files)}] "
            f"Processing {csv_path.stem} ..."
        )

        try:

            report = process_sequence(
                csv_path
            )

            successful.append(
                report["sequence_id"]
            )

            total_rows += report["input_rows"]
            total_valid_labels += (
                report["valid_label_rows"]
            )

            print(
                f"[PASS] {csv_path.stem}: "
                f"{report['valid_label_rows']:,} "
                f"valid labels"
            )

        except Exception as exc:

            failed.append({
                "sequence_id": csv_path.stem,
                "error": str(exc),
            })

            print(
                f"[FAIL] {csv_path.stem}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "stage": "7",
        "stage_name": "Ground-Truth Error Label Generation",

        "sequences_found": len(csv_files),
        "successful": len(successful),
        "failed": len(failed),

        "total_input_rows": total_rows,
        "total_valid_labels": total_valid_labels,

        "label_columns": [
            "position_error_m",
            "velocity_error_m_s",
        ],

        "raw_data_modified": False,
        "stage1_to_stage6_modified": False,
        "model_training_performed": False,

        "successful_sequences": successful,
        "failed_sequences": failed,

        "status": (
            "PASS"
            if len(failed) == 0
            else "WARNING"
        ),
    }

    summary_json = (
        REPORT_DIR
        / "stage7_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # Human-readable summary
    summary_txt = (
        REPORT_DIR
        / "stage7_summary.txt"
    )

    with open(
        summary_txt,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "NAV-SHIELD STAGE 7: "
            "GROUND-TRUTH ERROR LABEL GENERATION\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Sequences found      : {len(csv_files)}\n"
        )

        f.write(
            f"Successful           : {len(successful)}\n"
        )

        f.write(
            f"Failed               : {len(failed)}\n"
        )

        f.write(
            f"Input rows           : {total_rows:,}\n"
        )

        f.write(
            f"Valid label rows     : "
            f"{total_valid_labels:,}\n"
        )

        f.write(
            "\nTarget labels:\n"
        )

        f.write(
            "- position_error_m\n"
        )

        f.write(
            "- velocity_error_m_s\n"
        )

        f.write(
            "\nRaw data modified    : NO\n"
        )

        f.write(
            "Stage 1-6 modified   : NO\n"
        )

        f.write(
            "Model training       : NO\n"
        )

        f.write(
            f"\nSTATUS: {summary['status']}\n"
        )

    print()
    print("=" * 70)
    print("STAGE 7 SUMMARY")
    print("=" * 70)

    print(
        f"Sequences found      : {len(csv_files)}"
    )

    print(
        f"Successful            : {len(successful)}"
    )

    print(
        f"Failed                : {len(failed)}"
    )

    print(
        f"Input rows            : {total_rows:,}"
    )

    print(
        f"Valid label rows      : "
        f"{total_valid_labels:,}"
    )

    print()
    print(
        "Target labels:"
    )

    print(
        "  position_error_m"
    )

    print(
        "  velocity_error_m_s"
    )

    print()
    print(
        "Raw data modified     : NO"
    )

    print(
        "Stage 1-6 modified    : NO"
    )

    print(
        "Model training        : NO"
    )

    print()
    print(
        f"Status                 : "
        f"{summary['status']}"
    )

    print()
    print(
        "Labels saved to:"
    )

    print(LABEL_DIR)

    print()
    print(
        "Reports saved to:"
    )

    print(REPORT_DIR)

    print("=" * 70)


if __name__ == "__main__":
    main()