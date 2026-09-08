# NAV-SHIELD Stage 3

Stage 3 converts the verified Stage 2 aligned S/V CSV files into numerical, AI-ready feature datasets. It does not train a model, detect drift, create predictions, or invent labels.

## Input

By default the pipeline discovers every `*_aligned.csv` file under:

```text
results/data/aligned/
```

These files are produced by Stage 2 and are not modified. The primary time fields are `S_time_seconds` and `V_time_seconds`; source columns remain in the output with their original names.

## Run

```powershell
python stage3_features.py --input results/data/aligned --output results --window 10
```

Process one file while testing:

```powershell
python stage3_features.py --input results/data/aligned/vta11_aligned.csv --output results --window 10
```

Use `python stage3_features.py --help` for all options.

## Cleaning and time

Numeric feature candidates are converted with `pandas.to_numeric(errors="coerce")`; invalid and infinite values become `NaN`. Valid zero values are preserved. Rows are not dropped for missing sensor values. Every sequence report records input rows, output rows, retention, important-feature missingness, and quality statistics.

Rows are stably sorted by `timestamp`, which uses `S_time_seconds` and falls back to `V_time_seconds` when S time is missing. The following fields are generated:

- `timestamp`: chronological synchronized time in seconds.
- `elapsed_time`: timestamp relative to the first valid timestamp.
- `delta_time`: non-negative time difference from the preceding row.
- `time_gap_flag`: true when `delta_time` is larger than the sequence's robust gap threshold. The threshold is the maximum of three times the median positive interval, median plus six median absolute deviations, and median plus 0.05 seconds.

## Raw canonical features

The detector matches columns by case-insensitive content, not position. It prefers S-prefixed fields and falls back to compatible V fields.

- Accelerometer: `accel_x`, `accel_y`, `accel_z`
- Gyroscope: `gyro_yaw`, `gyro_pitch`, `gyro_roll`
- Magnetometer: `mag_x`, `mag_y`, `mag_z`
- Gravity: `gravity_x`, `gravity_y`, `gravity_z`
- Orientation: `orientation_yaw`, `orientation_pitch`, `orientation_roll`
- GPS: `gps_latitude`, `gps_longitude`, `gps_altitude`, `gps_speed`, `gps_accuracy`, `gps_satellites`

The original source columns are retained alongside these canonical numeric columns.

## Derived features

Vector magnitudes use Euclidean norm:

```text
magnitude = sqrt(x*x + y*y + z*z)
```

Generated magnitudes include `accel_magnitude`, `gyro_magnitude`, `mag_magnitude`, and `gravity_magnitude`.

Change and rate features include `accel_change`, `gyro_change`, `magnetic_field_change`, `speed_change`, their `_rate` variants, and `yaw_change`, `pitch_change`, `roll_change` with corresponding rates. Angular differences use the shortest circular difference in a 360-degree period, so a transition from 359 to 1 degrees is +2 degrees rather than -358 degrees.

## Causal rolling features

For each suitable signal, Stage 3 creates trailing `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, and `rolling_range` features. The default window is 10 samples and can be changed with `--window`. Rolling windows include the current row and prior rows only. No centered windows or future rows are used.

## Quality indicators

- `gps_valid`: latitude and longitude are present and within physical ranges.
- `gps_accuracy_valid`: GPS accuracy is present and non-negative.
- `satellite_count_valid`: satellite count is present and non-negative.
- `missing_sensor_flag`: at least one required raw sensor component is missing.
- `sensor_quality_score`: the unweighted mean of the three GPS validity indicators and `1 - missing_sensor_flag`. Each available indicator contributes equally; no arbitrary weights are used.

These are data-quality indicators, not drift labels.

## Labels and normalization

Existing explicit label-like columns are preserved and listed in each report. If none are found, the report states:

```text
No explicit drift ground-truth label was found.
```

No fake drift labels are generated. No global normalization is performed. A future training stage must fit any scaler using training data only and apply the fitted configuration to validation/test data; Stage 3 intentionally leaves raw engineered values available for that workflow.

## Outputs

```text
results/
  features/categorised/<sequence>_features.csv
  feature_reports/<sequence>_feature_report.json
  feature_reports/<sequence>_feature_report.txt
  feature_reports/plots/<representative>_feature_diagnostics.png
  feature_reports/plot_selection.txt
  feature_summary.csv
  feature_summary.json
```

Three representative successful sequences are plotted: the first, middle, and last successful inputs. Each diagnostic figure contains accelerometer magnitude, gyroscope magnitude, GPS speed, GPS accuracy, and rolling feature plots.

## Validation

The pipeline checks sequence identity, chronological timestamps, numeric rolling features, infinite values, row consistency, and required rolling outputs. Per-sequence failures are printed and included in the summary while other sequences continue. A nonzero exit code is returned if any discovered sequence fails.

Stage 3 ends at the AI-ready feature dataset. AI drift detection belongs to a later stage.
