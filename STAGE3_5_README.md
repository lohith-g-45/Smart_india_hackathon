# NAV-SHIELD Stage 3.5

Stage 3.5 is a read-only quality audit of the existing Stage 3 feature CSVs. It does not modify the original dataset, Stage 2 aligned files, Stage 3 feature files, or any feature values. It does not impute, remove, normalize, split, or train data.

## Run

From the project directory:

```powershell
python stage3_5_audit.py --input results\features\categorised --output results\data_reports\stage3_5
```

Use `python stage3_5_audit.py --help` to override the input or report directories. The default input is `results/features/categorised`; the default output is `results/data_reports/stage3_5`.

## Reports

The audit writes:

```text
results/data_reports/stage3_5/
  stage3_5_summary.json
  stage3_5_summary.txt
  feature_quality_report.csv
  sequence_quality_report.csv
  missing_value_report.csv
  feature_statistics.csv
  plots/
    missing_percentage_by_feature.png
    missing_percentage_distribution.png
    missing_percentage_by_sequence.png
```

## Status rules

- `PASS`: no missing values, infinite values, duplicate rows, timestamp issues, constant/near-constant numeric features, or flagged range/outlier values.
- `WARNING`: the sequence is readable but has expected missingness, duplicate rows/timestamps, gaps, constant/near-constant features, or flagged ranges/outliers.
- `FAIL`: the file cannot be read, has no usable rows, or has a fatal schema/data problem.

The audit does not call ordinary missingness a failure automatically. The final dataset recommendation becomes `REQUIRES CORRECTION` when a deterministic engineered-feature defect, infinite values, or unreadable sequence is detected.

## Missingness causes

Cause labels are based on the existing Stage 3 code and observed schema:

- Differencing/derivative fields: first-row initialization or unavailable preceding sample.
- Rate fields: unavailable `delta_time` or division by zero.
- Rolling fields: trailing-window initialization or propagation from a missing base feature.
- `V_` source fields: unmatched Stage 2 rows can leave vehicle values missing.
- `S_` source fields: missingness inherited from the synchronized source.
- Gyroscope-derived fields: the current Stage 3 implementation detects gyroscope axes as yaw/pitch/roll but `vector_magnitude()` requests x/y/z. This is reported as an engineered-feature bug, not legitimate sensor missingness.
- Anything not supported by the code/data is reported as `Cause requires further inspection.`

The audit found that the gyro issue must be corrected before Stage 4. It should not be solved by imputing a feature that is 100% missing due to a deterministic implementation mismatch.