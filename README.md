# NAV-SHIELD — Member 5 AI Drift Correction Module

**STAGE 1 ONLY: IO-VNBD Data Loading, Validation & S/V Synchronization**

This repository implements *only* Stage 1 of the future NAV-SHIELD pipeline:

```
IO-VNBD Dataset → Data Loading → Data Validation → S/V Synchronization
   → (Stage 2+: Preprocessing & Calibration → Feature Engineering → LSTM
      → GNSS-Denied Dead Reckoning → Drift Calculation → AI Drift
      Correction → Evaluation → Model Export)
```

No ML model, dead-reckoning logic, drift correction, GNSS-outage
simulation, sensor fusion, map matching, or mobile/ONNX/TFLite export is
implemented here. Those belong to later stages.

## Dataset

This project uses **only** the official **IO-VNBD Synchronised V and S
Dataset**. The dataset is **not bundled** with this repository — you must
supply the path to your own local, extracted copy at runtime.

Do not use the Unsynchronised V+S dataset, KITTI, Oxford RobotCar, EuRoC,
or any other dataset with this code.

## Project structure

```
member5_ai_drift/
├── data/
│   ├── raw/io_vnbd/        # (optional) place a local copy/symlink here, or
│   │                         point --dataset anywhere else on disk
│   └── processed/          # Stage 1 writes <sequence>_aligned.csv here
├── src/
│   ├── data_loader.py      # dataset path resolution, S/V file discovery, CSV loading
│   ├── data_validator.py   # missing/duplicate/infinite/lat-lon checks, sampling stats
│   └── synchronizer.py     # time-field derivation + time-based S/V synchronization
├── results/data_reports/   # Stage 1 writes JSON/TXT reports + diagnostic PNG here
├── tests/test_stage1.py    # unit + integration tests (synthetic data only)
├── main.py                 # CLI entry point
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies are intentionally lightweight: `pandas`, `numpy`,
`matplotlib`, `pytest`. No PyTorch/TensorFlow/ONNX/CUDA is installed or
required for Stage 1.

## Running Stage 1

```bash
python main.py --dataset "D:\SIH\IO-VNBD\Synchronised V and S Dataset" --sequence Vfa01
```

Optional flags:

- `--tolerance <seconds>` — maximum allowed `|S_time - V_time|` for a
  synchronization match (default `0.05`, i.e. half of V's native 0.1 s
  sample period). Increase this if you see a low match percentage on a
  different sequence with a coarser V sampling rate.

The `--dataset` path is **never hard-coded**: point it at wherever you
extracted the dataset on your machine. The `--sequence` value is likewise
not hard-coded to `Vfa01` — any sequence with a matching
`S-<sequence>.csv` / `V-<sequence>.csv` pair anywhere under the dataset
root (searched recursively) will work, e.g.:

```bash
python main.py --dataset "D:\SIH\IO-VNBD\Synchronised V and S Dataset" --sequence Vfa02
python main.py --dataset "D:\SIH\IO-VNBD\Synchronised V and S Dataset" --sequence Vfa03
```

If the dataset path does not exist, or the requested sequence's files
cannot be found, the program stops immediately with a clear, actionable
error message (see `src/data_loader.py`).

## What Stage 1 actually does

1. **Locate** `S-<sequence>.csv` and `V-<sequence>.csv` under the supplied
   dataset root (recursive search — works regardless of which
   driver/sub-folder they live in).
2. **Load** both CSVs without modifying the original files. Loading is
   encoding-tolerant (tries UTF-8, then Windows-1252, then Latin-1) because
   the real V files are UTF-8 while the real S files use characters like
   `°` and `µ` that are commonly Windows-1252/Latin-1 encoded.
3. **Inspect** the actual column names, dtypes, and candidate time-related
   fields in both files (never assumed from memory).
4. **Validate** both DataFrames: row/column counts, missing values, full
   duplicate rows, duplicate timestamps, infinite values, invalid numeric
   values, min/max per numeric column, and latitude/longitude range
   validity. Nothing is silently dropped, filled, or corrected — every
   problem is only reported.
5. **Determine the real time base of each file** (see "Time fields" below)
   and compute each file's approximate sampling interval/frequency.
6. **Synchronize S against V by time**, using nearest-time matching
   (`pandas.merge_asof`, `direction="nearest"`) within a configurable
   tolerance — never by row index and never by truncating the longer file.
   S samples with no V sample within tolerance are kept and explicitly
   marked `matched = False`, with `NaN` (never a fabricated value) in every
   `V_*` column.
7. **Assembles one aligned DataFrame** with `S_<original column>` and
   `V_<original column>` prefixes so no column name is ambiguous, plus:
   - `timestamp_seconds` — the canonical synchronization time (S's derived
     seconds-since-midnight, since S is the primary sensor stream that
     later stages will feed to the LSTM),
   - `time_diff_seconds` — `|S_time − matched V_time|`,
   - `matched` — boolean.
8. **Saves** the aligned dataset, a JSON report, a TXT report, and a basic
   synchronization diagnostic PNG (S coverage / V coverage / match
   coverage over the timeline — this is *not* the final drift/trajectory
   plot, only a Stage 1 sanity check).

## Time fields actually used (Vfa01, verified against the real files)

| File | Column(s) inspected | Selected as sync field? | Notes |
|---|---|---|---|
| V-Vfa01.csv | `Time Since Start of Day (seconds)` | **Yes** | Numeric, seconds since local midnight, perfectly constant 0.1 s step (10 Hz) across the whole file. |
| V-Vfa01.csv | `Sample period (seconds)` | No (cross-check only) | Matches the observed 0.1 s diff of the time column — used only to corroborate the sampling rate. |
| S-Vfa01.csv | `TIME SINCE START (ms)` | No | Milliseconds since the *phone app's own logging start* — an arbitrary device-local epoch with no fixed relationship to V's "since start of day" reference. Cannot be used directly against V. |
| S-Vfa01.csv | `DATE (YYYY-MO-DD HH-MI-SS_SSS)` | **Yes (derived)** | Absolute wall-clock timestamp. The column header advertises an underscore before the millisecond field, but the *actual* stored values use a colon (e.g. `2019-11-08 09:20:26:236`). Stage 1 parses this real format and converts it into seconds-since-midnight, which is directly comparable to V's native time column (both reference the same real-world clock/day). |

This decision was made by **inspecting the real row values and time
deltas**, not by assuming a column name — see the "WHY TIME-BASED, NOT
ROW-BASED" docstring at the top of `src/synchronizer.py` for full
reasoning, and `src/data_validator.py:detect_time_column_candidates()` for
the (name-keyword-based) candidate proposal step that a human/analyst
reviews before the final field is hard-selected in `main.py`.

## Important guarantees

- Original CSV files are **never modified**.
- No records are **silently dropped**.
- No missing values are **silently filled**.
- Synchronization is **never** done by row number.
- The sequence is **not hard-coded** to `Vfa01` — any valid `S-<seq>.csv`
  / `V-<seq>.csv` pair works via `--sequence`.
- Stage 1 does **not** decide the final LSTM input/target split and does
  **not** create ML-ready features — that is deliberately deferred to
  Stage 2+ to avoid prematurely locking in a design that could introduce
  ground-truth leakage.

## Outputs

For `--sequence Vfa01`:

- `data/processed/Vfa01_aligned.csv`
- `results/data_reports/Vfa01_report.json`
- `results/data_reports/Vfa01_report.txt`
- `results/data_reports/Vfa01_sync_diagnostic.png`

## Testing

```bash
pytest tests/test_stage1.py -v
```

Tests use a small **synthetic** S/V dataset built entirely in-memory /
in a pytest `tmp_path` — this synthetic data is used *only* to verify the
pipeline's logic (loading, error handling, time detection, sampling-rate
math, synchronization, unmatched-sample handling, duplicate-column checks,
save/reload round-trips, and report generation). It is never treated as,
or substituted for, the real IO-VNBD dataset.

## Stage boundary

This module stops after producing the aligned dataset and Stage 1
reports/plot. Preprocessing & calibration, feature engineering, the LSTM
model, dead reckoning, drift calculation, AI drift correction, evaluation,
and model export are all out of scope here and will be implemented in
later, separate stages.
