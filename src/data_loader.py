"""
data_loader.py
--------------
Stage 1 module for NAV-SHIELD (Member 5 - AI Drift Correction pipeline).

Responsibilities (Stage 1 ONLY):
    - Resolve and validate the local IO-VNBD "Synchronised V and S Dataset" path.
    - Discover S/V file pairs for a given sequence name (e.g. "Vfa01"), regardless
      of which sub-folder they live in inside the dataset tree.
    - Load the raw CSV files into pandas DataFrames without modifying the
      original files on disk, tolerating the mixed text encodings actually
      present in the dataset (the vehicle "V" files are UTF-8, the smartphone
      "S" files are commonly Windows-1252 / Latin-1 because of the "°" and
      "µ" characters in their headers).

This module does NOT validate data quality (see data_validator.py) and does
NOT perform S/V synchronization (see synchronizer.py). It only finds and
loads data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetPathError(FileNotFoundError):
    """Raised when the supplied dataset root path does not exist."""


class SVFileNotFoundError(FileNotFoundError):
    """Raised when the S or V file for a requested sequence cannot be found."""


class CSVLoadError(RuntimeError):
    """Raised when a CSV file exists but cannot be parsed with any known encoding."""


# Encodings attempted, in order, when reading IO-VNBD CSV files.
# The dataset mixes UTF-8 (vehicle files) and Windows-1252/Latin-1
# (smartphone files, due to '°' and 'µ' symbols in column headers).
CANDIDATE_ENCODINGS = ("utf-8", "cp1252", "latin-1")


@dataclass
class SVFilePair:
    """Resolved paths for one S/V sequence pair."""

    sequence: str
    s_path: Path
    v_path: Path


def resolve_dataset_root(dataset_path: str) -> Path:
    """
    Validate that the user-supplied dataset path exists on disk.

    Parameters
    ----------
    dataset_path : str
        Path to the root of the locally extracted
        "IO-VNBD Synchronised V and S Dataset" folder.

    Returns
    -------
    Path
        The resolved, validated dataset root.

    Raises
    ------
    DatasetPathError
        If the path does not exist or is not a directory.
    """
    root = Path(dataset_path).expanduser().resolve()

    if not root.exists():
        raise DatasetPathError(
            f"Dataset path does not exist: '{root}'.\n"
            "Please supply the correct local path to the extracted "
            "IO-VNBD 'Synchronised V and S Dataset' folder, e.g.\n"
            '  python main.py --dataset "D:\\SIH\\IO-VNBD\\Synchronised V and S Dataset" '
            "--sequence Vfa01"
        )

    if not root.is_dir():
        raise DatasetPathError(
            f"Dataset path exists but is not a directory: '{root}'."
        )

    return root


def _find_file_case_insensitive(root: Path, filename: str) -> Optional[Path]:
    """
    Recursively search `root` for a file named `filename` (case-insensitive).
    Returns the first match, or None if not found.
    """
    target = filename.lower()
    for candidate in root.rglob("*.csv"):
        if candidate.name.lower() == target:
            return candidate
    return None


def discover_sv_pair(dataset_root: Path, sequence: str) -> SVFilePair:
    """
    Locate the S-<sequence>.csv and V-<sequence>.csv files anywhere inside
    the dataset root (searched recursively, since IO-VNBD ships each driver's
    data in its own sub-folder, e.g. "Vf (Driver E) / V-Vfa01").

    Parameters
    ----------
    dataset_root : Path
        Validated dataset root (see resolve_dataset_root).
    sequence : str
        Sequence identifier, e.g. "Vfa01" (WITHOUT the leading "S-"/"V-").

    Returns
    -------
    SVFilePair

    Raises
    ------
    SVFileNotFoundError
        If either the S or V file cannot be located.
    """
    s_filename = f"S-{sequence}.csv"
    v_filename = f"V-{sequence}.csv"

    s_path = _find_file_case_insensitive(dataset_root, s_filename)
    v_path = _find_file_case_insensitive(dataset_root, v_filename)

    missing = []
    if s_path is None:
        missing.append(s_filename)
    if v_path is None:
        missing.append(v_filename)

    if missing:
        raise SVFileNotFoundError(
            f"Could not locate the following required file(s) under '{dataset_root}': "
            f"{', '.join(missing)}.\n"
            "Confirm that the --dataset path points to the extracted "
            "IO-VNBD 'Synchronised V and S Dataset' folder and that the "
            "--sequence value matches an existing S-<seq>.csv / V-<seq>.csv pair."
        )

    return SVFilePair(sequence=sequence, s_path=s_path, v_path=v_path)


def load_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Load a CSV file trying several encodings until one succeeds.
    Does not modify, rewrite, or overwrite the original file.

    Returns
    -------
    (DataFrame, encoding_used)

    Raises
    ------
    CSVLoadError
        If no candidate encoding can parse the file.
    """
    if not path.exists():
        raise SVFileNotFoundError(f"File does not exist: '{path}'")

    last_error: Optional[Exception] = None
    for encoding in CANDIDATE_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df, encoding
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 - surfaced to caller below
            last_error = exc
            continue

    raise CSVLoadError(
        f"Failed to load '{path}' with any of the attempted encodings "
        f"{CANDIDATE_ENCODINGS}. Last error: {last_error}"
    )


def load_sv_pair(pair: SVFilePair) -> tuple[pd.DataFrame, str, pd.DataFrame, str]:
    """
    Load both files of an SVFilePair.

    Returns
    -------
    (df_s, s_encoding, df_v, v_encoding)
    """
    df_s, s_encoding = load_csv_robust(pair.s_path)
    df_v, v_encoding = load_csv_robust(pair.v_path)
    return df_s, s_encoding, df_v, v_encoding


def list_available_sequences(dataset_root: Path) -> list[str]:
    """
    Convenience helper (not required by the CLI, useful for debugging):
    scans the dataset root for all "V-<sequence>.csv" files and returns the
    list of sequence names for which a V file exists. Does not guarantee a
    matching S file exists; discover_sv_pair() still performs that check.
    """
    sequences = []
    for candidate in dataset_root.rglob("V-*.csv"):
        name = candidate.stem  # "V-Vfa01"
        if name.startswith("V-"):
            sequences.append(name[2:])
    return sorted(set(sequences))


if __name__ == "__main__":
    # Minimal manual smoke-test entry point.
    if len(sys.argv) != 3:
        print("Usage: python data_loader.py <dataset_path> <sequence>")
        sys.exit(1)
    root = resolve_dataset_root(sys.argv[1])
    pair = discover_sv_pair(root, sys.argv[2])
    print(f"Found S file: {pair.s_path}")
    print(f"Found V file: {pair.v_path}")
