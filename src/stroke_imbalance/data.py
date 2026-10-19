"""Download, verify and load the Kaggle stroke dataset.

Run ``python -m stroke_imbalance.data`` to fetch the CSV into ``data/raw/``.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from stroke_imbalance.config import PROJECT_ROOT

KAGGLE_DATASET = "fedesoriano/stroke-prediction-dataset"
KAGGLE_URL = "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"
FILENAME = "healthcare-dataset-stroke-data.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PATH = RAW_DIR / FILENAME

# Recorded from the first download (kagglehub, dataset version 1, 2026-10-19).
# Checked on every load.
EXPECTED_SHA256 = "644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420"
EXPECTED_ROWS = 5110
EXPECTED_POSITIVES = 249


def sha256sum(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_kagglehub() -> Path | None:
    try:
        import kagglehub

        folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:  # noqa: BLE001 (any failure falls through to the next method)
        print(f"kagglehub download failed: {exc}")
        return None
    found = folder / FILENAME
    return found if found.exists() else None


def _download_kaggle_cli() -> Path | None:
    if shutil.which("kaggle") is None:
        print("Kaggle CLI not found.")
        return None
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR), "--unzip"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Kaggle CLI download failed: {result.stderr.strip()}")
        return None
    return DEFAULT_PATH if DEFAULT_PATH.exists() else None


def download(dest: Path = DEFAULT_PATH) -> Path:
    """Fetch the CSV via kagglehub, then the Kaggle CLI. Print manual steps if both fail."""
    if dest.exists():
        print(f"Already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = _download_kagglehub() or _download_kaggle_cli()
    if source is None:
        sys.exit(
            "Automatic download failed. Download the dataset manually:\n"
            f"  1. Open {KAGGLE_URL}\n"
            f"  2. Download and unzip it, then place {FILENAME} in {RAW_DIR}\n"
        )
    if source.resolve() != dest.resolve():
        shutil.copyfile(source, dest)
    print(f"Saved {dest}\nSHA-256: {sha256sum(dest)}")
    return dest


def load_raw(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the CSV after checking its checksum, row count and number of stroke cases."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m stroke_imbalance.data` first.")
    if EXPECTED_SHA256 is not None:
        actual = sha256sum(path)
        if actual != EXPECTED_SHA256:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {EXPECTED_SHA256}")
    df = pd.read_csv(path)
    if len(df) != EXPECTED_ROWS or int(df["stroke"].sum()) != EXPECTED_POSITIVES:
        raise ValueError(
            f"Unexpected data: {len(df)} rows, {int(df['stroke'].sum())} positives "
            f"(expected {EXPECTED_ROWS} and {EXPECTED_POSITIVES})."
        )
    return df


if __name__ == "__main__":
    load_raw(download())
    print(f"Verified: {EXPECTED_ROWS} rows, {EXPECTED_POSITIVES} positives.")
