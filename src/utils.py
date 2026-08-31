"""Shared path and file I/O helpers used across the cleaning, feature
engineering, training, and database-loading scripts."""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DB_DIR = os.path.join(BASE_DIR, "db")


def ensure_dirs():
    """Creates every output directory the pipeline writes to. Safe to call
    repeatedly - exist_ok=True means it's a no-op once they already exist."""
    for path in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DB_DIR):
        os.makedirs(path, exist_ok=True)


def load_csv(filename, folder=RAW_DIR):
    """Reads a CSV from the given folder (data/raw by default). Raises a
    clear error instead of a bare pandas traceback if the file is missing -
    almost always means an earlier pipeline step hasn't been run yet."""
    path = os.path.join(folder, filename)
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find {path}. If this is a processed/ file, run the "
            f"earlier pipeline steps first (data_cleaning.py, "
            f"feature_engineering.py)."
        ) from exc


def save_csv(df, filename, folder=PROCESSED_DIR):
    """Writes a DataFrame to the given folder (data/processed by default)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print(f"Saved file: {path}")
    return path


def save_raw_csv(df, filename):
    """Writes a DataFrame into data/raw/ - used only by
    generate_synthetic_data.py, which is the one script that produces raw
    input data rather than consuming it."""
    return save_csv(df, filename, folder=RAW_DIR)
