"""Shared path and file I/O helpers used across the pipeline scripts."""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DB_DIR = os.path.join(BASE_DIR, "db")


def ensure_dirs():
    for path in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DB_DIR):
        os.makedirs(path, exist_ok=True)


def load_csv(filename, folder=RAW_DIR):
    # clearer error than a bare pandas traceback - usually means an earlier
    # pipeline step hasn't run yet
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
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print(f"Saved file: {path}")
    return path


def save_raw_csv(df, filename):
    # only generate_synthetic_data.py writes to data/raw/, everything else reads it
    return save_csv(df, filename, folder=RAW_DIR)
