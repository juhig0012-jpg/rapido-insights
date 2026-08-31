"""Cleans the five raw CSVs and writes a `_cleaned.csv` version of each
into data/processed/.

Run after generate_synthetic_data.py (or with your own raw CSVs in
data/raw/, same schema):
    python src/data_cleaning.py
"""

import numpy as np
import pandas as pd

from utils import ensure_dirs, load_csv, save_csv

RAW_FILES = [
    "bookings.csv",
    "customers.csv",
    "drivers.csv",
    "location_demand.csv",
    "time_features.csv",
]


def clean_columns(df):
    # lowercase_with_underscores, strip anything but letters/digits/underscore
    # (Excel exports like to sneak in a BOM or stray punctuation)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def fill_missing(df):
    # numeric -> median (robust to outliers), text -> "Unknown" so it doesn't
    # just vanish from a groupby/one-hot
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")
    return df


def parse_datetime_columns(df):
    # any column with "time"/"date" in the name gets tried as a datetime.
    # trip_duration_min matches too but isn't a real date, hence the try/except
    for col in df.columns:
        if "time" in col or "date" in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass
    return df


def standardize_status(df):
    # title-case so "completed"/"Completed"/"COMPLETED" aren't 3 categories
    if "ride_status" in df.columns:
        df["ride_status"] = df["ride_status"].astype(str).str.strip().str.title()
    if "cancelled_by" in df.columns:
        df["cancelled_by"] = df["cancelled_by"].astype(str).str.strip().str.title()
    return df


def basic_numeric_clean(df):
    # turn inf/-inf into NaN first, fill_missing() only handles NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def clean_file(filename):
    df = load_csv(filename)
    before_rows = len(df)

    df = clean_columns(df)
    df = df.drop_duplicates()
    df = parse_datetime_columns(df)
    df = basic_numeric_clean(df)
    df = fill_missing(df)
    df = standardize_status(df)

    after_rows = len(df)
    print(f"{filename}: {before_rows} -> {after_rows} rows")

    cleaned_name = filename.replace(".csv", "_cleaned.csv")
    save_csv(df, cleaned_name)
    return df


def main():
    ensure_dirs()
    for filename in RAW_FILES:
        try:
            clean_file(filename)
        except FileNotFoundError as exc:
            print(f"Skipping {filename}: {exc}")


if __name__ == "__main__":
    main()
