"""Cleans the five raw CSVs (bookings/customers/drivers/location_demand/
time_features) and writes a `_cleaned.csv` version of each into
data/processed/.

Run after generate_synthetic_data.py (or with your own raw CSVs dropped
into data/raw/ in the same schema):
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
    """lowercase_with_underscores column names, stripped of anything that
    isn't a letter/number/underscore (guards against stray BOM characters
    or punctuation sneaking in from an Excel export)."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def fill_missing(df):
    """Object columns get an explicit "Unknown" instead of staying null (so
    a missing category doesn't silently vanish from a groupby or a one-hot
    encoder); numeric columns get the column's own median, which is a
    reasonable stand-in for the handful of missing values this dataset
    actually has and won't blow up outliers the way a mean would."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")
    return df


def parse_datetime_columns(df):
    """Any column with "time" or "date" in its name gets a shot at
    pd.to_datetime - wrapped in try/except because a couple of columns in
    this dataset (e.g. trip_duration_min) happen to contain "time"-ish
    substrings without actually holding date data, and should just be left
    alone rather than crash the whole cleaning run."""
    for col in df.columns:
        if "time" in col or "date" in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass
    return df


def standardize_status(df):
    """Title-cases the two free-text status columns so "completed",
    "Completed", and "COMPLETED" don't end up as three different
    categories to a model or a groupby."""
    if "ride_status" in df.columns:
        df["ride_status"] = df["ride_status"].astype(str).str.strip().str.title()
    if "cancelled_by" in df.columns:
        df["cancelled_by"] = df["cancelled_by"].astype(str).str.strip().str.title()
    return df


def basic_numeric_clean(df):
    """Divide-by-zero and similar edge cases upstream can leave literal
    inf/-inf values in a numeric column - fill_missing() only catches NaN,
    so this runs first to turn infinities into NaN as well."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def clean_file(filename):
    """Runs one raw CSV through the full cleaning sequence and writes the
    result to data/processed/<name>_cleaned.csv."""
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
