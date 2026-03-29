import pandas as pd
import numpy as np
from utils import ensure_dirs, load_csv, save_csv

def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

def fill_missing(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

def parse_datetime_columns(df):
    for col in df.columns:
        if "time" in col or "date" in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df

def standardize_status(df):
    if "ride_status" in df.columns:
        df["ride_status"] = df["ride_status"].astype(str).str.strip().str.title()
    if "cancelled_by" in df.columns:
        df["cancelled_by"] = df["cancelled_by"].astype(str).str.strip().str.title()
    return df

def basic_numeric_clean(df):
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
    print(f"{filename}: {before_rows} -> {after_rows}")

    cleaned_name = filename.replace(".csv", "_cleaned.csv")
    save_csv(df, cleaned_name)
    return df

def main():
    ensure_dirs()
    files = [
        "bookings.csv",
        "customers.csv",
        "drivers.csv",
        "location_demand.csv",
        "time_features.csv"
    ]
    for f in files:
        clean_file(f)

if __name__ == "__main__":
    main()