"""Merges the five cleaned tables into one modeling dataset and derives the
features the models train on (fare ratios, rush-hour/long-distance flags,
city_pair, reliability/loyalty scores, and the model targets).

Run after data_cleaning.py:
    python src/feature_engineering.py

Writes data/processed/final_merged_data.csv.
"""

import numpy as np
import pandas as pd

from utils import PROCESSED_DIR, ensure_dirs, load_csv, save_csv


def load_cleaned_tables():
    return {
        "bookings": load_csv("bookings_cleaned.csv", folder=PROCESSED_DIR),
        "customers": load_csv("customers_cleaned.csv", folder=PROCESSED_DIR),
        "drivers": load_csv("drivers_cleaned.csv", folder=PROCESSED_DIR),
        "location": load_csv("location_demand_cleaned.csv", folder=PROCESSED_DIR),
        "time": load_csv("time_features_cleaned.csv", folder=PROCESSED_DIR),
    }


def add_time_parts(bookings):
    # break the timestamp into parts a tree model can actually use
    bookings["booking_time"] = pd.to_datetime(bookings["booking_time"], errors="coerce")
    bookings["hour"] = bookings["booking_time"].dt.hour
    bookings["day_of_week"] = bookings["booking_time"].dt.day_name()
    bookings["is_weekend"] = bookings["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return bookings


def merge_tables(tables):
    # left joins throughout - a booking should survive even if e.g. its
    # pickup/drop pair has no location_demand entry
    bookings = add_time_parts(tables["bookings"])

    df = bookings.merge(tables["customers"], on="customer_id", how="left", suffixes=("", "_customer"))
    df = df.merge(tables["drivers"], on="driver_id", how="left", suffixes=("", "_driver"))
    df = df.merge(tables["location"], on=["pickup_location", "drop_location"], how="left")
    df = df.merge(tables["time"], on="hour", how="left")

    # both bookings and time_features have is_weekend -> suffixed to _x/_y,
    # keep bookings' version (time_features' copy is just a placeholder 0)
    if "is_weekend_x" in df.columns:
        df["is_weekend"] = df["is_weekend_x"]
    df = df.drop(columns=[c for c in ("is_weekend_x", "is_weekend_y") if c in df.columns])

    df = df.drop(columns=[c for c in ("customer_name", "driver_name") if c in df.columns])

    return df


def add_fare_features(df):
    df["estimated_fare"] = df["base_fare"] * df["surge_multiplier"]

    df["fare_per_km"] = df["estimated_fare"] / df["distance_km"].replace(0, np.nan)
    df["fare_per_min"] = df["estimated_fare"] / df["trip_duration_min"].replace(0, np.nan)
    df["fare_per_km"] = df["fare_per_km"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["fare_per_min"] = df["fare_per_min"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def add_trip_flags(df):
    df["long_distance_flag"] = (df["distance_km"] >= df["distance_km"].median()).astype(int)
    df["rush_hour_flag"] = df["is_peak_hour"].fillna(0).astype(int)
    df["city_pair"] = df["pickup_location"].astype(str) + " -> " + df["drop_location"].astype(str)
    return df


def add_reliability_scores(df):
    # weights below (40/40/20) are a judgment call, not fitted to anything
    df["customer_cancellation_rate"] = (
        df["total_cancelled_rides"]
        / (df["total_completed_rides"] + df["total_cancelled_rides"]).replace(0, np.nan)
    ).fillna(0)

    df["driver_cancellation_rate"] = (
        df["total_cancelled_rides_driver"]
        / (df["total_completed_rides_driver"] + df["total_cancelled_rides_driver"]).replace(0, np.nan)
    ).fillna(0)

    df["driver_reliability_score"] = (
        (df["driver_rating"] * 20) * 0.4
        + (df["acceptance_rate"] * 100) * 0.4
        + ((1 / (1 + df["avg_delay_min"])) * 100) * 0.2
    )

    df["customer_loyalty_score"] = (
        (df["customer_rating"] * 20) * 0.4
        + (df["total_completed_rides"] * 0.4)
        + ((1 - df["customer_cancellation_rate"]) * 100 * 0.2)
    )

    return df


def add_model_targets(df):
    df["ride_outcome_target"] = df["ride_status"]
    df["customer_cancel_flag"] = (df["cancelled_by"] == "Customer").astype(int)
    df["driver_delay_flag"] = (df["driver_delay_min"] >= 10).astype(int)
    return df


def final_cleanup(df):
    # left joins can introduce new NaNs that weren't in any source table
    for col in df.columns:
        is_text = not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])
        if is_text:
            df[col] = df[col].fillna("Unknown")

    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def build_features():
    ensure_dirs()
    tables = load_cleaned_tables()

    df = merge_tables(tables)
    df = add_fare_features(df)
    df = add_trip_flags(df)
    df = add_reliability_scores(df)
    df = add_model_targets(df)
    df = final_cleanup(df)

    save_csv(df, "final_merged_data.csv")
    print("Saved: data/processed/final_merged_data.csv")
    print("Final shape:", df.shape)
    return df


def main():
    build_features()


if __name__ == "__main__":
    main()
