"""Merges the five cleaned tables into one modeling dataset and derives the
features every model downstream trains on (fare ratios, rush-hour/long-
distance flags, City_Pair, reliability/loyalty scores, and the three model
targets).

Run after data_cleaning.py:
    python src/feature_engineering.py

Writes data/processed/final_merged_data.csv.
"""

import numpy as np
import pandas as pd

from utils import PROCESSED_DIR, ensure_dirs, load_csv, save_csv


def load_cleaned_tables():
    """Loads the five _cleaned.csv files produced by data_cleaning.py."""
    return {
        "bookings": load_csv("bookings_cleaned.csv", folder=PROCESSED_DIR),
        "customers": load_csv("customers_cleaned.csv", folder=PROCESSED_DIR),
        "drivers": load_csv("drivers_cleaned.csv", folder=PROCESSED_DIR),
        "location": load_csv("location_demand_cleaned.csv", folder=PROCESSED_DIR),
        "time": load_csv("time_features_cleaned.csv", folder=PROCESSED_DIR),
    }


def add_time_parts(bookings):
    """Breaks booking_time into hour/day-of-week/weekend flag - the raw
    timestamp itself isn't useful to a tree model, but its components are."""
    bookings["booking_time"] = pd.to_datetime(bookings["booking_time"], errors="coerce")
    bookings["hour"] = bookings["booking_time"].dt.hour
    bookings["day_of_week"] = bookings["booking_time"].dt.day_name()
    bookings["is_weekend"] = bookings["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return bookings


def merge_tables(tables):
    """Left-joins customer, driver, location, and time-bucket detail onto
    every booking row. Left joins throughout because every booking should
    survive even if, say, a pickup/drop pair has no entry in
    location_demand - we'd rather have a row with some missing columns than
    silently lose the booking."""
    bookings = add_time_parts(tables["bookings"])

    df = bookings.merge(tables["customers"], on="customer_id", how="left", suffixes=("", "_customer"))
    df = df.merge(tables["drivers"], on="driver_id", how="left", suffixes=("", "_driver"))
    df = df.merge(tables["location"], on=["pickup_location", "drop_location"], how="left")
    df = df.merge(tables["time"], on="hour", how="left")

    # bookings and the time-features table both have an is_weekend column;
    # the merge suffixes them into is_weekend_x/_y - keep the real one from
    # bookings (time-features' copy is a placeholder, always 0)
    if "is_weekend_x" in df.columns:
        df["is_weekend"] = df["is_weekend_x"]
    df = df.drop(columns=[c for c in ("is_weekend_x", "is_weekend_y") if c in df.columns])

    # names are useful for a human reading the dashboard, not for a model
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
    """Historical cancellation rates and the two composite scores the spec
    asks for (Driver_Reliability_Score, Customer_Loyalty_Score). Weights
    below are a judgment call, not derived from anything - rating counts
    for 40%, behavior (acceptance/completed-ride volume) for 40%, and
    delay/cancellation history for the remaining 20%."""
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
    """The three (soon four) labels the training scripts predict."""
    df["ride_outcome_target"] = df["ride_status"]
    df["customer_cancel_flag"] = (df["cancelled_by"] == "Customer").astype(int)
    df["driver_delay_flag"] = (df["driver_delay_min"] >= 10).astype(int)
    return df


def final_cleanup(df):
    """One more missing-value pass after all the merges/derived columns -
    a left join can introduce fresh NaNs (e.g. a location pair with no
    demand-table entry) that didn't exist in any single source table."""
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
