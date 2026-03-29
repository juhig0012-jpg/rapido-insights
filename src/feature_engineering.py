import numpy as np
import pandas as pd
from utils import ensure_dirs, load_csv, save_csv


def main():
    ensure_dirs()

    bookings = load_csv("bookings_cleaned.csv", folder="data/processed")
    customers = load_csv("customers_cleaned.csv", folder="data/processed")
    drivers = load_csv("drivers_cleaned.csv", folder="data/processed")
    location = load_csv("location_demand_cleaned.csv", folder="data/processed")
    time_df = load_csv("time_features_cleaned.csv", folder="data/processed")

    bookings["booking_time"] = pd.to_datetime(bookings["booking_time"], errors="coerce")

    bookings["hour"] = bookings["booking_time"].dt.hour
    bookings["day_of_week"] = bookings["booking_time"].dt.day_name()
    bookings["is_weekend"] = bookings["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    df = bookings.merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
    df = df.merge(drivers, on="driver_id", how="left", suffixes=("", "_driver"))
    df = df.merge(location, on=["pickup_location", "drop_location"], how="left")
    df = df.merge(time_df, on="hour", how="left")

    # Fix duplicate weekend columns after merge
    if "is_weekend_x" in df.columns:
        df["is_weekend"] = df["is_weekend_x"]

    dup_weekend_cols = [c for c in ["is_weekend_x", "is_weekend_y"] if c in df.columns]
    if dup_weekend_cols:
        df.drop(columns=dup_weekend_cols, inplace=True)

    # Drop name columns - useful for display, not for ML
    for col in ["customer_name", "driver_name"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df["estimated_fare"] = df["base_fare"] * df["surge_multiplier"]

    df["fare_per_km"] = df["estimated_fare"] / df["distance_km"].replace(0, np.nan)
    df["fare_per_min"] = df["estimated_fare"] / df["trip_duration_min"].replace(0, np.nan)

    df["fare_per_km"] = df["fare_per_km"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["fare_per_min"] = df["fare_per_min"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["long_distance_flag"] = (df["distance_km"] >= df["distance_km"].median()).astype(int)
    df["rush_hour_flag"] = df["is_peak_hour"].fillna(0).astype(int)

    df["customer_cancellation_rate"] = (
        df["total_cancelled_rides"] /
        (df["total_completed_rides"] + df["total_cancelled_rides"]).replace(0, np.nan)
    ).fillna(0)

    df["driver_cancellation_rate"] = (
        df["total_cancelled_rides_driver"] /
        (df["total_completed_rides_driver"] + df["total_cancelled_rides_driver"]).replace(0, np.nan)
    ).fillna(0)

    df["driver_reliability_score"] = (
        (df["driver_rating"] * 20) * 0.4 +
        (df["acceptance_rate"] * 100) * 0.4 +
        ((1 / (1 + df["avg_delay_min"])) * 100) * 0.2
    )

    df["customer_loyalty_score"] = (
        (df["customer_rating"] * 20) * 0.4 +
        (df["total_completed_rides"] * 0.4) +
        ((1 - df["customer_cancellation_rate"]) * 100 * 0.2)
    )

    df["ride_outcome_target"] = df["ride_status"]
    df["customer_cancel_flag"] = (df["cancelled_by"] == "Customer").astype(int)
    df["driver_delay_flag"] = (df["driver_delay_min"] >= 10).astype(int)

    # Clean object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].fillna("Unknown")

    # Clean numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)

    save_csv(df, "final_merged_data.csv")
    print("Saved: data/processed/final_merged_data.csv")
    print("Final shape:", df.shape)
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()