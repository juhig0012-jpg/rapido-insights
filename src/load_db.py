"""Loads the cleaned CSVs into a local SQLite database using db/schema.sql.

SQLite because it's a single file with no server/credentials to set up -
the DDL should port to Postgres/MySQL later without much trouble.

Run after data_cleaning.py:
    python src/load_db.py
"""

import sqlite3

from utils import DB_DIR, PROCESSED_DIR, ensure_dirs, load_csv

DB_PATH = f"{DB_DIR}/rapido.db"
SCHEMA_PATH = f"{DB_DIR}/schema.sql"

# (table name, source CSV, columns to keep - in schema order)
TABLE_SOURCES = [
    ("customers", "customers_cleaned.csv", [
        "customer_id", "customer_name", "customer_rating",
        "total_completed_rides", "total_cancelled_rides",
        "avg_monthly_spend", "preferred_vehicle",
    ]),
    ("drivers", "drivers_cleaned.csv", [
        "driver_id", "driver_name", "driver_rating",
        "total_completed_rides", "total_cancelled_rides",
        "acceptance_rate", "avg_delay_min", "vehicle_type",
    ]),
    ("location_demand", "location_demand_cleaned.csv", [
        "pickup_location", "drop_location", "demand_index",
        "avg_eta_min", "zone_type",
    ]),
    ("time_features", "time_features_cleaned.csv", [
        "hour", "is_peak_hour", "is_weekend", "time_bucket",
    ]),
    ("bookings", "bookings_cleaned.csv", [
        "booking_id", "customer_id", "driver_id", "booking_time",
        "pickup_location", "drop_location", "city", "vehicle_type",
        "payment_method", "distance_km", "trip_duration_min", "base_fare",
        "surge_multiplier", "traffic_level", "weather_condition",
        "ride_status", "cancelled_by", "driver_delay_min",
    ]),
]


def build_schema(conn):
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    try:
        conn.executescript(schema_sql)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to apply {SCHEMA_PATH}: {exc}") from exc


def load_table(conn, table_name, csv_filename, columns):
    # delete existing rows first so reruns don't duplicate everything
    df = load_csv(csv_filename, folder=PROCESSED_DIR)[columns]
    try:
        conn.execute(f"DELETE FROM {table_name}")
        df.to_sql(table_name, conn, if_exists="append", index=False)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to load {csv_filename} into {table_name}: {exc}") from exc
    print(f"Loaded {len(df):,} rows into {table_name}")


def main():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        build_schema(conn)
        # order matters - bookings has FKs into the other three tables
        for table_name, csv_filename, columns in TABLE_SOURCES:
            load_table(conn, table_name, csv_filename, columns)
        conn.commit()
        print(f"\nDatabase ready: {DB_PATH}")
    except (RuntimeError, sqlite3.Error) as exc:
        conn.rollback()
        print(f"Load failed, rolled back: {exc}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
