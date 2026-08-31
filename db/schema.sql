-- Normalized schema for the Rapido mobility dataset.
--
-- customers and drivers are independent dimension tables; location_demand
-- is a dimension table keyed on a pickup/drop pair; time_features is a
-- small fixed lookup table keyed on hour-of-day. bookings is the one fact
-- table, referencing all four via foreign keys instead of repeating
-- customer/driver/location attributes on every row - that repetition is
-- exactly what the raw CSVs do (every booking row carries its own copy of
-- the driver's rating, acceptance rate, etc.), which is fine for a pandas
-- pipeline but is the redundancy a normalized schema is meant to avoid.
--
-- Written for SQLite (see src/load_db.py) - the same DDL runs on
-- Postgres/MySQL with minor type-name changes (TEXT -> VARCHAR, etc.).

CREATE TABLE IF NOT EXISTS customers (
    customer_id            TEXT PRIMARY KEY,
    customer_name           TEXT NOT NULL,
    customer_rating         REAL NOT NULL,
    total_completed_rides   INTEGER NOT NULL DEFAULT 0,
    total_cancelled_rides   INTEGER NOT NULL DEFAULT 0,
    avg_monthly_spend       REAL,
    preferred_vehicle       TEXT
);

CREATE TABLE IF NOT EXISTS drivers (
    driver_id               TEXT PRIMARY KEY,
    driver_name              TEXT NOT NULL,
    driver_rating             REAL NOT NULL,
    total_completed_rides     INTEGER NOT NULL DEFAULT 0,
    total_cancelled_rides     INTEGER NOT NULL DEFAULT 0,
    acceptance_rate           REAL,
    avg_delay_min             REAL,
    vehicle_type              TEXT
);

CREATE TABLE IF NOT EXISTS location_demand (
    pickup_location   TEXT NOT NULL,
    drop_location     TEXT NOT NULL,
    demand_index      INTEGER,
    avg_eta_min       INTEGER,
    zone_type         TEXT,
    PRIMARY KEY (pickup_location, drop_location)
);

CREATE TABLE IF NOT EXISTS time_features (
    hour            INTEGER PRIMARY KEY,
    is_peak_hour    INTEGER NOT NULL,
    is_weekend      INTEGER NOT NULL,
    time_bucket     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bookings (
    booking_id          TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id),
    driver_id           TEXT NOT NULL REFERENCES drivers(driver_id),
    booking_time         TEXT NOT NULL,
    pickup_location      TEXT NOT NULL,
    drop_location         TEXT NOT NULL,
    city                  TEXT NOT NULL,
    vehicle_type          TEXT NOT NULL,
    payment_method        TEXT NOT NULL,
    distance_km           REAL NOT NULL,
    trip_duration_min     REAL NOT NULL,
    base_fare             REAL NOT NULL,
    surge_multiplier      REAL NOT NULL,
    traffic_level         TEXT,
    weather_condition     TEXT,
    ride_status           TEXT NOT NULL,
    cancelled_by          TEXT,
    driver_delay_min      REAL,
    FOREIGN KEY (pickup_location, drop_location)
        REFERENCES location_demand(pickup_location, drop_location)
);

-- Indexes on the columns the dashboard and EDA queries actually filter or
-- group by - customer/driver lookups, city-level aggregation, and the
-- ride_status breakdown used on almost every page of the Streamlit app.
CREATE INDEX IF NOT EXISTS idx_bookings_customer_id ON bookings(customer_id);
CREATE INDEX IF NOT EXISTS idx_bookings_driver_id ON bookings(driver_id);
CREATE INDEX IF NOT EXISTS idx_bookings_city ON bookings(city);
CREATE INDEX IF NOT EXISTS idx_bookings_ride_status ON bookings(ride_status);
CREATE INDEX IF NOT EXISTS idx_bookings_booking_time ON bookings(booking_time);
