-- Normalized schema for the Rapido mobility dataset.
--
-- customers, drivers, location_demand are dimension tables; time_features
-- is a small fixed lookup keyed on hour-of-day; bookings is the fact table,
-- referencing all four by FK instead of repeating customer/driver/location
-- columns on every row like the raw CSVs do.
--
-- Written for SQLite (see src/load_db.py). Should run on Postgres/MySQL
-- with minor type changes (TEXT -> VARCHAR etc).

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

-- indexes for the columns the dashboard/EDA queries filter or group by
CREATE INDEX IF NOT EXISTS idx_bookings_customer_id ON bookings(customer_id);
CREATE INDEX IF NOT EXISTS idx_bookings_driver_id ON bookings(driver_id);
CREATE INDEX IF NOT EXISTS idx_bookings_city ON bookings(city);
CREATE INDEX IF NOT EXISTS idx_bookings_ride_status ON bookings(ride_status);
CREATE INDEX IF NOT EXISTS idx_bookings_booking_time ON bookings(booking_time);
