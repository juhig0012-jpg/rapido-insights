-- Ad-hoc analysis queries against db/rapido.db (built by src/load_db.py).
-- Run with: sqlite3 db/rapido.db < db/analysis_queries.sql

-- Ride volume by hour of day
SELECT
    CAST(strftime('%H', booking_time) AS INTEGER) AS hour,
    COUNT(*) AS ride_count
FROM bookings
GROUP BY hour
ORDER BY hour;

-- Cancellation rate by city
SELECT
    city,
    COUNT(*) AS total_rides,
    SUM(CASE WHEN ride_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_rides,
    ROUND(100.0 * SUM(CASE WHEN ride_status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct
FROM bookings
GROUP BY city
ORDER BY cancellation_rate_pct DESC;

-- Traffic level vs cancellation rate
SELECT
    traffic_level,
    COUNT(*) AS total_rides,
    ROUND(100.0 * SUM(CASE WHEN ride_status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct
FROM bookings
GROUP BY traffic_level
ORDER BY cancellation_rate_pct DESC;

-- Weather condition vs cancellation rate
SELECT
    weather_condition,
    COUNT(*) AS total_rides,
    ROUND(100.0 * SUM(CASE WHEN ride_status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct
FROM bookings
GROUP BY weather_condition
ORDER BY cancellation_rate_pct DESC;

-- Payment method usage share
SELECT
    payment_method,
    COUNT(*) AS ride_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM bookings), 2) AS pct_of_rides
FROM bookings
GROUP BY payment_method
ORDER BY ride_count DESC;

-- Top 10 drivers by average delay, among drivers with at least 20 completed rides
SELECT
    driver_id, driver_name, driver_rating, total_completed_rides, avg_delay_min
FROM drivers
WHERE total_completed_rides >= 20
ORDER BY avg_delay_min DESC
LIMIT 10;

-- Top 10 customers by cancellation rate (min 10 rides)
SELECT
    customer_id, customer_name, customer_rating,
    total_completed_rides, total_cancelled_rides,
    ROUND(100.0 * total_cancelled_rides / (total_completed_rides + total_cancelled_rides), 2) AS cancellation_rate_pct
FROM customers
WHERE (total_completed_rides + total_cancelled_rides) >= 10
ORDER BY cancellation_rate_pct DESC
LIMIT 10;

-- Average fare per km by vehicle type
SELECT
    vehicle_type,
    ROUND(AVG(base_fare * surge_multiplier), 2) AS avg_estimated_fare,
    ROUND(AVG((base_fare * surge_multiplier) / distance_km), 2) AS avg_fare_per_km,
    COUNT(*) AS ride_count
FROM bookings
GROUP BY vehicle_type
ORDER BY avg_fare_per_km DESC;
