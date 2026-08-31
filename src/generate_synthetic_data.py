"""Generates a larger, realistic set of raw CSVs in the shape the rest of
the pipeline expects (bookings/customers/drivers/location_demand/time_features).

Why this exists: the dataset that shipped with this project had the right
columns for every downstream model (fare, surge, traffic, weather, driver
delay, cancellation reason) but only 15 booking rows - nowhere near enough
to train anything or trust a reported accuracy/RMSE number. There's also a
much larger `data/raw/rides_data.csv` (50k rows) sitting in the repo, but
it's a different dataset entirely - no customer/driver IDs, no traffic or
weather columns, no surge multiplier - so it can't answer three of the four
questions this project needs (it's fine for ride-outcome/fare trends in
isolation, but not for anything customer- or driver-specific). Rather than
force that file into a schema it doesn't have the columns for, this script
generates a synthetic sample *in the schema the project actually needs*,
scaled up enough for the models and metrics to mean something.

Run directly:
    python src/generate_synthetic_data.py
"""

import os

import numpy as np
import pandas as pd

from utils import ensure_dirs, save_raw_csv

RNG_SEED = 42
N_CUSTOMERS = 400
N_DRIVERS = 150
N_BOOKINGS = 6000

CITIES = ["Delhi", "Bengaluru", "Mumbai", "Hyderabad", "Pune"]

LOCATIONS_BY_CITY = {
    "Delhi": ["Connaught Place", "Karol Bagh", "Saket", "Hauz Khas", "Dwarka",
              "Janakpuri", "Rohini", "Pitampura", "Lajpat Nagar", "Noida Sector 18"],
    "Bengaluru": ["Koramangala", "Indiranagar", "Whitefield", "HSR Layout",
                  "Electronic City", "Jayanagar", "MG Road", "Marathahalli"],
    "Mumbai": ["Andheri", "Bandra", "Dadar", "Powai", "Borivali",
               "Chembur", "Colaba", "Malad"],
    "Hyderabad": ["Gachibowli", "Madhapur", "Banjara Hills", "Kukatpally",
                  "Secunderabad", "Ameerpet"],
    "Pune": ["Hinjewadi", "Kothrud", "Viman Nagar", "Baner", "Kharadi", "Wakad"],
}

VEHICLE_TYPES = ["Bike", "Auto", "Cab"]
PAYMENT_METHODS = ["UPI", "Cash", "Card", "Wallet"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Smog"]
ZONE_TYPES = ["Residential", "Commercial", "Tourist", "Transit"]

BASE_FARE_BY_VEHICLE = {"Bike": 40, "Auto": 60, "Cab": 90}
PER_KM_BY_VEHICLE = {"Bike": 6, "Auto": 9, "Cab": 14}


def _make_time_features():
    """Fixed hour -> peak/time-bucket lookup table. Small on purpose - this
    is a dimension table (24 possible hours), not a fact table, so it
    doesn't need scaling up like bookings/customers/drivers do."""
    rows = []
    for hour in range(24):
        is_peak = int(hour in (7, 8, 9, 17, 18, 19))
        if 5 <= hour < 11:
            bucket = "Morning"
        elif 11 <= hour < 14:
            bucket = "Late Morning"
        elif 14 <= hour < 17:
            bucket = "Afternoon"
        elif 17 <= hour < 21:
            bucket = "Evening"
        else:
            bucket = "Night"
        rows.append({"hour": hour, "is_peak_hour": is_peak, "is_weekend": 0, "time_bucket": bucket})
    return pd.DataFrame(rows)


def _make_customers(rng):
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        completed = int(rng.integers(2, 300))
        # each customer has a persistent "flakiness" trait, drawn from a
        # right-skewed distribution so most customers rarely cancel but a
        # meaningful minority cancel often - a tight, low-variance rate here
        # would make customer_cancellation_rate nearly identical for every
        # outcome, giving the cancellation-risk model nothing to learn from
        cancel_rate = np.clip(rng.beta(2, 14), 0.0, 0.6)
        cancelled = int(round(completed * cancel_rate / max(1 - cancel_rate, 0.01)))
        rating = np.clip(rng.normal(4.3 - cancel_rate, 0.3), 1.0, 5.0)
        rows.append({
            "customer_id": f"C{i:04d}",
            "customer_name": f"Customer_{i}",
            "customer_rating": round(rating, 2),
            "total_completed_rides": completed,
            "total_cancelled_rides": cancelled,
            "avg_monthly_spend": int(np.clip(rng.normal(600 + completed * 8, 300), 100, None)),
            "preferred_vehicle": rng.choice(VEHICLE_TYPES, p=[0.35, 0.35, 0.30]),
        })
    return pd.DataFrame(rows)


def _make_drivers(rng):
    rows = []
    for i in range(1, N_DRIVERS + 1):
        completed = int(rng.integers(20, 800))
        acceptance = np.clip(rng.normal(0.90, 0.08), 0.4, 1.0)
        # low acceptance rate tends to go with more delay - gives
        # driver_reliability_score real predictive signal
        avg_delay = np.clip(rng.normal(8 - acceptance * 6, 3), 0, 30)
        cancelled = int(rng.integers(0, max(int(completed * 0.15), 1)))
        rating = np.clip(rng.normal(4.5 - avg_delay / 25, 0.25), 1.0, 5.0)
        rows.append({
            "driver_id": f"D{i:04d}",
            "driver_name": f"Driver_{i}",
            "driver_rating": round(rating, 2),
            "total_completed_rides": completed,
            "total_cancelled_rides": cancelled,
            "acceptance_rate": round(acceptance, 2),
            "avg_delay_min": round(avg_delay, 1),
            "vehicle_type": rng.choice(VEHICLE_TYPES, p=[0.35, 0.35, 0.30]),
        })
    return pd.DataFrame(rows)


def _make_location_demand(rng):
    rows = []
    seen_pairs = set()
    for city, spots in LOCATIONS_BY_CITY.items():
        for pickup in spots:
            for drop in spots:
                if pickup == drop or (pickup, drop) in seen_pairs:
                    continue
                seen_pairs.add((pickup, drop))
                zone = rng.choice(ZONE_TYPES, p=[0.35, 0.35, 0.15, 0.15])
                demand = int(rng.integers(40, 95))
                rows.append({
                    "pickup_location": pickup,
                    "drop_location": drop,
                    "demand_index": demand,
                    "avg_eta_min": int(rng.integers(3, 12)),
                    "zone_type": zone,
                })
    return pd.DataFrame(rows)


def _make_bookings(rng, customers, drivers, location_demand):
    customer_ids = customers["customer_id"].tolist()
    driver_ids = drivers["driver_id"].tolist()
    # weight booking assignment by each customer/driver's ride volume, so the
    # historical-rate features computed later aren't dominated by customers
    # who only appear once or twice
    customer_weights = (customers["total_completed_rides"] + 1).to_numpy()
    customer_weights = customer_weights / customer_weights.sum()
    driver_weights = (drivers["total_completed_rides"] + 1).to_numpy()
    driver_weights = driver_weights / driver_weights.sum()

    start = pd.Timestamp("2026-01-01")
    rows = []

    for i in range(1, N_BOOKINGS + 1):
        city = rng.choice(CITIES)
        spots = LOCATIONS_BY_CITY[city]
        pickup, drop = rng.choice(spots, size=2, replace=False)

        day_offset = int(rng.integers(0, 90))
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        booking_time = start + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)

        vehicle_type = rng.choice(VEHICLE_TYPES, p=[0.35, 0.35, 0.30])
        distance_km = round(float(np.clip(rng.exponential(6) + 1, 0.5, 45)), 2)
        is_peak = hour in (7, 8, 9, 17, 18, 19)
        traffic_level = rng.choice(
            TRAFFIC_LEVELS, p=[0.2, 0.35, 0.45] if is_peak else [0.5, 0.35, 0.15]
        )
        weather_condition = rng.choice(WEATHER_CONDITIONS, p=[0.55, 0.25, 0.15, 0.05])

        traffic_speed_factor = {"Low": 1.0, "Medium": 1.3, "High": 1.7}[traffic_level]
        trip_duration_min = round(distance_km * traffic_speed_factor * rng.uniform(1.6, 2.4), 1)

        base_fare = round(
            BASE_FARE_BY_VEHICLE[vehicle_type] + distance_km * PER_KM_BY_VEHICLE[vehicle_type], 2
        )
        surge_multiplier = round(
            1.0
            + (0.4 if is_peak else 0.0)
            + (0.3 if traffic_level == "High" else 0.0)
            + (0.2 if weather_condition == "Rain" else 0.0)
            + rng.uniform(0, 0.15),
            2,
        )

        customer_id = rng.choice(customer_ids, p=customer_weights)
        driver_id = rng.choice(driver_ids, p=driver_weights)
        customer_row = customers.loc[customers["customer_id"] == customer_id].iloc[0]
        driver_row = drivers.loc[drivers["driver_id"] == driver_id].iloc[0]

        # cancellation/delay probability is driven by the same signals a real
        # platform would see: bad weather + high traffic + a driver who's
        # historically unreliable + a customer who cancels a lot
        customer_hist_cancel_rate = customer_row["total_cancelled_rides"] / max(
            customer_row["total_completed_rides"] + customer_row["total_cancelled_rides"], 1
        )
        driver_unreliability = driver_row["avg_delay_min"] / 30 + (1 - driver_row["acceptance_rate"])

        cancel_prob = np.clip(
            0.03 + customer_hist_cancel_rate * 1.3 + (0.10 if traffic_level == "High" else 0)
            + (0.10 if weather_condition in ("Rain", "Smog") else 0),
            0.01, 0.75,
        )
        incomplete_prob = np.clip(0.02 + driver_unreliability * 0.22, 0.01, 0.3)

        roll = rng.random()
        if roll < cancel_prob:
            ride_status = "Cancelled"
            cancelled_by = rng.choice(["Customer", "Driver"], p=[0.7, 0.3])
        elif roll < cancel_prob + incomplete_prob:
            ride_status = "Incomplete"
            cancelled_by = "None"
        else:
            ride_status = "Completed"
            cancelled_by = "None"

        driver_delay_min = max(0, round(rng.normal(driver_row["avg_delay_min"], 3), 1))
        if traffic_level == "High":
            driver_delay_min += rng.uniform(0, 4)

        rows.append({
            "booking_id": f"B{i:05d}",
            "customer_id": customer_id,
            "driver_id": driver_id,
            "booking_time": booking_time,
            "pickup_location": pickup,
            "drop_location": drop,
            "city": city,
            "vehicle_type": vehicle_type,
            "payment_method": rng.choice(PAYMENT_METHODS, p=[0.45, 0.2, 0.25, 0.1]),
            "distance_km": distance_km,
            "trip_duration_min": trip_duration_min,
            "base_fare": base_fare,
            "surge_multiplier": surge_multiplier,
            "traffic_level": traffic_level,
            "weather_condition": weather_condition,
            "ride_status": ride_status,
            "cancelled_by": cancelled_by,
            "driver_delay_min": round(driver_delay_min, 1),
        })

    return pd.DataFrame(rows)


def generate_all():
    """Builds every raw CSV and writes it to data/raw/, overwriting the
    small starter files that shipped with the project."""
    rng = np.random.default_rng(RNG_SEED)
    ensure_dirs()

    customers = _make_customers(rng)
    drivers = _make_drivers(rng)
    location_demand = _make_location_demand(rng)
    time_features = _make_time_features()
    bookings = _make_bookings(rng, customers, drivers, location_demand)

    save_raw_csv(customers, "customers.csv")
    save_raw_csv(drivers, "drivers.csv")
    save_raw_csv(location_demand, "location_demand.csv")
    save_raw_csv(time_features, "time_features.csv")
    save_raw_csv(bookings, "bookings.csv")

    print(f"\nGenerated {len(bookings):,} bookings, {len(customers)} customers, "
          f"{len(drivers)} drivers, {len(location_demand)} location pairs.")
    print("Ride status split:")
    print(bookings["ride_status"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    generate_all()
