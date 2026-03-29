import joblib
import pandas as pd

def predict_ride_outcome(input_df):
    model = joblib.load("models/ride_outcome_model.pkl")
    label_encoder = joblib.load("models/ride_outcome_label_encoder.pkl")
    pred = model.predict(input_df)
    return label_encoder.inverse_transform(pred)[0]

def predict_fare(input_df):
    model = joblib.load("models/fare_model.pkl")
    pred = model.predict(input_df)
    return round(float(pred[0]), 2)

def predict_customer_cancel(input_df):
    model = joblib.load("models/customer_cancel_model.pkl")
    pred = model.predict(input_df)
    return int(pred[0])

if __name__ == "__main__":
    sample = pd.DataFrame([{
        "pickup_location": "Connaught Place",
        "drop_location": "Karol Bagh",
        "city": "Delhi",
        "vehicle_type": "Bike",
        "payment_method": "UPI",
        "distance_km": 6.5,
        "trip_duration_min": 22,
        "base_fare": 95,
        "surge_multiplier": 1.2,
        "traffic_level": "High",
        "weather_condition": "Clear",
        "driver_delay_min": 4,
        "hour": 8,
        "day_of_week": "Monday",
        "is_weekend": 0,
        "customer_rating": 4.5,
        "total_completed_rides": 52,
        "total_cancelled_rides": 4,
        "avg_monthly_spend": 1850,
        "preferred_vehicle": "Bike",
        "driver_rating": 4.6,
        "total_completed_rides_driver": 140,
        "total_cancelled_rides_driver": 8,
        "acceptance_rate": 0.94,
        "avg_delay_min": 4,
        "vehicle_type_driver": "Cab",
        "demand_index": 82,
        "avg_eta_min": 6,
        "zone_type": "Commercial",
        "is_peak_hour": 1,
        "time_bucket": "Morning",
        "fare_per_km": 17.54,
        "fare_per_min": 5.18,
        "long_distance_flag": 1,
        "rush_hour_flag": 1,
        "customer_cancellation_rate": 0.07,
        "driver_cancellation_rate": 0.05,
        "driver_reliability_score": 83,
        "customer_loyalty_score": 61
    }])

    print("Ride outcome:", predict_ride_outcome(sample))
    print("Fare:", predict_fare(sample))
    print("Customer cancel:", predict_customer_cancel(sample))