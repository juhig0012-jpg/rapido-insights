import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Rapido Mobility Insights", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/final_merged_data.csv")

@st.cache_resource
def load_models():
    ride_model = joblib.load("models/ride_outcome_model.pkl")
    ride_le = joblib.load("models/ride_outcome_label_encoder.pkl")
    fare_model = joblib.load("models/fare_model.pkl")
    cancel_model = joblib.load("models/customer_cancel_model.pkl")
    return ride_model, ride_le, fare_model, cancel_model

st.title("Rapido: Intelligent Mobility Insights")

page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "EDA", "Ride Outcome Prediction", "Fare Prediction", "Customer Risk"]
)

df = load_data()
ride_model, ride_le, fare_model, cancel_model = load_models()

if page == "Dashboard":
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rides", len(df))
    col2.metric("Completed Rides", int((df["ride_status"] == "Completed").sum()))
    col3.metric("Cancellation Rate", f"{(df['ride_status'].eq('Cancelled').mean() * 100):.2f}%")

    st.subheader("Ride Volume by Hour")
    ride_by_hour = df.groupby("hour").size()
    st.line_chart(ride_by_hour)

    st.subheader("Ride Status Distribution")
    st.bar_chart(df["ride_status"].value_counts())

elif page == "EDA":
    st.subheader("Explore Data")
    city = st.selectbox("Select city", sorted(df["city"].dropna().unique()))
    filtered = df[df["city"] == city]

    st.write("Filtered rows:", len(filtered))
    st.bar_chart(filtered.groupby("hour").size())
    st.bar_chart(filtered["vehicle_type"].value_counts())
    st.bar_chart(filtered["payment_method"].value_counts())

elif page == "Ride Outcome Prediction":
    st.subheader("Predict Ride Outcome")

    input_data = {
        "pickup_location": st.selectbox("Pickup", sorted(df["pickup_location"].dropna().unique())),
        "drop_location": st.selectbox("Drop", sorted(df["drop_location"].dropna().unique())),
        "city": st.selectbox("City", sorted(df["city"].dropna().unique())),
        "vehicle_type": st.selectbox("Vehicle Type", sorted(df["vehicle_type"].dropna().unique())),
        "payment_method": st.selectbox("Payment Method", sorted(df["payment_method"].dropna().unique())),
        "distance_km": st.number_input("Distance (km)", 0.1, 50.0, 5.0),
        "trip_duration_min": st.number_input("Trip Duration (min)", 1, 180, 20),
        "base_fare": st.number_input("Base Fare", 20, 1000, 100),
        "surge_multiplier": st.number_input("Surge", 1.0, 3.0, 1.2),
        "traffic_level": st.selectbox("Traffic", sorted(df["traffic_level"].dropna().unique())),
        "weather_condition": st.selectbox("Weather", sorted(df["weather_condition"].dropna().unique())),
        "driver_delay_min": st.number_input("Driver Delay", 0, 60, 5),
        "hour": st.slider("Hour", 0, 23, 9),
        "day_of_week": st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
        "is_weekend": st.selectbox("Weekend", [0,1]),
        "customer_rating": st.number_input("Customer Rating", 1.0, 5.0, 4.2),
        "total_completed_rides": st.number_input("Customer Completed Rides", 0, 500, 30),
        "total_cancelled_rides": st.number_input("Customer Cancelled Rides", 0, 100, 4),
        "avg_monthly_spend": st.number_input("Avg Monthly Spend", 0, 10000, 1500),
        "preferred_vehicle": st.selectbox("Preferred Vehicle", ["Bike","Auto","Cab"]),
        "driver_rating": st.number_input("Driver Rating", 1.0, 5.0, 4.3),
        "total_completed_rides_driver": st.number_input("Driver Completed Rides", 0, 1000, 100),
        "total_cancelled_rides_driver": st.number_input("Driver Cancelled Rides", 0, 200, 8),
        "acceptance_rate": st.number_input("Acceptance Rate", 0.0, 1.0, 0.90),
        "avg_delay_min": st.number_input("Driver Avg Delay", 0, 60, 5),
        "vehicle_type_driver": st.selectbox("Driver Vehicle Type", ["Bike","Auto","Cab"]),
        "demand_index": st.number_input("Demand Index", 0, 100, 70),
        "avg_eta_min": st.number_input("Avg ETA", 1, 60, 6),
        "zone_type": st.selectbox("Zone Type", ["Residential","Commercial","Tourist","Transit"]),
        "is_peak_hour": st.selectbox("Is Peak Hour", [0,1]),
        "time_bucket": st.selectbox("Time Bucket", ["Morning","Late Morning","Afternoon","Evening","Night"])
    }

    estimated_fare = input_data["base_fare"] * input_data["surge_multiplier"]
    input_data["fare_per_km"] = estimated_fare / max(input_data["distance_km"], 0.1)
    input_data["fare_per_min"] = estimated_fare / max(input_data["trip_duration_min"], 1)
    input_data["long_distance_flag"] = int(input_data["distance_km"] >= 5)
    input_data["rush_hour_flag"] = input_data["is_peak_hour"]
    input_data["customer_cancellation_rate"] = input_data["total_cancelled_rides"] / max(
        input_data["total_completed_rides"] + input_data["total_cancelled_rides"], 1
    )
    input_data["driver_cancellation_rate"] = input_data["total_cancelled_rides_driver"] / max(
        input_data["total_completed_rides_driver"] + input_data["total_cancelled_rides_driver"], 1
    )
    input_data["driver_reliability_score"] = (
        (input_data["driver_rating"] * 20) * 0.4 +
        (input_data["acceptance_rate"] * 100) * 0.4 +
        ((1 / (1 + input_data["avg_delay_min"])) * 100) * 0.2
    )
    input_data["customer_loyalty_score"] = (
        (input_data["customer_rating"] * 20) * 0.4 +
        input_data["total_completed_rides"] * 0.4 +
        (1 - input_data["customer_cancellation_rate"]) * 100 * 0.2
    )

    if st.button("Predict Ride Outcome"):
        input_df = pd.DataFrame([input_data])
        pred = ride_model.predict(input_df)
        label = ride_le.inverse_transform(pred)[0]
        st.success(f"Predicted Ride Outcome: {label}")

elif page == "Fare Prediction":
    st.subheader("Predict Fare")

    distance_km = st.number_input("Distance (km)", 0.1, 50.0, 6.0)
    trip_duration_min = st.number_input("Trip Duration (min)", 1, 180, 20)
    base_fare = st.number_input("Base Fare", 20, 1000, 100)
    surge_multiplier = st.number_input("Surge Multiplier", 1.0, 3.0, 1.2)

    sample = df.iloc[[0]].copy()
    sample["distance_km"] = distance_km
    sample["trip_duration_min"] = trip_duration_min
    sample["base_fare"] = base_fare
    sample["surge_multiplier"] = surge_multiplier
    sample["fare_per_km"] = (base_fare * surge_multiplier) / max(distance_km, 0.1)
    sample["fare_per_min"] = (base_fare * surge_multiplier) / max(trip_duration_min, 1)

    drop_cols = ["estimated_fare", "ride_status", "ride_outcome_target", "cancelled_by", "booking_id", "customer_id", "driver_id", "booking_time"]
    sample = sample.drop(columns=[c for c in drop_cols if c in sample.columns])

    if st.button("Predict Fare"):
        pred = fare_model.predict(sample)[0]
        st.success(f"Estimated Fare: Rs {pred:.2f}")

elif page == "Customer Risk":
    st.subheader("Customer Cancellation Risk")

    sample = df.iloc[[0]].copy()
    sample["total_cancelled_rides"] = st.number_input("Cancelled Rides", 0, 100, 5)
    sample["total_completed_rides"] = st.number_input("Completed Rides", 0, 500, 30)
    sample["customer_rating"] = st.number_input("Customer Rating", 1.0, 5.0, 4.0)

    sample["customer_cancellation_rate"] = sample["total_cancelled_rides"] / (
        sample["total_completed_rides"] + sample["total_cancelled_rides"] + 1
    )

    drop_cols = ["customer_cancel_flag", "ride_status", "ride_outcome_target", "cancelled_by", "booking_id", "customer_id", "driver_id", "booking_time"]
    sample = sample.drop(columns=[c for c in drop_cols if c in sample.columns])

    if st.button("Predict Customer Risk"):
        pred = cancel_model.predict(sample)[0]
        if pred == 1:
            st.warning("High chance of customer cancellation")
        else:
            st.success("Low chance of customer cancellation")