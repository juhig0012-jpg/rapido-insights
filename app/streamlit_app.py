import streamlit as st
import pandas as pd
import pickle

st.title("🚖 Rapido Mobility Dashboard")

# Load models
cancel_model = pickle.load(open("../models/cancellation_model.pkl", "rb"))
fare_model = pickle.load(open("../models/fare_model.pkl", "rb"))

# User Inputs
distance = st.slider("Distance (km)", 1, 50)
hour = st.slider("Hour of Day", 0, 23)
rush_hour = 1 if hour in range(7,10) or hour in range(17,21) else 0

# Prediction
if st.button("Predict"):

    input_data = [[distance, hour, rush_hour]]

    cancel_pred = cancel_model.predict(input_data)
    fare_pred = fare_model.predict(input_data)

    st.write(f"🚨 Ride Status Prediction: {cancel_pred[0]}")
    st.write(f"💰 Estimated Fare: ₹{fare_pred[0]:.2f}")