"""Loads the four trained models and exposes one predict_* function per
model. Each expects a single-row DataFrame shaped like final_merged_data.csv
minus the target columns.

Run directly for a quick smoke test against a real row:
    python src/predict.py
"""

import joblib
import pandas as pd

from utils import MODELS_DIR, PROCESSED_DIR, load_csv


def _load_model(filename):
    path = f"{MODELS_DIR}/{filename}"
    try:
        return joblib.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find {path}. Run src/train_model.py first."
        ) from exc


def predict_ride_outcome(input_df):
    model = _load_model("ride_outcome_model.pkl")
    label_encoder = _load_model("ride_outcome_label_encoder.pkl")
    pred = model.predict(input_df)
    return label_encoder.inverse_transform(pred)[0]


def predict_fare(input_df):
    model = _load_model("fare_model.pkl")
    pred = model.predict(input_df)
    return round(float(pred[0]), 2)


def predict_customer_cancel_risk(input_df):
    model = _load_model("customer_cancel_model.pkl")
    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]
    return int(pred[0]), round(float(proba), 4)


def predict_driver_delay_risk(input_df):
    model = _load_model("driver_delay_model.pkl")
    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]
    return int(pred[0]), round(float(proba), 4)


if __name__ == "__main__":
    df = load_csv("final_merged_data.csv", folder=PROCESSED_DIR)
    sample = df.iloc[[0]].copy()

    # none of these are legit features for any of the four models
    common_drop = ["ride_status", "ride_outcome_target", "cancelled_by",
                   "customer_cancel_flag", "driver_delay_flag",
                   "booking_id", "customer_id", "driver_id", "booking_time",
                   "customer_name", "driver_name"]
    fare_drop = common_drop + ["estimated_fare"]
    delay_drop = common_drop + ["driver_delay_min"]

    def _prep(drop_cols):
        return sample.drop(columns=[c for c in drop_cols if c in sample.columns])

    print("Ride outcome:", predict_ride_outcome(_prep(common_drop)))
    print("Fare:", predict_fare(_prep(fare_drop)))
    print("Customer cancel risk:", predict_customer_cancel_risk(_prep(common_drop)))
    print("Driver delay risk:", predict_driver_delay_risk(_prep(delay_drop)))
