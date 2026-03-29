import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from utils import ensure_dirs, load_csv

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
    return preprocessor

def train_classification(df):
    drop_cols = [
        "booking_id",
        "customer_id",
        "driver_id",
        "booking_time",
        "ride_status",
        "ride_outcome_target",
        "cancelled_by",
        "estimated_fare",
        "customer_cancel_flag",
        "driver_delay_flag",
        "customer_name",
        "driver_name"
    ]

    # drop duplicate weekend columns if present
    for col in ["is_weekend_x", "is_weekend_y"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["ride_outcome_target"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Ride Outcome Model")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/ride_outcome_model.pkl")
    joblib.dump(le, "models/ride_outcome_label_encoder.pkl")

def train_regression(df):
    drop_cols = [
        "booking_id", "customer_id", "driver_id", "booking_time",
        "ride_status", "ride_outcome_target", "cancelled_by",
        "estimated_fare"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["estimated_fare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Fare Model")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
    print("R2:", r2_score(y_test, preds))

    joblib.dump(model, "models/fare_model.pkl")

def train_customer_cancel_model(df):
    drop_cols = [
        "booking_id", "customer_id", "driver_id", "booking_time",
        "ride_status", "ride_outcome_target", "cancelled_by",
        "customer_cancel_flag"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["customer_cancel_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Customer Cancellation Model")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/customer_cancel_model.pkl")

def main():
    ensure_dirs()
    df = load_csv("final_merged_data.csv", folder="data/processed")
    train_classification(df)
    train_regression(df)
    train_customer_cancel_model(df)

if __name__ == "__main__":
    main()