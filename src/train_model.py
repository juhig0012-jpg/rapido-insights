"""Trains the four models the project spec asks for and writes a persisted
evaluation report so the accuracy/RMSE numbers can actually be checked
against the target benchmarks later, instead of only ever appearing once
in a terminal that's already scrolled past.

Run after feature_engineering.py:
    python src/train_model.py

Produces:
    models/*.pkl                    - one file per trained model/encoder
    reports/model_metrics.json      - machine-readable metrics per model
    reports/MODEL_EVALUATION.md     - the same metrics, written up
"""

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from utils import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, load_csv

# Columns that either identify a row (no predictive value) or leak the
# answer to one of the four targets - excluded from every model's inputs,
# then each training function excludes a couple more that are specific to
# its own target.
ALWAYS_DROP = [
    "booking_id", "customer_id", "driver_id", "booking_time",
    "customer_name", "driver_name", "is_weekend_x", "is_weekend_y",
]

TARGET_COLUMNS = [
    "ride_status", "ride_outcome_target", "cancelled_by",
    "customer_cancel_flag", "driver_delay_flag",
]

CLASSIFICATION_ACCURACY_BENCHMARK = 0.85
REGRESSION_RMSE_PCT_BENCHMARK = 0.10  # RMSE must be within +/-10% of mean fare


def build_preprocessor(X):
    """One-hot encodes categorical columns, median-imputes and scales
    numeric ones. Shared by every model so a change here (say, a different
    imputation strategy) applies consistently across all four."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])


def split_xy(df, target_col, extra_drop):
    """Builds X/y for one model: drop everything in ALWAYS_DROP plus this
    model's own extra leak columns, and set aside the target column."""
    drop_cols = set(ALWAYS_DROP) | set(extra_drop) | {target_col}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    return X, y


def tune_classifier(X_train, y_train):
    """Small GridSearchCV sweep over the two hyperparameters that matter
    most for a RandomForest on a dataset this size - kept small (4
    combinations x 3 folds) so tuning finishes in seconds rather than
    minutes."""
    # class_weight="balanced" matters here - cancellations and delays are
    # both minority outcomes (roughly 1-in-10), and without it the model
    # just learns to always predict the majority class and still scores a
    # deceptively high accuracy while missing every positive case
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    param_grid = {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [None, 12],
    }
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_regressor(X_train, y_train):
    """Same idea for the fare model, but with XGBoost instead of
    RandomForest - fare is a smooth numeric target with real linear
    structure (distance/duration/surge all combine roughly additively into
    price), which gradient boosting tends to fit a bit more precisely than
    bagged trees."""
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("regressor", XGBRegressor(random_state=42, n_jobs=-1)),
    ])
    param_grid = {
        "regressor__n_estimators": [200, 400],
        "regressor__max_depth": [4, 6],
        "regressor__learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_classifier(name, model, X_test, y_test, class_names):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    proba = model.predict_proba(X_test)
    try:
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class="ovr")
    except ValueError:
        # AUC is undefined if the test split ends up missing a class -
        # shouldn't happen with stratified splits, but better to report
        # "not available" than crash the whole training run over it
        auc = None

    report = classification_report(
        y_test, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, preds).tolist()

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}" + (f" | AUC: {auc:.4f}" if auc is not None else ""))
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    return {
        "model": name,
        "task": "classification",
        "accuracy": round(float(accuracy), 4),
        "auc": round(float(auc), 4) if auc is not None else None,
        "confusion_matrix": matrix,
        "class_names": list(class_names),
        "classification_report": report,
        "meets_benchmark": bool(accuracy >= CLASSIFICATION_ACCURACY_BENCHMARK),
        "benchmark": f">= {CLASSIFICATION_ACCURACY_BENCHMARK:.0%} accuracy",
    }


def evaluate_regressor(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    rmse_pct = rmse / y_test.mean()

    print(f"\n{name}")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} ({rmse_pct:.1%} of mean fare) | R2: {r2:.4f}")

    return {
        "model": name,
        "task": "regression",
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "rmse_pct_of_mean": round(float(rmse_pct), 4),
        "r2": round(float(r2), 4),
        "meets_benchmark": bool(rmse_pct <= REGRESSION_RMSE_PCT_BENCHMARK),
        "benchmark": f"RMSE within {REGRESSION_RMSE_PCT_BENCHMARK:.0%} of mean fare",
    }


def train_ride_outcome_model(df):
    # ride_status and cancelled_by are the same information as the target
    # under a different name (ride_outcome_target is a direct copy of
    # ride_status) - leaving either in X lets the model "predict" by just
    # reading its own answer back, which is how an earlier version of this
    # function ended up with a suspicious 100% test accuracy
    X, y = split_xy(
        df, "ride_outcome_target",
        extra_drop=["ride_status", "cancelled_by", "customer_cancel_flag", "driver_delay_flag"],
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model, best_params = tune_classifier(X_train, y_train)
    metrics = evaluate_classifier("Ride Outcome Model", model, X_test, y_test, label_encoder.classes_)
    metrics["best_params"] = best_params

    joblib.dump(model, f"{MODELS_DIR}/ride_outcome_model.pkl")
    joblib.dump(label_encoder, f"{MODELS_DIR}/ride_outcome_label_encoder.pkl")
    return metrics


def train_fare_model(df):
    X, y = split_xy(
        df, "estimated_fare",
        extra_drop=["ride_status", "ride_outcome_target", "cancelled_by",
                    "customer_cancel_flag", "driver_delay_flag"],
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, best_params = tune_regressor(X_train, y_train)
    metrics = evaluate_regressor("Fare Prediction Model", model, X_test, y_test)
    metrics["best_params"] = best_params

    joblib.dump(model, f"{MODELS_DIR}/fare_model.pkl")
    return metrics


def train_customer_cancel_model(df):
    X, y = split_xy(
        df, "customer_cancel_flag",
        extra_drop=["ride_status", "ride_outcome_target", "cancelled_by", "driver_delay_flag"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, best_params = tune_classifier(X_train, y_train)
    metrics = evaluate_classifier(
        "Customer Cancellation Risk Model", model, X_test, y_test, ["No Cancel", "Cancel"]
    )
    metrics["best_params"] = best_params

    joblib.dump(model, f"{MODELS_DIR}/customer_cancel_model.pkl")
    return metrics


def train_driver_delay_model(df):
    """The model the spec asked for that was never actually built - predicts
    whether a driver is likely to cause a delay (driver_delay_min >= 10,
    computed in feature_engineering.py) using their historical
    reliability/acceptance behavior and this trip's traffic exposure.
    driver_delay_min itself is excluded from the inputs since it's exactly
    what the target is thresholded from - leaving it in would let the model
    "predict" its own label."""
    X, y = split_xy(
        df, "driver_delay_flag",
        extra_drop=["ride_status", "ride_outcome_target", "cancelled_by",
                    "customer_cancel_flag", "driver_delay_min"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, best_params = tune_classifier(X_train, y_train)
    metrics = evaluate_classifier(
        "Driver Delay Prediction Model", model, X_test, y_test, ["On Time", "Delayed"]
    )
    metrics["best_params"] = best_params

    joblib.dump(model, f"{MODELS_DIR}/driver_delay_model.pkl")
    return metrics


def write_metrics_report(all_metrics):
    """Saves the metrics both as JSON (for the Streamlit dashboard or any
    future script to read back) and as a short Markdown write-up (for a
    human skimming the repo)."""
    with open(f"{REPORTS_DIR}/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    lines = ["# Model Evaluation Report", ""]
    for metrics in all_metrics:
        lines.append(f"## {metrics['model']}")
        lines.append("")
        if metrics["task"] == "classification":
            auc_text = f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "n/a"
            lines.append(f"- Accuracy: **{metrics['accuracy']:.2%}**")
            lines.append(f"- AUC: **{auc_text}**")
            lines.append(f"- Best params: `{metrics['best_params']}`")
            lines.append(f"- Confusion matrix ({', '.join(metrics['class_names'])}): `{metrics['confusion_matrix']}`")
        else:
            lines.append(f"- MAE: **{metrics['mae']:.2f}**")
            lines.append(f"- RMSE: **{metrics['rmse']:.2f}** ({metrics['rmse_pct_of_mean']:.1%} of mean fare)")
            lines.append(f"- R²: **{metrics['r2']:.4f}**")
            lines.append(f"- Best params: `{metrics['best_params']}`")
        status = "MEETS" if metrics["meets_benchmark"] else "BELOW"
        lines.append(f"- Benchmark ({metrics['benchmark']}): **{status}**")
        lines.append("")

    with open(f"{REPORTS_DIR}/MODEL_EVALUATION.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {REPORTS_DIR}/model_metrics.json and {REPORTS_DIR}/MODEL_EVALUATION.md")


def main():
    ensure_dirs()
    df = load_csv("final_merged_data.csv", folder=PROCESSED_DIR)

    all_metrics = [
        train_ride_outcome_model(df),
        train_fare_model(df),
        train_customer_cancel_model(df),
        train_driver_delay_model(df),
    ]
    write_metrics_report(all_metrics)


if __name__ == "__main__":
    main()
