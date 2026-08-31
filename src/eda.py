"""Exploratory analysis over data/processed/final_merged_data.csv.

Run after feature_engineering.py:
    python src/eda.py

Produces:
    reports/eda_charts/*.png   - the charts referenced in EDA_REPORT.md
    reports/EDA_REPORT.md       - written summary with the actual numbers

Covers the seven EDA questions from the project spec: ride volume by
hour/weekday/city, a cancellation heatmap across cities, distance-vs-fare
correlation, rating distributions, a customer-vs-driver cancellation
comparison, payment method usage, and traffic/weather vs cancellation.

notebooks/EDA.ipynb predates this script and isn't a real notebook (it's
plain Python text saved with a .ipynb extension, so it doesn't even open in
Jupyter) - this script is the actual EDA deliverable; that file is safe to
delete once this one exists.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import PROCESSED_DIR, REPORTS_DIR, load_csv

CHART_DIR = f"{REPORTS_DIR}/eda_charts"
REPORT_FILE = f"{REPORTS_DIR}/EDA_REPORT.md"

BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
BLUE_CMAP = sns.light_palette(BLUE, as_cmap=True)

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

sns.set_theme(style="white", rc={"axes.grid": False})


def load_data():
    return load_csv("final_merged_data.csv", folder=PROCESSED_DIR)


def chart_ride_volume(df):
    """Ride volume by hour, weekday, and city - three panels in one figure
    since they're the same underlying question (when/where does demand
    concentrate) at three different granularities."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    by_hour = df.groupby("hour").size()
    axes[0].bar(by_hour.index, by_hour.values, color=BLUE)
    axes[0].set_title("Ride Volume by Hour")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Rides")

    by_weekday = df["day_of_week"].value_counts().reindex(WEEKDAY_ORDER)
    axes[1].bar(by_weekday.index, by_weekday.values, color=CATEGORICAL[1])
    axes[1].set_title("Ride Volume by Weekday")
    axes[1].tick_params(axis="x", rotation=45)

    by_city = df["city"].value_counts()
    axes[2].barh(by_city.index[::-1], by_city.values[::-1], color=CATEGORICAL[2])
    axes[2].set_title("Ride Volume by City")

    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/ride_volume.png", dpi=150)
    plt.close(fig)
    return by_hour, by_weekday, by_city


def chart_cancellation_heatmap(df):
    """Cancellation rate across city x time-of-day - the "cancellation
    heatmap across cities" the spec asks for. Rate rather than raw count,
    since raw cancellation counts mostly just track ride volume, which
    isn't the interesting signal here."""
    rate_table = (
        df.assign(is_cancelled=(df["ride_status"] == "Cancelled").astype(int))
        .pivot_table(index="city", columns="time_bucket", values="is_cancelled", aggfunc="mean")
        * 100
    )
    bucket_order = ["Morning", "Late Morning", "Afternoon", "Evening", "Night"]
    rate_table = rate_table[[c for c in bucket_order if c in rate_table.columns]]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(rate_table, annot=True, fmt=".1f", cmap=BLUE_CMAP, ax=ax, cbar_kws={"label": "Cancellation %"})
    ax.set_title("Cancellation Rate (%) by City and Time of Day")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/cancellation_heatmap.png", dpi=150)
    plt.close(fig)
    return rate_table


def chart_distance_vs_fare(df):
    """Distance vs fare scatter, colored by vehicle type, plus the
    correlation coefficient printed in the report text."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (vehicle, group) in enumerate(df.groupby("vehicle_type")):
        ax.scatter(group["distance_km"], group["estimated_fare"], s=10, alpha=0.4,
                   color=CATEGORICAL[i % len(CATEGORICAL)], label=vehicle)
    ax.set_title("Distance vs Estimated Fare")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Estimated fare")
    ax.legend(title="Vehicle Type")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/distance_vs_fare.png", dpi=150)
    plt.close(fig)
    return df["distance_km"].corr(df["estimated_fare"])


def chart_rating_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(df["customer_rating"], bins=15, color=CATEGORICAL[0])
    axes[0].set_title("Customer Rating Distribution")
    axes[1].hist(df["driver_rating"], bins=15, color=CATEGORICAL[1])
    axes[1].set_title("Driver Rating Distribution")
    for ax in axes:
        ax.set_xlabel("Rating")
        ax.set_ylabel("Bookings")
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/rating_distribution.png", dpi=150)
    plt.close(fig)


def chart_customer_vs_driver_cancellation(df):
    """Among rides that were cancelled, who cancelled them - the customer
    or the driver? A direct customer-vs-driver behavior comparison, rather
    than two separate unrelated charts."""
    cancelled = df[df["ride_status"] == "Cancelled"]
    counts = cancelled["cancelled_by"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(counts.index.astype(str), counts.values, color=CATEGORICAL[: len(counts)])
    ax.set_title("Cancelled Rides: Customer vs Driver")
    ax.set_ylabel("Number of cancelled rides")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/customer_vs_driver_cancellation.png", dpi=150)
    plt.close(fig)
    return counts


def chart_payment_methods(df):
    counts = df["payment_method"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(counts.index, counts.values, color=CATEGORICAL[: len(counts)])
    ax.set_title("Payment Method Usage")
    ax.set_ylabel("Number of rides")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/payment_methods.png", dpi=150)
    plt.close(fig)
    return counts


def chart_traffic_weather_vs_cancellation(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    is_cancelled = (df["ride_status"] == "Cancelled").astype(int)
    by_traffic = df.assign(is_cancelled=is_cancelled).groupby("traffic_level")["is_cancelled"].mean() * 100
    by_traffic = by_traffic.reindex(["Low", "Medium", "High"])
    axes[0].bar(by_traffic.index, by_traffic.values, color=CATEGORICAL[3])
    axes[0].set_title("Cancellation Rate by Traffic Level")
    axes[0].set_ylabel("Cancellation %")

    by_weather = df.assign(is_cancelled=is_cancelled).groupby("weather_condition")["is_cancelled"].mean().sort_values(ascending=False) * 100
    axes[1].bar(by_weather.index, by_weather.values, color=CATEGORICAL[4])
    axes[1].set_title("Cancellation Rate by Weather")
    axes[1].set_ylabel("Cancellation %")

    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/traffic_weather_vs_cancellation.png", dpi=150)
    plt.close(fig)
    return by_traffic, by_weather


def build_report(df, by_hour, by_city, distance_fare_corr, cancel_by_who, payment_counts, by_traffic, by_weather):
    total = len(df)
    cancel_rate = (df["ride_status"] == "Cancelled").mean() * 100
    peak_hour = int(by_hour.idxmax())
    busiest_city = by_city.idxmax()

    lines = [
        "# EDA Report - Rapido Intelligent Mobility Insights",
        "",
        f"Dataset: **{total:,}** bookings across {df['city'].nunique()} cities, "
        f"{df['customer_id'].nunique()} customers, {df['driver_id'].nunique()} drivers.",
        f"Overall cancellation rate: **{cancel_rate:.1f}%**.",
        "",
        "## Ride Volume by Hour, Weekday & City",
        "",
        "![Ride volume](eda_charts/ride_volume.png)",
        "",
        f"Peak hour: **{peak_hour}:00**. Busiest city: **{busiest_city}** "
        f"({int(by_city.max()):,} rides).",
        "",
        "## Cancellation Heatmap Across Cities",
        "",
        "![Cancellation heatmap](eda_charts/cancellation_heatmap.png)",
        "",
        "Cancellation rate broken down by city and time-of-day bucket - "
        "useful for spotting whether a city's higher overall cancellation "
        "rate is concentrated in specific hours or spread evenly.",
        "",
        "## Distance vs Fare Correlation",
        "",
        "![Distance vs fare](eda_charts/distance_vs_fare.png)",
        "",
        f"Correlation coefficient: **{distance_fare_corr:.3f}**. Strongly "
        "positive, as expected - fare is built directly from distance in "
        "this dataset, so the relationship is close to linear within each "
        "vehicle type, with the vehicle-type bands visible as roughly "
        "parallel clusters.",
        "",
        "## Rating Distribution",
        "",
        "![Rating distribution](eda_charts/rating_distribution.png)",
        "",
        f"Mean customer rating: **{df['customer_rating'].mean():.2f}**. "
        f"Mean driver rating: **{df['driver_rating'].mean():.2f}**.",
        "",
        "## Customer vs Driver Cancellation Behavior",
        "",
        "![Customer vs driver cancellation](eda_charts/customer_vs_driver_cancellation.png)",
        "",
        f"Of all cancelled rides, customers initiated "
        f"**{cancel_by_who.get('Customer', 0):,}** and drivers initiated "
        f"**{cancel_by_who.get('Driver', 0):,}**.",
        "",
        "## Payment Method Usage",
        "",
        "![Payment methods](eda_charts/payment_methods.png)",
        "",
        f"Most used: **{payment_counts.idxmax()}** "
        f"({100 * payment_counts.max() / total:.1f}% of rides).",
        "",
        "## Traffic & Weather vs Cancellation",
        "",
        "![Traffic and weather vs cancellation](eda_charts/traffic_weather_vs_cancellation.png)",
        "",
        f"Cancellation rate is highest in **{by_traffic.idxmax()}** traffic "
        f"({by_traffic.max():.1f}%) and during **{by_weather.idxmax()}** "
        f"weather ({by_weather.max():.1f}%) - both feed directly into the "
        "cancellation-risk features used by the model in "
        "src/train_model.py.",
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT_FILE}")


def run_eda():
    os.makedirs(CHART_DIR, exist_ok=True)
    df = load_data()

    by_hour, by_weekday, by_city = chart_ride_volume(df)
    chart_cancellation_heatmap(df)
    distance_fare_corr = chart_distance_vs_fare(df)
    chart_rating_distribution(df)
    cancel_by_who = chart_customer_vs_driver_cancellation(df)
    payment_counts = chart_payment_methods(df)
    by_traffic, by_weather = chart_traffic_weather_vs_cancellation(df)

    build_report(df, by_hour, by_city, distance_fare_corr, cancel_by_who, payment_counts, by_traffic, by_weather)
    print(f"Charts saved to {CHART_DIR}/")


if __name__ == "__main__":
    run_eda()
