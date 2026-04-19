from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "us_10yr_monthly_cluster_forecast_with_irridance.csv"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import apply_heat_trace_theme


LOCATION_PRESETS = {
    "San Diego, CA": (32.72, -117.16),
    "Phoenix, AZ": (33.45, -112.07),
    "Austin, TX": (30.27, -97.74),
    "Portland, OR": (45.52, -122.68),
    "Newark, NJ": (40.74, -74.17),
}


@st.cache_data
def load_forecast() -> pd.DataFrame:
    df = pd.read_csv(FORECAST_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    return df


def nearest_cluster(forecast_df: pd.DataFrame, latitude: float, longitude: float) -> pd.Series:
    clusters = forecast_df[["cluster_id", "cluster_lat", "cluster_lon"]].drop_duplicates().copy()
    clusters["distance"] = np.sqrt(
        (clusters["cluster_lat"] - latitude) ** 2 + (clusters["cluster_lon"] - longitude) ** 2
    )
    return clusters.loc[clusters["distance"].idxmin()]


def estimate_monthly_kwh(
    kw_capacity: float,
    irradiance: pd.Series,
    avg_temp_c: pd.Series,
    days_in_month: pd.Series,
    performance_ratio: float = 0.75,
    temp_coeff: float = 0.004,
    min_temp_loss: float = 0.80,
) -> pd.Series:
    temp_loss = 1 - np.maximum(avg_temp_c - 25, 0) * temp_coeff
    temp_loss = np.maximum(temp_loss, min_temp_loss)
    return kw_capacity * irradiance * days_in_month * performance_ratio * temp_loss


def predict_year(
    forecast_df: pd.DataFrame,
    kw_capacity: float,
    year: int,
    latitude: float,
    longitude: float,
) -> tuple[pd.DataFrame, pd.Series]:
    cluster = nearest_cluster(forecast_df, latitude, longitude)
    cluster_df = forecast_df[
        (forecast_df["cluster_id"] == cluster["cluster_id"]) & (forecast_df["year"] == year)
    ].copy()

    if cluster_df.empty:
        raise ValueError(f"No forecast rows found for {year}.")

    cluster_df["days_in_month"] = cluster_df["time"].dt.days_in_month
    cluster_df["predicted_monthly_kwh"] = estimate_monthly_kwh(
        kw_capacity=kw_capacity,
        irradiance=cluster_df["irradiance"],
        avg_temp_c=cluster_df["pred_tavg"],
        days_in_month=cluster_df["days_in_month"],
    )
    cluster_df["month_label"] = cluster_df["time"].dt.strftime("%b")
    return cluster_df.sort_values("time"), cluster


st.set_page_config(page_title="Heat Trace | kWh Prediction Model", layout="wide")
apply_heat_trace_theme()

st.title("kWh Prediction Model")
st.caption(
    "This page uses the formula-backed predictor from the model repo to estimate annual solar "
    "production in kWh from system capacity, forecast temperature, location, and irradiance. "
    "In plain terms: choose how large the solar system is, pick a place and year, and Heat Trace "
    "estimates how much electricity that system could generate over the year. kW describes the size "
    "of the solar system; kWh describes the amount of electricity it produces."
)

forecast_df = load_forecast()
available_years = sorted(forecast_df["year"].unique())

summary_cols = st.columns(3)
summary_cols[0].metric("Input", "System size")
summary_cols[1].metric("Output", "Annual kWh")
summary_cols[2].metric("Forecast years", f"{min(available_years)}-{max(available_years)}")

st.subheader("Try a solar system size")

input_col, output_col = st.columns([1, 1])

with input_col:
    st.write("Adjust the values below to see how the yearly production estimate changes.")
    selected_location = st.selectbox("Location preset", list(LOCATION_PRESETS.keys()))
    preset_lat, preset_lon = LOCATION_PRESETS[selected_location]

    capacity_kw = st.slider(
        "System capacity (kW)",
        min_value=1.0,
        max_value=40.0,
        value=8.0,
        step=0.5,
    )
    selected_year = st.selectbox("Prediction year", available_years, index=0)

    with st.expander("Fine tune location"):
        latitude = st.number_input("Latitude", value=float(preset_lat), format="%.4f")
        longitude = st.number_input("Longitude", value=float(preset_lon), format="%.4f")

monthly_df, cluster = predict_year(forecast_df, capacity_kw, selected_year, latitude, longitude)
annual_kwh = monthly_df["predicted_monthly_kwh"].sum()
monthly_average = annual_kwh / 12
per_kw_output = annual_kwh / capacity_kw
best_month = monthly_df.loc[monthly_df["predicted_monthly_kwh"].idxmax()]
lowest_month = monthly_df.loc[monthly_df["predicted_monthly_kwh"].idxmin()]

with output_col:
    st.metric("Predicted annual production", f"{annual_kwh:,.0f} kWh")
    st.metric("Average monthly production", f"{monthly_average:,.0f} kWh")
    st.metric("Best month", f"{best_month['month_label']} ({best_month['predicted_monthly_kwh']:,.0f} kWh)")
    st.write(
        f"A {capacity_kw:g} kW system near {selected_location} is estimated to produce "
        f"{annual_kwh:,.0f} kWh in {selected_year}. Production is strongest in "
        f"{best_month['month_label']} and weakest in {lowest_month['month_label']}, which reflects "
        "seasonal changes in sunlight and weather."
    )
    st.caption(f"Behind the scenes, this location maps to forecast region #{int(cluster['cluster_id'])}.")

monthly_fig = px.bar(
    monthly_df,
    x="month_label",
    y="predicted_monthly_kwh",
    labels={"month_label": "Month", "predicted_monthly_kwh": "Predicted monthly production (kWh)"},
    title=f"Monthly production forecast for {selected_year}",
)
monthly_fig.update_traces(marker_color="#F57C30", marker_line_color="#F5A623", marker_line_width=1)
monthly_fig.update_layout(
    height=360,
    margin=dict(l=10, r=10, t=45, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(23,16,14,0.52)",
    font=dict(color="#F6ECDD"),
)
monthly_fig.update_xaxes(gridcolor="rgba(246,236,221,0.12)", categoryorder="array", categoryarray=monthly_df["month_label"])
monthly_fig.update_yaxes(gridcolor="rgba(246,236,221,0.12)", zerolinecolor="rgba(246,236,221,0.2)")
st.plotly_chart(monthly_fig, use_container_width=True, config={"displayModeBar": False})

st.info(
    f"Rule of thumb for this setup: each 1 kW of solar capacity produces about "
    f"{per_kw_output:,.0f} kWh per year in this forecast location."
)
