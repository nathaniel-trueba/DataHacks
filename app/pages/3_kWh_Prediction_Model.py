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
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "monthly_kwh_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "models" / "monthly_kwh_features.joblib"

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


@st.cache_resource
def load_model_artifacts() -> tuple[object, list[str]]:
    import joblib

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Model artifacts are not available.")
    if MODEL_PATH.stat().st_size < 1_000_000:
        raise FileNotFoundError("Model artifact appears to be a Git LFS pointer, not the full model file.")

    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


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


def build_model_features(
    cluster_df: pd.DataFrame,
    kw_capacity: float,
    month: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    x_pred = pd.DataFrame(
        {
            "kilowatt_value": kw_capacity,
            "latitude": cluster_df["cluster_lat"].values,
            "longitude": cluster_df["cluster_lon"].values,
            "irradiance": cluster_df["irradiance"].values,
            "avg_temp": cluster_df["pred_tavg"].values,
            "days_in_month": cluster_df["days_in_month"].values,
        }
    )
    x_pred["kw_x_irradiance"] = x_pred["kilowatt_value"] * x_pred["irradiance"]
    x_pred["temp_above_25"] = np.maximum(x_pred["avg_temp"] - 25, 0)
    x_pred["month_sin"] = np.sin(2 * np.pi * month.values / 12)
    x_pred["month_cos"] = np.cos(2 * np.pi * month.values / 12)
    return x_pred[feature_names]


def predict_year(
    forecast_df: pd.DataFrame,
    kw_capacity: float,
    year: int,
    latitude: float,
    longitude: float,
    method: str,
    model: object | None = None,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    cluster = nearest_cluster(forecast_df, latitude, longitude)
    cluster_df = forecast_df[
        (forecast_df["cluster_id"] == cluster["cluster_id"]) & (forecast_df["year"] == year)
    ].copy()

    if cluster_df.empty:
        raise ValueError(f"No forecast rows found for {year}.")

    cluster_df["days_in_month"] = cluster_df["time"].dt.days_in_month
    if method == "ML model" and model is not None and feature_names is not None:
        x_pred = build_model_features(cluster_df, kw_capacity, cluster_df["month"], feature_names)
        cluster_df["predicted_monthly_kwh"] = model.predict(x_pred)
    else:
        cluster_df["predicted_monthly_kwh"] = estimate_monthly_kwh(
            kw_capacity=kw_capacity,
            irradiance=cluster_df["irradiance"],
            avg_temp_c=cluster_df["pred_tavg"],
            days_in_month=cluster_df["days_in_month"],
        )
    cluster_df["month_label"] = cluster_df["time"].dt.strftime("%b")
    return cluster_df.sort_values("time"), cluster


def capacity_curve(
    forecast_df: pd.DataFrame,
    selected_year: int,
    latitude: float,
    longitude: float,
    method: str,
    model: object | None,
    feature_names: list[str] | None,
) -> pd.DataFrame:
    capacities = [round(value * 0.5, 1) for value in range(2, 81)]
    rows = []
    for capacity in capacities:
        predicted, _ = predict_year(
            forecast_df,
            capacity,
            selected_year,
            latitude,
            longitude,
            method,
            model,
            feature_names,
        )
        rows.append(
            {
                "System capacity (kW)": capacity,
                "Predicted annual production (kWh)": predicted["predicted_monthly_kwh"].sum(),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Heat Trace | kWh Prediction Model", layout="wide")
apply_heat_trace_theme()

st.title("kWh Prediction Model")
st.caption(
    "This page uses the formula-backed predictor from the model repo to estimate annual solar "
    "production in kWh from system capacity, forecast temperature, location, and irradiance."
)

forecast_df = load_forecast()
available_years = sorted(forecast_df["year"].unique())
model = None
feature_names = None
model_ready = False
model_message = ""

try:
    model, feature_names = load_model_artifacts()
    model_ready = True
except Exception as exc:
    model_message = str(exc)

summary_cols = st.columns(3)
summary_cols[0].metric("Prediction target", "Annual kWh")
summary_cols[1].metric("Forecast clusters", f"{forecast_df['cluster_id'].nunique():,}")
summary_cols[2].metric("Model status", "ML model" if model_ready else "Formula fallback")

if not model_ready:
    st.info(
        "The formula predictor is active because the ML artifact could not be loaded. "
        f"Details: {model_message}"
    )

st.subheader("Solar production playground")

input_col, output_col = st.columns([1, 1])

with input_col:
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
    prediction_method = st.radio(
        "Prediction method",
        ["ML model", "Formula predictor"] if model_ready else ["Formula predictor"],
        horizontal=True,
    )

    with st.expander("Location inputs"):
        latitude = st.number_input("Latitude", value=float(preset_lat), format="%.4f")
        longitude = st.number_input("Longitude", value=float(preset_lon), format="%.4f")

monthly_df, cluster = predict_year(
    forecast_df,
    capacity_kw,
    selected_year,
    latitude,
    longitude,
    prediction_method,
    model,
    feature_names,
)
annual_kwh = monthly_df["predicted_monthly_kwh"].sum()
monthly_average = annual_kwh / 12
per_kw_output = annual_kwh / capacity_kw
best_month = monthly_df.loc[monthly_df["predicted_monthly_kwh"].idxmax()]

with output_col:
    st.metric("Predicted annual production", f"{annual_kwh:,.0f} kWh")
    st.metric("Average monthly production", f"{monthly_average:,.0f} kWh")
    st.metric("Annual output per kW", f"{per_kw_output:,.0f} kWh/kW")
    st.write(
        f"The closest forecast cluster is #{int(cluster['cluster_id'])}, centered near "
        f"{cluster['cluster_lat']:.2f}, {cluster['cluster_lon']:.2f}. "
        f"The strongest predicted month is {best_month['month_label']} at "
        f"{best_month['predicted_monthly_kwh']:,.0f} kWh using the {prediction_method.lower()}."
    )

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

curve_df = capacity_curve(
    forecast_df,
    selected_year,
    latitude,
    longitude,
    prediction_method,
    model,
    feature_names,
)
curve_fig = px.line(
    curve_df,
    x="System capacity (kW)",
    y="Predicted annual production (kWh)",
)
curve_fig.add_scatter(
    x=[capacity_kw],
    y=[annual_kwh],
    mode="markers",
    marker=dict(size=13, color="#F5A623", line=dict(color="#050403", width=1.5)),
    name="Selected capacity",
)
curve_fig.update_traces(line=dict(color="#E96225", width=3), selector=dict(type="scatter", mode="lines"))
curve_fig.update_layout(
    height=390,
    margin=dict(l=10, r=10, t=25, b=10),
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(23,16,14,0.52)",
    font=dict(color="#F6ECDD"),
)
curve_fig.update_xaxes(gridcolor="rgba(246,236,221,0.12)", zerolinecolor="rgba(246,236,221,0.2)")
curve_fig.update_yaxes(gridcolor="rgba(246,236,221,0.12)", zerolinecolor="rgba(246,236,221,0.2)")
st.plotly_chart(curve_fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("Forecast details for the selected cluster"):
    st.dataframe(
        monthly_df[
            ["month_label", "pred_tavg", "irradiance", "days_in_month", "predicted_monthly_kwh"]
        ].rename(
            columns={
                "month_label": "Month",
                "pred_tavg": "Avg temp (C)",
                "irradiance": "Irradiance",
                "days_in_month": "Days",
                "predicted_monthly_kwh": "Predicted kWh",
            }
        ).style.format(
            {
                "Avg temp (C)": "{:.1f}",
                "Irradiance": "{:.2f}",
                "Predicted kWh": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
