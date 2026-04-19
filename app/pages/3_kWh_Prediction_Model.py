from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "us_10yr_monthly_cluster_forecast_with_irridance.csv"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import (
    HEAT_AMBER,
    HEAT_CONTINUOUS_SCALE,
    HEAT_MUTED,
    HEAT_SURFACE,
    HEAT_TEXT,
    apply_heat_trace_theme,
)


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


@st.cache_data
def build_prediction_grid(forecast_df: pd.DataFrame, kw_capacity: float, year: int) -> pd.DataFrame:
    year_df = forecast_df[forecast_df["year"] == year].copy()
    year_df["days_in_month"] = year_df["time"].dt.days_in_month
    year_df["predicted_monthly_kwh"] = estimate_monthly_kwh(
        kw_capacity=kw_capacity,
        irradiance=year_df["irradiance"],
        avg_temp_c=year_df["pred_tavg"],
        days_in_month=year_df["days_in_month"],
    )

    return (
        year_df.groupby(["cluster_id", "cluster_lat", "cluster_lon"], as_index=False)
        .agg(
            predicted_annual_kwh=("predicted_monthly_kwh", "sum"),
            avg_temp_c=("pred_tavg", "mean"),
            avg_irradiance=("irradiance", "mean"),
        )
        .sort_values("predicted_annual_kwh", ascending=False)
    )


@st.cache_data
def build_grid_geojson(grid_df: pd.DataFrame, half_size: float = 0.55) -> dict:
    features = []
    for row in grid_df.itertuples(index=False):
        lat = float(row.cluster_lat)
        lon = float(row.cluster_lon)
        cluster_id = str(int(row.cluster_id))
        features.append(
            {
                "type": "Feature",
                "id": cluster_id,
                "properties": {"cluster_id": cluster_id},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lon - half_size, lat - half_size],
                            [lon + half_size, lat - half_size],
                            [lon + half_size, lat + half_size],
                            [lon - half_size, lat + half_size],
                            [lon - half_size, lat - half_size],
                        ]
                    ],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def render_prediction_grid_map(
    grid_df: pd.DataFrame,
    selected_cluster: pd.Series,
    selected_location: str,
) -> go.Figure:
    fig = go.Figure()
    grid_geojson = build_grid_geojson(grid_df)

    fig.add_trace(
        go.Choropleth(
            geojson=grid_geojson,
            locations=grid_df["cluster_id"].astype(str),
            z=grid_df["predicted_annual_kwh"],
            featureidkey="id",
            colorscale=HEAT_CONTINUOUS_SCALE,
            marker_line_color="rgba(246, 236, 221, 0.18)",
            marker_line_width=0.35,
            colorbar=dict(
                title=dict(text="Predicted kWh", font=dict(color=HEAT_MUTED)),
                tickfont=dict(color=HEAT_MUTED),
                bgcolor="rgba(0,0,0,0)",
                outlinecolor="rgba(246, 236, 221, 0.18)",
            ),
            customdata=np.stack(
                [
                    grid_df["cluster_lat"],
                    grid_df["cluster_lon"],
                    grid_df["avg_temp_c"],
                    grid_df["avg_irradiance"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Forecast grid %{location}<br>"
                "Predicted annual production: %{z:,.0f} kWh<br>"
                "Lat/Lon: %{customdata[0]:.2f}, %{customdata[1]:.2f}<br>"
                "Avg temp: %{customdata[2]:.1f} C<br>"
                "Avg irradiance: %{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[selected_cluster["cluster_lat"]],
            lon=[selected_cluster["cluster_lon"]],
            mode="markers+text",
            text=["Selected area"],
            textposition="top center",
            marker=dict(
                size=13,
                color=HEAT_AMBER,
                line=dict(color=HEAT_TEXT, width=2),
            ),
            hovertemplate=(
                f"{selected_location}<br>"
                f"Forecast region #{int(selected_cluster['cluster_id'])}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        bgcolor="rgba(0,0,0,0)",
        showland=True,
        landcolor=HEAT_SURFACE,
        showlakes=False,
        showocean=False,
        showcoastlines=False,
        showcountries=False,
        showsubunits=True,
        subunitcolor="rgba(246, 236, 221, 0.16)",
        showframe=False,
    )
    fig.update_layout(
        height=610,
        dragmode=False,
        margin=dict(l=0, r=0, t=12, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=HEAT_TEXT),
    )
    return fig


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
grid_df = build_prediction_grid(forecast_df, capacity_kw, selected_year)
annual_kwh = monthly_df["predicted_monthly_kwh"].sum()
monthly_average = annual_kwh / 12
per_kw_output = annual_kwh / capacity_kw
best_month = monthly_df.loc[monthly_df["predicted_monthly_kwh"].idxmax()]
lowest_month = monthly_df.loc[monthly_df["predicted_monthly_kwh"].idxmin()]
best_grid_cell = grid_df.iloc[0]

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

st.subheader("Predicted production grid")
st.caption(
    "Each colored square represents one forecast region from the latitude/longitude dataset. "
    "Warmer cells predict more annual kWh for the selected system size and year; darker cells "
    "predict less. The marker shows the forecast region used for the selected location."
)

grid_fig = render_prediction_grid_map(grid_df, cluster, selected_location)
st.plotly_chart(
    grid_fig,
    use_container_width=True,
    config={"displayModeBar": False, "scrollZoom": False, "staticPlot": False},
)

grid_cols = st.columns(3)
grid_cols[0].metric("Forecast grid cells", f"{len(grid_df):,}")
grid_cols[1].metric("Highest predicted cell", f"{best_grid_cell['predicted_annual_kwh']:,.0f} kWh")
grid_cols[2].metric("Selected cell", f"{annual_kwh:,.0f} kWh")

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

with st.expander("Seasonal pattern for the selected location"):
    st.plotly_chart(monthly_fig, use_container_width=True, config={"displayModeBar": False})

st.info(
    f"Rule of thumb for this setup: each 1 kW of solar capacity produces about "
    f"{per_kw_output:,.0f} kWh per year in this forecast location."
)
