from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import apply_heat_trace_theme


LOCATION_PRESETS = {
    "San Diego, CA": {
        "latitude": 32.72,
        "longitude": -117.16,
        "avg_temperature_f": 65.0,
        "sunlight_hours": 5.7,
    },
    "Phoenix, AZ": {
        "latitude": 33.45,
        "longitude": -112.07,
        "avg_temperature_f": 75.0,
        "sunlight_hours": 6.6,
    },
    "Austin, TX": {
        "latitude": 30.27,
        "longitude": -97.74,
        "avg_temperature_f": 69.0,
        "sunlight_hours": 5.3,
    },
    "Portland, OR": {
        "latitude": 45.52,
        "longitude": -122.68,
        "avg_temperature_f": 54.0,
        "sunlight_hours": 3.8,
    },
}


def baseline_kwh_prediction(capacity_kw: float, sunlight_hours: float, avg_temperature_f: float) -> float:
    """Temporary solar production estimate until the trained model is connected."""
    performance_ratio = 0.82
    temp_penalty = max(0.88, 1 - max(avg_temperature_f - 77, 0) * 0.003)
    return capacity_kw * sunlight_hours * 365 * performance_ratio * temp_penalty


def capacity_curve(sunlight_hours: float, avg_temperature_f: float) -> pd.DataFrame:
    capacities = [round(value * 0.5, 1) for value in range(2, 81)]
    return pd.DataFrame(
        {
            "System capacity (kW)": capacities,
            "Predicted annual production (kWh)": [
                baseline_kwh_prediction(capacity, sunlight_hours, avg_temperature_f)
                for capacity in capacities
            ],
        }
    )


st.set_page_config(page_title="Heat Trace | kWh Prediction Model", layout="wide")
apply_heat_trace_theme()

st.title("kWh Prediction Model")
st.caption(
    "Interactive template for a future Heat Trace model that predicts annual solar production in kWh "
    "from system capacity, weather, location, and sunlight."
)

st.info(
    "This page currently uses a transparent baseline estimate so the demo is interactive. "
    "Later, this calculation can be replaced with the trained machine learning model."
)

summary_cols = st.columns(3)
summary_cols[0].metric("Prediction target", "Annual kWh")
summary_cols[1].metric("Primary input", "kW capacity")
summary_cols[2].metric("Model status", "Template")

st.subheader("Solar production playground")

input_col, output_col = st.columns([1, 1])

with input_col:
    selected_location = st.selectbox("Location preset", list(LOCATION_PRESETS.keys()))
    preset = LOCATION_PRESETS[selected_location]

    capacity_kw = st.slider(
        "System capacity (kW)",
        min_value=1.0,
        max_value=40.0,
        value=8.0,
        step=0.5,
    )

    with st.expander("Weather and sunlight assumptions"):
        latitude = st.number_input("Latitude", value=preset["latitude"], format="%.4f")
        longitude = st.number_input("Longitude", value=preset["longitude"], format="%.4f")
        avg_temperature_f = st.slider(
            "Average temperature (F)",
            min_value=35.0,
            max_value=95.0,
            value=float(preset["avg_temperature_f"]),
            step=1.0,
        )
        sunlight_hours = st.slider(
            "Average daily sunlight / irradiance proxy (hours)",
            min_value=2.0,
            max_value=8.0,
            value=float(preset["sunlight_hours"]),
            step=0.1,
        )

prediction_kwh = baseline_kwh_prediction(capacity_kw, sunlight_hours, avg_temperature_f)
monthly_average = prediction_kwh / 12
per_kw_output = prediction_kwh / capacity_kw

with output_col:
    st.metric("Predicted annual production", f"{prediction_kwh:,.0f} kWh")
    st.metric("Average monthly production", f"{monthly_average:,.0f} kWh")
    st.metric("Annual output per kW", f"{per_kw_output:,.0f} kWh/kW")
    st.write(
        f"For a {capacity_kw:g} kW system near {selected_location}, the template estimate is "
        f"{prediction_kwh:,.0f} kWh per year using {sunlight_hours:.1f} average daily sunlight hours."
    )

curve_df = capacity_curve(sunlight_hours, avg_temperature_f)
fig = px.line(
    curve_df,
    x="System capacity (kW)",
    y="Predicted annual production (kWh)",
    markers=False,
)
fig.add_scatter(
    x=[capacity_kw],
    y=[prediction_kwh],
    mode="markers",
    marker=dict(size=13, color="#F5A623", line=dict(color="#050403", width=1.5)),
    name="Selected capacity",
)
fig.update_traces(line=dict(color="#E96225", width=3), selector=dict(type="scatter", mode="lines"))
fig.update_layout(
    height=390,
    margin=dict(l=10, r=10, t=20, b=10),
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(23,16,14,0.52)",
    font=dict(color="#F6ECDD"),
)
fig.update_xaxes(gridcolor="rgba(246,236,221,0.12)", zerolinecolor="rgba(246,236,221,0.2)")
fig.update_yaxes(gridcolor="rgba(246,236,221,0.12)", zerolinecolor="rgba(246,236,221,0.2)")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("Model inputs planned for the full version"):
    st.write(
        "The production model is expected to use kW capacity, latitude, longitude, weather-derived "
        "average temperature, and sunlight or irradiance features. The user-facing workflow will keep "
        "capacity easy to adjust while location and environmental features can be filled from data APIs."
    )
