from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import CHART_METRICS, METRIC_LABELS, load_state_timeseries, state_summary, time_series_chart


st.set_page_config(page_title="Heat Trace | State Explorer", layout="wide")

st.title("State Explorer")
st.caption("Compare energy, solar, emissions, and air-quality trends for a selected state.")

df = load_state_timeseries()
states = df["state"].drop_duplicates().sort_values().tolist()

selector_col, _ = st.columns([1, 2])
with selector_col:
    selected_state = st.selectbox(
        "State",
        states,
        index=states.index("California") if "California" in states else 0,
    )
state_df = df[df["state"] == selected_state].sort_values("date")

latest = state_df.iloc[-1]
cols = st.columns(4)
cols[0].metric("Latest solar added", f"{latest['solar_capacity_added']:,.0f}")
cols[1].metric("Latest CO2", f"{latest['co2_emissions']:,.0f}")
cols[2].metric("Latest AQI", f"{latest['air_quality_index']:.0f}")
cols[3].metric("Clean ratio", f"{latest['clean_ratio']:.3f}")

st.subheader(selected_state)
st.write(state_summary(state_df))

tab_labels = [METRIC_LABELS[metric] for metric in CHART_METRICS]
tabs = st.tabs(tab_labels)
for tab, metric in zip(tabs, CHART_METRICS):
    with tab:
        st.plotly_chart(time_series_chart(state_df, metric), use_container_width=True)

with st.expander("State data"):
    st.dataframe(
        state_df[
            [
                "year",
                "energy_consumption",
                "energy_production",
                "solar_capacity_added",
                "co2_emissions",
                "air_quality_index",
                "clean_ratio",
                "emissions_intensity",
                "solar_growth_rate",
                "impact_gap_flag",
            ]
        ].style.format(
            {
                "energy_consumption": "{:,.0f}",
                "energy_production": "{:,.0f}",
                "solar_capacity_added": "{:,.0f}",
                "co2_emissions": "{:,.0f}",
                "air_quality_index": "{:.0f}",
                "clean_ratio": "{:.3f}",
                "emissions_intensity": "{:.3f}",
                "solar_growth_rate": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
