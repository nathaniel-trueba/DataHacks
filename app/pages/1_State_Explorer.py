from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import (
    CHART_METRICS,
    METRIC_LABELS,
    apply_light_mode_background,
    load_homepage_map_data,
    load_state_timeseries,
    solar_production_comparison_chart,
    state_summary,
    time_series_chart,
)


st.set_page_config(page_title="Heat Trace | State Explorer", layout="wide")
apply_light_mode_background()

st.title("State Explorer")
st.caption(
    "Explore annual energy consumption for all 50 states, with estimated solar production shown "
    "where the current ZenPower subset has coverage."
)

df = load_state_timeseries()
map_df = load_homepage_map_data()
energy_abbrs = set(map_df.loc[map_df["has_energy"], "state_abbr"])
df = df[df["state_abbr"].isin(energy_abbrs)].copy()
states = df["state"].drop_duplicates().sort_values().tolist()

selector_col, _ = st.columns([1, 2])
with selector_col:
    selected_state = st.selectbox(
        "State",
        states,
        index=states.index("California") if "California" in states else 0,
    )
state_df = df[df["state"] == selected_state].sort_values("date")
selected_abbr = state_df["state_abbr"].iloc[0]
solar_row = map_df[map_df["state_abbr"] == selected_abbr]
has_solar = not solar_row.empty and bool(solar_row["has_solar"].iloc[0])

latest = state_df.iloc[-1]
first = state_df.iloc[0]
peak = state_df.loc[state_df["energy_kwh"].idxmax()]
cols = st.columns(4)
cols[0].metric("Latest energy (kWh)", f"{latest['energy_kwh']:,.0f}")
cols[1].metric("Peak year", f"{int(peak['year'])}")
cols[2].metric("Latest YoY change", f"{latest['year_over_year_change']:.1%}")
cols[3].metric("Years shown", f"{state_df['year'].nunique()}")

st.subheader(selected_state)
st.write(state_summary(state_df))

tab_labels = [METRIC_LABELS[metric] for metric in CHART_METRICS] + ["Estimated Solar Production"]
tabs = st.tabs(tab_labels)
for tab, metric in zip(tabs[: len(CHART_METRICS)], CHART_METRICS):
    with tab:
        st.plotly_chart(time_series_chart(state_df, metric), use_container_width=True)

with tabs[-1]:
    if has_solar:
        solar_value = float(solar_row["solar_production"].iloc[0])
        latest_energy = float(latest["energy_kwh"])
        solar_share = solar_value / latest_energy if latest_energy else 0.0
        solar_rank = int(map_df["solar_production"].rank(method="min", ascending=False).loc[solar_row.index[0]])

        solar_cols = st.columns(3)
        solar_cols[0].metric("Estimated solar production", f"{solar_value:,.0f} kWh")
        solar_cols[1].metric("Solar rank in subset", f"#{solar_rank} of 21")
        solar_cols[2].metric("Share of latest consumption", f"{solar_share:.2%}")

        st.plotly_chart(
            solar_production_comparison_chart(map_df, selected_abbr),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info(
            "Estimated solar production is not available for this state in the current ZenPower subset. "
            "Energy consumption trends are still available because the clean energy dataset covers all 50 states."
        )

with st.expander("State data"):
    st.dataframe(
        state_df[["year", "energy_btu", "energy_kwh", "year_over_year_change"]].style.format(
            {
                "energy_btu": "{:,.0f}",
                "energy_kwh": "{:,.0f}",
                "year_over_year_change": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
