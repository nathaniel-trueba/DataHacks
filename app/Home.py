from __future__ import annotations

import streamlit as st

from utils import (
    apply_heat_trace_theme,
    energy_solar_overlay_map,
    homepage_rankings,
    load_homepage_map_data,
    render_homepage_map_cards,
)


st.set_page_config(page_title="Heat Trace", layout="wide")
apply_heat_trace_theme()

st.title("Heat Trace")
st.caption(
    "Heat Trace is a data analytics and machine learning project for DataHacks 2026. "
    "Our model predicts annual kWh production using solar system capacity, location-based weather, "
    "average temperature, and sunlight or irradiance inputs. "
    "State-level variation in U.S. energy consumption and solar production reveals where "
    "solar meaningfully offsets demand and where its impact remains marginal. "
    "A kilowatt hour, or kWh, is a measure of energy use: one kWh is the energy needed to "
    "run a 1,000-watt appliance for one hour."
)

map_df = load_homepage_map_data()

st.subheader("National energy and solar overview")
render_homepage_map_cards(map_df)

st.caption(
    "State fill shows energy consumption. Green bubbles show estimated solar production. "
)

st.plotly_chart(
    energy_solar_overlay_map(map_df),
    use_container_width=True,
    config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
)

st.subheader("Top states by available metric")
consumption_col, solar_col = st.columns(2)

with consumption_col:
    st.write("Energy consumption")
    st.dataframe(
        homepage_rankings(map_df, "energy_consumption_kwh").style.format({"Value (kWh)": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with solar_col:
    st.write("Estimated solar production")
    st.dataframe(
        homepage_rankings(map_df, "solar_production").style.format({"Value (kWh)": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("About this prototype"):
    st.write(
        "The Home map combines the latest state energy consumption data with available estimated "
        "solar production rows. States can still show an energy fill even when solar production is "
        "missing, and solar bubbles can appear wherever solar production is available. Gray is "
        "reserved for states where neither metric is currently loaded."
    )
