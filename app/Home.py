from __future__ import annotations

import streamlit as st

from utils import (
    apply_light_mode_background,
    energy_solar_overlay_map,
    homepage_rankings,
    load_homepage_map_data,
    render_homepage_map_cards,
)


st.set_page_config(page_title="Heat Trace", layout="wide")
apply_light_mode_background()

st.title("Heat Trace")
st.caption(
    "State-level variation in U.S. energy consumption and solar production reveals where "
    "solar meaningfully offsets demand and where its impact remains marginal."
)

map_df = load_homepage_map_data()

st.subheader("National energy and solar overview")
render_homepage_map_cards(map_df)

st.caption(
    "State fill shows energy consumption. Green bubbles show estimated solar production. "
    "Gray states do not have overlapping data for both metrics."
)

st.plotly_chart(
    energy_solar_overlay_map(map_df),
    use_container_width=True,
    config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
)

st.subheader("Top states in the overlapping dataset")
consumption_col, solar_col = st.columns(2)

with consumption_col:
    st.write("Energy consumption")
    st.dataframe(
        homepage_rankings(map_df, "energy_consumption").style.format({"Value": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with solar_col:
    st.write("Estimated solar production")
    st.dataframe(
        homepage_rankings(map_df, "solar_production").style.format({"Value": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("About this prototype"):
    st.write(
        "The Home map uses state-level rows with overlapping energy consumption and estimated "
        "solar production data. Other states are intentionally shown in gray so missing coverage "
        "is visible rather than hidden."
    )
