from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import build_insights, dataframe_with_formats, insight_count_chart, load_state_timeseries


st.set_page_config(page_title="Heat Trace | Insights", layout="wide")

st.title("Insights")
st.caption("Rule-based flags that surface interesting state patterns for follow-up analysis.")

df = load_state_timeseries()
insights = build_insights(df)

if insights.empty:
    st.info("No insight flags were found in the current dataset.")
else:
    cols = st.columns(3)
    cols[0].metric("Flagged states", insights["State"].nunique())
    cols[1].metric("Total flags", len(insights))
    cols[2].metric("Insight types", insights["Insight"].nunique())

    st.plotly_chart(insight_count_chart(insights), use_container_width=True)

    selected_flags = st.multiselect(
        "Filter insight types",
        options=sorted(insights["Insight"].unique()),
        default=sorted(insights["Insight"].unique()),
    )
    filtered = insights[insights["Insight"].isin(selected_flags)].copy()

    st.dataframe(
        dataframe_with_formats(filtered, percent_columns=["Solar growth, 5yr", "CO2 change, 5yr"]),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Insight rules"):
    st.write(
        "- Solar growth increased but emissions did not decrease: five-year solar growth is above 25%, "
        "while five-year CO2 emissions are flat or higher.\n"
        "- High pollution and low solar adoption: latest AQI is in the top quartile, while clean ratio is in the bottom quartile.\n"
        "- Strong solar adoption and improving environmental indicators: latest clean ratio is in the top quartile, "
        "with falling CO2 emissions and improving AQI over the last five years."
    )
