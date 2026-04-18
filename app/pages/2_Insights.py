from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


st.set_page_config(page_title="Heat Trace | Insights", layout="wide")

st.title("Insights")
st.caption("Solar permit patterns reveal two very different adoption stories.")

permit_chart_path = APP_DIR / "assets" / "solar_permit_individual_analysis.png"

st.image(permit_chart_path, use_container_width=True)

st.subheader("Tiny towns, huge systems. Big cities, everyday rooftops.")
st.write(
    "The cities with the highest average system size, such as Point Reyes at 741 kWh "
    "and Detour at 630 kWh, are tiny towns most people have not heard of. The cities "
    "with the most permits, such as Oakland with 3,455 permits and Hanford with 1,776, "
    "have much smaller average systems around 7-10 kWh."
)

st.subheader("What this means for regular people")
st.write(
    "Small rural towns are installing massive solar systems, likely farms, ranches, "
    "or commercial properties that need huge energy capacity. Meanwhile, big cities "
    "like Oakland and San Diego have thousands of regular homeowners installing modest "
    "rooftop panels."
)

st.write("So there are actually two completely different types of solar buyers in this data:")
st.markdown(
    "- **The big installer:** one farm in Point Reyes putting in a 741 kWh system.\n"
    "- **The everyday homeowner:** thousands of Oakland residents each putting in a 7-10 kWh rooftop panel."
)
