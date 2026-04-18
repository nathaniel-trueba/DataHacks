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

combined_chart_path = APP_DIR / "assets" / "solar_permit_individual_analysis.png"
permit_count_chart_path = APP_DIR / "assets" / "top_cities_by_permit_count.png"
avg_system_chart_path = APP_DIR / "assets" / "top_cities_by_avg_kwh_per_permit.png"
adoption_time_chart_path = APP_DIR / "assets" / "solar_permit_adoption_over_time.png"

st.image(combined_chart_path, use_container_width=True)

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

st.divider()

st.subheader("Where individual solar buying is happening at scale")
st.image(permit_count_chart_path, use_container_width=True)
st.write(
    "Permit volume shows where lots of individual buyers are entering the solar market. "
    "Oakland leads by a wide margin with 3,455 permits, while Hanford, Indio, and San Diego "
    "also show strong adoption. The interesting part is that permit count is not just a "
    "sun-exposure story: places with very different climates and city sizes can still show "
    "high solar activity when household demand, incentives, costs, and local permitting line up."
)

st.subheader("Where system size tells a different story")
st.image(avg_system_chart_path, use_container_width=True)
st.write(
    "Average system size points to a separate type of solar buyer. Cities like Point Reyes "
    "and Detour have enormous average systems, with Point Reyes reaching 741 kWh per permit "
    "and Detour reaching 630 kWh. Those numbers are far beyond a typical single-family rooftop "
    "setup, which suggests farms, ranches, or commercial properties with much larger energy needs."
)

st.subheader("What this means")
st.write(
    "These charts point to two different solar markets. One is the everyday homeowner market, "
    "where thousands of people in places like Oakland and San Diego install modest rooftop systems. "
    "The other is the big-installer market, where a smaller number of rural or commercial buyers "
    "install very large systems. That distinction matters for modeling: sun exposure alone may not "
    "predict adoption as well as the mix of household demand, property type, incentives, and local economics."
)

st.divider()

st.subheader("Solar adoption shifted from flatline to breakout")
st.image(adoption_time_chart_path, use_container_width=True)
st.write(
    "From 2018 through 2022, solar adoption was essentially flat, with monthly permit counts "
    "hovering around only 100-150. Starting in early 2023, the pattern changed: permits spiked "
    "to roughly 800 in a month, which may line up with Inflation Reduction Act tax credits making "
    "solar cheaper for homeowners. By 2024 and 2025, adoption was no longer just a one-month spike; "
    "it became a sustained climb from roughly 400 to 1,400 permits per month."
)
st.write(
    "March 2026 stands out as the major anomaly, peaking at 4,592 permits. That sudden jump could "
    "reflect a policy deadline, a rush to file before incentives changed, a large data batch upload, "
    "or a genuine surge in demand. Even after April 2026 falls back to about 1,200 permits, the level "
    "is still far above the historical baseline, suggesting that the floor for solar adoption has "
    "permanently moved higher."
)
