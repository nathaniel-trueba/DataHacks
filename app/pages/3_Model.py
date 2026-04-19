from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from utils import apply_light_mode_background


st.set_page_config(page_title="Heat Trace | Model", layout="wide")
apply_light_mode_background()

st.title("Model")
st.caption("Placeholder page for the next version of Heat Trace's prediction and scoring workflow.")

st.info(
    "This page is reserved for the model prototype. A future version can compare solar production, "
    "energy demand, policy timing, and location features to estimate where solar adoption may have "
    "the strongest impact."
)

cols = st.columns(3)
cols[0].metric("Model status", "Draft")
cols[1].metric("Training data", "Pending")
cols[2].metric("Target", "TBD")
