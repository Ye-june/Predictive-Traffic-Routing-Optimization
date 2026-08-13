"""TrafficFlow Streamlit entrypoint — named navigation for Community Cloud."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

st.set_page_config(
    page_title="TrafficFlow",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.navigation(
    [
        st.Page("home.py", title="Home", default=True),
        st.Page("pages/1_Route_Planner.py", title="Route Planner"),
        st.Page("pages/2_Traffic_Forecast.py", title="Traffic Forecast"),
        st.Page("pages/3_Model_Performance.py", title="Model Performance"),
        st.Page("pages/4_Network_Explorer.py", title="Network Explorer"),
        st.Page("pages/5_About.py", title="About"),
    ]
)
page.run()
