"""Project context for reviewers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from utils.load import render_disclaimer, require_bundle

st.set_page_config(page_title="About · TrafficFlow", layout="wide")
bundle = require_bundle()

st.title("About TrafficFlow")
st.write(
    "TrafficFlow is an end-to-end spatiotemporal forecasting and network-optimization "
    "project. It asks whether spatial sensor relationships improve forecasts, and "
    "whether those forecasts improve routing decisions."
)

st.markdown("### Architecture")
st.code(
    """
Historical traffic data
        ↓
Feature engineering (lags, calendar, neighbor speeds)
        ↓
Traffic forecasting (persistence, temporal XGBoost, spatiotemporal XGBoost)
        ↓
Predicted speeds → travel-time edge weights
        ↓
NetworkX Dijkstra routing
        ↓
Static vs current-state vs predictive vs oracle
        ↓
Streamlit historical replay interface
""",
    language="text",
)

st.markdown("### Dataset")
st.write(
    "METR-LA: 207 Los Angeles freeway loop detectors, 5-minute speeds, March–June 2012 "
    "(Li et al., DCRNN, ICLR 2018). The downloaded file ends 27 June 2012 23:55."
)

st.markdown("### Tools actually used")
st.write("Python · pandas · scikit-learn · XGBoost · NetworkX · Plotly · Streamlit")

st.markdown("### Limitations")
st.markdown(
    """
- Sensor adjacency is not a complete road map.
- The demo does not use live traffic, weather, or incidents.
- Routing does not model congestion feedback from the routed vehicle.
- METR-LA covers freeways, not an urban arterial grid.
- A 5 mph speed floor is applied so travel-time weights stay finite.
"""
)

st.markdown("### Reproduce")
st.code(
    "python scripts/download_data.py\n"
    "python scripts/prepare_data.py\n"
    "python scripts/build_deployment_assets.py\n"
    "streamlit run app/streamlit_app.py",
    language="bash",
)
st.write("GitHub: https://github.com/Ye-june/Predictive-Traffic-Routing-Optimization")
st.caption(f"TrafficFlow v{bundle.manifest.get('app_version', '1.0.0')} · Historical Replay Demo")
render_disclaimer()
