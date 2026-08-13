"""Project context for reviewers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from styles.theme import apply_theme, callout, footer, page_header, sidebar_brand
from utils.load import require_bundle

st.set_page_config(page_title="About · TrafficFlow", page_icon="◇", layout="wide")
apply_theme()
sidebar_brand()
bundle = require_bundle()

page_header(
    "About TrafficFlow",
    "An end-to-end spatiotemporal forecasting and network-optimization project: predict future traffic, then ask whether those predictions improve the route.",
    badges=[(f"v{bundle.manifest.get('app_version', '1.0.0')}", "blue"), ("Historical Replay", "warm")],
)

st.markdown(
    """
    <div class="tf-steps">
      <div class="tf-step"><div class="n">PREDICT</div><h3>Forecast speeds</h3><p>Temporal and neighbor-aware XGBoost models on METR-LA.</p></div>
      <div class="tf-step"><div class="n">UNDERSTAND</div><h3>Turn mph into minutes</h3><p>Edge cost = miles / max(speed, 5 mph) × 60.</p></div>
      <div class="tf-step"><div class="n">ROUTE</div><h3>Search the graph</h3><p>NetworkX Dijkstra on a k-nearest sensor network.</p></div>
      <div class="tf-step"><div class="n">EVALUATE</div><h3>Replay reality</h3><p>Score paths on traffic that actually occurred.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Dataset")
st.write(
    "METR-LA: 207 Los Angeles freeway loop detectors, 5-minute speeds, 1 March 2012 through 27 June 2012 "
    "(Li et al., DCRNN, ICLR 2018)."
)

st.markdown("#### Tools actually used")
st.write("Python · pandas · scikit-learn · XGBoost · NetworkX · Plotly · Streamlit")

st.markdown("#### Limitations")
st.markdown(
    """
- Sensor adjacency is not a complete road map.
- The demo does not use live traffic, weather, or incidents.
- Routing does not model congestion feedback from the routed vehicle.
- METR-LA covers freeways, not an urban arterial grid.
- A 5 mph speed floor keeps travel-time weights finite.
"""
)

callout(
    "Not a navigation product",
    "TrafficFlow is a research and portfolio demonstration based on historical sensor data. It is not intended for real-world or safety-critical routing.",
    "warn",
)

st.markdown("#### Reproduce")
st.code(
    "python scripts/download_data.py\n"
    "python scripts/prepare_data.py\n"
    "python scripts/build_deployment_assets.py\n"
    "streamlit run app/streamlit_app.py",
    language="bash",
)
st.markdown("[GitHub repository](https://github.com/Ye-june/Predictive-Traffic-Routing-Optimization)")

footer(bundle.manifest)
