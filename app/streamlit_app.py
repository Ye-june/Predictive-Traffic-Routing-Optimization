"""TrafficFlow home page — historical replay product."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from utils.charts import ROUTE_COLORS
from utils.load import format_minutes, render_disclaimer, require_bundle

st.set_page_config(
    page_title="TrafficFlow",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

bundle = require_bundle()
manifest = bundle.manifest
metrics = bundle.metrics
routing = bundle.routing_summary
delta = {row["horizon_minutes"]: row for row in metrics.get("spatial_vs_temporal", [])}

st.title("TrafficFlow")
st.subheader("Predict future traffic. Then ask whether those predictions improve the route.")

st.markdown(
    """
TrafficFlow forecasts speeds on a freeway sensor network and converts those
forecasts into travel-time costs. A shortest-path search then compares
**static**, **current-state**, and **predictive** routes.

This public demo runs in **Historical Replay Mode**: you pick a past departure.
The system uses only information that would have been available then, chooses a
route, and scores it on the traffic that actually occurred.
"""
)

left, right = st.columns((1.15, 1), gap="large")
with left:
    st.markdown("#### How a decision is made")
    st.code(
        "Traffic history → spatiotemporal forecast → predicted speeds\n"
        "→ travel-time edge weights → NetworkX route → realized travel time",
        language="text",
    )
    st.page_link("pages/1_Route_Planner.py", label="Open the route planner", icon="🗺️")
    st.caption(
        "Routes connect loop detectors. They are not turn-by-turn driving directions."
    )

with right:
    st.markdown("#### What the evaluation shows")
    c1, c2 = st.columns(2)
    spatial_30 = delta.get(30, {})
    vs_static = routing.get("predictive_vs_static", {})
    c1.metric("Demo sensors", f"{manifest.get('num_nodes', '—')}")
    c2.metric("Routing edges", f"{manifest.get('num_routing_edges', '—')}")
    c1.metric(
        "Spatial MAE gain (30 min)",
        f"{spatial_30.get('mae_improvement_pct', float('nan')):.1f}%"
        if spatial_30
        else "—",
        help="Percent MAE improvement of spatiotemporal XGBoost vs temporal XGBoost on the test subsample.",
    )
    c2.metric(
        "Mean savings vs static",
        format_minutes(vs_static.get("mean_savings_min")),
        help="Average realized minutes saved by the predictive route versus the free-flow static route.",
    )

st.divider()
st.markdown("#### Featured replay")
scenarios = bundle.scenarios
if not scenarios.empty:
    pred = scenarios[scenarios["strategy"] == "predictive"].copy()
    static = scenarios[scenarios["strategy"] == "static"][
        ["origin", "destination", "departure_time", "realized_minutes"]
    ].rename(columns={"realized_minutes": "static_realized"})
    merged = pred.merge(static, on=["origin", "destination", "departure_time"], how="left")
    merged["saved"] = merged["static_realized"] - merged["realized_minutes"]
    featured = merged.sort_values("saved", ascending=False).iloc[0]
    st.caption("Largest realized savings among the precomputed demo trips — not a typical-case claim.")
    current_row = scenarios[
        (scenarios["strategy"] == "current")
        & (scenarios["origin"] == featured["origin"])
        & (scenarios["destination"] == featured["destination"])
        & (scenarios["departure_time"] == featured["departure_time"])
    ]
    current_real = float(current_row["realized_minutes"].iloc[0]) if not current_row.empty else float("nan")
    st.info(
        f"**{str(featured['traffic_condition']).replace('_', ' ').title()}** · "
        f"departure `{featured['departure_time']}` · "
        f"sensor {featured['origin']} → {featured['destination']}"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Static realized", format_minutes(featured["static_realized"]))
    m2.metric("Current-state realized", format_minutes(current_real))
    m3.metric("Predictive realized", format_minutes(featured["realized_minutes"]))
    m4.metric("Saved vs static", format_minutes(featured["saved"]))
    st.caption(
        "These numbers come from `artifacts/demo/routing_scenarios.parquet`, not from a hand-picked story."
    )
else:
    st.warning("No precomputed routing scenarios were found in the artifacts.")

st.divider()
cols = st.columns(3)
with cols[0]:
    st.markdown("**Forecast**")
    st.write("Compare persistence, temporal XGBoost, and neighbor-aware XGBoost on held-out time.")
    st.page_link("pages/2_Traffic_Forecast.py", label="Inspect a forecast", icon="📈")
with cols[1]:
    st.markdown("**Evaluate**")
    st.write("See whether spatial features help, and whether better forecasts move routes.")
    st.page_link("pages/3_Model_Performance.py", label="Model & routing metrics", icon="📊")
with cols[2]:
    st.markdown("**Explore**")
    st.write("The map is a sensor graph, not a complete street network.")
    st.page_link("pages/4_Network_Explorer.py", label="Open the network", icon="🕸️")

st.divider()
render_disclaimer()
st.caption(
    f"Dataset {manifest.get('dataset')} · demo {manifest.get('demo_start')} → {manifest.get('demo_end')} · "
    f"horizons {manifest.get('forecast_horizons')} min · artifacts {manifest.get('total_artifact_mb')} MB"
)
