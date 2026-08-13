"""TrafficFlow home page — historical replay product."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from styles.theme import apply_theme, callout, footer, metric_card, metrics_row, sidebar_brand
from utils.charts import network_preview
from utils.compat import plotly_chart
from utils.load import format_minutes, require_bundle

st.set_page_config(
    page_title="TrafficFlow",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
sidebar_brand()
bundle = require_bundle()
manifest = bundle.manifest
metrics = bundle.metrics
routing = bundle.routing_summary
delta = {row["horizon_minutes"]: row for row in metrics.get("spatial_vs_temporal", [])}
spatial_30 = delta.get(30, {})
vs_static = routing.get("predictive_vs_static", {})

st.markdown(
    """
    <div class="tf-hero">
      <div class="tf-eyebrow">PREDICTIVE MOBILITY INTELLIGENCE</div>
      <h1>Forecast traffic.<br/>Route before congestion happens.</h1>
      <p>
        TrafficFlow predicts future speeds across a freeway sensor network, converts those
        forecasts into travel-time costs, and compares prediction-aware routes with static
        and current-state plans. This demo replays historical METR-LA traffic — it is not live navigation.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

cta1, cta2, _ = st.columns([1, 1.1, 2])
with cta1:
    st.page_link("pages/1_Route_Planner.py", label="Plan a route")
with cta2:
    st.page_link("pages/3_Model_Performance.py", label="Explore model performance")

left, right = st.columns((1.15, 1), gap="large")
with left:
    st.markdown("#### How a decision is made")
    st.markdown(
        """
        <div class="tf-steps">
          <div class="tf-step"><div class="n">01</div><h3>Predict</h3><p>Forecast future sensor speeds from history and neighbors.</p></div>
          <div class="tf-step"><div class="n">02</div><h3>Understand</h3><p>Turn predicted mph into minutes on each network edge.</p></div>
          <div class="tf-step"><div class="n">03</div><h3>Route</h3><p>Search the graph with Dijkstra using those future costs.</p></div>
          <div class="tf-step"><div class="n">04</div><h3>Evaluate</h3><p>Score the chosen path on traffic that actually occurred.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    plotly_chart(network_preview(bundle.sensor_metadata))
    st.caption("Loop detectors, not a street map. Routes follow sensor relationships.")

st.markdown("#### Evaluation snapshot")
mae_gain = spatial_30.get("mae_improvement_pct")
metrics_row(
    [
        metric_card("Sensors in demo", f"{manifest.get('num_nodes', '—')}", "METR-LA loop detectors", "blue"),
        metric_card("Routing edges", f"{manifest.get('num_routing_edges', '—')}", "k-nearest sensor graph", "info"),
        metric_card(
            "Spatial MAE gain · 30 min",
            f"{mae_gain:.1f}%" if mae_gain is not None else "—",
            "Spatiotemporal vs temporal XGBoost",
            "info",
        ),
        metric_card(
            "Mean savings vs static",
            format_minutes(vs_static.get("mean_savings_min")),
            "Realized travel time, 72 demo trips",
            "positive" if (vs_static.get("mean_savings_min") or 0) > 0 else "neutral",
        ),
    ]
)

st.markdown("#### Featured historical replay")
scenarios = bundle.scenarios
if not scenarios.empty:
    pred = scenarios[scenarios["strategy"] == "predictive"].copy()
    static = scenarios[scenarios["strategy"] == "static"][
        ["origin", "destination", "departure_time", "realized_minutes"]
    ].rename(columns={"realized_minutes": "static_realized"})
    merged = pred.merge(static, on=["origin", "destination", "departure_time"], how="left")
    merged["saved"] = merged["static_realized"] - merged["realized_minutes"]
    featured = merged.sort_values("saved", ascending=False).iloc[0]
    current_row = scenarios[
        (scenarios["strategy"] == "current")
        & (scenarios["origin"] == featured["origin"])
        & (scenarios["destination"] == featured["destination"])
        & (scenarios["departure_time"] == featured["departure_time"])
    ]
    current_real = float(current_row["realized_minutes"].iloc[0]) if not current_row.empty else float("nan")
    callout(
        f"{str(featured['traffic_condition']).replace('_', ' ').title()} · {featured['departure_time']}",
        f"Largest realized savings in the precomputed demo set — sensor {featured['origin']} → {featured['destination']}. "
        "This is the best case in that set, not a typical-trip claim.",
        "info",
    )
    metrics_row(
        [
            metric_card("Static realized", format_minutes(featured["static_realized"]), variant="neutral"),
            metric_card("Current-state realized", format_minutes(current_real), variant="info"),
            metric_card("Predictive realized", format_minutes(featured["realized_minutes"]), variant="blue"),
            metric_card("Saved vs static", format_minutes(featured["saved"]), variant="positive"),
        ]
    )
else:
    callout("No precomputed trips", "Routing scenarios were not found in the deployment artifacts.", "warn")

st.markdown("#### Continue")
n1, n2, n3 = st.columns(3)
with n1:
    st.markdown('<div class="tf-card"><h3>Traffic forecast</h3><p>Compare temporal and neighbor-aware predictions on a single sensor.</p></div>', unsafe_allow_html=True)
    st.page_link("pages/2_Traffic_Forecast.py", label="Inspect a forecast")
with n2:
    st.markdown('<div class="tf-card"><h3>Model performance</h3><p>See whether spatial features help, and whether better forecasts move routes.</p></div>', unsafe_allow_html=True)
    st.page_link("pages/3_Model_Performance.py", label="Open evaluation")
with n3:
    st.markdown('<div class="tf-card"><h3>Network explorer</h3><p>Inspect sensor locations, neighbors, and observed speeds.</p></div>', unsafe_allow_html=True)
    st.page_link("pages/4_Network_Explorer.py", label="Explore the network")

footer(manifest)
