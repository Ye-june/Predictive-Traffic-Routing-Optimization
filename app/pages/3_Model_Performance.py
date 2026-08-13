"""Forecast and routing evaluation for technical reviewers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.load import format_minutes, render_disclaimer, require_bundle

st.set_page_config(page_title="Model performance · TrafficFlow", layout="wide")
bundle = require_bundle()

st.title("Model performance")
st.write(
    "Prediction alone is not the goal. This page reports forecast error and whether "
    "prediction-aware routing reduced realized travel time on the demo trips."
)

results = pd.DataFrame(bundle.metrics["results"])
st.markdown("### Forecast error on the chronological test subsample")
st.caption("MAE / RMSE in mph. Persistence is the required naive baseline.")
pivot_mae = results.pivot(index="model", columns="horizon_minutes", values="mae_mph").round(3)
pivot_rmse = results.pivot(index="model", columns="horizon_minutes", values="rmse_mph").round(3)
st.write("MAE (mph)")
st.dataframe(pivot_mae, use_container_width=True)
st.write("RMSE (mph)")
st.dataframe(pivot_rmse, use_container_width=True)

delta = pd.DataFrame(bundle.metrics.get("spatial_vs_temporal", []))
if not delta.empty:
    st.markdown("### Did neighboring sensors help?")
    st.dataframe(
        delta.rename(
            columns={
                "horizon_minutes": "Horizon (min)",
                "temporal_mae": "Temporal MAE",
                "spatial_mae": "Spatiotemporal MAE",
                "mae_improvement": "MAE improvement",
                "mae_improvement_pct": "MAE improvement %",
            }
        ).round(3),
        hide_index=True,
        use_container_width=True,
    )
    fig = px.bar(
        delta,
        x="horizon_minutes",
        y="mae_improvement_pct",
        title="MAE improvement from adding neighbor speeds (%)",
        labels={"horizon_minutes": "Horizon (min)", "mae_improvement_pct": "Improvement (%)"},
        color_discrete_sequence=["#0e7c66"],
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Routing simulation")
summary = bundle.routing_summary
st.write(f"Precomputed trips: **{summary.get('n_trips', 0)}**")
cols = st.columns(2)
vs_static = summary.get("predictive_vs_static", {})
vs_current = summary.get("predictive_vs_current", {})
with cols[0]:
    st.markdown("**Predictive vs static**")
    st.metric("Mean savings", format_minutes(vs_static.get("mean_savings_min")))
    st.metric("Trips improved", f"{vs_static.get('pct_improved', float('nan')):.1f}%")
    st.metric("Trips worsened", f"{vs_static.get('pct_worsened', float('nan')):.1f}%")
with cols[1]:
    st.markdown("**Predictive vs current-state**")
    st.metric("Mean savings", format_minutes(vs_current.get("mean_savings_min")))
    st.metric("Trips improved", f"{vs_current.get('pct_improved', float('nan')):.1f}%")
    st.metric("Trips worsened", f"{vs_current.get('pct_worsened', float('nan')):.1f}%")

regret = summary.get("predictive_regret_vs_oracle_min", {})
if regret:
    st.caption(
        f"Mean regret versus the hindsight oracle: {regret.get('mean', float('nan')):.2f} min "
        f"(median {regret.get('median', float('nan')):.2f})."
    )

scenarios = bundle.scenarios
if not scenarios.empty:
    wide = scenarios.pivot_table(
        index=["origin", "destination", "departure_time", "traffic_condition"],
        columns="strategy",
        values="realized_minutes",
        aggfunc="first",
    ).reset_index()
    if {"static", "predictive"}.issubset(wide.columns):
        wide["savings_vs_static"] = wide["static"] - wide["predictive"]
        fig = px.histogram(
            wide,
            x="savings_vs_static",
            color="traffic_condition",
            nbins=20,
            title="Realized minutes saved by predictive routing vs static",
            labels={"savings_vs_static": "Minutes saved (negative = worse)"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Negative values are trips where the predictive route was slower in hindsight. They are not hidden.")

st.markdown("### Leakage audit")
config = bundle.manifest
st.json(
    {
        "split": "chronological 70/15/15",
        "demo_window": [bundle.manifest.get("demo_start"), bundle.manifest.get("demo_end")],
        "speed_floor_mph": bundle.manifest.get("speed_floor_mph"),
        "note": "Historical means and free-flow speeds are fit on the training window only.",
    }
)

render_disclaimer()
