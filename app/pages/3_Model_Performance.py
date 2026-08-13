"""Forecast and routing evaluation for technical reviewers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from styles.theme import apply_theme, callout, footer, metric_card, metrics_row, page_header, sidebar_brand
from utils.charts import styled_bar, styled_hist
from utils.load import format_minutes, require_bundle

st.set_page_config(page_title="Model performance · TrafficFlow", page_icon="◇", layout="wide")
apply_theme()
sidebar_brand()
bundle = require_bundle()

page_header(
    "Model performance",
    "Prediction alone is not the goal. This page reports forecast error and whether prediction-aware routing reduced realized travel time.",
    badges=[("Chronological test", "blue"), ("72 replay trips", "teal")],
)

results = pd.DataFrame(bundle.metrics["results"])
delta = pd.DataFrame(bundle.metrics.get("spatial_vs_temporal", []))
summary = bundle.routing_summary
vs_static = summary.get("predictive_vs_static", {})
vs_current = summary.get("predictive_vs_current", {})

best_row = results.loc[results["mae_mph"].idxmin()]
best_spatial = None if delta.empty else delta.sort_values("mae_improvement_pct", ascending=False).iloc[0]
metrics_row(
    [
        metric_card("Lowest MAE model", str(best_row["model"]).replace("_", " "), f"{best_row['horizon_minutes']} min horizon", "blue"),
        metric_card(
            "Best spatial MAE gain",
            f"{best_spatial['mae_improvement_pct']:.1f}%" if best_spatial is not None else "—",
            f"at {int(best_spatial['horizon_minutes'])} min" if best_spatial is not None else "",
            "info",
        ),
        metric_card(
            "Trips improved vs static",
            f"{vs_static.get('pct_improved', float('nan')):.1f}%",
            f"worsened {vs_static.get('pct_worsened', float('nan')):.1f}%",
            "positive",
        ),
        metric_card("Mean savings vs static", format_minutes(vs_static.get("mean_savings_min")), "Realized minutes", "positive"),
    ]
)

st.markdown("#### Forecast error by horizon")
st.caption("MAE and RMSE in mph on the chronological test subsample. Persistence is the required naive baseline.")
c1, c2 = st.columns(2)
with c1:
    st.write("MAE (mph)")
    st.dataframe(results.pivot(index="model", columns="horizon_minutes", values="mae_mph").round(3), use_container_width=True)
with c2:
    st.write("RMSE (mph)")
    st.dataframe(results.pivot(index="model", columns="horizon_minutes", values="rmse_mph").round(3), use_container_width=True)

if not delta.empty:
    st.markdown("#### Did neighboring sensors help?")
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
    st.plotly_chart(
        styled_bar(
            delta,
            "horizon_minutes",
            "mae_improvement_pct",
            "Forecast error improvement from adding neighbor speeds",
            "Horizon (min)",
            "MAE improvement (%)",
        ),
        use_container_width=True,
    )

st.markdown("#### Routing simulation")
st.caption(f"Precomputed trips: {summary.get('n_trips', 0)}. Negative savings are trips where predictive routing was slower.")
left, right = st.columns(2)
with left:
    st.markdown("**Predictive vs static**")
    st.metric("Mean savings", format_minutes(vs_static.get("mean_savings_min")))
    st.metric("Trips improved", f"{vs_static.get('pct_improved', float('nan')):.1f}%")
    st.metric("Trips worsened", f"{vs_static.get('pct_worsened', float('nan')):.1f}%")
with right:
    st.markdown("**Predictive vs current-state**")
    st.metric("Mean savings", format_minutes(vs_current.get("mean_savings_min")))
    st.metric("Trips improved", f"{vs_current.get('pct_improved', float('nan')):.1f}%")
    st.metric("Trips worsened", f"{vs_current.get('pct_worsened', float('nan')):.1f}%")

regret = summary.get("predictive_regret_vs_oracle_min", {})
if regret:
    callout(
        "Distance to a perfect-information oracle",
        f"Mean regret {regret.get('mean', float('nan')):.2f} min (median {regret.get('median', float('nan')):.2f}).",
        "info",
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
        st.plotly_chart(
            styled_hist(
                wide,
                "savings_vs_static",
                "traffic_condition",
                "Realized minutes saved by predictive routing vs static",
                "Minutes saved (negative = worse)",
            ),
            use_container_width=True,
        )

with st.expander("Leakage audit"):
    st.write(
        {
            "split": "chronological 70/15/15",
            "demo_window": [bundle.manifest.get("demo_start"), bundle.manifest.get("demo_end")],
            "speed_floor_mph": bundle.manifest.get("speed_floor_mph"),
            "note": "Historical means and free-flow speeds are fit on the training window only.",
        }
    )

footer(bundle.manifest)
