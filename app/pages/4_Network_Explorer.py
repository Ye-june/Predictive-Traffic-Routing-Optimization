"""Explore the METR-LA sensor relationship graph."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from styles.theme import apply_theme, footer, metric_card, metrics_row, page_header, sidebar_brand
from utils.charts import speed_map
from utils.load import require_bundle, sensor_options

st.set_page_config(page_title="Network explorer · TrafficFlow", page_icon="◇", layout="wide")
apply_theme()
sidebar_brand()
bundle = require_bundle()

page_header(
    "Network explorer",
    "Each node is a loop detector. Edges are the k-nearest road-network neighbors used for routing — not a complete street map.",
    badges=[("207 sensors", "blue"), ("k = 8 neighbors", "teal")],
)

ctrl, _ = st.columns((1.2, 2))
with ctrl:
    timestamp = st.selectbox(
        "Observed timestamp",
        list(bundle.speeds.index),
        index=min(24, len(bundle.speeds) - 1),
        format_func=lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
    )
timestamp = pd.Timestamp(timestamp)
speeds = {
    str(col): float(bundle.speeds.loc[timestamp, col])
    for col in bundle.speeds.columns
    if pd.notna(bundle.speeds.loc[timestamp, col])
}
st.plotly_chart(
    speed_map(bundle.sensor_metadata, speeds, f"Observed speed at {timestamp.strftime('%Y-%m-%d %H:%M')}"),
    use_container_width=True,
)

options = sensor_options(bundle)
sensor_label = st.selectbox("Inspect a sensor", list(options.keys()))
sensor_id = options[sensor_label]
meta = bundle.sensor_metadata.set_index("sensor_id").loc[sensor_id]
neighbors = bundle.neighbors.get(sensor_id, [])
metrics_row(
    [
        metric_card("Free-flow speed", f"{meta['free_flow_mph']:.1f} mph", "Train-set 95th percentile", "info"),
        metric_card("Graph degree", str(int(meta["degree"])), "Routing-graph connections", "blue"),
        metric_card("Region", str(meta["region"]), "Relative location in the sensor set", "neutral"),
    ]
)
st.markdown("#### Neighbors")
st.write(", ".join(neighbors) if neighbors else "None")
st.markdown("#### Observed speed history")
st.line_chart(bundle.speeds[sensor_id].rename("Speed (mph)"))

g = bundle.graph
with st.expander("Graph facts"):
    st.write(
        {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "note": bundle.manifest.get("graph_notes"),
        }
    )

footer(bundle.manifest)
