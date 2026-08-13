"""Explore the METR-LA sensor relationship graph."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from utils.charts import speed_map
from utils.load import render_disclaimer, require_bundle, sensor_options

st.set_page_config(page_title="Network explorer · TrafficFlow", layout="wide")
bundle = require_bundle()

st.title("Network explorer")
st.write(
    "Each node is a loop detector. Edges are the k-nearest road-network neighbors "
    "used for routing. This is not a complete street map."
)

timestamp = st.selectbox(
    "Timestamp",
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
    speed_map(bundle.sensor_metadata, speeds, f"Observed speed at {timestamp}"),
    use_container_width=True,
)

options = sensor_options(bundle)
sensor_label = st.selectbox("Inspect sensor", list(options.keys()))
sensor_id = options[sensor_label]
meta = bundle.sensor_metadata.set_index("sensor_id").loc[sensor_id]
neighbors = bundle.neighbors.get(sensor_id, [])
c1, c2, c3 = st.columns(3)
c1.metric("Free-flow speed", f"{meta['free_flow_mph']:.1f} mph")
c2.metric("Graph degree", str(int(meta["degree"])))
c3.metric("Region", str(meta["region"]))
st.write("Neighbors:", ", ".join(neighbors) if neighbors else "None")
st.line_chart(bundle.speeds[sensor_id].rename("Speed (mph)"))

g = bundle.graph
st.markdown("#### Graph facts")
st.write(
    {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "note": bundle.manifest.get("graph_notes"),
    }
)
render_disclaimer()
