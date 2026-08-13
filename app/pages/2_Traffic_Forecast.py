"""Sensor-level forecast explorer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from utils.charts import forecast_chart
from utils.load import render_disclaimer, require_bundle, sensor_options

st.set_page_config(page_title="Traffic forecast · TrafficFlow", layout="wide")
bundle = require_bundle()
options = sensor_options(bundle)
labels = list(options.keys())

st.title("Traffic forecast")
st.write(
    "Traffic at one location depends on its own recent history and on nearby sensors. "
    "Compare a temporal model with a neighbor-aware model on the same timestamp."
)

c1, c2, c3 = st.columns(3)
sensor_label = c1.selectbox("Sensor", labels)
horizon = c2.selectbox("Horizon (minutes)", [15, 30, 60], index=1)
departure = c3.selectbox(
    "Forecast issued at",
    list(bundle.speeds.index),
    index=min(48, len(bundle.speeds) - 1),
    format_func=lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
)
sensor_id = options[sensor_label]
departure = pd.Timestamp(departure)
horizon = int(horizon)

history = bundle.speeds[sensor_id].loc[:departure].tail(36)
future_start = departure
future_end = departure + pd.Timedelta(minutes=horizon)
actual_future = bundle.speeds[sensor_id].loc[future_start:future_end]

subset = bundle.forecasts[
    (bundle.forecasts["sensor_id"] == sensor_id)
    & (bundle.forecasts["horizon_minutes"] == horizon)
    & (bundle.forecasts["timestamp"] == departure)
]

if subset.empty:
    st.error("Forecast data is unavailable for the selected timestamp.")
    render_disclaimer()
    st.stop()

row = subset.iloc[0]
pred_index = pd.DatetimeIndex([future_end])
temporal_series = pd.Series([row["pred_temporal"]], index=pred_index)
spatial_series = pd.Series([row["pred_spatial"]], index=pred_index)
persist_series = pd.Series([row["pred_persistence"]], index=pred_index)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        forecast_chart(history, actual_future, spatial_series, "Spatiotemporal XGBoost"),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        forecast_chart(history, actual_future, temporal_series, "Temporal XGBoost"),
        use_container_width=True,
    )

actual = row["target"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Actual future speed", f"{actual:.1f} mph" if pd.notna(actual) else "—")
m2.metric("Spatiotemporal", f"{row['pred_spatial']:.1f} mph")
m3.metric("Temporal", f"{row['pred_temporal']:.1f} mph")
m4.metric("Persistence", f"{row['pred_persistence']:.1f} mph")

neighbors = bundle.neighbors.get(sensor_id, [])
meta = bundle.sensor_metadata.set_index("sensor_id")
st.markdown("#### Nearby sensors used as spatial context")
if neighbors:
    st.write(", ".join(neighbors[:8]))
else:
    st.write("No graph neighbors stored for this sensor.")

if pd.notna(actual):
    err_s = abs(float(row["pred_spatial"]) - float(actual))
    err_t = abs(float(row["pred_temporal"]) - float(actual))
    st.caption(f"Absolute error · spatial {err_s:.2f} mph · temporal {err_t:.2f} mph · persistence {abs(float(row['pred_persistence'])-float(actual)):.2f} mph")

render_disclaimer()
