"""Sensor-level forecast explorer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from styles.theme import apply_theme, callout, footer, metric_card, metrics_row, page_header, sidebar_brand
from utils.charts import forecast_chart
from utils.load import require_bundle, sensor_options

st.set_page_config(page_title="Traffic forecast · TrafficFlow", page_icon="◇", layout="wide")
apply_theme()
sidebar_brand()
bundle = require_bundle()
options = sensor_options(bundle)
labels = list(options.keys())

page_header(
    "Traffic forecast",
    "See how conditions are expected to evolve, and whether neighboring sensors improve the prediction.",
    badges=[("15 / 30 / 60 min", "blue"), ("Temporal vs spatial", "teal")],
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
    callout(
        "Traffic data is unavailable for this time.",
        "Try another historical departure inside the demo window.",
        "warn",
    )
    footer(bundle.manifest)
    st.stop()

row = subset.iloc[0]
pred_index = pd.DatetimeIndex([future_end])
temporal_series = pd.Series([row["pred_temporal"]], index=pred_index)
spatial_series = pd.Series([row["pred_spatial"]], index=pred_index)

actual = row["target"]
metrics_row(
    [
        metric_card(
            "Actual future speed",
            f"{actual:.1f} mph" if pd.notna(actual) else "—",
            "Observed at the horizon",
            "neutral",
        ),
        metric_card("Spatiotemporal", f"{row['pred_spatial']:.1f} mph", "Uses neighbor speeds", "blue"),
        metric_card("Temporal", f"{row['pred_temporal']:.1f} mph", "Own history + calendar", "info"),
        metric_card("Persistence", f"{row['pred_persistence']:.1f} mph", "Last observed speed", "neutral"),
    ]
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        forecast_chart(history, actual_future, spatial_series, "Predicted vs actual traffic speed · spatiotemporal"),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        forecast_chart(history, actual_future, temporal_series, "Predicted vs actual traffic speed · temporal"),
        use_container_width=True,
    )

neighbors = bundle.neighbors.get(sensor_id, [])
if pd.notna(actual):
    err_s = abs(float(row["pred_spatial"]) - float(actual))
    err_t = abs(float(row["pred_temporal"]) - float(actual))
    err_p = abs(float(row["pred_persistence"]) - float(actual))
    callout(
        "Absolute forecast error on this point",
        f"Spatiotemporal {err_s:.2f} mph · temporal {err_t:.2f} mph · persistence {err_p:.2f} mph.",
        "info",
    )

st.markdown("#### Nearby sensors used as spatial context")
st.write(", ".join(neighbors[:8]) if neighbors else "No graph neighbors stored for this sensor.")

footer(bundle.manifest)
