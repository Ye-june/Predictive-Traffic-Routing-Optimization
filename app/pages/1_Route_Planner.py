"""Historical replay route planner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from trafficflow.routing.engine import RoutingError
from trafficflow.serving import compare_routes
from utils.charts import ROUTE_LABELS, route_map
from utils.load import format_miles, format_minutes, render_disclaimer, require_bundle, sensor_options

st.set_page_config(page_title="Route planner · TrafficFlow", layout="wide")
bundle = require_bundle()
options = sensor_options(bundle)
labels = list(options.keys())
id_to_label = {value: key for key, value in options.items()}

st.title("Route planner")
st.caption("Historical Replay Mode — choose a past departure, then compare routes on realized traffic.")

scenarios = bundle.scenarios
example_lookup = {
    "Morning rush": "morning_rush",
    "Evening rush": "evening_rush",
    "Off-peak": "off_peak",
}

with st.sidebar:
    st.header("Trip")
    example = st.radio("Quick example", ["Custom"] + list(example_lookup), index=1)
    default_origin = labels[0]
    default_dest = labels[min(20, len(labels) - 1)]
    default_time = bundle.speeds.index[len(bundle.speeds) // 2]
    if example != "Custom" and not scenarios.empty:
        subset = scenarios[scenarios["traffic_condition"] == example_lookup[example]]
        if not subset.empty:
            row = subset.iloc[0]
            default_origin = id_to_label.get(str(row["origin"]), default_origin)
            default_dest = id_to_label.get(str(row["destination"]), default_dest)
            default_time = pd.Timestamp(row["departure_time"])
    origin_label = st.selectbox("Origin", labels, index=labels.index(default_origin) if default_origin in labels else 0)
    dest_label = st.selectbox(
        "Destination",
        labels,
        index=labels.index(default_dest) if default_dest in labels else 1,
    )
    departure = st.selectbox(
        "Departure time",
        list(bundle.speeds.index),
        index=int(bundle.speeds.index.get_indexer([default_time], method="nearest")[0]),
        format_func=lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
    )
    horizon = st.selectbox("Forecast horizon", [15, 30, 60], index=1)
    model_key = st.selectbox(
        "Forecast model",
        ["spatial", "temporal", "persistence", "historical"],
        format_func=lambda key: {
            "spatial": "Spatiotemporal XGBoost",
            "temporal": "Temporal XGBoost",
            "persistence": "Persistence",
            "historical": "Historical weekday-hour mean",
        }[key],
    )
    show = st.multiselect(
        "Show on map",
        ["static", "current", "predictive"],
        default=["static", "predictive"],
        format_func=lambda key: ROUTE_LABELS[key],
    )
    run = st.button("Calculate routes", type="primary", use_container_width=True)

origin = options[origin_label]
destination = options[dest_label]

if "route_results" not in st.session_state:
    st.session_state.route_results = None

if run:
    with st.spinner("Generating traffic forecast and calculating routes..."):
        try:
            st.session_state.route_results = compare_routes(
                bundle,
                origin,
                destination,
                pd.Timestamp(departure),
                horizon_minutes=int(horizon),
                forecast_model=str(model_key),
            )
            st.session_state.route_meta = {
                "origin": origin,
                "destination": destination,
                "departure": str(pd.Timestamp(departure)),
                "horizon": int(horizon),
                "model": str(model_key),
            }
        except (RoutingError, KeyError) as exc:
            st.session_state.route_results = None
            st.error(str(exc))

results = st.session_state.route_results
if results is None:
    st.info("Choose origin, destination, and departure, then calculate routes.")
    render_disclaimer()
    st.stop()

pred = results["predictive"]
static = results["static"]
current = results["current"]
oracle = results["oracle"]

st.markdown("### Predictive route")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated time", format_minutes(pred.estimated_minutes))
k2.metric("Realized time", format_minutes(pred.realized_minutes))
k3.metric("Distance", format_miles(pred.distance_miles))
k4.metric("Segments", str(pred.n_edges))

if static.realized_minutes and pred.realized_minutes:
    saved = static.realized_minutes - pred.realized_minutes
    st.metric("Realized savings vs static", format_minutes(saved))

st.markdown("### Strategy comparison")
st.caption("Estimated uses the strategy's own costs. Realized replays the chosen path on actual future speeds.")
table = pd.DataFrame(
    [
        {
            "Strategy": ROUTE_LABELS[name],
            "Estimated (min)": round(result.estimated_minutes, 2),
            "Realized (min)": None if result.realized_minutes is None else round(result.realized_minutes, 2),
            "Distance (mi)": round(result.distance_miles, 2),
            "Segments": result.n_edges,
        }
        for name, result in results.items()
    ]
)
st.dataframe(table, hide_index=True, use_container_width=True)

paths = {name: result.path for name, result in results.items()}
st.plotly_chart(
    route_map(
        bundle.sensor_metadata,
        bundle.graph,
        paths,
        origin,
        destination,
        visible=show or ["predictive"],
    ),
    use_container_width=True,
)
st.warning(
    "Routes represent connectivity between traffic sensor locations and should not be "
    "interpreted as turn-by-turn road navigation."
)

with st.expander("Technical details"):
    meta = st.session_state.get("route_meta", {})
    st.write(
        {
            "model": meta.get("model"),
            "forecast_horizon_minutes": meta.get("horizon"),
            "departure": meta.get("departure"),
            "routing_algorithm": "NetworkX Dijkstra",
            "edge_weight": "minutes = miles / max(speed_mph, 5) × 60",
            "nodes": bundle.manifest.get("num_nodes"),
            "predictive_path": pred.path,
        }
    )

render_disclaimer()
