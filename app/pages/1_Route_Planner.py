"""Historical replay route planner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

from styles.theme import (
    apply_theme,
    callout,
    empty_state,
    footer,
    metric_card,
    metrics_row,
    page_header,
    route_compare,
    sidebar_brand,
)
from trafficflow.routing.engine import RoutingError
from trafficflow.serving import compare_routes
from utils.charts import ROUTE_COLORS, ROUTE_LABELS, route_map
from utils.load import format_miles, format_minutes, require_bundle, sensor_options

st.set_page_config(page_title="Route planner · TrafficFlow", page_icon="◇", layout="wide")
apply_theme()
sidebar_brand()
bundle = require_bundle()
options = sensor_options(bundle)
labels = list(options.keys())
id_to_label = {value: key for key, value in options.items()}

page_header(
    "Route planner",
    "Choose a historical departure. TrafficFlow uses only information available at that moment, then scores the path on traffic that actually occurred.",
    badges=[("Historical Replay", "warm"), ("METR-LA", "blue"), ("Dijkstra routing", "teal")],
)

scenarios = bundle.scenarios
example_lookup = {
    "Morning rush": "morning_rush",
    "Evening rush": "evening_rush",
    "Off-peak": "off_peak",
}

with st.sidebar:
    st.markdown("**Plan your trip**")
    st.caption("Starting point and destination are loop detectors, labeled by region.")
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
    origin_label = st.selectbox(
        "Choose your starting point",
        labels,
        index=labels.index(default_origin) if default_origin in labels else 0,
    )
    dest_label = st.selectbox(
        "Choose your destination",
        labels,
        index=labels.index(default_dest) if default_dest in labels else 1,
    )
    departure = st.selectbox(
        "Departure time",
        list(bundle.speeds.index),
        index=int(bundle.speeds.index.get_indexer([default_time], method="nearest")[0]),
        format_func=lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
        help="Choose a point within the historical replay period.",
    )
    with st.expander("Advanced model settings", expanded=False):
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
            "Layers on map",
            ["static", "current", "predictive"],
            default=["static", "predictive"],
            format_func=lambda key: ROUTE_LABELS[key],
        )
    run = st.button("Calculate predictive route", type="primary", use_container_width=True)

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
            message = str(exc)
            if "different origin" in message.lower():
                callout("Choose two different sensors", message, "warn")
            elif "no valid path" in message.lower():
                callout(
                    "We couldn't build a route between these two nodes.",
                    "The selected sensors may belong to disconnected parts of the network. Try another destination.",
                    "warn",
                )
            else:
                callout(
                    "Traffic data is unavailable for this time.",
                    "Try another historical departure inside the demo window.",
                    "warn",
                )

results = st.session_state.route_results
if results is None:
    empty_state(
        "Ready when you are",
        "Choose an origin, destination, and departure time, then calculate a predictive route to compare strategies.",
    )
    footer(bundle.manifest)
    st.stop()

pred = results["predictive"]
static = results["static"]
current = results["current"]
oracle = results["oracle"]
meta = st.session_state.get("route_meta", {})

st.success("Route comparison ready")
metrics_row(
    [
        metric_card("Predictive estimate", format_minutes(pred.estimated_minutes), "Model-based travel time", "blue"),
        metric_card("Predictive realized", format_minutes(pred.realized_minutes), "Actual future traffic", "info"),
        metric_card("Distance", format_miles(pred.distance_miles), f"{pred.n_edges} sensor segments", "neutral"),
        metric_card("Horizon", f"{meta.get('horizon', '—')} min", "Forecast lead time", "neutral"),
    ]
)

if static.realized_minutes is not None and pred.realized_minutes is not None:
    saved = static.realized_minutes - pred.realized_minutes
    if saved > 0.05:
        callout(
            "Using actual future traffic observations",
            f"The predictive route completed {saved:.1f} minutes faster than the static route.",
            "good",
        )
    elif saved < -0.05:
        callout(
            "Using actual future traffic observations",
            f"The predictive route was {abs(saved):.1f} minutes slower than the static route on this trip.",
            "warn",
        )
    else:
        callout(
            "Using actual future traffic observations",
            "Realized travel time was essentially the same as the static route.",
            "info",
        )

st.markdown("#### Route comparison")
st.caption("Realized minutes replay each chosen path on observed future speeds. Estimated minutes use the strategy's own costs.")
realized_vals = [
    result.realized_minutes
    for result in results.values()
    if result.realized_minutes is not None
]
max_real = max(realized_vals) if realized_vals else 1.0
best_name = min(
    results,
    key=lambda name: results[name].realized_minutes
    if results[name].realized_minutes is not None
    else float("inf"),
)
compare_rows = []
for name, result in results.items():
    minutes = result.realized_minutes
    compare_rows.append(
        {
            "name": ROUTE_LABELS[name],
            "minutes": format_minutes(minutes),
            "width_pct": 12 if minutes is None else max(12.0, 100.0 * float(minutes) / max_real),
            "best": name == best_name,
            "color": ROUTE_COLORS[name],
        }
    )
route_compare(compare_rows)

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

st.markdown("#### Network route")
st.caption(
    f"Predicted conditions near {meta.get('departure', 'the selected departure')}. "
    "Lines connect sensors; they are not turn-by-turn road geometry."
)
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

why_bits = []
if pred.realized_minutes is not None and static.realized_minutes is not None:
    delta = static.realized_minutes - pred.realized_minutes
    if abs(pred.distance_miles - static.distance_miles) > 0.05:
        longer = "longer" if pred.distance_miles > static.distance_miles else "shorter"
        why_bits.append(
            f"The predictive path is {abs(pred.distance_miles - static.distance_miles):.1f} miles {longer} "
            f"({pred.n_edges} segments vs {static.n_edges})."
        )
    if current.realized_minutes is not None:
        vs_cur = current.realized_minutes - pred.realized_minutes
        if abs(vs_cur) > 0.05:
            cmp = "faster" if vs_cur > 0 else "slower"
            why_bits.append(
                f"Against the current-state route it was {abs(vs_cur):.1f} minutes {cmp} in replay."
            )
if why_bits:
    callout("Why this route looks different", " ".join(why_bits), "info")

with st.expander("Technical details"):
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

footer(bundle.manifest)
