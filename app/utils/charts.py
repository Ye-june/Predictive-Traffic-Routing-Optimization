"""Plotly helpers for the Streamlit app."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROUTE_COLORS = {
    "static": "#6c757d",
    "current": "#e09f3e",
    "predictive": "#0e7c66",
    "oracle": "#6d597a",
}
ROUTE_LABELS = {
    "static": "Static",
    "current": "Current-state",
    "predictive": "Predictive",
    "oracle": "Oracle (hindsight)",
}


def speed_map(metadata: pd.DataFrame, speeds: dict[str, float], title: str) -> go.Figure:
    frame = metadata.copy()
    frame["speed_mph"] = frame["sensor_id"].map(lambda sid: speeds.get(str(sid)))
    fig = px.scatter(
        frame,
        x="longitude",
        y="latitude",
        color="speed_mph",
        color_continuous_scale="RdYlGn",
        hover_name="label",
        title=title,
        labels={"speed_mph": "Speed (mph)", "longitude": "Longitude", "latitude": "Latitude"},
    )
    fig.update_traces(marker={"size": 8})
    fig.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        coloraxis_colorbar_title="mph",
        plot_bgcolor="#f7f4ef",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def route_map(
    metadata: pd.DataFrame,
    graph,
    routes: dict[str, list[str]],
    origin: str,
    destination: str,
    visible: Iterable[str],
) -> go.Figure:
    lookup = metadata.set_index("sensor_id")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metadata["longitude"],
            y=metadata["latitude"],
            mode="markers",
            marker={"size": 6, "color": "#b8b2a7"},
            name="Sensors",
            hovertext=metadata["label"],
            hoverinfo="text",
        )
    )
    for strategy in visible:
        path = routes.get(strategy) or []
        if len(path) < 2:
            continue
        xs: list[float | None] = []
        ys: list[float | None] = []
        for src, dst in zip(path[:-1], path[1:]):
            if src not in lookup.index or dst not in lookup.index:
                continue
            xs.extend([float(lookup.loc[src, "longitude"]), float(lookup.loc[dst, "longitude"]), None])
            ys.extend([float(lookup.loc[src, "latitude"]), float(lookup.loc[dst, "latitude"]), None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": ROUTE_COLORS[strategy], "width": 4},
                name=ROUTE_LABELS[strategy],
            )
        )
    for node, color, name in ((origin, "#1d3557", "Origin"), (destination, "#9b2226", "Destination")):
        if node in lookup.index:
            fig.add_trace(
                go.Scatter(
                    x=[float(lookup.loc[node, "longitude"])],
                    y=[float(lookup.loc[node, "latitude"])],
                    mode="markers",
                    marker={"size": 14, "color": color, "symbol": "diamond"},
                    name=name,
                )
            )
    fig.update_layout(
        title="Sensor-network routes (not turn-by-turn road geometry)",
        height=540,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h"},
        plot_bgcolor="#f7f4ef",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def forecast_chart(
    history: pd.Series,
    actual_future: pd.Series,
    predicted: pd.Series,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Recent observed", line={"color": "#1d3557"}))
    fig.add_trace(
        go.Scatter(x=actual_future.index, y=actual_future.values, name="Actual future", line={"color": "#9b2226"})
    )
    fig.add_trace(
        go.Scatter(
            x=predicted.index,
            y=predicted.values,
            name="Predicted",
            line={"color": "#0e7c66", "dash": "dash"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Speed (mph)",
        height=380,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h"},
    )
    return fig
