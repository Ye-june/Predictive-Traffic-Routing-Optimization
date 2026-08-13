"""Plotly helpers aligned with the TrafficFlow product theme."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROUTE_COLORS = {
    "static": "#64717D",
    "current": "#0F8B8D",
    "predictive": "#2563EB",
    "oracle": "#D98C4A",
}
ROUTE_LABELS = {
    "static": "Static",
    "current": "Current-state",
    "predictive": "Predictive",
    "oracle": "Oracle (hindsight)",
}

_LAYOUT = {
    "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#17212B", "size": 13},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#FFFFFF",
    "margin": {"l": 48, "r": 18, "t": 56, "b": 48},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
}


def _apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Merge theme defaults with per-chart options without duplicate kwargs."""
    layout = dict(_LAYOUT)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def _axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F4", zeroline=False, linecolor="#E4E8EC")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F4", zeroline=False, linecolor="#E4E8EC")
    return fig


def speed_map(metadata: pd.DataFrame, speeds: dict[str, float], title: str) -> go.Figure:
    frame = metadata.copy()
    frame["speed_mph"] = frame["sensor_id"].map(lambda sid: speeds.get(str(sid)))
    fig = px.scatter(
        frame,
        x="longitude",
        y="latitude",
        color="speed_mph",
        color_continuous_scale=["#C94C4C", "#D49A2A", "#2E8B57"],
        hover_name="label",
        title=title,
        labels={"speed_mph": "Speed (mph)", "longitude": "Longitude", "latitude": "Latitude"},
    )
    fig.update_traces(marker={"size": 8, "line": {"width": 0}})
    _apply_layout(fig, height=520, coloraxis_colorbar_title="mph")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _axes(fig)


def network_preview(metadata: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metadata["longitude"],
            y=metadata["latitude"],
            mode="markers",
            marker={"size": 7, "color": "#2563EB", "opacity": 0.78},
            hovertext=metadata["label"],
            hoverinfo="text",
            name="Sensors",
        )
    )
    _apply_layout(
        fig,
        title="METR-LA sensor network",
        height=320,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        showlegend=False,
        margin={"l": 40, "r": 16, "t": 48, "b": 40},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _axes(fig)


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
            marker={"size": 5, "color": "#C5CDD4"},
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
                line={"color": ROUTE_COLORS[strategy], "width": 3.5},
                name=ROUTE_LABELS[strategy],
            )
        )
    if origin in lookup.index:
        fig.add_trace(
            go.Scatter(
                x=[float(lookup.loc[origin, "longitude"])],
                y=[float(lookup.loc[origin, "latitude"])],
                mode="markers",
                marker={"size": 13, "color": "#2563EB"},
                name="Origin",
            )
        )
    if destination in lookup.index:
        fig.add_trace(
            go.Scatter(
                x=[float(lookup.loc[destination, "longitude"])],
                y=[float(lookup.loc[destination, "latitude"])],
                mode="markers",
                marker={"size": 13, "color": "#D98C4A"},
                name="Destination",
            )
        )
    _apply_layout(
        fig,
        title="Network route overlay",
        height=540,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _axes(fig)


def forecast_chart(
    history: pd.Series,
    actual_future: pd.Series,
    predicted: pd.Series,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=history.index, y=history.values, name="Recent observed", line={"color": "#17212B", "width": 2})
    )
    fig.add_trace(
        go.Scatter(x=actual_future.index, y=actual_future.values, name="Actual future", line={"color": "#C94C4C", "width": 2})
    )
    fig.add_trace(
        go.Scatter(
            x=predicted.index,
            y=predicted.values,
            name="Predicted",
            line={"color": "#2563EB", "width": 2, "dash": "dash"},
        )
    )
    _apply_layout(fig, title=title, xaxis_title="Time", yaxis_title="Speed (mph)", height=380)
    return _axes(fig)


def styled_bar(frame: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str) -> go.Figure:
    fig = px.bar(frame, x=x, y=y, title=title, labels={x: xlabel, y: ylabel}, color_discrete_sequence=["#2563EB"])
    _apply_layout(fig, height=360)
    return _axes(fig)


def styled_hist(frame: pd.DataFrame, x: str, color: str, title: str, xlabel: str) -> go.Figure:
    fig = px.histogram(
        frame,
        x=x,
        color=color,
        nbins=20,
        title=title,
        labels={x: xlabel},
        color_discrete_sequence=["#2563EB", "#0F8B8D", "#D98C4A"],
    )
    _apply_layout(fig, height=380, bargap=0.08)
    return _axes(fig)
