"""Dijkstra routing on the sensor relationship graph.

Paths connect loop detectors. They are not turn-by-turn road geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd

from trafficflow.routing.weights import (
    SPEED_FLOOR_MPH,
    assign_travel_time_weights,
    combine_endpoint_speeds,
    travel_time_minutes,
)

STRATEGIES = ("static", "current", "predictive", "oracle")


@dataclass
class RouteResult:
    origin: str
    destination: str
    strategy: str
    path: list[str]
    n_edges: int
    distance_miles: float
    estimated_minutes: float
    realized_minutes: float | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = list(self.path)
        return payload


class RoutingError(ValueError):
    """Raised for invalid user-facing routing inputs."""


def shortest_path_route(
    graph: nx.DiGraph,
    origin: str,
    destination: str,
    *,
    weight: str = "travel_time_min",
) -> list[str]:
    """Return a node list for the least-cost path.

    Raises
    ------
    RoutingError
        If origin equals destination or no path exists.
    """
    origin = str(origin)
    destination = str(destination)
    if origin == destination:
        raise RoutingError("Please choose different origin and destination nodes.")
    if origin not in graph or destination not in graph:
        raise RoutingError("Origin or destination is not in the sensor network.")
    try:
        return nx.shortest_path(graph, origin, destination, weight=weight)
    except nx.NetworkXNoPath as exc:
        raise RoutingError(
            "No valid path exists between these locations in the available sensor network."
        ) from exc


def path_distance_miles(graph: nx.DiGraph, path: list[str]) -> float:
    """Sum edge ``distance_miles`` along a node path."""
    total = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        total += float(graph[src][dst]["distance_miles"])
    return total


def path_estimated_minutes(graph: nx.DiGraph, path: list[str], *, weight: str = "travel_time_min") -> float:
    """Sum weighted travel time along a path using current edge weights."""
    total = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        total += float(graph[src][dst][weight])
    return total


def realized_travel_time_minutes(
    graph: nx.DiGraph,
    path: list[str],
    speeds: pd.DataFrame,
    departure: pd.Timestamp,
    *,
    free_flow_by_sensor: Mapping[str, float] | None = None,
    speed_floor_mph: float = SPEED_FLOOR_MPH,
) -> float:
    """Replay a path against actual future speeds.

    The vehicle is assumed to start at ``departure``. After each edge the
    clock advances by that edge's realized travel time and the next edge
    uses the nearest later timestamp in ``speeds``.
    """
    if len(path) < 2:
        return 0.0
    free_flow = free_flow_by_sensor or {}
    index = speeds.index
    if departure not in index:
        position = int(index.searchsorted(departure))
        if position >= len(index):
            raise KeyError(f"Departure {departure} is after the demo period.")
        current_time = index[position]
    else:
        current_time = departure

    total = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        row = speeds.loc[current_time]
        fallback_vals = [free_flow.get(str(src), np.nan), free_flow.get(str(dst), np.nan)]
        finite = [float(v) for v in fallback_vals if v is not None and np.isfinite(v)]
        fallback = float(np.mean(finite)) if finite else float("nan")
        speed = combine_endpoint_speeds(
            float(row[src]) if src in row.index and pd.notna(row[src]) else float("nan"),
            float(row[dst]) if dst in row.index and pd.notna(row[dst]) else float("nan"),
            fallback_mph=fallback if np.isfinite(fallback) else speed_floor_mph,
        )
        minutes = travel_time_minutes(
            float(graph[src][dst]["distance_miles"]),
            speed,
            speed_floor_mph=speed_floor_mph,
        )
        total += minutes
        next_time = current_time + pd.Timedelta(minutes=minutes)
        next_pos = int(index.searchsorted(next_time))
        if next_pos >= len(index):
            current_time = index[-1]
        else:
            current_time = index[next_pos]
    return total


def route_with_speeds(
    graph: nx.DiGraph,
    origin: str,
    destination: str,
    speed_by_sensor: Mapping[str, float],
    *,
    strategy: str,
    realized_speeds: pd.DataFrame | None = None,
    departure: pd.Timestamp | None = None,
    free_flow_by_sensor: Mapping[str, float] | None = None,
    speed_floor_mph: float = SPEED_FLOOR_MPH,
) -> RouteResult:
    """Compute a route from a speed snapshot and optionally score it on realized traffic."""
    weighted = assign_travel_time_weights(
        graph,
        speed_by_sensor,
        free_flow_by_sensor=free_flow_by_sensor,
        speed_floor_mph=speed_floor_mph,
    )
    path = shortest_path_route(weighted, origin, destination)
    estimated = path_estimated_minutes(weighted, path)
    realized = None
    if realized_speeds is not None and departure is not None:
        realized = realized_travel_time_minutes(
            graph,
            path,
            realized_speeds,
            departure,
            free_flow_by_sensor=free_flow_by_sensor,
            speed_floor_mph=speed_floor_mph,
        )
    return RouteResult(
        origin=str(origin),
        destination=str(destination),
        strategy=strategy,
        path=path,
        n_edges=max(len(path) - 1, 0),
        distance_miles=path_distance_miles(graph, path),
        estimated_minutes=estimated,
        realized_minutes=realized,
        notes=[
            "Paths connect traffic sensors, not verified turn-by-turn road geometry.",
        ],
    )
