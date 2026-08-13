"""Convert traffic speeds into non-negative travel-time edge weights.

Units
-----
distance : miles
speed    : miles per hour (mph)
output   : minutes
"""

from __future__ import annotations

from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd

SPEED_FLOOR_MPH = 5.0


def travel_time_minutes(
    distance_miles: float,
    speed_mph: float,
    *,
    speed_floor_mph: float = SPEED_FLOOR_MPH,
) -> float:
    """Return travel time in minutes.

    Speeds below ``speed_floor_mph`` are clipped so edge weights stay
    finite and strictly positive. The floor is a routing safeguard, not a
    claim that traffic never moves more slowly.
    """
    if distance_miles < 0:
        raise ValueError("distance_miles must be non-negative.")
    if speed_floor_mph <= 0:
        raise ValueError("speed_floor_mph must be positive.")
    if not np.isfinite(distance_miles):
        raise ValueError("distance_miles must be finite.")
    effective = max(float(speed_mph) if np.isfinite(speed_mph) else 0.0, speed_floor_mph)
    return (float(distance_miles) / effective) * 60.0


def combine_endpoint_speeds(
    speed_u: float | None,
    speed_v: float | None,
    *,
    fallback_mph: float,
) -> float:
    """Average available endpoint speeds; otherwise use a fallback.

    Parameters
    ----------
    speed_u, speed_v:
        Speeds in mph. Non-finite or non-positive values are ignored.
    fallback_mph:
        Used when both endpoints are missing, typically free-flow speed.
    """
    values = [
        float(value)
        for value in (speed_u, speed_v)
        if value is not None and np.isfinite(value) and value > 0
    ]
    if values:
        return float(np.mean(values))
    if not np.isfinite(fallback_mph) or fallback_mph <= 0:
        return SPEED_FLOOR_MPH
    return float(fallback_mph)


def assign_travel_time_weights(
    graph: nx.DiGraph,
    speed_by_sensor: Mapping[str, float],
    *,
    free_flow_by_sensor: Mapping[str, float] | None = None,
    speed_floor_mph: float = SPEED_FLOOR_MPH,
    weight_attr: str = "travel_time_min",
) -> nx.DiGraph:
    """Return a copy of ``graph`` with travel-time weights in minutes.

    Each edge uses the mean of its endpoint speeds. Missing observations
    fall back to the edge's mean free-flow speed, then to the floor.
    """
    weighted = graph.copy()
    free_flow = free_flow_by_sensor or {}
    for src, dst, data in weighted.edges(data=True):
        distance_miles = float(data.get("distance_miles", 0.0))
        fallback_vals = [
            free_flow.get(str(src), np.nan),
            free_flow.get(str(dst), np.nan),
        ]
        finite = [float(v) for v in fallback_vals if v is not None and np.isfinite(v)]
        fallback = float(np.mean(finite)) if finite else float("nan")
        speed = combine_endpoint_speeds(
            speed_by_sensor.get(str(src)),
            speed_by_sensor.get(str(dst)),
            fallback_mph=fallback if np.isfinite(fallback) else speed_floor_mph,
        )
        data[weight_attr] = travel_time_minutes(
            distance_miles,
            speed,
            speed_floor_mph=speed_floor_mph,
        )
        data["speed_mph"] = max(speed, speed_floor_mph)
    return weighted


def speed_lookup(frame: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, float]:
    """Return {sensor_id: speed_mph} for a single timestamp."""
    if timestamp not in frame.index:
        raise KeyError(f"Timestamp {timestamp} is not in the speed matrix.")
    row = frame.loc[timestamp]
    return {str(col): float(val) if pd.notna(val) else float("nan") for col, val in row.items()}
