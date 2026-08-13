"""Routing weight and path tests."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from trafficflow.routing.engine import RoutingError, realized_travel_time_minutes, shortest_path_route
from trafficflow.routing.weights import assign_travel_time_weights, travel_time_minutes


def test_travel_time_units() -> None:
    minutes = travel_time_minutes(distance_miles=10.0, speed_mph=60.0)
    assert abs(minutes - 10.0) < 1e-9


def test_speed_floor_prevents_zero_division() -> None:
    minutes = travel_time_minutes(distance_miles=1.0, speed_mph=0.0, speed_floor_mph=5.0)
    assert minutes == 12.0


def test_negative_distance_rejected() -> None:
    with pytest.raises(ValueError):
        travel_time_minutes(-1.0, 30.0)


def test_same_origin_destination_message() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", travel_time_min=1.0, distance_miles=1.0)
    with pytest.raises(RoutingError, match="different origin"):
        shortest_path_route(graph, "a", "a")


def test_disconnected_route_message() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", travel_time_min=1.0, distance_miles=1.0)
    graph.add_node("c")
    with pytest.raises(RoutingError, match="No valid path"):
        shortest_path_route(graph, "a", "c")


def test_weights_are_positive() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", distance_miles=2.0)
    weighted = assign_travel_time_weights(graph, {"a": 40.0, "b": 40.0}, speed_floor_mph=5.0)
    assert weighted["a"]["b"]["travel_time_min"] > 0


def test_realized_replay_advances_in_time() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", distance_miles=1.0)
    index = pd.date_range("2012-06-21 08:00", periods=6, freq="5min")
    speeds = pd.DataFrame({"a": np.full(6, 60.0), "b": np.full(6, 60.0)}, index=index)
    minutes = realized_travel_time_minutes(graph, ["a", "b"], speeds, index[0])
    assert abs(minutes - 1.0) < 1e-9
