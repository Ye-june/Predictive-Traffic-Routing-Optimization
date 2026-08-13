"""Graph construction tests on a synthetic directed network."""

from __future__ import annotations

import pandas as pd

from trafficflow.features.graph import (
    build_sensor_distance_graph,
    k_nearest_neighbors,
    meters_to_miles,
    summarize_graph,
)


def test_meters_to_miles() -> None:
    assert abs(meters_to_miles(1609.344) - 1.0) < 1e-9


def test_build_and_summarize_synthetic_graph() -> None:
    sensor_ids = ["a", "b", "c"]
    locations = pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "latitude": [34.0, 34.01, 34.02],
            "longitude": [-118.0, -118.01, -118.02],
        }
    )
    distances = pd.DataFrame(
        {
            "from": ["a", "b", "b", "c"],
            "to": ["b", "a", "c", "b"],
            "cost": [100.0, 110.0, 200.0, 210.0],
        }
    )
    graph = build_sensor_distance_graph(sensor_ids, locations, distances)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 4
    assert graph["a"]["b"]["distance_m"] == 100.0
    summary = summarize_graph(graph)
    assert summary.n_isolated_nodes == 0
    assert summary.is_weakly_connected
    neighbors = k_nearest_neighbors(graph, k=1)
    assert neighbors["a"] == ["b"]
    assert neighbors["b"][0] in {"a", "c"}
