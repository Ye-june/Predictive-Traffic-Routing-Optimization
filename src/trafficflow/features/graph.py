"""Construct and validate the METR-LA sensor relationship graph.

Edges represent published road-network distances between loop detectors,
not a complete street-level map. Geographic coordinates are stored on
nodes for visualization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import pandas as pd

from trafficflow.utils.logging import get_logger

logger = get_logger(__name__)

METERS_PER_MILE = 1609.344


@dataclass
class GraphSummary:
    n_nodes: int
    n_edges: int
    n_weakly_connected_components: int
    n_strongly_connected_components: int
    n_isolated_nodes: int
    average_degree: float
    density: float
    largest_weak_component_size: int
    largest_strong_component_size: int
    is_weakly_connected: bool
    is_strongly_connected: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def meters_to_miles(meters: float) -> float:
    """Convert meters to miles."""
    return float(meters) / METERS_PER_MILE


def build_sensor_distance_graph(
    sensor_ids: list[str],
    locations: pd.DataFrame,
    distances: pd.DataFrame,
    *,
    max_distance_m: float | None = None,
) -> nx.DiGraph:
    """Build a directed sensor graph from DCRNN pairwise distances.

    Parameters
    ----------
    sensor_ids:
        Ordered METR-LA sensor identifiers.
    locations:
        Table with ``sensor_id``, ``latitude``, ``longitude``.
    distances:
        Table with ``from``, ``to``, ``cost`` where ``cost`` is meters.
    max_distance_m:
        Optional additional cutoff in meters. ``None`` keeps every
        METR-LA pair present in the distance file.

    Returns
    -------
    networkx.DiGraph
        Nodes are sensor ID strings. Edge attributes:

        * ``distance_m`` — road-network distance in meters
        * ``distance_miles`` — same distance in miles
    """
    graph = nx.DiGraph()
    loc_lookup = locations.set_index(locations["sensor_id"].astype(str))
    for sensor_id in sensor_ids:
        attrs: dict[str, Any] = {"sensor_id": sensor_id}
        if sensor_id in loc_lookup.index:
            row = loc_lookup.loc[sensor_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            attrs["latitude"] = float(row["latitude"])
            attrs["longitude"] = float(row["longitude"])
        graph.add_node(sensor_id, **attrs)

    id_set = set(sensor_ids)
    from_ids = distances["from"].astype(str)
    to_ids = distances["to"].astype(str)
    mask = from_ids.isin(id_set) & to_ids.isin(id_set) & (from_ids != to_ids)
    if max_distance_m is not None:
        mask = mask & (distances["cost"] <= max_distance_m)

    subset = distances.loc[mask, ["from", "to", "cost"]]
    for src, dst, cost in subset.itertuples(index=False, name=None):
        src_id, dst_id = str(src), str(dst)
        distance_m = float(cost)
        graph.add_edge(
            src_id,
            dst_id,
            distance_m=distance_m,
            distance_miles=meters_to_miles(distance_m),
        )

    logger.info(
        "Built directed sensor graph: %s nodes, %s edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def k_nearest_neighbors(
    graph: nx.DiGraph,
    *,
    k: int,
    weight: str = "distance_m",
) -> dict[str, list[str]]:
    """Return the ``k`` nearest successors for each node.

    Neighbors are ranked by edge weight. Isolated nodes map to an empty list.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    neighbors: dict[str, list[str]] = {}
    for node in graph.nodes:
        incident = []
        for _, dst, data in graph.out_edges(node, data=True):
            incident.append((float(data[weight]), str(dst)))
        incident.sort(key=lambda item: item[0])
        neighbors[str(node)] = [dst for _, dst in incident[:k]]
    return neighbors


def summarize_graph(graph: nx.DiGraph) -> GraphSummary:
    """Compute connectivity statistics used before routing experiments."""
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    weak = list(nx.weakly_connected_components(graph))
    strong = list(nx.strongly_connected_components(graph))
    isolated = [node for node, degree in graph.degree() if degree == 0]
    density = nx.density(graph)
    avg_degree = (2.0 * n_edges / n_nodes) if n_nodes else 0.0
    notes = [
        "Graph nodes are loop detectors, not intersections of a full road map.",
        "Edge distances are DCRNN road-network costs in meters.",
    ]
    if not nx.is_strongly_connected(graph):
        notes.append(
            "Graph is not strongly connected; origin-destination sampling "
            "must stay inside a reachable component."
        )

    summary = GraphSummary(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_weakly_connected_components=len(weak),
        n_strongly_connected_components=len(strong),
        n_isolated_nodes=len(isolated),
        average_degree=avg_degree,
        density=float(density),
        largest_weak_component_size=max((len(c) for c in weak), default=0),
        largest_strong_component_size=max((len(c) for c in strong), default=0),
        is_weakly_connected=bool(nx.is_weakly_connected(graph)) if n_nodes else False,
        is_strongly_connected=bool(nx.is_strongly_connected(graph)) if n_nodes else False,
        notes=notes,
    )
    logger.info(
        "Graph summary: nodes=%s edges=%s weak_cc=%s strong_cc=%s isolated=%s",
        summary.n_nodes,
        summary.n_edges,
        summary.n_weakly_connected_components,
        summary.n_strongly_connected_components,
        summary.n_isolated_nodes,
    )
    return summary
