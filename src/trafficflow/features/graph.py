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


def build_knn_routing_graph(
    full_graph: nx.DiGraph,
    *,
    k: int = 8,
    max_distance_m: float = 8000.0,
) -> nx.DiGraph:
    """Build a sparse directed k-NN graph suitable for shortest-path routing.

    The full DCRNN distance table links almost every nearby detector pair,
    which produces unrealistic "teleport" shortcuts. Routing therefore uses
    only the ``k`` nearest outgoing edges within ``max_distance_m``.

    If the k-NN graph is not weakly connected, additional shortest unused
    edges are added until a single weak component remains. Reverse edges are
    added when needed so the largest strongly connected component covers
    nearly all sensors.

    Parameters
    ----------
    full_graph:
        Directed graph with ``distance_m`` / ``distance_miles`` on edges.
    k:
        Number of nearest neighbors retained per node.
    max_distance_m:
        Maximum allowed neighbor distance in meters (8 km default).
    """
    routing = nx.DiGraph()
    routing.add_nodes_from(full_graph.nodes(data=True))
    for node in full_graph.nodes:
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for _, dst, data in full_graph.out_edges(node, data=True):
            distance_m = float(data["distance_m"])
            if distance_m <= max_distance_m:
                candidates.append((distance_m, str(dst), dict(data)))
        candidates.sort(key=lambda item: item[0])
        for _, dst, data in candidates[:k]:
            routing.add_edge(str(node), dst, **data)

    _connect_weak_components(full_graph, routing)
    _ensure_strong_connectivity(full_graph, routing)
    logger.info(
        "Built routing k-NN graph: %s nodes, %s edges (k=%s, max_m=%s)",
        routing.number_of_nodes(),
        routing.number_of_edges(),
        k,
        max_distance_m,
    )
    return routing


def _connect_weak_components(full_graph: nx.DiGraph, routing: nx.DiGraph) -> None:
    components = [set(part) for part in nx.weakly_connected_components(routing)]
    if len(components) <= 1:
        return
    remaining = sorted(components, key=len, reverse=True)
    main = remaining[0]
    for extra in remaining[1:]:
        best: tuple[float, str, str, dict[str, Any]] | None = None
        for src in extra:
            for _, dst, data in full_graph.out_edges(src, data=True):
                if dst in main:
                    distance_m = float(data["distance_m"])
                    if best is None or distance_m < best[0]:
                        best = (distance_m, str(src), str(dst), dict(data))
            for pred, _, data in full_graph.in_edges(src, data=True):
                if pred in main:
                    distance_m = float(data["distance_m"])
                    if best is None or distance_m < best[0]:
                        best = (distance_m, str(pred), str(src), dict(data))
        if best is None:
            continue
        _, src, dst, data = best
        routing.add_edge(src, dst, **data)
        main.update(extra)


def _ensure_strong_connectivity(full_graph: nx.DiGraph, routing: nx.DiGraph) -> None:
    if nx.is_strongly_connected(routing):
        return
    for src, dst, data in list(routing.edges(data=True)):
        if not routing.has_edge(dst, src):
            reverse = dict(data)
            routing.add_edge(dst, src, **reverse)
    logger.info(
        "Added reverse edges so routing graph is traversable in both directions. "
        "This is a connectivity approximation, not verified one-way road geometry."
    )


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
