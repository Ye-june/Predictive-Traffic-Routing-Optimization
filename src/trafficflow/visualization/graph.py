"""Sensor-network plots. Edges are detector relationships, not curb geometry."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_sensor_map(locations: pd.DataFrame, path: Path) -> Path:
    """Scatter plot of loop-detector coordinates."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        locations["longitude"],
        locations["latitude"],
        s=18,
        c="#1f4e79",
        alpha=0.85,
        linewidths=0,
    )
    ax.set_title("METR-LA loop detector locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensor_graph(
    graph: nx.DiGraph,
    path: Path,
    *,
    max_edges: int = 2500,
) -> Path:
    """Plot the sensor graph using geographic coordinates as layout.

    Drawn edges are spatial relationships among detectors. They should not
    be read as an exact roadway basemap.
    """
    pos = {
        node: (data["longitude"], data["latitude"])
        for node, data in graph.nodes(data=True)
        if "longitude" in data and "latitude" in data
    }
    fig, ax = plt.subplots(figsize=(9, 8))
    edges = list(graph.edges())
    if len(edges) > max_edges:
        edges = edges[:max_edges]
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=edges,
        ax=ax,
        arrows=False,
        alpha=0.15,
        width=0.4,
    )
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    ax.scatter(xs, ys, s=12, c="#1f4e79", zorder=3)
    ax.set_title("Sensor relationship graph (not exact road geometry)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
