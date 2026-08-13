"""Build and validate the METR-LA sensor graph from inspected metadata."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trafficflow.data.loader import (
    load_distance_table,
    load_sensor_ids,
    load_sensor_locations,
)
from trafficflow.features.graph import build_sensor_distance_graph, summarize_graph
from trafficflow.utils.config import load_config
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import resolve_path
from trafficflow.visualization.graph import plot_sensor_graph, plot_sensor_map

logger = get_logger(__name__)


def main() -> int:
    config = load_config("data")
    paths = config["paths"]
    sensor_ids = load_sensor_ids(resolve_path(paths["sensor_ids"]))
    locations = load_sensor_locations(resolve_path(paths["sensor_locations"]))
    distances = load_distance_table(resolve_path(paths["distances"]))

    graph = build_sensor_distance_graph(sensor_ids, locations, distances)
    summary = summarize_graph(graph)

    graph_path = resolve_path("data/processed/sensor_graph.pkl")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("wb") as handle:
        pickle.dump(graph, handle)

    summary_path = resolve_path("outputs/metrics/graph_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    payload["graph_path"] = str(graph_path)
    payload["units"] = {"edge_distance": "meters and miles"}
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figures = resolve_path("outputs/figures")
    plot_sensor_map(locations, figures / "01_sensor_map.png")
    plot_sensor_graph(graph, figures / "02_sensor_graph.png")
    logger.info("Wrote graph summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
