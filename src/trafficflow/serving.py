"""Load deployment artifacts and run inference / routing without training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficflow.features.supervised import TEMPORAL_LAGS, ROLLING_WINDOWS, feature_columns
from trafficflow.features.temporal import add_calendar_features
from trafficflow.models.xgboost_model import SpeedForecaster
from trafficflow.routing.engine import RouteResult, RoutingError, route_with_speeds
from trafficflow.utils.paths import get_project_root, resolve_path

ARTIFACT_DIR = "artifacts"


@dataclass
class ArtifactBundle:
    manifest: dict[str, Any]
    speeds: pd.DataFrame
    forecasts: pd.DataFrame
    sensor_metadata: pd.DataFrame
    historical_means: pd.DataFrame
    metrics: dict[str, Any]
    routing_summary: dict[str, Any]
    scenarios: pd.DataFrame
    graph: Any
    temporal_model: SpeedForecaster
    spatiotemporal_model: SpeedForecaster
    free_flow: dict[str, float]
    neighbors: dict[str, list[str]]


def artifact_root() -> Path:
    return get_project_root() / ARTIFACT_DIR


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def graph_from_json(payload: dict[str, Any]):
    import networkx as nx

    graph = nx.DiGraph()
    for node in payload["nodes"]:
        node_id = str(node["id"])
        attrs = {key: value for key, value in node.items() if key != "id"}
        graph.add_node(node_id, **attrs)
    for link in payload["links"]:
        src = str(link["source"])
        dst = str(link["target"])
        attrs = {key: value for key, value in link.items() if key not in {"source", "target"}}
        graph.add_edge(src, dst, **attrs)
    return graph


def load_artifacts() -> ArtifactBundle:
    """Load all files required by the Streamlit app."""
    root = artifact_root()
    manifest_path = root / "metadata" / "artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `python scripts/build_deployment_assets.py` first."
        )
    manifest = load_json(manifest_path)
    graph = graph_from_json(load_json(root / "graph" / "traffic_graph.json"))
    sensor_metadata = pd.read_parquet(root / "preprocessing" / "sensor_metadata.parquet")
    historical_means = pd.read_parquet(root / "preprocessing" / "historical_means.parquet")
    speeds = pd.read_parquet(root / "demo" / "demo_traffic.parquet")
    forecasts = pd.read_parquet(root / "demo" / "demo_forecasts.parquet")
    scenarios = pd.read_parquet(root / "demo" / "routing_scenarios.parquet")
    metrics = load_json(root / "demo" / "model_metrics.json")
    routing_summary = load_json(root / "demo" / "routing_summary.json")
    temporal_model = SpeedForecaster.load(root / "models" / "temporal_model.joblib")
    spatiotemporal_model = SpeedForecaster.load(root / "models" / "spatiotemporal_model.joblib")
    free_flow = {
        str(row.sensor_id): float(row.free_flow_mph)
        for row in sensor_metadata.itertuples(index=False)
    }
    neighbors = {}
    for row in sensor_metadata.itertuples(index=False):
        raw = row.neighbors
        if isinstance(raw, str):
            parsed = json.loads(raw)
        else:
            parsed = list(raw)
        neighbors[str(row.sensor_id)] = [str(item) for item in parsed]
    return ArtifactBundle(
        manifest=manifest,
        speeds=speeds,
        forecasts=forecasts,
        sensor_metadata=sensor_metadata,
        historical_means=historical_means,
        metrics=metrics,
        routing_summary=routing_summary,
        scenarios=scenarios,
        graph=graph,
        temporal_model=temporal_model,
        spatiotemporal_model=spatiotemporal_model,
        free_flow=free_flow,
        neighbors=neighbors,
    )


def features_at_timestamp(
    speeds: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    horizon_steps: int,
    neighbor_mean: pd.DataFrame | None,
    include_spatial: bool,
) -> pd.DataFrame:
    """Build a 207-row feature frame using only data at or before ``timestamp``."""
    if timestamp not in speeds.index:
        raise KeyError(f"Forecast data is unavailable for the selected timestamp ({timestamp}).")
    loc = int(speeds.index.get_loc(timestamp))
    calendar = add_calendar_features(pd.DatetimeIndex([timestamp])).iloc[0]
    rows: list[dict[str, Any]] = []
    for sensor in speeds.columns:
        series = speeds[sensor]
        row: dict[str, Any] = {
            "sensor_id": str(sensor),
            "speed_now": float(series.iloc[loc]) if pd.notna(series.iloc[loc]) else np.nan,
            "horizon_steps": int(horizon_steps),
            "hour_sin": float(calendar["hour_sin"]),
            "hour_cos": float(calendar["hour_cos"]),
            "dow_sin": float(calendar["dow_sin"]),
            "dow_cos": float(calendar["dow_cos"]),
            "is_weekend": int(calendar["is_weekend"]),
        }
        for lag in TEMPORAL_LAGS:
            row[f"lag_{lag}"] = (
                float(series.iloc[loc - lag])
                if loc >= lag and pd.notna(series.iloc[loc - lag])
                else np.nan
            )
        for window in ROLLING_WINDOWS:
            start = max(0, loc - window)
            window_vals = series.iloc[start:loc]
            row[f"roll_mean_{window}"] = float(window_vals.mean()) if len(window_vals) else np.nan
        if include_spatial:
            if neighbor_mean is None or sensor not in neighbor_mean.columns:
                row["neighbor_mean"] = np.nan
                row["neighbor_mean_lag1"] = np.nan
            else:
                nbr = neighbor_mean[sensor]
                row["neighbor_mean"] = float(nbr.iloc[loc]) if pd.notna(nbr.iloc[loc]) else np.nan
                row["neighbor_mean_lag1"] = (
                    float(nbr.iloc[loc - 1]) if loc >= 1 and pd.notna(nbr.iloc[loc - 1]) else np.nan
                )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("sensor_id")
    return frame.reindex(columns=feature_columns(include_spatial))


def lookup_precomputed_forecast(
    forecasts: pd.DataFrame,
    timestamp: pd.Timestamp,
    horizon_minutes: int,
    model_key: str,
) -> dict[str, float]:
    """Return predicted mph from the demo forecast table."""
    subset = forecasts[
        (forecasts["timestamp"] == timestamp)
        & (forecasts["horizon_minutes"] == horizon_minutes)
    ]
    if subset.empty:
        raise KeyError(f"Forecast data is unavailable for the selected timestamp ({timestamp}).")
    column = f"pred_{model_key}"
    if column not in subset.columns:
        raise KeyError(f"No precomputed forecast named {model_key}.")
    return {
        str(sensor_id): float(value) if pd.notna(value) else float("nan")
        for sensor_id, value in zip(subset["sensor_id"], subset[column])
    }


def historical_speed_snapshot(
    historical_means: pd.DataFrame,
    timestamp: pd.Timestamp,
    sensor_ids: list[str],
) -> dict[str, float]:
    """Train-set mean speed for each sensor at this weekday and hour."""
    dow = int(timestamp.dayofweek)
    hour = int(timestamp.hour)
    lookup = historical_means.set_index(["sensor_id", "dow", "hour"])["mean_speed"]
    snapshot: dict[str, float] = {}
    fallback = float(historical_means["mean_speed"].mean())
    for sensor in sensor_ids:
        key = (str(sensor), dow, hour)
        snapshot[str(sensor)] = float(lookup[key]) if key in lookup.index else fallback
    return snapshot


def compare_routes(
    bundle: ArtifactBundle,
    origin: str,
    destination: str,
    departure: pd.Timestamp,
    *,
    horizon_minutes: int,
    forecast_model: str,
) -> dict[str, RouteResult]:
    """Generate static, current, predictive, and oracle routes for one trip."""
    if departure not in bundle.speeds.index:
        raise RoutingError("Forecast data is unavailable for the selected timestamp.")
    sensors = [str(col) for col in bundle.speeds.columns]
    static_speeds = bundle.free_flow
    current_speeds = {
        sensor: float(bundle.speeds.loc[departure, sensor])
        if pd.notna(bundle.speeds.loc[departure, sensor])
        else bundle.free_flow.get(sensor, 55.0)
        for sensor in sensors
    }
    predictive_speeds = lookup_precomputed_forecast(
        bundle.forecasts,
        departure,
        horizon_minutes,
        forecast_model,
    )
    predictive_speeds = {
        sensor: value
        if value == value
        else bundle.free_flow.get(sensor, 55.0)
        for sensor, value in predictive_speeds.items()
    }
    future_index = bundle.speeds.index.get_loc(departure) + horizon_minutes // 5
    if future_index >= len(bundle.speeds.index):
        raise RoutingError("The selected departure is too close to the end of the demo period.")
    future_time = bundle.speeds.index[future_index]
    oracle_speeds = {
        sensor: float(bundle.speeds.loc[future_time, sensor])
        if pd.notna(bundle.speeds.loc[future_time, sensor])
        else bundle.free_flow.get(sensor, 55.0)
        for sensor in sensors
    }

    results: dict[str, RouteResult] = {}
    for strategy, snapshot in (
        ("static", static_speeds),
        ("current", current_speeds),
        ("predictive", predictive_speeds),
        ("oracle", oracle_speeds),
    ):
        results[strategy] = route_with_speeds(
            bundle.graph,
            origin,
            destination,
            snapshot,
            strategy=strategy,
            realized_speeds=bundle.speeds,
            departure=departure,
            free_flow_by_sensor=bundle.free_flow,
        )
    return results


def resolve_demo_path(relative: str) -> Path:
    return resolve_path(relative)
