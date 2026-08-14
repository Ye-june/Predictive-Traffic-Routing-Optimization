"""Train compact models and write Streamlit deployment artifacts.

This script is the offline training/export path. The web app only loads
the files it writes under ``artifacts/``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import networkx as nx
import numpy as np
import pandas as pd

from trafficflow.data.cleaning import estimate_free_flow_speed
from trafficflow.data.loader import (
    load_distance_table,
    load_sensor_ids,
    load_sensor_locations,
)
from trafficflow.features.graph import (
    build_knn_routing_graph,
    build_sensor_distance_graph,
    k_nearest_neighbors,
    summarize_graph,
)
from trafficflow.features.spatial import neighbor_mean_speed
from trafficflow.features.supervised import feature_columns, prepare_xy
from trafficflow.features.temporal import (
    add_calendar_features,
    chronological_split_indices,
    lag_feature,
    rolling_mean,
)
from trafficflow.models.baseline import historical_pattern_forecast
from trafficflow.models.evaluation import regression_metrics
from trafficflow.models.xgboost_model import train_speed_forecaster
from trafficflow.routing.engine import route_with_speeds
from trafficflow.utils.config import load_config
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import resolve_path
from trafficflow.utils.seeds import set_seeds

logger = get_logger(__name__)

HORIZONS = {"15": 3, "30": 6, "60": 12}
TRAIN_ROWS_PER_SENSOR = 80
TEST_STRIDE = 6
ROUTING_K = 8
MAX_DISTANCE_M = 8000.0
DEMO_DAYS = 7


def _graph_to_json(graph) -> dict:
    nodes = []
    for node, data in graph.nodes(data=True):
        item = {"id": str(node)}
        item.update({key: _jsonify(value) for key, value in data.items()})
        nodes.append(item)
    links = []
    for src, dst, data in graph.edges(data=True):
        item = {"source": str(src), "target": str(dst)}
        item.update({key: _jsonify(value) for key, value in data.items()})
        links.append(item)
    return {"directed": True, "multigraph": False, "nodes": nodes, "links": links}


def _jsonify(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def region_name(lat: float, lon: float, lat_med: float, lon_med: float) -> str:
    ns = "North" if lat >= lat_med else "South"
    ew = "east" if lon >= lon_med else "west"
    return f"{ns}{ew}"


def sampled_table(
    speeds: pd.DataFrame,
    neighbor_mean: pd.DataFrame,
    *,
    horizon_steps: int,
    include_spatial: bool,
    keep_index: pd.DatetimeIndex,
    per_sensor: int,
    seed: int,
) -> pd.DataFrame:
    calendar = add_calendar_features(speeds.index)
    blocks: list[pd.DataFrame] = []
    for offset, sensor in enumerate(speeds.columns):
        series = speeds[str(sensor)]
        data: dict[str, pd.Series] = {
            "speed_now": series,
            "target": series.shift(-horizon_steps),
        }
        for lag in (1, 2, 3, 6, 12, 24):
            data[f"lag_{lag}"] = lag_feature(series, lag)
        for window in (3, 6, 12):
            data[f"roll_mean_{window}"] = lag_feature(rolling_mean(series, window), 1)
        if include_spatial:
            nbr = neighbor_mean[str(sensor)]
            data["neighbor_mean"] = nbr
            data["neighbor_mean_lag1"] = lag_feature(nbr, 1)
        frame = pd.DataFrame(data)
        frame = frame.join(calendar)
        frame = frame.loc[keep_index]
        frame["horizon_steps"] = horizon_steps
        frame = frame.dropna(subset=feature_columns(include_spatial) + ["target"])
        if frame.empty:
            continue
        if len(frame) > per_sensor:
            frame = frame.sample(per_sensor, random_state=seed + offset)
        blocks.append(frame)
    if not blocks:
        raise RuntimeError("No training rows were produced.")
    return pd.concat(blocks, axis=0)


def build_eval_table(
    speeds: pd.DataFrame,
    neighbor_mean: pd.DataFrame,
    *,
    horizon_steps: int,
    include_spatial: bool,
    keep_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    calendar = add_calendar_features(speeds.index)
    blocks: list[pd.DataFrame] = []
    for sensor in speeds.columns:
        series = speeds[str(sensor)]
        data: dict[str, pd.Series] = {
            "speed_now": series,
            "target": series.shift(-horizon_steps),
            "sensor_id": pd.Series(str(sensor), index=series.index),
        }
        for lag in (1, 2, 3, 6, 12, 24):
            data[f"lag_{lag}"] = lag_feature(series, lag)
        for window in (3, 6, 12):
            data[f"roll_mean_{window}"] = lag_feature(rolling_mean(series, window), 1)
        if include_spatial:
            nbr = neighbor_mean[str(sensor)]
            data["neighbor_mean"] = nbr
            data["neighbor_mean_lag1"] = lag_feature(nbr, 1)
        frame = pd.DataFrame(data).join(calendar)
        frame["horizon_steps"] = horizon_steps
        frame = frame.loc[keep_index].dropna(subset=feature_columns(include_spatial) + ["target"])
        blocks.append(frame)
    return pd.concat(blocks, axis=0)


def sensor_metadata_frame(
    locations: pd.DataFrame,
    routing_graph,
    neighbors: dict[str, list[str]],
    free_flow: pd.Series,
) -> pd.DataFrame:
    lat_med = float(locations["latitude"].median())
    lon_med = float(locations["longitude"].median())
    rows = []
    for row in locations.itertuples(index=False):
        sensor_id = str(row.sensor_id)
        region = region_name(float(row.latitude), float(row.longitude), lat_med, lon_med)
        rows.append(
            {
                "sensor_id": sensor_id,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "region": region,
                "label": f"Sensor {sensor_id} — {region}",
                "free_flow_mph": float(free_flow.get(sensor_id, 65.0)),
                "degree": int(routing_graph.degree(sensor_id)) if routing_graph.has_node(sensor_id) else 0,
                "neighbors": json.dumps(neighbors.get(sensor_id, [])),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    started = time.perf_counter()
    data_cfg = load_config("data")
    set_seeds(int(data_cfg["seed"]))
    rng = np.random.default_rng(int(data_cfg["seed"]))

    speeds = pd.read_parquet(resolve_path(data_cfg["paths"]["cleaned_parquet"]))
    sensor_ids = load_sensor_ids(resolve_path(data_cfg["paths"]["sensor_ids"]))
    locations = load_sensor_locations(resolve_path(data_cfg["paths"]["sensor_locations"]))
    distances = load_distance_table(resolve_path(data_cfg["paths"]["distances"]))

    full_graph = build_sensor_distance_graph(sensor_ids, locations, distances)
    routing_graph = build_knn_routing_graph(
        full_graph, k=ROUTING_K, max_distance_m=MAX_DISTANCE_M
    )
    graph_stats = summarize_graph(routing_graph)
    neighbors = k_nearest_neighbors(routing_graph, k=ROUTING_K)
    neighbor_mean = neighbor_mean_speed(speeds, neighbors)

    train_slc, val_slc, test_slc = chronological_split_indices(len(speeds))
    train_index = speeds.index[train_slc]
    test_index = speeds.index[test_slc]
    logger.info(
        "Split train=%s:%s test=%s:%s",
        train_index[0],
        train_index[-1],
        test_index[0],
        test_index[-1],
    )

    free_flow = estimate_free_flow_speed(speeds.loc[train_index], percentile=95)
    historical = historical_pattern_forecast(
        speeds, train_index=train_index, horizon_steps=1
    )

    train_tables_t = []
    train_tables_s = []
    for horizon_steps in HORIZONS.values():
        train_tables_t.append(
            sampled_table(
                speeds,
                neighbor_mean,
                horizon_steps=horizon_steps,
                include_spatial=False,
                keep_index=train_index,
                per_sensor=TRAIN_ROWS_PER_SENSOR,
                seed=42,
            )
        )
        train_tables_s.append(
            sampled_table(
                speeds,
                neighbor_mean,
                horizon_steps=horizon_steps,
                include_spatial=True,
                keep_index=train_index,
                per_sensor=TRAIN_ROWS_PER_SENSOR,
                seed=42,
            )
        )
    temporal_train = pd.concat(train_tables_t, axis=0)
    spatial_train = pd.concat(train_tables_s, axis=0)
    X_t, y_t = prepare_xy(temporal_train, include_spatial=False)
    X_s, y_s = prepare_xy(spatial_train, include_spatial=True)
    logger.info("Training rows temporal=%s spatial=%s", f"{len(X_t):,}", f"{len(X_s):,}")

    fit_t0 = time.perf_counter()
    temporal_model = train_speed_forecaster(
        X_t, y_t, include_spatial=False, name="temporal_xgboost", seed=42
    )
    temporal_fit_s = time.perf_counter() - fit_t0
    fit_s0 = time.perf_counter()
    spatial_model = train_speed_forecaster(
        X_s, y_s, include_spatial=True, name="spatiotemporal_xgboost", seed=42
    )
    spatial_fit_s = time.perf_counter() - fit_s0
    logger.info("Fit seconds temporal=%.1f spatial=%.1f", temporal_fit_s, spatial_fit_s)

    test_eval_index = test_index[::TEST_STRIDE]
    metrics_rows = []
    for name, steps in HORIZONS.items():
        # Evaluate every model on the same (feature time t → target t+h) pairs.
        # Persistence at origin t is speed_now; seasonal mean uses the known
        # calendar of the target timestamp t+h (no recent-speed leakage).
        eval_t = build_eval_table(
            speeds,
            neighbor_mean,
            horizon_steps=steps,
            include_spatial=False,
            keep_index=test_eval_index,
        )
        eval_s = build_eval_table(
            speeds,
            neighbor_mean,
            horizon_steps=steps,
            include_spatial=True,
            keep_index=test_eval_index,
        )
        target_times = eval_t.index + pd.Timedelta(minutes=5 * steps)
        hist_at_target = np.array(
            [
                float(historical.loc[ts, sid])
                if ts in historical.index and sid in historical.columns
                else np.nan
                for ts, sid in zip(target_times, eval_t["sensor_id"].astype(str))
            ]
        )
        persist_m = regression_metrics(eval_t["target"].to_numpy(), eval_t["speed_now"].to_numpy())
        hist_m = regression_metrics(eval_t["target"].to_numpy(), hist_at_target)
        pred_t = temporal_model.predict(eval_t[feature_columns(False)])
        pred_s = spatial_model.predict(eval_s[feature_columns(True)])
        t_m = regression_metrics(eval_t["target"].to_numpy(), pred_t)
        s_m = regression_metrics(eval_s["target"].to_numpy(), pred_s)
        for model_name, mets, extra in (
            ("persistence", persist_m, {"feature_set": "current speed only", "train_seconds": 0.0}),
            ("historical_dow_hour", hist_m, {"feature_set": "sensor+dow+hour mean", "train_seconds": 0.0}),
            ("temporal_xgboost", t_m, {"feature_set": "lags+calendar+rolling", "train_seconds": temporal_fit_s}),
            (
                "spatiotemporal_xgboost",
                s_m,
                {"feature_set": "temporal+neighbor mean", "train_seconds": spatial_fit_s},
            ),
        ):
            metrics_rows.append(
                {
                    "model": model_name,
                    "horizon_minutes": int(name),
                    "mae_mph": mets["mae"],
                    "rmse_mph": mets["rmse"],
                    "n_eval_cells": mets["n"],
                    **extra,
                }
            )
            logger.info(
                "%s %smin MAE=%.3f RMSE=%.3f n=%s",
                model_name,
                name,
                mets["mae"],
                mets["rmse"],
                f"{mets['n']:,}",
            )

    demo_end = speeds.index.max()
    demo_start = demo_end - pd.Timedelta(days=DEMO_DAYS) + pd.Timedelta(minutes=5)
    context_start = demo_start - pd.Timedelta(hours=3)
    context = speeds.loc[context_start:demo_end]
    demo = speeds.loc[demo_start:demo_end]
    demo_neighbor = neighbor_mean.loc[context.index]
    logger.info("Demo window %s → %s (%s timestamps)", demo.index.min(), demo.index.max(), len(demo))

    forecast_frames = []
    infer_t0 = time.perf_counter()
    for horizon_name, steps in HORIZONS.items():
        eval_t = build_eval_table(
            context,
            demo_neighbor,
            horizon_steps=steps,
            include_spatial=False,
            keep_index=demo.index,
        )
        eval_s = build_eval_table(
            context,
            demo_neighbor,
            horizon_steps=steps,
            include_spatial=True,
            keep_index=demo.index,
        )
        eval_t = eval_t.copy()
        eval_s = eval_s.copy()
        eval_t["pred_temporal"] = temporal_model.predict(eval_t[feature_columns(False)])
        eval_s["pred_spatial"] = spatial_model.predict(eval_s[feature_columns(True)])
        merged = eval_t.reset_index().rename(columns={"index": "timestamp"})[
            ["timestamp", "sensor_id", "target", "speed_now", "pred_temporal"]
        ]
        spatial_part = eval_s.reset_index().rename(columns={"index": "timestamp"})[
            ["timestamp", "sensor_id", "pred_spatial"]
        ]
        merged = merged.merge(spatial_part, on=["timestamp", "sensor_id"], how="left")
        merged["horizon_minutes"] = int(horizon_name)
        # Fair h-step baselines at feature time t: persistence = speed_now;
        # historical = train mean for the calendar of target t+h.
        merged["pred_persistence"] = merged["speed_now"]
        target_times = pd.to_datetime(merged["timestamp"]) + pd.Timedelta(
            minutes=int(horizon_name)
        )
        merged["pred_historical"] = [
            float(historical.loc[ts, sid])
            if ts in historical.index and sid in historical.columns
            else np.nan
            for ts, sid in zip(target_times, merged["sensor_id"].astype(str))
        ]
        forecast_frames.append(merged)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    infer_s = time.perf_counter() - infer_t0
    logger.info("Demo forecast inference seconds=%.1f rows=%s", infer_s, f"{len(forecasts):,}")

    metadata = sensor_metadata_frame(locations, routing_graph, neighbors, free_flow)
    hist_long = (
        speeds.loc[train_index]
        .melt(ignore_index=False, var_name="sensor_id", value_name="speed")
        .dropna()
    )
    hist_long["dow"] = hist_long.index.dayofweek
    hist_long["hour"] = hist_long.index.hour
    historical_means = (
        hist_long.groupby(["sensor_id", "dow", "hour"], observed=True)["speed"]
        .mean()
        .reset_index()
        .rename(columns={"speed": "mean_speed"})
    )

    scc = max(nx.strongly_connected_components(routing_graph), key=len)
    scc_nodes = [node for node in routing_graph.nodes if node in scc]
    od_pairs: list[tuple[str, str]] = []
    attempts = 0
    while len(od_pairs) < 24 and attempts < 400:
        attempts += 1
        origin, dest = rng.choice(scc_nodes, size=2, replace=False)
        origin, dest = str(origin), str(dest)
        if origin == dest:
            continue
        if not nx.has_path(routing_graph, origin, dest):
            continue
        path = nx.shortest_path(routing_graph, origin, dest, weight="distance_miles")
        if 4 <= len(path) - 1 <= 25:
            od_pairs.append((origin, dest))
    od_pairs = list(dict.fromkeys(od_pairs))
    logger.info("Sampled %s origin-destination pairs", len(od_pairs))

    demo_hours = {8: "morning_rush", 17: "evening_rush", 21: "off_peak"}
    candidate_times = [
        ts
        for ts in demo.index
        if ts.hour in demo_hours and ts.minute == 0 and ts.dayofweek < 5
    ]
    if len(candidate_times) > 12:
        candidate_times = list(rng.choice(candidate_times, size=12, replace=False))
        candidate_times = sorted(pd.to_datetime(candidate_times))

    scenario_rows = []
    for origin, dest in od_pairs[:12]:
        for departure in candidate_times[:6]:
            current = {
                str(col): float(demo.loc[departure, col])
                if pd.notna(demo.loc[departure, col])
                else float(free_flow.get(str(col), 55.0))
                for col in demo.columns
            }
            pred_slice = forecasts[
                (forecasts["timestamp"] == departure) & (forecasts["horizon_minutes"] == 30)
            ]
            if pred_slice.empty:
                continue
            predictive = {
                str(row.sensor_id): float(row.pred_spatial)
                for row in pred_slice.itertuples(index=False)
                if pd.notna(row.pred_spatial)
            }
            future_pos = demo.index.get_loc(departure) + 6
            if future_pos >= len(demo.index):
                continue
            future_time = demo.index[future_pos]
            oracle = {
                str(col): float(demo.loc[future_time, col])
                if pd.notna(demo.loc[future_time, col])
                else float(free_flow.get(str(col), 55.0))
                for col in demo.columns
            }
            try:
                results = {
                    "static": route_with_speeds(
                        routing_graph,
                        origin,
                        dest,
                        {str(k): float(v) for k, v in free_flow.items()},
                        strategy="static",
                        realized_speeds=demo,
                        departure=departure,
                        free_flow_by_sensor=free_flow.to_dict(),
                    ),
                    "current": route_with_speeds(
                        routing_graph,
                        origin,
                        dest,
                        current,
                        strategy="current",
                        realized_speeds=demo,
                        departure=departure,
                        free_flow_by_sensor=free_flow.to_dict(),
                    ),
                    "predictive": route_with_speeds(
                        routing_graph,
                        origin,
                        dest,
                        predictive,
                        strategy="predictive",
                        realized_speeds=demo,
                        departure=departure,
                        free_flow_by_sensor=free_flow.to_dict(),
                    ),
                    "oracle": route_with_speeds(
                        routing_graph,
                        origin,
                        dest,
                        oracle,
                        strategy="oracle",
                        realized_speeds=demo,
                        departure=departure,
                        free_flow_by_sensor=free_flow.to_dict(),
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping trip %s -> %s at %s (%s)", origin, dest, departure, exc)
                continue
            traffic_condition = demo_hours.get(int(departure.hour), "other")
            static_real = results["static"].realized_minutes
            for strategy, result in results.items():
                scenario_rows.append(
                    {
                        "origin": origin,
                        "destination": dest,
                        "departure_time": str(departure),
                        "traffic_condition": traffic_condition,
                        "strategy": strategy,
                        "estimated_minutes": result.estimated_minutes,
                        "realized_minutes": result.realized_minutes,
                        "distance_miles": result.distance_miles,
                        "n_edges": result.n_edges,
                        "path": json.dumps(result.path),
                        "horizon_minutes": 30,
                        "forecast_model": "spatial",
                        "savings_vs_static_min": None
                        if result.realized_minutes is None or static_real is None
                        else static_real - result.realized_minutes,
                    }
                )
    scenarios = pd.DataFrame(scenario_rows)
    routing_summary = _summarize_routing(scenarios)
    logger.info("Routing scenarios: %s rows", len(scenarios))

    root = resolve_path("artifacts")
    for folder in ("models", "preprocessing", "graph", "demo", "metadata"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    temporal_model.save(root / "models" / "temporal_model.joblib")
    spatial_model.save(root / "models" / "spatiotemporal_model.joblib")
    (root / "graph" / "traffic_graph.json").write_text(
        json.dumps(_graph_to_json(routing_graph)), encoding="utf-8"
    )
    metadata.to_parquet(root / "preprocessing" / "sensor_metadata.parquet", index=False)
    historical_means.to_parquet(root / "preprocessing" / "historical_means.parquet", index=False)
    feature_config = {
        "lags": [1, 2, 3, 6, 12, 24],
        "rolling_windows": [3, 6, 12],
        "horizons_minutes": [15, 30, 60],
        "include_spatial_features": ["neighbor_mean", "neighbor_mean_lag1"],
        "speed_unit": "mph",
        "speed_floor_mph": 5.0,
        "split": {
            "train_start": str(train_index[0]),
            "train_end": str(train_index[-1]),
            "test_start": str(test_index[0]),
            "test_end": str(test_index[-1]),
        },
        "leakage_audit": [
            "Lags use shift(positive) only.",
            "Rolling means are lagged by one step.",
            "Historical means and free-flow percentiles use the training window only.",
            "Neighbor features use time t, never t+h.",
            "Chronological split; no row shuffling.",
            "Short missing gaps are forward-filled only (no future-looking interpolation).",
            "Baselines and XGBoost are scored on the same t → t+h pairs.",
        ],
    }
    (root / "preprocessing" / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2), encoding="utf-8"
    )

    demo.astype(np.float32).to_parquet(root / "demo" / "demo_traffic.parquet")
    forecasts.to_parquet(root / "demo" / "demo_forecasts.parquet", index=False)
    scenarios.to_parquet(root / "demo" / "routing_scenarios.parquet", index=False)
    metrics_payload = {
        "target_unit": "mph",
        "test_stride": TEST_STRIDE,
        "results": metrics_rows,
        "spatial_vs_temporal": _spatial_delta(metrics_rows),
    }
    (root / "demo" / "model_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    (root / "demo" / "routing_summary.json").write_text(
        json.dumps(routing_summary, indent=2), encoding="utf-8"
    )

    sizes = {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }
    manifest = {
        "app_version": "1.0.0",
        "mode": "historical_replay",
        "dataset": "METR-LA",
        "model_name": "spatiotemporal_xgboost",
        "forecast_horizons": [15, 30, 60],
        "demo_start": str(demo.index.min()),
        "demo_end": str(demo.index.max()),
        "num_nodes": int(routing_graph.number_of_nodes()),
        "num_routing_edges": int(routing_graph.number_of_edges()),
        "graph_notes": graph_stats.notes,
        "speed_unit": "mph",
        "distance_unit": "miles",
        "travel_time_unit": "minutes",
        "speed_floor_mph": 5.0,
        "artifact_bytes": sizes,
        "total_artifact_mb": round(sum(sizes.values()) / 1e6, 2),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    (root / "metadata" / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wrote artifacts to %s (%.2f MB, %.1fs)",
        root,
        manifest["total_artifact_mb"],
        manifest["build_seconds"],
    )
    return 0


def _spatial_delta(rows: list[dict]) -> list[dict]:
    out = []
    by_key = {(row["model"], row["horizon_minutes"]): row for row in rows}
    for horizon in (15, 30, 60):
        temporal = by_key.get(("temporal_xgboost", horizon))
        spatial = by_key.get(("spatiotemporal_xgboost", horizon))
        if not temporal or not spatial:
            continue
        mae_delta = temporal["mae_mph"] - spatial["mae_mph"]
        rmse_delta = temporal["rmse_mph"] - spatial["rmse_mph"]
        out.append(
            {
                "horizon_minutes": horizon,
                "temporal_mae": temporal["mae_mph"],
                "spatial_mae": spatial["mae_mph"],
                "mae_improvement": mae_delta,
                "mae_improvement_pct": 100.0 * mae_delta / temporal["mae_mph"]
                if temporal["mae_mph"]
                else None,
                "rmse_improvement": rmse_delta,
                "rmse_improvement_pct": 100.0 * rmse_delta / temporal["rmse_mph"]
                if temporal["rmse_mph"]
                else None,
            }
        )
    return out


def _summarize_routing(scenarios: pd.DataFrame) -> dict:
    if scenarios.empty:
        return {"n_trips": 0, "note": "No successful routing scenarios were generated."}
    wide = scenarios.pivot_table(
        index=["origin", "destination", "departure_time"],
        columns="strategy",
        values="realized_minutes",
        aggfunc="first",
    )
    n = int(len(wide))
    summary: dict = {"n_trips": n}
    if {"static", "predictive"}.issubset(wide.columns):
        delta = wide["static"] - wide["predictive"]
        summary["predictive_vs_static"] = {
            "mean_savings_min": float(delta.mean()),
            "median_savings_min": float(delta.median()),
            "pct_improved": float((delta > 0.05).mean() * 100),
            "pct_worsened": float((delta < -0.05).mean() * 100),
        }
    if {"current", "predictive"}.issubset(wide.columns):
        delta = wide["current"] - wide["predictive"]
        summary["predictive_vs_current"] = {
            "mean_savings_min": float(delta.mean()),
            "median_savings_min": float(delta.median()),
            "pct_improved": float((delta > 0.05).mean() * 100),
            "pct_worsened": float((delta < -0.05).mean() * 100),
        }
    if {"oracle", "predictive"}.issubset(wide.columns):
        regret = wide["predictive"] - wide["oracle"]
        summary["predictive_regret_vs_oracle_min"] = {
            "mean": float(regret.mean()),
            "median": float(regret.median()),
        }
    return summary


if __name__ == "__main__":
    sys.exit(main())
