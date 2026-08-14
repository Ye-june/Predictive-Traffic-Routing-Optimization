"""Leakage, alignment, and artifact integrity checks."""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from trafficflow.data.cleaning import clean_speed_frame
from trafficflow.models.baseline import persistence_forecast
from trafficflow.models.evaluation import regression_metrics
from trafficflow.routing.engine import realized_travel_time_minutes
from trafficflow.serving import load_artifacts


def test_cleaning_does_not_use_future_values() -> None:
    """Interior gaps must not be filled with information from later observations."""
    index = pd.date_range("2012-03-01", periods=4, freq="5min")
    frame = pd.DataFrame({"s": [10.0, np.nan, np.nan, 40.0]}, index=index)
    result = clean_speed_frame(frame, interpolate_limit_steps=12)
    # Forward-fill keeps the past value; linear interp would produce 20/30.
    assert result.speeds["s"].tolist() == [10.0, 10.0, 10.0, 40.0]
    assert not result.missing_mask.iloc[0, 0]
    assert result.imputed_mask.iloc[1, 0]
    assert result.imputed_mask.iloc[2, 0]


def test_cleaning_does_not_backfill_leading_gap() -> None:
    index = pd.date_range("2012-03-01", periods=4, freq="5min")
    frame = pd.DataFrame({"s": [np.nan, np.nan, 40.0, 50.0]}, index=index)
    result = clean_speed_frame(frame, interpolate_limit_steps=12)
    assert pd.isna(result.speeds["s"].iloc[0])
    assert pd.isna(result.speeds["s"].iloc[1])
    assert result.speeds["s"].iloc[2] == 40.0


def test_persistence_aligned_with_target_shift() -> None:
    """h-step persistence at feature time t is speed[t]; target is speed[t+h]."""
    index = pd.date_range("2012-03-01", periods=10, freq="5min")
    speeds = pd.DataFrame({"a": np.arange(10, dtype=float)}, index=index)
    horizon = 3
    feature_times = index[2:7]
    target = speeds["a"].shift(-horizon).loc[feature_times]
    persist = speeds["a"].loc[feature_times]
    # Equivalent classic form aligned to the target clock:
    classic = persistence_forecast(speeds, horizon)["a"].loc[feature_times + pd.Timedelta(minutes=5 * horizon)]
    np.testing.assert_allclose(persist.to_numpy(), classic.to_numpy())
    metrics = regression_metrics(target.to_numpy(), persist.to_numpy())
    assert metrics["n"] == len(feature_times)
    assert metrics["mae"] == pytest.approx(float(horizon), rel=0, abs=1e-9)


def test_realized_replay_multi_edge_advances_clock() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", distance_miles=1.0)
    graph.add_edge("b", "c", distance_miles=1.0)
    index = pd.date_range("2012-06-21 08:00", periods=12, freq="5min")
    # After the first 1-minute edge, the clock jumps to the next 5-min stamp (08:05).
    # Put the slowdown on that stamp so the second edge must see updated traffic.
    speeds = pd.DataFrame(
        {
            "a": [60.0] * 12,
            "b": [60.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            "c": [30.0] * 12,
        },
        index=index,
    )
    minutes = realized_travel_time_minutes(graph, ["a", "b", "c"], speeds, index[0])
    # Edge1 at 08:00 uses (a=60,b=60) → 1 min. Edge2 at 08:05 uses (b=30,c=30) → 2 min.
    assert minutes == pytest.approx(3.0, abs=1e-9)


def test_artifacts_load_and_graph_connected() -> None:
    bundle = load_artifacts()
    assert bundle.speeds.shape[1] == 207
    assert bundle.graph.number_of_nodes() == 207
    assert nx.is_strongly_connected(bundle.graph)
    assert {"results", "spatial_vs_temporal"}.issubset(bundle.metrics)
    assert bundle.routing_summary["n_trips"] > 0
    # Demo forecasts include fair persistence (= speed_now) after rebuild.
    sample = bundle.forecasts.dropna(
        subset=["speed_now", "pred_persistence", "target"]
    ).head(200)
    if not sample.empty:
        # Allow tiny float noise from parquet round-trip.
        assert np.allclose(
            sample["pred_persistence"].to_numpy(),
            sample["speed_now"].to_numpy(),
            equal_nan=True,
            atol=1e-5,
        )


def test_feature_config_mentions_causal_fill() -> None:
    from trafficflow.utils.paths import get_project_root

    payload = json.loads(
        (get_project_root() / "artifacts/preprocessing/feature_config.json").read_text(
            encoding="utf-8"
        )
    )
    joined = " ".join(payload.get("leakage_audit", [])).lower()
    assert "forward" in joined
    assert "t+h" in joined
