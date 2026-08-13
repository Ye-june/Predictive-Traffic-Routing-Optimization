"""Artifact and feature-at-timestamp checks that do not need trained models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trafficflow.features.supervised import feature_columns
from trafficflow.serving import features_at_timestamp, historical_speed_snapshot
from trafficflow.features.temporal import lag_feature


def test_features_at_timestamp_use_past_only() -> None:
    index = pd.date_range("2012-06-21 08:00", periods=30, freq="5min")
    values = np.arange(30, dtype=float)
    speeds = pd.DataFrame({"s1": values, "s2": values + 10}, index=index)
    neighbor_mean = pd.DataFrame({"s1": speeds["s2"], "s2": speeds["s1"]}, index=index)
    timestamp = index[20]
    features = features_at_timestamp(
        speeds,
        timestamp,
        horizon_steps=3,
        neighbor_mean=neighbor_mean,
        include_spatial=True,
    )
    assert list(features.columns) == feature_columns(True)
    assert features.loc["s1", "speed_now"] == 20.0
    assert features.loc["s1", "lag_1"] == 19.0
    assert features.loc["s1", "lag_3"] == 17.0
    assert features.loc["s1", "neighbor_mean"] == 30.0


def test_historical_snapshot_uses_lookup_table() -> None:
    means = pd.DataFrame(
        {
            "sensor_id": ["a", "a", "b"],
            "dow": [0, 0, 0],
            "hour": [8, 9, 8],
            "mean_speed": [40.0, 50.0, 60.0],
        }
    )
    timestamp = pd.Timestamp("2012-06-18 08:10")  # Monday
    snapshot = historical_speed_snapshot(means, timestamp, ["a", "b"])
    assert snapshot["a"] == 40.0
    assert snapshot["b"] == 60.0


def test_lag_feature_never_sees_future() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 99.0])
    lagged = lag_feature(series, 1)
    assert lagged.iloc[3] == 3.0
    assert lagged.iloc[3] != 99.0
