"""Leakage and split tests for temporal features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trafficflow.features.temporal import (
    chronological_split_indices,
    lag_feature,
    rolling_mean,
)
from trafficflow.models.baseline import persistence_forecast
from trafficflow.models.evaluation import regression_metrics


def test_lag_uses_only_past_values() -> None:
    series = pd.Series([10.0, 20.0, 30.0, 40.0])
    lagged = lag_feature(series, 2)
    assert np.isnan(lagged.iloc[0])
    assert np.isnan(lagged.iloc[1])
    assert lagged.iloc[2] == 10.0
    assert lagged.iloc[3] == 20.0


def test_lag_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError):
        lag_feature(pd.Series([1.0, 2.0]), 0)


def test_rolling_mean_is_backward_looking() -> None:
    series = pd.Series([1.0, 3.0, 5.0, 7.0])
    rolled = rolling_mean(series, window=2)
    assert rolled.iloc[0] == 1.0
    assert rolled.iloc[1] == 2.0
    assert rolled.iloc[2] == 4.0


def test_chronological_split_is_ordered_and_contiguous() -> None:
    train, val, test = chronological_split_indices(100, train_ratio=0.7, val_ratio=0.15)
    assert train == slice(0, 70)
    assert val == slice(70, 85)
    assert test == slice(85, 100)


def test_persistence_alignment() -> None:
    index = pd.date_range("2012-03-01", periods=4, freq="5min")
    speeds = pd.DataFrame({"s": [10.0, 11.0, 12.0, 13.0]}, index=index)
    pred = persistence_forecast(speeds, horizon_steps=1)
    assert np.isnan(pred.iloc[0, 0])
    assert pred.iloc[1, 0] == 10.0
    assert pred.iloc[3, 0] == 12.0


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0]))
    assert metrics["n"] == 3
    assert abs(metrics["mae"] - 1.0) < 1e-9
    assert abs(metrics["rmse"] - (5.0 / 3.0) ** 0.5) < 1e-9
