"""Build leakage-safe supervised tables for traffic-speed forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trafficflow.features.temporal import add_calendar_features, lag_feature, rolling_mean

TEMPORAL_LAGS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (3, 6, 12)


def long_feature_table(
    speeds: pd.DataFrame,
    *,
    horizon_steps: int,
    neighbor_mean: pd.DataFrame | None = None,
    include_spatial: bool = False,
    lags: tuple[int, ...] = TEMPORAL_LAGS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Return one row per (timestamp, sensor) with features known at time t.

    The target is speed at ``t + horizon_steps``. Rolling means are lagged
    by one step so they never include the current unused future, and they
    only use ``[t-window, t-1]`` after the extra shift.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1.")

    calendar = add_calendar_features(speeds.index)
    frames: list[pd.DataFrame] = []
    for sensor in speeds.columns:
        series = speeds[sensor]
        parts: dict[str, pd.Series] = {
            "speed_now": series,
            "target": series.shift(-horizon_steps),
        }
        for lag in lags:
            parts[f"lag_{lag}"] = lag_feature(series, lag)
        for window in rolling_windows:
            parts[f"roll_mean_{window}"] = lag_feature(rolling_mean(series, window), 1)
        if include_spatial:
            if neighbor_mean is None:
                raise ValueError("neighbor_mean is required when include_spatial=True.")
            if sensor in neighbor_mean.columns:
                parts["neighbor_mean"] = neighbor_mean[sensor]
                parts["neighbor_mean_lag1"] = lag_feature(neighbor_mean[sensor], 1)
            else:
                parts["neighbor_mean"] = pd.Series(np.nan, index=series.index)
                parts["neighbor_mean_lag1"] = pd.Series(np.nan, index=series.index)
        sensor_frame = pd.DataFrame(parts)
        sensor_frame["sensor_id"] = str(sensor)
        frames.append(sensor_frame)

    long = pd.concat(frames, axis=0)
    long = long.join(calendar, how="left")
    long["horizon_steps"] = horizon_steps
    return long


def feature_columns(include_spatial: bool) -> list[str]:
    """Return the ordered feature names used by the compact XGBoost models."""
    columns = [f"lag_{lag}" for lag in TEMPORAL_LAGS]
    columns.extend(f"roll_mean_{window}" for window in ROLLING_WINDOWS)
    columns.extend(
        [
            "speed_now",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "horizon_steps",
        ]
    )
    if include_spatial:
        columns.extend(["neighbor_mean", "neighbor_mean_lag1"])
    return columns


def prepare_xy(
    table: pd.DataFrame,
    *,
    include_spatial: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop incomplete rows and return ``X, y`` for model fitting."""
    columns = feature_columns(include_spatial)
    required = columns + ["target"]
    clean = table.dropna(subset=required)
    return clean[columns], clean["target"]
