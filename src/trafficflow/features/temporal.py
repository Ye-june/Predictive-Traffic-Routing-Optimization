"""Leakage-safe temporal features for traffic speed forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build calendar columns aligned to a timestamp index.

    Cyclical encodings use:

    * ``hour_sin/cos`` with period 24
    * ``dow_sin/cos`` with period 7

    Returns
    -------
    pandas.DataFrame
        One row per timestamp. No traffic values are used.
    """
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek.astype(float)
    frame = pd.DataFrame(
        {
            "hour": index.hour.astype(int),
            "minute": index.minute.astype(int),
            "day_of_week": index.dayofweek.astype(int),
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        },
        index=index,
    )
    return frame


def lag_feature(series: pd.Series, steps: int) -> pd.Series:
    """Shift a series forward in time by ``steps`` observations.

    ``value[t]`` in the result equals ``series[t - steps]``. Future values
    never enter the lagged feature.
    """
    if steps < 1:
        raise ValueError("Lag steps must be >= 1 to avoid leakage.")
    return series.shift(steps)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Backward-looking rolling mean.

    Pandas ``rolling(window).mean()`` at time t uses ``[t-window+1, t]``.
    Callers who want a strictly causal mean excluding the current point
    should lag the result by one step.
    """
    if window < 1:
        raise ValueError("Rolling window must be >= 1.")
    return series.rolling(window=window, min_periods=1).mean()


def chronological_split_indices(
    n_timestamps: int,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[slice, slice, slice]:
    """Return train/validation/test slices along time.

    Splits are contiguous and ordered. Random shuffling is intentionally
    unsupported.
    """
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Ratios must be in (0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    n_train = int(n_timestamps * train_ratio)
    n_val = int(n_timestamps * val_ratio)
    n_test = n_timestamps - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Split produced an empty partition.")
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_timestamps)
