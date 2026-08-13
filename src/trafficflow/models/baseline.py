"""Simple forecasting baselines that must be beaten by learned models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(speeds: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Predict speed at ``t + h`` as the speed observed at ``t``.

    Parameters
    ----------
    speeds:
        Timestamp × sensor matrix in mph.
    horizon_steps:
        Forecast lead time in sampling steps (1 step = 5 minutes on METR-LA).

    Returns
    -------
    pandas.DataFrame
        Predictions aligned to the **target** timestamp. The first
        ``horizon_steps`` rows are NaN.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1.")
    return speeds.shift(horizon_steps)


def historical_pattern_forecast(
    speeds: pd.DataFrame,
    *,
    train_index: pd.DatetimeIndex,
    horizon_steps: int,
) -> pd.DataFrame:
    """Predict with the train-set mean for (sensor, weekday, hour).

    The lookup is built **only** from timestamps in ``train_index``.
    Validation and test rows therefore cannot leak future means.

    The ``horizon_steps`` argument is accepted for a consistent interface;
    the seasonal mean does not depend on recent observations.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1.")
    train = speeds.loc[train_index]
    long = train.melt(ignore_index=False, var_name="sensor_id", value_name="speed")
    long["dow"] = long.index.dayofweek
    long["hour"] = long.index.hour
    lookup = (
        long.groupby(["sensor_id", "dow", "hour"], observed=True)["speed"]
        .mean()
        .rename("pred")
    )
    global_mean = float(train.to_numpy(dtype=float).ravel()[~np.isnan(train.to_numpy(dtype=float).ravel())].mean())

    current = speeds.melt(ignore_index=False, var_name="sensor_id", value_name="_unused")
    current["dow"] = current.index.dayofweek
    current["hour"] = current.index.hour
    merged = current.join(lookup, on=["sensor_id", "dow", "hour"])
    merged["pred"] = merged["pred"].fillna(global_mean)
    predicted = merged.pivot_table(index=merged.index, columns="sensor_id", values="pred")
    predicted = predicted.reindex(index=speeds.index, columns=speeds.columns)
    return predicted
