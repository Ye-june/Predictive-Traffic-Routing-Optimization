"""Forecast evaluation metrics. Units follow the target (mph for METR-LA)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE and RMSE on finite paired observations.

    MAPE is omitted here because speeds near zero would inflate it; those
    zeros are missing sentinels in METR-LA, not true standstill.
    """
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = pred[mask] - true[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"mae": mae, "rmse": rmse, "n": int(mask.sum())}


def frame_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, Any]:
    """Evaluate two aligned timestamp × sensor frames."""
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
    metrics = regression_metrics(aligned_true.to_numpy(), aligned_pred.to_numpy())
    metrics["n_timestamps"] = int(aligned_true.shape[0])
    metrics["n_sensors"] = int(aligned_true.shape[1])
    return metrics
