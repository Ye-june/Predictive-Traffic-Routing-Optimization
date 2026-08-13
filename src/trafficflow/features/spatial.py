"""Spatial neighbor features derived from the sensor graph.

Neighbor statistics at timestamp ``t`` use only contemporaneous
observations. They do not include future traffic and therefore do not
leak the forecast target at ``t + h``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def neighbor_mean_speed(
    speeds: pd.DataFrame,
    neighbors: dict[str, list[str]],
) -> pd.DataFrame:
    """Return the mean speed of each sensor's graph neighbors.

    Parameters
    ----------
    speeds:
        Timestamp × sensor matrix in mph.
    neighbors:
        Mapping of sensor ID → neighbor sensor IDs (typically k-nearest).

    Returns
    -------
    pandas.DataFrame
        Same shape as ``speeds``. Cells are NaN when a sensor has no
        neighbors with observed speeds.
    """
    columns = [str(col) for col in speeds.columns]
    result = pd.DataFrame(index=speeds.index, columns=columns, dtype=float)
    for sensor in columns:
        nbrs = [node for node in neighbors.get(sensor, []) if node in speeds.columns]
        if not nbrs:
            continue
        result[sensor] = speeds[nbrs].mean(axis=1, skipna=True)
    return result


def neighbor_min_speed(
    speeds: pd.DataFrame,
    neighbors: dict[str, list[str]],
) -> pd.DataFrame:
    """Return the minimum neighbor speed (mph) for each sensor."""
    columns = [str(col) for col in speeds.columns]
    result = pd.DataFrame(index=speeds.index, columns=columns, dtype=float)
    for sensor in columns:
        nbrs = [node for node in neighbors.get(sensor, []) if node in speeds.columns]
        if not nbrs:
            continue
        result[sensor] = speeds[nbrs].min(axis=1, skipna=True)
    return result


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in kilometers (WGS84 degrees in, km out)."""
    radius_km = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
