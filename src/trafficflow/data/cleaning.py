"""Cleaning utilities for traffic speed matrices.

Raw source files are never modified. Cleaning writes new interim artifacts
and an imputed-value mask so downstream models can ignore or weight filled
observations if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trafficflow.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningResult:
    """Cleaned speed matrix plus provenance flags."""

    speeds: pd.DataFrame
    missing_mask: pd.DataFrame
    imputed_mask: pd.DataFrame
    n_originally_missing: int
    n_imputed: int
    n_remaining_missing: int
    interpolate_limit_steps: int


def clean_speed_frame(
    frame: pd.DataFrame,
    *,
    treat_nonpositive_as_missing: bool = True,
    max_speed_mph: float = 90.0,
    interpolate_limit_steps: int = 12,
    interpolation_method: str = "time",
) -> CleaningResult:
    """Return a cleaned speed matrix without touching the raw file.

    Policy
    ------
    1. Non-positive values may be treated as missing (METR-LA convention).
    2. Speeds above ``max_speed_mph`` are set to missing as physically
       implausible for this freeway loop-detector corpus.
    3. Short gaps (up to ``interpolate_limit_steps`` observations) are
       interpolated along time within each sensor.
    4. Remaining gaps are left as NaN. They are **not** filled with global
       means, which would hide long outages.

    Parameters
    ----------
    frame:
        Raw timestamp x sensor speed matrix, miles per hour.
    treat_nonpositive_as_missing:
        If True, values ``<= 0`` become missing.
    max_speed_mph:
        Values strictly above this threshold become missing.
    interpolate_limit_steps:
        Maximum consecutive missing steps filled by interpolation.
        At 5-minute sampling, 12 steps equals one hour.
    interpolation_method:
        Pandas interpolation method. ``time`` uses the DatetimeIndex.
    """
    speeds = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    original_missing = speeds.isna()
    if treat_nonpositive_as_missing:
        original_missing = original_missing | (speeds <= 0)
    original_missing = original_missing | (speeds > max_speed_mph)

    cleaned = speeds.mask(original_missing)
    interpolated = cleaned.interpolate(
        method=interpolation_method,
        axis=0,
        limit=interpolate_limit_steps,
        limit_direction="both",
    )
    imputed_mask = original_missing & interpolated.notna()
    remaining_missing = interpolated.isna()

    n_orig = int(original_missing.to_numpy().sum())
    n_imputed = int(imputed_mask.to_numpy().sum())
    n_remain = int(remaining_missing.to_numpy().sum())
    logger.info(
        "Cleaning: originally_missing=%s imputed=%s remaining_missing=%s "
        "(limit=%s steps, method=%s)",
        f"{n_orig:,}",
        f"{n_imputed:,}",
        f"{n_remain:,}",
        interpolate_limit_steps,
        interpolation_method,
    )

    return CleaningResult(
        speeds=interpolated,
        missing_mask=original_missing,
        imputed_mask=imputed_mask,
        n_originally_missing=n_orig,
        n_imputed=n_imputed,
        n_remaining_missing=n_remain,
        interpolate_limit_steps=interpolate_limit_steps,
    )


def congestion_ratio(
    speeds: pd.DataFrame,
    *,
    free_flow: pd.Series,
    floor: float = 1e-6,
) -> pd.DataFrame:
    """Compute ``current_speed / free_flow_speed`` per sensor.

    Parameters
    ----------
    speeds:
        Speed matrix in mph.
    free_flow:
        Per-sensor free-flow speed in mph, typically a high percentile.
    floor:
        Minimum divisor to avoid division by zero.

    Returns
    -------
    pandas.DataFrame
        Unitless ratio. Values near 1 indicate free-flow; lower is slower.
    """
    denom = free_flow.clip(lower=floor)
    return speeds.div(denom, axis=1)


def estimate_free_flow_speed(
    speeds: pd.DataFrame,
    *,
    percentile: float = 95.0,
    overnight_hours: tuple[int, int] = (0, 5),
) -> pd.Series:
    """Estimate free-flow speed per sensor.

    Uses the given percentile of observed (non-missing) speeds. Overnight
    hours are recorded for documentation; the percentile is computed on all
    valid observations so a few noisy night spikes do not dominate.

    Units
    -----
    Input and output are miles per hour.
    """
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100].")
    free_flow = speeds.quantile(percentile / 100.0)
    logger.info(
        "Estimated free-flow speed at p%s: median=%.2f mph, overnight_hours=%s",
        percentile,
        float(np.nanmedian(free_flow.to_numpy())),
        overnight_hours,
    )
    return free_flow
