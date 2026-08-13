"""Structural validation of traffic matrices and sensor metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trafficflow.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Machine-readable summary of structural checks."""

    n_timestamps: int
    n_sensors: int
    start_timestamp: str | None
    end_timestamp: str | None
    inferred_frequency: str | None
    inferred_frequency_minutes: float | None
    n_duplicate_timestamps: int
    n_unsorted_timestamps: int
    n_null_values: int
    n_nonpositive_values: int
    n_negative_values: int
    min_value: float | None
    max_value: float | None
    median_value: float | None
    n_missing_expected_timestamps: int
    sensor_id_overlap_with_metadata: int | None = None
    n_metadata_sensors: int | None = None
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_traffic_frame(
    frame: pd.DataFrame,
    *,
    sensor_ids: list[str] | None = None,
    locations: pd.DataFrame | None = None,
    expected_frequency_minutes: int | None = 5,
    max_speed_mph: float = 90.0,
) -> ValidationReport:
    """Inspect a speed matrix without changing it.

    Parameters
    ----------
    frame:
        Timestamp x sensor speed matrix. Units are miles per hour for METR-LA.
    sensor_ids:
        Optional ordered ID list from ``graph_sensor_ids.txt``.
    locations:
        Optional sensor coordinate table.
    expected_frequency_minutes:
        Sampling interval claimed by literature; compared with the index.
    max_speed_mph:
        Speeds above this threshold are flagged as suspicious, not dropped.
    """
    issues: list[str] = []
    notes: list[str] = []

    if frame.empty:
        issues.append("Traffic frame is empty.")

    if not isinstance(frame.index, pd.DatetimeIndex):
        issues.append("Index is not a DatetimeIndex.")

    duplicate_timestamps = int(frame.index.duplicated().sum())
    if duplicate_timestamps:
        issues.append(f"Found {duplicate_timestamps} duplicate timestamps.")

    unsorted = 0
    if len(frame.index) > 1:
        unsorted = int((frame.index[1:] < frame.index[:-1]).sum())
        if unsorted:
            issues.append(f"Found {unsorted} out-of-order timestamp steps.")

    freq = frame.index.inferred_freq if isinstance(frame.index, pd.DatetimeIndex) else None
    freq_minutes = _infer_frequency_minutes(frame.index)
    if expected_frequency_minutes is not None and freq_minutes is not None:
        if abs(freq_minutes - expected_frequency_minutes) > 0.1:
            issues.append(
                f"Inferred frequency {freq_minutes} minutes differs from "
                f"expected {expected_frequency_minutes} minutes."
            )

    missing_expected = 0
    if (
        isinstance(frame.index, pd.DatetimeIndex)
        and len(frame.index) > 1
        and expected_frequency_minutes is not None
    ):
        expected_index = pd.date_range(
            frame.index.min(),
            frame.index.max(),
            freq=f"{expected_frequency_minutes}min",
        )
        missing_expected = int(expected_index.difference(frame.index).shape[0])
        if missing_expected:
            issues.append(
                f"Missing {missing_expected} timestamps relative to a regular "
                f"{expected_frequency_minutes}-minute grid."
            )

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    n_null = int(numeric.isna().sum().sum())
    n_negative = int((numeric < 0).sum().sum())
    n_nonpositive = int((numeric <= 0).sum().sum())
    finite = numeric.to_numpy(dtype=float, copy=False)
    finite = finite[np.isfinite(finite)]

    min_value = float(np.min(finite)) if finite.size else None
    max_value = float(np.max(finite)) if finite.size else None
    median_value = float(np.median(finite)) if finite.size else None

    if n_negative:
        issues.append(f"Found {n_negative} negative speed values.")
    if max_value is not None and max_value > max_speed_mph:
        issues.append(
            f"Maximum speed {max_value:.2f} mph exceeds suspicious threshold "
            f"{max_speed_mph} mph."
        )
    if n_nonpositive:
        notes.append(
            f"{n_nonpositive} non-positive values present. METR-LA commonly "
            "encodes missing readings as 0; confirm before imputation."
        )

    overlap = None
    n_metadata = None
    if sensor_ids is not None:
        frame_cols = set(frame.columns.astype(str))
        id_set = set(sensor_ids)
        overlap = len(frame_cols.intersection(id_set))
        if frame_cols != id_set:
            issues.append(
                "Traffic columns and sensor ID list do not match exactly "
                f"(overlap={overlap}, columns={len(frame_cols)}, ids={len(id_set)})."
            )
        elif list(frame.columns.astype(str)) != sensor_ids:
            notes.append("Sensor ID list matches columns as a set but not in order.")

    if locations is not None:
        n_metadata = int(locations["sensor_id"].nunique())
        loc_ids = set(locations["sensor_id"].astype(str))
        frame_cols = set(frame.columns.astype(str))
        if loc_ids != frame_cols:
            issues.append(
                "Sensor location IDs do not match traffic columns "
                f"(location_ids={len(loc_ids)}, columns={len(frame_cols)}, "
                f"overlap={len(loc_ids & frame_cols)})."
            )

    start = frame.index.min() if len(frame.index) else None
    end = frame.index.max() if len(frame.index) else None

    report = ValidationReport(
        n_timestamps=int(frame.shape[0]),
        n_sensors=int(frame.shape[1]),
        start_timestamp=None if start is None else str(start),
        end_timestamp=None if end is None else str(end),
        inferred_frequency=str(freq) if freq is not None else None,
        inferred_frequency_minutes=freq_minutes,
        n_duplicate_timestamps=duplicate_timestamps,
        n_unsorted_timestamps=unsorted,
        n_null_values=n_null,
        n_nonpositive_values=n_nonpositive,
        n_negative_values=n_negative,
        min_value=min_value,
        max_value=max_value,
        median_value=median_value,
        n_missing_expected_timestamps=missing_expected,
        sensor_id_overlap_with_metadata=overlap,
        n_metadata_sensors=n_metadata,
        issues=issues,
        notes=notes,
    )
    logger.info(
        "Validation complete: %s timestamps, %s sensors, %s issues",
        report.n_timestamps,
        report.n_sensors,
        len(issues),
    )
    return report


def _infer_frequency_minutes(index: pd.Index) -> float | None:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    deltas = index[1:] - index[:-1]
    if len(deltas) == 0:
        return None
    median_delta = pd.Series(deltas).median()
    if pd.isna(median_delta):
        return None
    return float(median_delta / pd.Timedelta(minutes=1))
