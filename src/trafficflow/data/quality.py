"""Reproducible data-quality reporting for the traffic matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficflow.data.validation import ValidationReport, validate_traffic_frame
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import resolve_path

logger = get_logger(__name__)


def build_quality_report(
    frame: pd.DataFrame,
    *,
    validation: ValidationReport | None = None,
    sensor_ids: list[str] | None = None,
    locations: pd.DataFrame | None = None,
    literature: dict[str, Any] | None = None,
    missing_sentinel: float = 0.0,
    top_n: int = 15,
) -> dict[str, Any]:
    """Compute dataset-level quality statistics from the actual files.

    Missingness is reported both as pandas NA and as sentinel zeros, because
    METR-LA commonly stores outages as ``0.0`` rather than NaN.
    """
    if validation is None:
        validation = validate_traffic_frame(
            frame,
            sensor_ids=sensor_ids,
            locations=locations,
        )

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    sentinel_missing = numeric == missing_sentinel
    na_missing = numeric.isna()
    combined_missing = sentinel_missing | na_missing
    n_cells = int(numeric.size)
    n_sentinel = int(sentinel_missing.to_numpy().sum())
    n_na = int(na_missing.to_numpy().sum())
    n_combined = int(combined_missing.to_numpy().sum())

    missing_by_sensor = (
        combined_missing.mean()
        .sort_values(ascending=False)
        .head(top_n)
        .mul(100)
        .round(3)
        .to_dict()
    )
    missing_by_timestamp = combined_missing.mean(axis=1)
    timestamp_missing_summary = {
        "mean_pct": float(missing_by_timestamp.mean() * 100),
        "median_pct": float(missing_by_timestamp.median() * 100),
        "max_pct": float(missing_by_timestamp.max() * 100),
        "timestamps_all_missing": int((missing_by_timestamp >= 1.0).sum()),
    }

    valid = numeric.mask(combined_missing)
    flattened = valid.to_numpy(dtype=float).ravel()
    flattened = flattened[np.isfinite(flattened)]

    literature_comparison: dict[str, Any] = {}
    if literature:
        literature_comparison = {
            "n_sensors_reference": literature.get("n_sensors"),
            "n_sensors_observed": int(frame.shape[1]),
            "n_timesteps_reference": literature.get("n_timesteps"),
            "n_timesteps_observed": int(frame.shape[0]),
            "frequency_minutes_reference": literature.get("frequency_minutes"),
            "frequency_minutes_observed": validation.inferred_frequency_minutes,
            "start_date_reference": literature.get("start_date"),
            "end_date_reference": literature.get("end_date"),
            "reported_missing_pct": literature.get("reported_missing_pct"),
            "observed_combined_missing_pct": round(100.0 * n_combined / n_cells, 4)
            if n_cells
            else None,
        }

    report: dict[str, Any] = {
        "n_observations": n_cells,
        "n_timestamps": int(frame.shape[0]),
        "n_sensors": int(frame.shape[1]),
        "date_range": {
            "start": validation.start_timestamp,
            "end": validation.end_timestamp,
        },
        "sampling_interval_minutes": validation.inferred_frequency_minutes,
        "inferred_frequency": validation.inferred_frequency,
        "missing": {
            "n_null": n_na,
            "n_sentinel": n_sentinel,
            "n_combined": n_combined,
            "pct_null": round(100.0 * n_na / n_cells, 4) if n_cells else None,
            "pct_sentinel": round(100.0 * n_sentinel / n_cells, 4) if n_cells else None,
            "pct_combined": round(100.0 * n_combined / n_cells, 4) if n_cells else None,
            "sentinel_value": missing_sentinel,
            "by_sensor_top": missing_by_sensor,
            "by_timestamp": timestamp_missing_summary,
        },
        "duplicates": {
            "n_duplicate_timestamps": validation.n_duplicate_timestamps,
        },
        "values": {
            "min": validation.min_value,
            "max": validation.max_value,
            "median": validation.median_value,
            "mean_valid": float(np.mean(flattened)) if flattened.size else None,
            "p01_valid": float(np.percentile(flattened, 1)) if flattened.size else None,
            "p99_valid": float(np.percentile(flattened, 99)) if flattened.size else None,
            "n_negative": validation.n_negative_values,
            "n_nonpositive": validation.n_nonpositive_values,
        },
        "temporal_gaps": {
            "n_missing_expected_timestamps": validation.n_missing_expected_timestamps,
        },
        "metadata": {
            "sensor_id_overlap": validation.sensor_id_overlap_with_metadata,
            "n_metadata_sensors": validation.n_metadata_sensors,
        },
        "literature_comparison": literature_comparison,
        "issues": validation.issues,
        "notes": validation.notes,
    }
    return report


def write_quality_report(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Persist the quality report as JSON and Markdown."""
    json_path = resolve_path(json_path)
    markdown_path = resolve_path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(_to_markdown(report), encoding="utf-8")
    logger.info("Wrote quality report to %s and %s", json_path, markdown_path)


def _to_markdown(report: dict[str, Any]) -> str:
    missing = report.get("missing", {})
    values = report.get("values", {})
    date_range = report.get("date_range", {})
    literature = report.get("literature_comparison", {})
    issues = report.get("issues") or ["None"]
    notes = report.get("notes") or ["None"]

    lines = [
        "# METR-LA Data Quality Report",
        "",
        "Generated from the downloaded files. Figures are computed, not assumed.",
        "",
        "## Summary",
        "",
        f"- Observations: **{report.get('n_observations')}**",
        f"- Timestamps: **{report.get('n_timestamps')}**",
        f"- Sensors: **{report.get('n_sensors')}**",
        f"- Date range: **{date_range.get('start')}** to **{date_range.get('end')}**",
        f"- Sampling interval: **{report.get('sampling_interval_minutes')}** minutes",
        "",
        "## Missingness",
        "",
        f"- Null values: {missing.get('n_null')} ({missing.get('pct_null')}%)",
        f"- Sentinel values ({missing.get('sentinel_value')}): "
        f"{missing.get('n_sentinel')} ({missing.get('pct_sentinel')}%)",
        f"- Combined missing: {missing.get('n_combined')} ({missing.get('pct_combined')}%)",
        "",
        "## Value range (including sentinels unless noted)",
        "",
        f"- Min: {values.get('min')}",
        f"- Max: {values.get('max')}",
        f"- Median: {values.get('median')}",
        f"- Mean of valid (non-missing) speeds: {values.get('mean_valid')}",
        f"- P01 / P99 of valid speeds: {values.get('p01_valid')} / {values.get('p99_valid')}",
        f"- Negative values: {values.get('n_negative')}",
        f"- Non-positive values: {values.get('n_nonpositive')}",
        "",
        "## Literature comparison",
        "",
    ]
    if literature:
        for key, value in literature.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- No literature reference provided.")

    lines.extend(
        [
            "",
            "## Issues",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in issues)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {item}" for item in notes)
    lines.append("")
    return "\n".join(lines)
