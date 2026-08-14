"""Evaluate persistence and historical-pattern baselines on a chronological split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from trafficflow.features.temporal import chronological_split_indices
from trafficflow.models.baseline import historical_pattern_forecast
from trafficflow.models.evaluation import frame_metrics
from trafficflow.utils.config import load_config
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import resolve_path
from trafficflow.utils.seeds import set_seeds

logger = get_logger(__name__)


def _split_labels(index: pd.DatetimeIndex, slices: tuple[slice, slice, slice]) -> dict[str, str]:
    names = ("train", "validation", "test")
    out: dict[str, str] = {}
    for name, slc in zip(names, slices, strict=True):
        part = index[slc]
        out[f"{name}_start"] = str(part[0])
        out[f"{name}_end"] = str(part[-1])
        out[f"{name}_n"] = str(len(part))
    return out


def main() -> int:
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    set_seeds(int(data_cfg["seed"]))

    speeds = pd.read_parquet(resolve_path(data_cfg["paths"]["cleaned_parquet"]))
    splits = chronological_split_indices(
        len(speeds),
        train_ratio=float(data_cfg["splits"]["train_ratio"]),
        val_ratio=float(data_cfg["splits"]["val_ratio"]),
    )
    train_slc, val_slc, test_slc = splits
    split_info = _split_labels(speeds.index, splits)
    logger.info("Chronological split: %s", split_info)

    horizons = model_cfg["horizons_steps"]
    rows: list[dict] = []
    train_index = speeds.index[train_slc]

    for horizon_name, horizon_steps in horizons.items():
        horizon_steps = int(horizon_steps)
        seasonal = historical_pattern_forecast(
            speeds,
            train_index=train_index,
            horizon_steps=horizon_steps,
        )
        for split_name, slc in (("validation", val_slc), ("test", test_slc)):
            # Same (t → t+h) pairing used by the XGBoost export path.
            feature_times = speeds.index[slc]
            target = speeds.shift(-horizon_steps).loc[feature_times]
            persist_pred = speeds.loc[feature_times]
            target_times = feature_times + pd.Timedelta(minutes=5 * horizon_steps)
            # Drop rows whose target falls outside the frame.
            valid = target_times.isin(speeds.index)
            feature_times = feature_times[valid]
            target = target.loc[feature_times]
            persist_pred = persist_pred.loc[feature_times]
            hist_pred = seasonal.loc[target_times[valid]].copy()
            hist_pred.index = feature_times
            for model_name, pred in (
                ("persistence", persist_pred),
                ("historical_dow_hour", hist_pred),
            ):
                metrics = frame_metrics(target, pred)
                row = {
                    "model": model_name,
                    "feature_set": "none" if model_name == "persistence" else "sensor+dow+hour mean (train only)",
                    "horizon_name": horizon_name,
                    "horizon_steps": horizon_steps,
                    "horizon_minutes": horizon_steps * 5,
                    "split": split_name,
                    "mae_mph": metrics["mae"],
                    "rmse_mph": metrics["rmse"],
                    "n_eval_cells": metrics["n"],
                }
                rows.append(row)
                logger.info(
                    "%s | %s | %s min | MAE=%.3f mph RMSE=%.3f mph n=%s",
                    model_name,
                    split_name,
                    row["horizon_minutes"],
                    metrics["mae"],
                    metrics["rmse"],
                    f"{metrics['n']:,}",
                )

    frame = pd.DataFrame(rows)
    out_csv = resolve_path("outputs/metrics/baseline_forecast_metrics.csv")
    out_json = resolve_path("outputs/metrics/baseline_forecast_metrics.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    payload = {"split": split_info, "target_unit": "mph", "results": rows}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote baseline metrics to %s", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
