"""Compact XGBoost speed forecaster used for both training and serving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from trafficflow.features.supervised import feature_columns

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 120,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "n_jobs": 2,
    "tree_method": "hist",
    "objective": "reg:squarederror",
}


@dataclass
class SpeedForecaster:
    """Wrapper around a trained ``XGBRegressor`` plus feature metadata."""

    model: XGBRegressor
    include_spatial: bool
    feature_names: list[str]
    name: str
    params: dict[str, Any]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        ordered = features.reindex(columns=self.feature_names)
        return np.asarray(self.model.predict(ordered), dtype=float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "include_spatial": self.include_spatial,
                "feature_names": self.feature_names,
                "name": self.name,
                "params": self.params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SpeedForecaster":
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            include_spatial=bool(payload["include_spatial"]),
            feature_names=list(payload["feature_names"]),
            name=str(payload["name"]),
            params=dict(payload.get("params") or {}),
        )


def train_speed_forecaster(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    include_spatial: bool,
    name: str,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> SpeedForecaster:
    """Fit a compact XGBoost regressor on the provided design matrix."""
    settings = dict(DEFAULT_XGB_PARAMS)
    if params:
        settings.update(params)
    settings["random_state"] = seed
    model = XGBRegressor(**settings)
    model.fit(X, y)
    return SpeedForecaster(
        model=model,
        include_spatial=include_spatial,
        feature_names=feature_columns(include_spatial),
        name=name,
        params=settings,
    )
