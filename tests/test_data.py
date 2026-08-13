"""Tests for configuration, path resolution, and synthetic data checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trafficflow.data.cleaning import clean_speed_frame
from trafficflow.data.validation import validate_traffic_frame
from trafficflow.utils.config import load_config
from trafficflow.utils.paths import get_project_root, resolve_path
from trafficflow.utils.seeds import set_seeds


def test_project_root_contains_src() -> None:
    root = get_project_root()
    assert (root / "src" / "trafficflow").is_dir()
    assert (root / "configs" / "data.yaml").is_file()


def test_resolve_path_joins_root() -> None:
    path = resolve_path("data/raw/metr-la.h5")
    assert path == get_project_root() / "data" / "raw" / "metr-la.h5"


def test_load_configs() -> None:
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    routing_cfg = load_config("routing")
    assert data_cfg["dataset"]["name"] == "metr-la"
    assert data_cfg["dataset"]["speed_unit"] == "mph"
    assert "horizons_steps" in model_cfg
    assert routing_cfg["units"]["travel_time"] == "minutes"


def test_missing_config_raises() -> None:
    load_config.cache_clear()
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
    load_config.cache_clear()


def test_set_seeds_reproducible() -> None:
    set_seeds(123)
    first = np.random.rand(4)
    set_seeds(123)
    second = np.random.rand(4)
    np.testing.assert_array_equal(first, second)


def _synthetic_speed_frame() -> pd.DataFrame:
    index = pd.date_range("2012-03-01", periods=20, freq="5min")
    data = np.full((20, 3), 50.0)
    data[5, 0] = 0.0
    data[6, 0] = 0.0
    data[10, 1] = np.nan
    columns = ["773869", "767541", "767542"]
    return pd.DataFrame(data, index=index, columns=columns)


def test_validate_synthetic_frame_sorted_and_regular() -> None:
    frame = _synthetic_speed_frame()
    report = validate_traffic_frame(frame, expected_frequency_minutes=5)
    assert report.n_timestamps == 20
    assert report.n_sensors == 3
    assert report.n_duplicate_timestamps == 0
    assert report.inferred_frequency_minutes == 5
    assert report.n_missing_expected_timestamps == 0


def test_cleaning_interpolates_short_zero_gaps() -> None:
    frame = _synthetic_speed_frame()
    result = clean_speed_frame(
        frame,
        treat_nonpositive_as_missing=True,
        interpolate_limit_steps=4,
    )
    assert result.missing_mask.iloc[5, 0]
    assert result.imputed_mask.iloc[5, 0]
    assert np.isfinite(result.speeds.iloc[5, 0])
    assert result.n_remaining_missing == 0


def test_cleaning_does_not_fill_long_gaps() -> None:
    index = pd.date_range("2012-03-01", periods=30, freq="5min")
    data = np.full((30, 1), 40.0)
    data[5:20, 0] = 0.0
    frame = pd.DataFrame(data, index=index, columns=["s1"])
    result = clean_speed_frame(
        frame,
        treat_nonpositive_as_missing=True,
        interpolate_limit_steps=3,
    )
    assert result.n_remaining_missing > 0
    assert result.speeds.isna().sum().sum() == result.n_remaining_missing
