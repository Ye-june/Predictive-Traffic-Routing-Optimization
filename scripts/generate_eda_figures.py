"""Generate exploratory traffic figures from cleaned interim data."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from trafficflow.utils.config import load_config
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import resolve_path
from trafficflow.visualization.traffic import (
    plot_mean_speed_over_time,
    plot_missingness_by_sensor,
    plot_speed_by_hour,
    plot_speed_by_weekday,
    plot_speed_distribution,
)

logger = get_logger(__name__)


def main() -> int:
    config = load_config("data")
    paths = config["paths"]
    speeds = pd.read_parquet(resolve_path(paths["cleaned_parquet"]))
    missing = pd.read_parquet(resolve_path(paths["missing_mask_parquet"]))
    figures = resolve_path("outputs/figures")

    plot_mean_speed_over_time(speeds, figures / "03_mean_speed_daily.png")
    plot_speed_by_hour(speeds, figures / "04_speed_by_hour.png")
    plot_speed_by_weekday(speeds, figures / "05_speed_by_weekday.png")
    plot_speed_distribution(speeds, figures / "06_speed_distribution.png")
    plot_missingness_by_sensor(missing, figures / "07_missing_by_sensor.png")
    logger.info("Wrote EDA figures to %s", figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
