"""Chart helpers must not pass duplicate Plotly layout kwargs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_APP = Path(__file__).resolve().parents[1] / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from utils.charts import network_preview, speed_map  # noqa: E402


def test_network_preview_builds_without_duplicate_margin() -> None:
    frame = pd.DataFrame(
        {
            "sensor_id": ["a", "b"],
            "longitude": [-118.3, -118.2],
            "latitude": [34.1, 34.15],
            "label": ["Sensor a", "Sensor b"],
        }
    )
    fig = network_preview(frame)
    assert fig.layout.margin.t == 48
    assert len(fig.data) == 1


def test_speed_map_builds() -> None:
    frame = pd.DataFrame(
        {
            "sensor_id": ["a"],
            "longitude": [-118.3],
            "latitude": [34.1],
            "label": ["Sensor a"],
        }
    )
    fig = speed_map(frame, {"a": 55.0}, "Observed speed")
    assert fig.layout.height == 520
