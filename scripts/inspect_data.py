"""Inspect downloaded METR-LA files and write a data-quality report.

This script does not modify raw files. Run after ``scripts/download_data.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trafficflow.data.loader import (
    load_distance_table,
    load_sensor_ids,
    load_sensor_locations,
    load_traffic_frame,
)
from trafficflow.data.quality import build_quality_report, write_quality_report
from trafficflow.data.validation import validate_traffic_frame
from trafficflow.utils.config import load_config
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import resolve_path

logger = get_logger(__name__)


def main() -> int:
    config = load_config("data")
    paths = config["paths"]

    frame = load_traffic_frame(resolve_path(paths["traffic_h5"]))
    sensor_ids = load_sensor_ids(resolve_path(paths["sensor_ids"]))
    locations = load_sensor_locations(resolve_path(paths["sensor_locations"]))

    distance_overlap = {
        "n_distance_rows": None,
        "n_ids_in_distance_from": None,
        "n_ids_in_distance_to": None,
        "overlap_with_traffic_from": None,
        "overlap_with_traffic_to": None,
        "note": (
            "The published distances_la_2012.csv often uses a different "
            "identifier space than METR-LA sensor IDs. Overlap is measured, "
            "not assumed."
        ),
    }
    distance_path = resolve_path(paths["distances"])
    if distance_path.exists():
        distances = load_distance_table(distance_path)
        traffic_ids = set(frame.columns.astype(str))
        from_ids = set(distances["from"].astype(str))
        to_ids = set(distances["to"].astype(str))
        distance_overlap.update(
            {
                "n_distance_rows": int(len(distances)),
                "n_ids_in_distance_from": len(from_ids),
                "n_ids_in_distance_to": len(to_ids),
                "overlap_with_traffic_from": len(traffic_ids & from_ids),
                "overlap_with_traffic_to": len(traffic_ids & to_ids),
            }
        )

    validation = validate_traffic_frame(
        frame,
        sensor_ids=sensor_ids,
        locations=locations,
        expected_frequency_minutes=config["literature_reference"]["frequency_minutes"],
        max_speed_mph=float(config["cleaning"]["max_speed_mph"]),
    )
    report = build_quality_report(
        frame,
        validation=validation,
        sensor_ids=sensor_ids,
        locations=locations,
        literature=config["literature_reference"],
        missing_sentinel=float(config["literature_reference"]["missing_sentinel"]),
    )
    report["distance_table"] = distance_overlap
    report["schema"] = {
        "index_name": str(frame.index.name),
        "index_dtype": str(frame.index.dtype),
        "n_columns": int(frame.shape[1]),
        "column_sample": list(frame.columns[:10].astype(str)),
        "value_dtype": str(frame.dtypes.iloc[0]) if len(frame.dtypes) else None,
        "units": {
            "speed": config["dataset"]["speed_unit"],
            "distance_file_cost": config["dataset"]["distance_unit"],
        },
    }

    write_quality_report(
        report,
        json_path=paths["quality_report_json"],
        markdown_path=paths["quality_report_md"],
    )
    logger.info("Inspection summary:\n%s", json.dumps({
        "n_timestamps": report["n_timestamps"],
        "n_sensors": report["n_sensors"],
        "date_range": report["date_range"],
        "sampling_interval_minutes": report["sampling_interval_minutes"],
        "missing_pct_combined": report["missing"]["pct_combined"],
        "issues": report["issues"],
        "distance_overlap_from": distance_overlap["overlap_with_traffic_from"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
