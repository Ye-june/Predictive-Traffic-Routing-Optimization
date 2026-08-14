"""Clean the traffic matrix and write interim parquet artifacts.

Raw HDF5 files are left unchanged. Cleaning decisions are logged and stored
alongside an imputed-value mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trafficflow.data.cleaning import clean_speed_frame
from trafficflow.data.loader import load_traffic_frame
from trafficflow.utils.config import load_config
from trafficflow.utils.logging import get_logger
from trafficflow.utils.paths import resolve_path

logger = get_logger(__name__)


def main() -> int:
    config = load_config("data")
    paths = config["paths"]
    cleaning_cfg = config["cleaning"]

    frame = load_traffic_frame(resolve_path(paths["traffic_h5"]))
    result = clean_speed_frame(
        frame,
        treat_nonpositive_as_missing=bool(cleaning_cfg["treat_nonpositive_as_missing"]),
        max_speed_mph=float(cleaning_cfg["max_speed_mph"]),
        interpolate_limit_steps=int(cleaning_cfg["interpolate_limit_steps"]),
        interpolation_method=str(cleaning_cfg["interpolation_method"]),
    )

    speed_path = resolve_path(paths["cleaned_parquet"])
    mask_path = resolve_path(paths["missing_mask_parquet"])
    imputed_path = resolve_path(paths["imputed_flag_parquet"])
    speed_path.parent.mkdir(parents=True, exist_ok=True)

    result.speeds.to_parquet(speed_path)
    result.missing_mask.to_parquet(mask_path)
    result.imputed_mask.to_parquet(imputed_path)

    n_cells = int(result.speeds.size)
    summary = {
        "cleaned_parquet": str(speed_path),
        "missing_mask_parquet": str(mask_path),
        "imputed_flag_parquet": str(imputed_path),
        "n_originally_missing": result.n_originally_missing,
        "n_imputed": result.n_imputed,
        "n_remaining_missing": result.n_remaining_missing,
        "pct_originally_missing": round(100.0 * result.n_originally_missing / n_cells, 4),
        "pct_imputed": round(100.0 * result.n_imputed / n_cells, 4),
        "pct_remaining_missing": round(100.0 * result.n_remaining_missing / n_cells, 4),
        "interpolate_limit_steps": result.interpolate_limit_steps,
        "method": "ffill",
        "treat_nonpositive_as_missing": cleaning_cfg["treat_nonpositive_as_missing"],
        "max_speed_mph": cleaning_cfg["max_speed_mph"],
        "limitations": [
            "Short gaps are forward-filled from each sensor's own past only.",
            "Gaps longer than interpolate_limit_steps remain NaN.",
            "Sentinel zeros are treated as missing when configured; this follows "
            "the common METR-LA convention and is verified in the quality report.",
        ],
    }
    summary_path = resolve_path("outputs/metrics/cleaning_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Cleaning summary written to %s", summary_path)
    logger.info(
        "Imputed %s / %s originally missing cells; %s remain missing",
        f"{result.n_imputed:,}",
        f"{result.n_originally_missing:,}",
        f"{result.n_remaining_missing:,}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
