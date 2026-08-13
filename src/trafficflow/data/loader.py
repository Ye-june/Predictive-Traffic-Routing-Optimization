"""Load raw METR-LA traffic and sensor metadata without modifying source files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trafficflow.utils.logging import get_logger

logger = get_logger(__name__)


def load_traffic_frame(path: str | Path) -> pd.DataFrame:
    """Load the METR-LA speed matrix from an HDF5 file.

    The canonical DCRNN file stores a pandas DataFrame in HDF5 with:

    * index: timestamps
    * columns: sensor IDs
    * values: traffic speed (miles per hour in the published METR-LA release)

    Raw files are never overwritten. Missingness encoding (often zeros) is
    left intact for the validation and cleaning stages.

    Parameters
    ----------
    path:
        Path to ``metr-la.h5``.

    Returns
    -------
    pandas.DataFrame
        Speed matrix with a DatetimeIndex and sensor-ID columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Traffic file not found: {path}. Run `python scripts/download_data.py` first."
        )

    logger.info("Loading traffic HDF5 from %s", path)
    try:
        frame = pd.read_hdf(path)
    except (ImportError, ValueError, OSError):
        frame = _load_traffic_with_h5py(path)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {path}, got {type(frame)!r}.")

    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    frame.sort_index(inplace=True)
    frame.columns = [str(col) for col in frame.columns]
    logger.info(
        "Loaded traffic frame: %s timestamps x %s sensors",
        f"{len(frame):,}",
        frame.shape[1],
    )
    return frame


def _load_traffic_with_h5py(path: Path) -> pd.DataFrame:
    """Fallback reader when PyTables is unavailable."""
    import h5py

    logger.info("Reading HDF5 via h5py fallback: %s", path)
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        logger.info("HDF5 top-level keys: %s", keys)
        if "df" in handle:
            return _pandas_like_from_group(handle["df"])
        if len(keys) == 1:
            obj = handle[keys[0]]
            if isinstance(obj, h5py.Group):
                return _pandas_like_from_group(obj)
        raise ValueError(
            f"Unrecognized HDF5 layout in {path}. Keys: {keys}. "
            "Install PyTables (`tables`) for pandas HDFStore support."
        )


def _pandas_like_from_group(group: object) -> pd.DataFrame:
    """Best-effort reconstruction of a pandas HDFStore block from h5py."""
    import h5py

    if not isinstance(group, h5py.Group):
        raise TypeError("Expected an HDF5 group.")

    axis0 = _decode_axis(group["axis0"][()]) if "axis0" in group else None
    axis1 = _decode_axis(group["axis1"][()]) if "axis1" in group else None
    value_key = next(
        (key for key in group.keys() if key.endswith("_values")),
        None,
    )
    if value_key is None:
        raise ValueError(f"No value block found in HDF5 group keys: {list(group.keys())}")
    values = group[value_key][()]
    columns = axis0
    index = pd.to_datetime(axis1) if axis1 is not None else None
    return pd.DataFrame(values, index=index, columns=columns)


def _decode_axis(raw: object) -> list[str]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    decoded: list[str] = []
    for item in raw:  # type: ignore[union-attr]
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def load_sensor_locations(path: str | Path) -> pd.DataFrame:
    """Load sensor coordinates.

    Expected columns: ``index``, ``sensor_id``, ``latitude``, ``longitude``.

    Coordinates are WGS84 decimal degrees. They locate loop detectors, not
    full road-centerline geometry.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor location file not found: {path}")

    locations = pd.read_csv(path)
    required = {"sensor_id", "latitude", "longitude"}
    missing = required.difference(locations.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    locations = locations.copy()
    locations["sensor_id"] = locations["sensor_id"].astype(str)
    logger.info("Loaded %s sensor locations from %s", len(locations), path)
    return locations


def load_sensor_ids(path: str | Path) -> list[str]:
    """Load the ordered sensor ID list used by the traffic matrix columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor ID file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Sensor ID file is empty: {path}")
    sensor_ids = [item.strip() for item in text.split(",") if item.strip()]
    logger.info("Loaded %s sensor IDs from %s", len(sensor_ids), path)
    return sensor_ids


def load_distance_table(path: str | Path) -> pd.DataFrame:
    """Load the DCRNN pairwise distance table if present.

    Columns are ``from``, ``to``, ``cost`` with ``cost`` in meters. The
    published ``distances_la_2012.csv`` uses a broader PeMS identifier space
    that often does **not** match METR-LA ``sensor_id`` values. Callers must
    verify overlap before using this table as graph adjacency.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Distance file not found: {path}")

    distances = pd.read_csv(path)
    required = {"from", "to", "cost"}
    missing = required.difference(distances.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    distances = distances.copy()
    distances["from"] = distances["from"].astype(str)
    distances["to"] = distances["to"].astype(str)
    logger.info("Loaded %s distance rows from %s", f"{len(distances):,}", path)
    return distances
