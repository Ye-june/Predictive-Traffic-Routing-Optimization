"""YAML configuration loader."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from trafficflow.utils.paths import get_project_root


@lru_cache(maxsize=8)
def load_config(name: str) -> dict[str, Any]:
    """Load a YAML file from ``configs/{name}.yaml``.

    Parameters
    ----------
    name:
        Config stem without extension, e.g. ``"data"`` or ``"model"``.

    Returns
    -------
    dict
        Parsed YAML mapping.

    Raises
    ------
    FileNotFoundError
        If the requested config file does not exist.
    """
    path = get_project_root() / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return loaded
