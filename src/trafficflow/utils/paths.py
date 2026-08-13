"""Project path helpers.

All generated artifacts are resolved relative to the repository root so
scripts remain runnable from any working directory.
"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (the directory that contains ``src/``)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a config-relative path against the project root.

    Parameters
    ----------
    relative_path:
        Path as stored in YAML configs, typically relative to the repo root.

    Returns
    -------
    pathlib.Path
        Absolute path. Absolute inputs are returned unchanged.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return get_project_root() / path
