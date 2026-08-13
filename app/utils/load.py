"""Cached artifact loaders for Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from trafficflow.serving import ArtifactBundle, load_artifacts


@st.cache_resource
def get_bundle() -> ArtifactBundle:
    return load_artifacts()


def sensor_options(bundle: ArtifactBundle) -> dict[str, str]:
    """Map display label → sensor id."""
    frame = bundle.sensor_metadata.sort_values(["region", "sensor_id"])
    return {str(row.label): str(row.sensor_id) for row in frame.itertuples(index=False)}


def format_minutes(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1f} min"


def format_miles(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1f} mi"


def render_disclaimer() -> None:
    st.caption(
        "TrafficFlow v1.0 · Historical Replay Demo · METR-LA freeway sensors. "
        "This is a research and portfolio demonstration, not real-world navigation."
    )


def require_bundle():
    try:
        return get_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Local setup: `python scripts/build_deployment_assets.py`")
        st.stop()
        raise
