"""Streamlit API shims for Community Cloud (1.61+) and older local versions."""

from __future__ import annotations

import inspect
from typing import Any

import streamlit as st


def plotly_chart(fig: Any) -> None:
    params = inspect.signature(st.plotly_chart).parameters
    if "width" in params:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def dataframe(data: Any, **kwargs: Any) -> None:
    params = inspect.signature(st.dataframe).parameters
    if "width" in params:
        kwargs.setdefault("width", "stretch")
        kwargs.pop("use_container_width", None)
    else:
        kwargs.setdefault("use_container_width", True)
    st.dataframe(data, **kwargs)
