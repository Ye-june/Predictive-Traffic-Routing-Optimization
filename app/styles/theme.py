"""Design tokens and reusable Streamlit UI helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).with_name("theme.css")

COLORS = {
    "bg": "#F5F7F8",
    "surface": "#FFFFFF",
    "text": "#17212B",
    "muted": "#64717D",
    "line": "#E4E8EC",
    "blue": "#2563EB",
    "teal": "#0F8B8D",
    "warm": "#D98C4A",
    "positive": "#2E8B57",
    "warning": "#D49A2A",
    "negative": "#C94C4C",
    "navy": "#18232E",
}


def apply_theme() -> None:
    """Inject the product stylesheet once per session render."""
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="tf-brand">
          <div class="tf-mark">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <circle cx="3.5" cy="14" r="2" fill="white"/>
              <circle cx="9" cy="5" r="2" fill="white"/>
              <circle cx="14.5" cy="12.5" r="2" fill="white"/>
              <path d="M5 12.6 L7.7 6.6" stroke="white" stroke-width="1.5"/>
              <path d="M10.7 6.3 L13 11" stroke="white" stroke-width="1.5"/>
            </svg>
          </div>
          <div>
            <h1>TrafficFlow</h1>
            <p>Predictive mobility intelligence</p>
          </div>
        </div>
        <div class="tf-status"><i></i> Historical Replay</div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, badges: list[tuple[str, str]] | None = None) -> None:
    chips = ""
    if badges:
        parts = [f'<span class="tf-badge {kind}">{label}</span>' for label, kind in badges]
        chips = f'<div class="tf-badges">{"".join(parts)}</div>'
    st.markdown(
        f"""
        <div class="tf-page-head">
          <h1>{title}</h1>
          <p>{subtitle}</p>
          {chips}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str = "", variant: str = "neutral") -> str:
    cls = f"tf-metric {variant}".strip()
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    return (
        f'<div class="{cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{hint_html}</div>'
    )


def metrics_row(cards: list[str]) -> None:
    st.markdown(
        '<div class="tf-metrics-row">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def callout(title: str, body: str, variant: str = "info") -> None:
    st.markdown(
        f'<div class="tf-callout {variant}"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="tf-empty"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def badge_row(items: list[tuple[str, str]]) -> None:
    chips = "".join(f'<span class="tf-badge {kind}">{label}</span>' for label, kind in items)
    st.markdown(f'<div class="tf-badges">{chips}</div>', unsafe_allow_html=True)


def route_compare(rows: list[dict]) -> None:
    """rows: name, minutes, width_pct, best, color."""
    blocks = []
    for row in rows:
        best = " best" if row.get("best") else ""
        color = row.get("color", COLORS["blue"])
        blocks.append(
            f"""
            <div class="tf-row{best}">
              <div class="name">{row["name"]}</div>
              <div class="tf-bar"><span style="width:{row["width_pct"]:.0f}%;background:{color};"></span></div>
              <div class="time">{row["minutes"]}</div>
            </div>
            """
        )
    st.markdown('<div class="tf-compare">' + "".join(blocks) + "</div>", unsafe_allow_html=True)


def info_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="tf-card"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def footer(manifest: dict | None = None) -> None:
    version = (manifest or {}).get("app_version", "1.0.0")
    dataset = (manifest or {}).get("dataset", "METR-LA")
    st.markdown(
        f"""
        <div class="tf-footer">
          TrafficFlow v{version} · Historical traffic simulation using {dataset}.
          Not intended for real-world navigation.<br/>
          Built with Python, XGBoost, NetworkX, and Streamlit ·
          <a href="https://github.com/Ye-june/Predictive-Traffic-Routing-Optimization">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_page(title: str) -> None:
    """Shared page chrome: config-safe theme + sidebar brand."""
    apply_theme()
    sidebar_brand()
    st.sidebar.caption("Choose a destination on the sensor network, then compare routing strategies.")
