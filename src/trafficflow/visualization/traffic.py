"""Traffic time-series and missingness plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mean_speed_over_time(
    speeds: pd.DataFrame,
    path: Path,
    *,
    resample: str = "D",
) -> Path:
    """Plot network-wide mean speed over time.

    Parameters
    ----------
    speeds:
        Timestamp × sensor matrix in mph.
    path:
        Output image path.
    resample:
        Pandas offset alias used to aggregate the series.
    """
    series = speeds.mean(axis=1, skipna=True).resample(resample).mean()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(series.index, series.values, color="#1f4e79", linewidth=1.4)
    ax.set_title(f"Network-wide mean traffic speed ({resample} average)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Speed (mph)")
    return _save(fig, path)


def plot_speed_by_hour(speeds: pd.DataFrame, path: Path) -> Path:
    """Plot mean speed by hour of day."""
    hourly = speeds.mean(axis=1, skipna=True).groupby(speeds.index.hour).mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hourly.index, hourly.values, marker="o", color="#1f4e79")
    ax.set_title("Average traffic speed by hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Speed (mph)")
    ax.set_xticks(range(0, 24))
    return _save(fig, path)


def plot_speed_by_weekday(speeds: pd.DataFrame, path: Path) -> Path:
    """Plot mean speed by weekday."""
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = speeds.mean(axis=1, skipna=True).groupby(speeds.index.dayofweek).mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(7), [daily.get(i, float("nan")) for i in range(7)], color="#4c78a8")
    ax.set_xticks(range(7), labels)
    ax.set_title("Average traffic speed by weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Speed (mph)")
    return _save(fig, path)


def plot_speed_distribution(speeds: pd.DataFrame, path: Path) -> Path:
    """Histogram of valid (non-missing) speeds in mph."""
    values = speeds.to_numpy().ravel()
    values = values[pd.notna(values)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=50, color="#4c78a8", edgecolor="white")
    ax.set_title("Distribution of observed traffic speeds")
    ax.set_xlabel("Speed (mph)")
    ax.set_ylabel("Count")
    return _save(fig, path)


def plot_missingness_by_sensor(
    missing_mask: pd.DataFrame,
    path: Path,
    *,
    top_n: int = 20,
) -> Path:
    """Bar chart of sensors with the highest original missing rates."""
    rates = missing_mask.mean().sort_values(ascending=False).head(top_n) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(rates.index.astype(str)[::-1], rates.values[::-1], color="#d62728")
    ax.set_title(f"Highest missing rates by sensor (top {top_n})")
    ax.set_xlabel("Missing observations (%)")
    ax.set_ylabel("Sensor ID")
    return _save(fig, path)
