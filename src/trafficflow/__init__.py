"""TrafficFlow: spatiotemporal traffic forecasting and predictive routing."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trafficflow")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
