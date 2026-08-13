"""Logging helpers with a consistent, low-noise default format."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Return a module logger, attaching handlers only once.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__``.
    level:
        Default log level. Debug output is off unless raised explicitly.
    log_file:
        Optional file destination in addition to stderr.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
