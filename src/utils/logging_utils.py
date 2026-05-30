"""Centralised logging configuration.

WHY THIS FILE EXISTS
--------------------
The original notebooks scattered ``print(...)`` statements everywhere, which made
it impossible to control verbosity or capture a run log. This module provides a
single ``get_logger`` helper so every module in ``src/`` logs in a consistent,
timestamped format and the whole pipeline can be made quiet or verbose from one
place.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO, fmt: str = _DEFAULT_FORMAT) -> None:
    """Configure the root logger once for the whole process.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
    fmt:
        ``logging`` format string.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Ensures :func:`configure_logging` has run so loggers behave consistently even
    when a module is imported and used outside the main pipeline (e.g. in a
    notebook or a unit test).
    """
    configure_logging(level=level)
    return logging.getLogger(name if name else "cogdist")
