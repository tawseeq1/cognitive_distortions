"""Small I/O helpers shared across the pipeline.

WHY THIS FILE EXISTS
--------------------
The notebooks repeated the same ``to_csv`` / ``read_csv`` / ``os.makedirs``
boilerplate (often with the path left as the literal string ``'your_path'``).
Centralising these helpers guarantees output directories always exist and that
DataFrames and figures are written consistently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create the directory (and parents) for ``path`` and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, path: PathLike, index: bool = False) -> Path:
    """Write ``df`` to CSV, creating parent directories as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)
    logger.info("Wrote %d rows -> %s", len(df), p)
    return p


def read_csv(path: PathLike, **kwargs) -> pd.DataFrame:
    """Read a CSV, tolerating malformed lines the way the original notebooks did.

    The teen comment dumps contained a handful of bad rows; the notebooks read
    them with ``on_bad_lines='skip', engine='python'``. We default to the robust
    settings but allow callers to override via ``**kwargs``.
    """
    defaults = {"on_bad_lines": "skip", "engine": "python"}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def save_figure(fig, path: PathLike, dpi: int = 200) -> Path:
    """Save a matplotlib figure to ``path`` (parents created), then close it."""
    import matplotlib.pyplot as plt

    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", p)
    return p
