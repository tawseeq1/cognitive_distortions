"""Plotting helpers for the manuscript figures.

WHY THIS FILE EXISTS
--------------------
The figures in ``paper/figures`` were produced by long, copy-pasted matplotlib
blocks in ``plots_masked.ipynb``, ``correlation.ipynb`` and the topic-modelling
notebooks. Each plot type is reduced here to one parameterised function that
returns a :class:`matplotlib.figure.Figure`, so notebooks and scripts can render
and save figures without duplicating styling code.

All functions use a non-interactive Agg backend when imported headlessly and
never call ``plt.show`` -- callers save the returned figure with
``src.utils.io.save_figure``.
"""
from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # safe default for scripts/CI; notebooks can override.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:  # seaborn only needed for heatmaps; degrade gracefully if missing.
    import seaborn as sns

    _HAS_SNS = True
except Exception:  # pragma: no cover
    _HAS_SNS = False

COVID_ONSET = datetime.date(2020, 1, 22)  # r/SGExams COVID marker used in notebooks.


def timeseries_tiles(
    series: pd.DataFrame,
    dates: Optional[Sequence] = None,
    title: str = "Cognitive distortion time series",
    covid_marker: Optional[datetime.date] = COVID_ONSET,
    ylim: Optional[tuple] = None,
    ncols: int = 3,
) -> plt.Figure:
    """Render one small-multiples tile per distortion category.

    Reproduces the 4x3 z-score/raw/filtered tile grids from ``plots_masked.ipynb``.
    """
    cols = list(series.columns)
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    x = list(dates) if dates is not None else list(range(len(series)))
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        ax.plot(x, series[cols[i]].to_numpy())
        if covid_marker is not None and dates is not None:
            ax.axvline(x=covid_marker, color="red", linestyle="--")
        ax.set_title(cols[i])
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def timeseries_overlay(
    series: pd.DataFrame,
    dates: Optional[Sequence] = None,
    title: str = "All distortion series",
) -> plt.Figure:
    """Overlay all distortion series on a single axis (the 'multiple' figures)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(dates) if dates is not None else list(range(len(series)))
    for col in series.columns:
        ax.plot(x, series[col].to_numpy(), label=col)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Correlation matrix",
    labels: Optional[List[str]] = None,
    vmin: float = -0.4,
    vmax: float = 1.0,
) -> plt.Figure:
    """Render a correlation heatmap (seaborn if available, else matplotlib)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ticklabels = labels if labels is not None else list(corr.columns)
    if _HAS_SNS:
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=ticklabels, yticklabels=ticklabels,
            vmin=vmin, vmax=vmax, ax=ax,
        )
    else:  # pragma: no cover - fallback path
        im = ax.imshow(np.asarray(corr), cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ticklabels)))
        ax.set_yticks(range(len(ticklabels)))
        ax.set_xticklabels(ticklabels, rotation=90)
        ax.set_yticklabels(ticklabels)
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def davies_bouldin_plot(
    scores_by_label: Dict[str, Sequence[float]],
    k_values: Sequence[int],
    title: str = "Davies-Bouldin score vs number of clusters",
) -> plt.Figure:
    """Plot Davies-Bouldin score against cluster count for one or more series."""
    fig, ax = plt.subplots(figsize=(12, 8))
    for label, scores in scores_by_label.items():
        ax.plot(list(k_values), list(scores), marker="o", label=label)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Davies-Bouldin score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def cluster_frequency_bar(
    labels: Sequence[int], title: str = "Cluster frequency distribution"
) -> plt.Figure:
    """Bar chart of how many sentences fall into each topic cluster."""
    from collections import Counter

    dist = Counter(labels)
    keys = sorted(dist)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(keys, [dist[k] for k in keys], color="skyblue")
    ax.set_xlabel("Cluster number")
    ax.set_ylabel("Number of sentences")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig
