"""Temporal-analysis operations on weekly distortion time series.

WHY THIS FILE EXISTS
--------------------
``plots_masked.ipynb`` defined ``moving_average`` and ``filter_and_identify_spikes``
and then performed z-scoring, normalisation and COVID-period splitting inline,
repeating the arithmetic for all 12 categories. These transforms are collected
here as small, tested, reusable functions:

* ``normalize_by_volume`` -- divide raw weekly counts by weekly word volume
  (Methods Eq. 2).
* ``zscore`` -- standardise each series (Methods Eq. 3).
* ``moving_average`` / ``filter_and_identify_spikes`` -- the smoothing and
  spike-detection used to define "major spikes" (>1 SD above the trailing
  4-week mean).
* ``covid_period`` / ``split_by_covid`` -- the before/during/after segmentation
  driven by the configurable COVID timestamps.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Distortion-category columns are everything except the volume column.
VOLUME_COL = "total_unigrams"


def category_columns(ts: pd.DataFrame) -> List[str]:
    """Return the distortion-category columns (i.e. all columns except volume)."""
    return [c for c in ts.columns if c != VOLUME_COL]


def normalize_by_volume(ts: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Normalise raw weekly counts by the weekly word volume.

    Implements ``X_bar_j(c) = X_j(c) / N_j`` from the Methods. ``scale`` reproduces
    the notebooks' habit of multiplying by a constant (e.g. 100) to keep the very
    small ratios in a readable range; it cancels out of z-scores and correlations.
    """
    cols = category_columns(ts)
    volume = ts[VOLUME_COL].replace(0, np.nan)
    normalized = ts[cols].div(volume, axis=0) * scale
    return normalized.fillna(0.0)


def zscore(series: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Return the z-score standardisation of a 1-D series (zero mean, unit SD)."""
    arr = np.asarray(series, dtype=float)
    sd = arr.std()
    if sd == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / sd


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`zscore` to every column of ``df``."""
    return df.apply(lambda col: zscore(col.to_numpy()), axis=0)


def moving_average(data: Sequence[float], window_size: int) -> List[float]:
    """Trailing simple moving average (verbatim port from ``plots_masked.ipynb``)."""
    data = list(data)
    out: List[float] = []
    for i in range(window_size - 1, len(data)):
        window_sum = sum(data[i - window_size + 1 : i + 1])
        out.append(window_sum / window_size)
    return out


def filter_and_identify_spikes(
    data: Sequence[float], window_size: int = 4
) -> Tuple[np.ndarray, List[int]]:
    """Identify "major spikes": points exceeding the trailing mean by >1 SD.

    Returns the filtered series (non-spike weeks zeroed out) and the list of spike
    week indices. This is the project's operational definition of a distortion
    spike used for cross-category co-occurrence analysis.
    """
    data = list(data)
    filtered: List[float] = []
    spikes: List[int] = []
    for i in range(len(data)):
        if i >= window_size:
            avg_prev = float(np.mean(data[i - window_size : i]))
            std_dev = float(np.std(data[i - window_size : i]))
            if data[i] - avg_prev > std_dev:
                spikes.append(i)
                filtered.append(data[i])
            else:
                filtered.append(0.0)
        else:
            filtered.append(0.0)
    return np.array(filtered), spikes


def covid_period(utc: float, covid_start: float, covid_end: float) -> str:
    """Classify a UTC timestamp as ``before`` / ``during`` / ``after`` COVID."""
    if utc < covid_start:
        return "before"
    if covid_start <= utc <= covid_end:
        return "during"
    return "after"


def split_by_covid(
    df: pd.DataFrame,
    covid_start: float,
    covid_end: float,
    end_date: float | None = None,
    time_col: str = "created_utc",
) -> Dict[str, pd.DataFrame]:
    """Split a document frame into before/during/after-COVID subsets.

    ``end_date`` optionally caps the "after" period (the notebooks used a fixed
    ``end_date`` so that a trailing partial window did not distort the analysis).
    """
    before = df[df[time_col] < covid_start]
    during = df[(df[time_col] >= covid_start) & (df[time_col] <= covid_end)]
    after = df[df[time_col] > covid_end]
    if end_date is not None:
        after = after[after[time_col] <= end_date]
    return {"before": before, "during": during, "after": after}
