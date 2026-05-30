"""Weekly cognitive-distortion time-series construction.

WHY THIS FILE EXISTS
--------------------
The notebooks built weekly counts with ad-hoc ``np.floor((reference - utc) /
604800)`` arithmetic and a hand-sized ``weeks = np.zeros(267)`` array, processing
one distortion category at a time by uncommenting ``for`` loops. This module
turns that into a single reusable function that, given a dataframe with a UTC
timestamp column and one or more text columns, returns a tidy weekly time-series
table: one column per distortion category plus the weekly total-unigram count
used as the normalisation denominator (Methods, Eq. 2 of the manuscript).
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from src.cognitive_distortions.detect import build_distortion_sets
from src.preprocessing.tokenize import generate_ngrams, word_tokens
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SECONDS_PER_WEEK = 604800
DEFAULT_N_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5)


def assign_week(utc: float, start_utc: float, seconds_per_week: int = SECONDS_PER_WEEK) -> int:
    """Return the zero-based week index of ``utc`` relative to ``start_utc``."""
    return int(np.floor((utc - start_utc) / seconds_per_week))


def weekly_distortion_counts(
    df: pd.DataFrame,
    text_cols: Iterable[str],
    time_col: str = "created_utc",
    n_orders: Sequence[int] = DEFAULT_N_ORDERS,
    start_utc: float | None = None,
    seconds_per_week: int = SECONDS_PER_WEEK,
    distortion_sets: Dict[str, set] | None = None,
) -> pd.DataFrame:
    """Aggregate distortion counts and total unigrams into a weekly table.

    Parameters
    ----------
    df:
        Source documents with a UTC timestamp column and text column(s).
    text_cols:
        Columns to scan (e.g. ``["body"]`` for comments, ``["title", "body"]``
        for posts).
    time_col:
        Name of the integer/float UTC-seconds column.
    n_orders:
        N-gram orders to generate (default 1..5).
    start_utc:
        UTC of week 0. Defaults to the minimum timestamp in ``df``.
    seconds_per_week:
        Week length in seconds (604800 = 7 days).
    distortion_sets:
        Optional precomputed category->set mapping.

    Returns
    -------
    pandas.DataFrame
        Index ``week`` (0..N), one column per distortion category, plus
        ``total_unigrams`` (the weekly word volume used for normalisation).
    """
    distortion_sets = distortion_sets or build_distortion_sets()
    text_cols = list(text_cols)
    categories = list(distortion_sets.keys())

    df = df.dropna(subset=[time_col])
    if len(df) == 0:
        raise ValueError("No rows with a valid timestamp in time_col=%r" % time_col)
    start = float(df[time_col].min()) if start_utc is None else float(start_utc)
    n_weeks = assign_week(float(df[time_col].max()), start, seconds_per_week) + 1

    counts = {name: np.zeros(n_weeks, dtype=np.int64) for name in categories}
    total_unigrams = np.zeros(n_weeks, dtype=np.int64)

    times = df[time_col].to_numpy()
    for pos in range(len(df)):
        week = assign_week(float(times[pos]), start, seconds_per_week)
        if week < 0 or week >= n_weeks:
            continue
        row = df.iloc[pos]
        for col in text_cols:
            tokens = word_tokens(row[col])
            total_unigrams[week] += len(tokens)
            ngram_data = generate_ngrams(tokens, n_orders)
            for n in n_orders:
                for ngram_str in ngram_data[n]:
                    for name, ngram_set in distortion_sets.items():
                        if ngram_str in ngram_set:
                            counts[name][week] += 1

    out = pd.DataFrame(counts)
    out["total_unigrams"] = total_unigrams
    out.index.name = "week"
    logger.info("Built weekly time series with %d weeks.", n_weeks)
    return out
