"""Dictionary-based cognitive-distortion detection.

WHY THIS FILE EXISTS
--------------------
This is the single, de-duplicated home for the detection logic that appeared
(in slightly different forms) in ``comments_masked.ipynb``,
``posts_masked.ipynb`` and ``corr_row.ipynb``. Each of those notebooks rebuilt
the ``distortion_ngram_sets`` and re-wrote a ``process_data`` function. Here the
logic exists once and is reused by the per-document detector (for correlation
matrices) and by the weekly time-series builder (for temporal analysis).

The matching rule is identical to the original: a text is tokenised, all 1- to
5-grams are generated, and every n-gram is checked for exact membership in each
category's set of surface forms.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd

from src.cognitive_distortions.target_words import DISTORTIONS, category_names
from src.preprocessing.tokenize import generate_ngrams, word_tokens
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_N_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5)


def build_distortion_sets() -> Dict[str, set]:
    """Return ``{category_name: set(ngram_surface_forms)}`` for fast lookup."""
    return {name: set(ngrams) for name, ngrams in DISTORTIONS.items()}


def count_distortions_in_text(
    text: object,
    distortion_sets: Dict[str, set],
    n_orders: Sequence[int] = DEFAULT_N_ORDERS,
) -> Dict[str, int]:
    """Count occurrences of each distortion category in a single text.

    Parameters
    ----------
    text:
        Raw document text (post body, comment, or title).
    distortion_sets:
        Mapping produced by :func:`build_distortion_sets`.
    n_orders:
        Which n-gram orders to generate (default 1..5).

    Returns
    -------
    dict
        ``{category_name: count}`` for all categories (zero when absent).
    """
    counts = {name: 0 for name in distortion_sets}
    tokens = word_tokens(text)
    ngram_data = generate_ngrams(tokens, n_orders)
    for n in n_orders:
        for ngram_str in ngram_data[n]:
            for name, ngram_set in distortion_sets.items():
                if ngram_str in ngram_set:
                    counts[name] += 1
    return counts


def detect_dataframe(
    df: pd.DataFrame,
    text_cols: Iterable[str],
    distortion_sets: Dict[str, set] | None = None,
    n_orders: Sequence[int] = DEFAULT_N_ORDERS,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Build a per-document distortion-count matrix.

    Each row of the returned frame corresponds to a row of ``df`` and has one
    integer column per distortion category. When several ``text_cols`` are given
    (e.g. ``title`` and ``body`` for posts) their counts are summed, matching the
    notebooks' treatment of posts.

    This per-document matrix is what feeds the per-comment/per-post correlation
    analysis (``corr_row.ipynb`` / ``correlation.ipynb``).
    """
    distortion_sets = distortion_sets or build_distortion_sets()
    text_cols = list(text_cols)
    categories = list(distortion_sets.keys())

    records: List[Dict[str, int]] = []
    for pos in range(len(df)):
        row_counts = {name: 0 for name in categories}
        for col in text_cols:
            text_counts = count_distortions_in_text(
                df.iloc[pos][col], distortion_sets, n_orders
            )
            for name in categories:
                row_counts[name] += text_counts[name]
        if id_col is not None:
            row_counts = {id_col: df.iloc[pos].get(id_col, pos), **row_counts}
        records.append(row_counts)

    result = pd.DataFrame.from_records(records)
    logger.info(
        "Detected distortions in %d documents across %d categories.",
        len(result),
        len(categories),
    )
    return result


def distortion_totals(detection_df: pd.DataFrame) -> pd.Series:
    """Return the column-wise totals for the 12 distortion categories only."""
    cols = [c for c in category_names() if c in detection_df.columns]
    return detection_df[cols].sum()
