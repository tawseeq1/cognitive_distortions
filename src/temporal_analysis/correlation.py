"""Correlation analysis between cognitive-distortion categories.

WHY THIS FILE EXISTS
--------------------
Correlation matrices were computed in three different notebooks
(``correlation.ipynb`` over per-document counts, ``plots_masked.ipynb`` over
normalised/raw weekly series, ``corr_row.ipynb`` over merged per-row counts). All
of them ultimately call ``np.corrcoef`` / ``DataFrame.corr`` over the 12 category
series. This module exposes the two flavours used in the manuscript:

* ``correlation_matrix`` -- Pearson correlation of the 12 weekly time series.
* ``per_document_correlation`` -- correlation across the per-comment/per-post
  detection matrix.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from src.temporal_analysis.timeseries_ops import category_columns


def correlation_matrix(series_frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between the distortion-category columns of a series frame.

    Works on either the raw or normalised/z-scored weekly series (correlation is
    invariant to the affine z-score transform, so both give the same matrix).
    """
    cols = category_columns(series_frame)
    return series_frame[cols].corr()


def per_document_correlation(detection_df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix across the per-document distortion-count matrix.

    Drops any id column automatically by keeping only numeric category columns.
    """
    numeric = detection_df.select_dtypes(include="number")
    return numeric.corr()


def correlations_by_period(period_series: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Compute a correlation matrix for each before/during/after weekly series."""
    return {period: correlation_matrix(frame) for period, frame in period_series.items()}
