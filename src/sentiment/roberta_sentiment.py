"""RoBERTa sentiment scoring of distortion-bearing text.

WHY THIS FILE EXISTS
--------------------
The manuscript (Methods, "Sentiment Analysis") and the README describe using
RoBERTa to score the sentiment (positive / neutral / negative probabilities) of
sentences containing cognitive-distortion markers, as a validation check that
those sentences skew negative. No notebook in the original repo contained a
finished, runnable version of this step, so this module provides a clean
reference implementation built on the public ``cardiffnlp/twitter-roberta-base-
sentiment-latest`` model (a Twitter/social-media RoBERTa, appropriate for Reddit
text).

``transformers``/``torch`` are imported lazily; if they are not installed the
function raises a clear, actionable error instead of failing at import time.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_LABELS = ["negative", "neutral", "positive"]


def _load_pipeline(model_name: str):
    """Lazily build a HuggingFace sentiment pipeline returning all class scores."""
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Sentiment analysis requires `transformers` and `torch`. "
            "Install them (see requirements.txt) or skip the 'sentiment' stage."
        ) from exc
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        top_k=None,  # return scores for every class
        truncation=True,
    )


def score_sentences(
    sentences: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Return a DataFrame of negative/neutral/positive probabilities per sentence.

    Parameters
    ----------
    sentences:
        Texts to score (e.g. sentences that contain distortion n-grams).
    model_name:
        HuggingFace model id.
    batch_size:
        Inference batch size.

    Returns
    -------
    pandas.DataFrame
        Columns ``text``, ``negative``, ``neutral``, ``positive`` and the
        ``label`` of the highest-probability class.
    """
    clf = _load_pipeline(model_name)
    sentences = list(sentences)
    rows: List[Dict[str, object]] = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        for text, scores in zip(batch, clf(batch)):
            probs = {s["label"].lower(): float(s["score"]) for s in scores}
            row: Dict[str, object] = {"text": text}
            for label in _LABELS:
                row[label] = probs.get(label, float("nan"))
            row["label"] = max(_LABELS, key=lambda lbl: probs.get(lbl, 0.0))
            rows.append(row)
    logger.info("Scored sentiment for %d sentences.", len(rows))
    return pd.DataFrame(rows)


def summarise_sentiment(scored: pd.DataFrame) -> pd.Series:
    """Return the mean probability per sentiment class across all rows."""
    return scored[_LABELS].mean()
