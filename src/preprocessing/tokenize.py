"""Tokenisation and n-gram generation.

WHY THIS FILE EXISTS
--------------------
Every notebook re-implemented the same three primitives:

* lower-case word tokenisation (``word_tokenize(str(text).lower())``),
* sentence tokenisation (``sent_tokenize``) for topic modelling, and
* fixed-order n-gram generation (``nltk.ngrams``).

They are collected here once. NLTK is used when available (matching the original
behaviour); otherwise we fall back to a regex tokeniser so that the data,
detection, temporal and visualisation stages of the pipeline remain runnable in
environments where NLTK is not installed.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Regex matching word characters plus a few intra-word punctuation marks so that
# contractions such as "don't" and "i'll" survive as single tokens (important,
# because many distortion n-grams contain apostrophes).
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_NLTK_READY: bool | None = None


def _try_init_nltk() -> bool:
    """Return True if NLTK tokenizers are importable and the 'punkt' data is present."""
    global _NLTK_READY
    if _NLTK_READY is not None:
        return _NLTK_READY
    try:
        import nltk  # noqa: F401
        from nltk.tokenize import word_tokenize  # noqa: F401

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        _NLTK_READY = True
    except Exception:  # pragma: no cover - depends on environment
        logger.warning("NLTK unavailable; using regex tokenizer fallback.")
        _NLTK_READY = False
    return _NLTK_READY


def word_tokens(text: object) -> List[str]:
    """Lower-case word-tokenise ``text`` (mirrors ``word_tokenize(str(text).lower())``)."""
    s = str(text).lower()
    if _try_init_nltk():
        from nltk.tokenize import word_tokenize

        return word_tokenize(s)
    return _WORD_RE.findall(s)


def sentence_tokens(text: object) -> List[str]:
    """Split ``text`` into sentences (mirrors ``nltk.sent_tokenize``)."""
    s = str(text)
    if _try_init_nltk():
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(s)
    return [part for part in _SENT_RE.split(s) if part.strip()]


def ngrams(tokens: Sequence[str], n: int) -> List[str]:
    """Return space-joined ``n``-grams from a token sequence."""
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def generate_ngrams(tokens: Sequence[str], n_orders: Sequence[int]) -> Dict[int, List[str]]:
    """Return a ``{n: [ngram, ...]}`` dict for each order in ``n_orders``.

    This is the function the notebooks called ``generate_ngrams``; it is the unit
    of work for dictionary-based distortion detection.
    """
    return {n: ngrams(tokens, n) for n in n_orders}


def tokenize_column(texts: Sequence[object]) -> List[List[str]]:
    """Word-tokenise an iterable of raw texts (e.g. a DataFrame column)."""
    return [word_tokens(t) for t in texts]
