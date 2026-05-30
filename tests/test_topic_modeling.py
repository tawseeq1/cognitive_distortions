"""Tests for the topic-modelling helpers.

The embedding/clustering steps require scikit-learn and sentence-transformers; if
those are not installed the relevant tests are skipped (the pure-Python filtering
and context-window helpers are always tested).
"""
import importlib.util

import pandas as pd

from src.topic_modeling.embed_cluster import (
    cluster_examples_dataframe,
    expand_around_ngram,
    filter_sentences,
    sentences_with_distortion,
)

try:  # pytest is optional in this environment.
    import pytest
except Exception:  # pragma: no cover
    pytest = None

_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None


def test_filter_sentences_keeps_only_matches():
    sents = ["i feel sad because i feel lost", "nothing to see here"]
    kept = filter_sentences(sents, ["because i feel"])
    assert kept == ["i feel sad because i feel lost"]


def test_expand_around_ngram_window():
    text = "one two three because i feel happy four five six"
    out = expand_around_ngram(text, ["because i feel"], window=2)
    assert len(out) == 1
    assert "because i feel" in out[0]
    # window=2 -> 2 words before + ngram(3) + 2 after = up to 7 tokens
    assert len(out[0].split()) <= 7


def test_sentences_with_distortion_from_documents():
    docs = ["I am fine. But i feel awful today.", "All good here."]
    out = sentences_with_distortion(docs, ["i feel"])
    assert any("i feel" in s.lower() for s in out)


def test_cluster_examples_dataframe_structure():
    df = cluster_examples_dataframe({0: ["a", "b"]}, {0: ["c"]})
    assert list(df.columns) == ["Cluster ID", "Random Sentence", "Closest Sentence"]
    assert (df["Cluster ID"] == 0).all()


def test_clustering_end_to_end_if_deps_available():
    if not (_HAS_SKLEARN and _HAS_ST):
        if pytest is not None:
            pytest.skip("sklearn/sentence-transformers not installed")
        return
    from src.topic_modeling.embed_cluster import find_optimal_clusters
    import numpy as np

    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.1, (20, 8)), rng.normal(5, 0.1, (20, 8))])
    k, scores = find_optimal_clusters(emb, range(2, 6))
    assert 2 <= k <= 5 and len(scores) == 4
