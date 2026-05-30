"""Topic modelling of distortion-bearing sentences (Sentence-BERT + K-means).

WHY THIS FILE EXISTS
--------------------
``topic_modelling.ipynb``, ``topic_modelling_all.ipynb`` and
``topicmodelling_latest.ipynb`` re-implemented the *same* pipeline a dozen times
(once per distortion, often with copy-paste bugs such as encoding the wrong
``embeddings`` variable). This module is the single, correct implementation:

    sentences with target n-grams  ->  Sentence-BERT embeddings
    ->  K-means over a range of k  ->  pick k minimising the Davies-Bouldin index
    ->  extract representative sentences (random + closest-to-centroid) per cluster

Heavy dependencies (``sentence_transformers``, ``scikit-learn``) are imported
lazily inside the functions that need them so that the rest of the pipeline can
be imported and run without them installed.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.tokenize import sentence_tokens, word_tokens
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------- filtering
def filter_sentences(sentences: Sequence[str], ngrams: Sequence[str]) -> List[str]:
    """Keep sentences that contain at least one target n-gram (case-insensitive)."""
    targets = [g.lower() for g in ngrams]
    out: List[str] = []
    for sentence in sentences:
        low = sentence.lower()
        if any(g in low for g in targets):
            out.append(sentence)
    return out


def sentences_with_distortion(texts: Sequence[object], ngrams: Sequence[str]) -> List[str]:
    """Sentence-tokenise ``texts`` then keep sentences containing a target n-gram."""
    sentences: List[str] = []
    for text in texts:
        if isinstance(text, str):
            sentences.extend(sentence_tokens(text))
    return filter_sentences(sentences, ngrams)


def expand_around_ngram(text: str, ngrams: Sequence[str], window: int = 10) -> List[str]:
    """Return ``window`` words either side of each matched target n-gram.

    Port of ``expand_sentence_with_ngram`` -- used as an alternative to sentence
    tokenisation so clusters are built from a fixed local context.
    """
    words = word_tokens(text)
    out: List[str] = []
    for i in range(len(words)):
        for target in ngrams:
            ngram_len = len(target.split())
            if " ".join(words[i : i + ngram_len]).lower() == target.lower():
                start = max(0, i - window)
                end = min(len(words), i + ngram_len + window)
                out.append(" ".join(words[start:end]))
                break
    return out


# --------------------------------------------------------------------- embedding
def embed_sentences(sentences: Sequence[str], model_name: str) -> np.ndarray:
    """Encode sentences with a Sentence-BERT model (lazy import)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Topic modelling requires `sentence-transformers` and `scikit-learn`. "
            "Install them (see requirements.txt) or skip the 'topic_model' stage."
        ) from exc

    model = SentenceTransformer(model_name)
    logger.info("Encoding %d sentences with %s", len(sentences), model_name)
    return np.asarray(model.encode(list(sentences)))


# --------------------------------------------------------------------- clustering
def evaluate_clusters(embeddings: np.ndarray, k_range: Sequence[int]) -> List[float]:
    """Davies-Bouldin score for each ``k`` in ``k_range`` (lower is better)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score

    scores: List[float] = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(embeddings).labels_
        scores.append(float(davies_bouldin_score(embeddings, labels)))
    return scores


def find_optimal_clusters(
    embeddings: np.ndarray, k_range: Sequence[int]
) -> Tuple[int, List[float]]:
    """Return ``(optimal_k, scores)`` where optimal_k minimises Davies-Bouldin."""
    scores = evaluate_clusters(embeddings, k_range)
    optimal_k = list(k_range)[int(np.argmin(scores))]
    logger.info("Optimal k = %d (min Davies-Bouldin = %.4f)", optimal_k, min(scores))
    return optimal_k, scores


def fit_kmeans(embeddings: np.ndarray, k: int):
    """Fit and return a K-means model with the project's fixed random seed."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit(embeddings)


# --------------------------------------------------- representative sentences
def random_sentences_per_cluster(
    sentences: Sequence[str], labels: Sequence[int], num_samples: int = 5, seed: int = 42
) -> Dict[int, List[str]]:
    """Return up to ``num_samples`` random sentences for each cluster label."""
    rng = random.Random(seed)
    out: Dict[int, List[str]] = {}
    for label in np.unique(labels):
        members = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
        out[int(label)] = rng.sample(members, min(num_samples, len(members)))
    return out


def closest_sentences_per_cluster(
    embeddings: np.ndarray, sentences: Sequence[str], kmeans, num_samples: int = 5
) -> Dict[int, List[str]]:
    """Return the ``num_samples`` sentences nearest each cluster centroid."""
    from sklearn.metrics.pairwise import euclidean_distances

    out: Dict[int, List[str]] = {}
    labels = kmeans.labels_
    for i, centroid in enumerate(kmeans.cluster_centers_):
        idx = [j for j in range(len(embeddings)) if labels[j] == i]
        if not idx:
            out[i] = []
            continue
        dists = euclidean_distances([centroid], [embeddings[j] for j in idx])[0]
        nearest = np.argsort(dists)[:num_samples]
        out[i] = [sentences[idx[k]] for k in nearest]
    return out


def cluster_examples_dataframe(
    random_by_cluster: Dict[int, List[str]],
    closest_by_cluster: Dict[int, List[str]],
) -> pd.DataFrame:
    """Combine random and centroid examples into a tidy per-cluster table.

    This is the spreadsheet that was sent for manual ChatGPT-assisted topic
    labelling (see the research journal, weeks 27-45).
    """
    rows = []
    for cid in random_by_cluster:
        rnd = random_by_cluster.get(cid, [])
        clo = closest_by_cluster.get(cid, [])
        for r, c in zip(rnd + [""] * (len(clo) - len(rnd)), clo + [""] * (len(rnd) - len(clo))):
            rows.append({"Cluster ID": cid, "Random Sentence": r, "Closest Sentence": c})
    return pd.DataFrame(rows)


def run_topic_model(
    texts: Sequence[object],
    ngrams: Sequence[str],
    model_name: str,
    k_range: Sequence[int],
    context: str = "sentence",
    window: int = 10,
) -> dict:
    """End-to-end topic model for one distortion category.

    Parameters
    ----------
    texts:
        Raw documents (comments and/or posts) to mine.
    ngrams:
        Target n-grams for the distortion category.
    model_name:
        Sentence-BERT model identifier.
    k_range:
        Candidate cluster counts to evaluate.
    context:
        ``"sentence"`` to use NLTK sentence tokenisation, or ``"window"`` to use
        a fixed +/- ``window``-word context around each matched n-gram.
    window:
        Half-window size when ``context == "window"``.

    Returns
    -------
    dict
        ``optimal_k``, ``scores``, ``labels``, ``examples`` (DataFrame),
        and the filtered ``sentences``.
    """
    if context == "window":
        sentences: List[str] = []
        for t in texts:
            if isinstance(t, str):
                sentences.extend(expand_around_ngram(t, ngrams, window))
    else:
        sentences = sentences_with_distortion(texts, ngrams)

    if len(sentences) < (min(k_range) if len(k_range) else 2):
        raise ValueError(
            f"Only {len(sentences)} matching sentences; need at least min(k_range)."
        )

    embeddings = embed_sentences(sentences, model_name)
    optimal_k, scores = find_optimal_clusters(embeddings, k_range)
    kmeans = fit_kmeans(embeddings, optimal_k)
    examples = cluster_examples_dataframe(
        random_sentences_per_cluster(sentences, kmeans.labels_),
        closest_sentences_per_cluster(embeddings, sentences, kmeans),
    )
    return {
        "optimal_k": optimal_k,
        "scores": scores,
        "labels": kmeans.labels_,
        "examples": examples,
        "sentences": sentences,
    }
