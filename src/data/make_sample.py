"""Synthetic sample-data generator.

WHY THIS FILE EXISTS
--------------------
The real corpus (~31M Reddit posts/comments) is private and is not shipped with
this repository. To keep the project *reproducible by a new collaborator* the
pipeline must still run end-to-end out of the box. This module fabricates a
small, deterministic dataset that has the same schema as the real data
(``title``, ``body``, ``created_utc``) and deliberately seeds it with real
distortion n-grams drawn from the dictionary, spread across the before / during /
after-COVID windows, so that every downstream stage (detection, weekly time
series, normalisation, correlation, plotting, topic modelling) produces non-empty
output.

The generator is seeded with a fixed RNG so runs are reproducible. It is NOT a
substitute for the real data and produces no scientific result -- it exists only
to exercise the code paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.cognitive_distortions.target_words import DISTORTIONS
from src.utils.config import Config
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Three COVID windows expressed as UTC seconds, matching the notebooks' constants.
_BEFORE = (1_546_300_000, 1_579_478_400)   # ~2019 -> COVID start
_DURING = (1_579_478_400, 1_594_598_400)   # COVID start -> assumed end
_AFTER = (1_594_598_400, 1_697_372_672)    # assumed end -> end_date

_FILLER = [
    "today was an interesting day at school",
    "i went to the library after class",
    "thanks for the advice everyone",
    "does anyone have tips for the exam",
    "the weather has been nice lately",
]


def _make_docs(rng: np.random.Generator, n: int, window: tuple) -> List[Dict]:
    """Create ``n`` synthetic documents whose text embeds real distortion n-grams."""
    categories = list(DISTORTIONS.items())
    docs: List[Dict] = []
    lo, hi = window
    for _ in range(n):
        name, ngrams = categories[int(rng.integers(0, len(categories)))]
        phrase = ngrams[int(rng.integers(0, len(ngrams)))]
        filler = _FILLER[int(rng.integers(0, len(_FILLER)))]
        body = f"{filler}, i think it {phrase} honestly. {filler}."
        title = f"feeling like everything {phrase} again"
        docs.append(
            {
                "title": title,
                "body": body,
                "created_utc": int(rng.integers(lo, hi)),
            }
        )
    return docs


def sample_community(cfg: Config, community: str, seed: int | None = None) -> Dict[str, pd.DataFrame]:
    """Return synthetic ``{"posts", "comments"}`` frames for a community.

    The two communities use different seeds so they are not identical, mimicking
    the real situation where r/teenagers is larger than r/SGExams.
    """
    seed = cfg.get("data.sample.seed", 42) if seed is None else seed
    base = {"sgexams": 0, "teenagers": 100}.get(community, 0)
    rng = np.random.default_rng(seed + base)

    n_posts = int(cfg.get("data.sample.n_posts", 120))
    n_comments = int(cfg.get("data.sample.n_comments", 300))
    # r/teenagers is roughly twice the size in this toy sample.
    factor = 2 if community == "teenagers" else 1

    post_docs: List[Dict] = []
    comment_docs: List[Dict] = []
    for window in (_BEFORE, _DURING, _AFTER):
        post_docs += _make_docs(rng, n_posts * factor // 3, window)
        comment_docs += _make_docs(rng, n_comments * factor // 3, window)

    posts = pd.DataFrame(post_docs)
    posts["text"] = posts["title"] + "." + posts["body"]
    comments = pd.DataFrame(comment_docs)[["body", "created_utc"]]
    logger.info(
        "Generated synthetic sample for '%s': %d posts, %d comments.",
        community, len(posts), len(comments),
    )
    return {"posts": posts, "comments": comments}


def write_sample(cfg: Config, out_dir: Path | None = None) -> Path:
    """Write synthetic CSVs for both communities into ``data/raw/sample/``.

    Used by ``scripts/run_pipeline.py --stage make_sample`` so the sample can be
    inspected on disk just like real data.
    """
    out_dir = out_dir or (cfg.path("paths.data_raw") / "sample")
    for community in ("sgexams", "teenagers"):
        data = sample_community(cfg, community)
        save_dataframe(data["posts"], out_dir / f"{community}_posts.csv")
        save_dataframe(data["comments"], out_dir / f"{community}_comments.csv")
    logger.info("Synthetic sample written to %s", out_dir)
    return out_dir
