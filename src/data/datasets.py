"""Dataset loading for the two Reddit communities.

WHY THIS FILE EXISTS
--------------------
Each notebook opened the same CSVs by absolute Google-Drive path and rebuilt the
``text = title + '.' + body`` column by hand. The teen comments were split into
four shards that had to be concatenated (two of them needing
``on_bad_lines='skip'``). This module reads dataset locations from
``configs/data.yaml`` and returns clean, concatenated DataFrames, so no path is
ever hard-coded in analysis code.

The raw CSVs are NOT distributed with this repository (they contain ~31M Reddit
posts/comments). When the configured files are absent and ``data.use_sample`` is
true, the loaders transparently fall back to the synthetic sample produced by
``src/data/make_sample.py`` so the whole pipeline remains runnable end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.utils.config import Config
from src.utils.io import read_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _exists_all(paths: List[Path]) -> bool:
    return all(p.exists() for p in paths)


def add_text_column(posts: pd.DataFrame) -> pd.DataFrame:
    """Add a combined ``text = title . body`` column to a posts frame (idempotent)."""
    if "text" not in posts.columns and {"title", "body"}.issubset(posts.columns):
        posts = posts.copy()
        posts["text"] = posts["title"].astype(str) + "." + posts["body"].astype(str)
    return posts


def load_community(cfg: Config, community: str) -> dict[str, pd.DataFrame]:
    """Load posts and comments for one community (``"sgexams"`` or ``"teenagers"``).

    Returns ``{"posts": DataFrame, "comments": DataFrame}``. Comment shards are
    concatenated. Falls back to the synthetic sample when configured files are
    missing and ``data.use_sample`` is enabled.
    """
    comment_files = [Config.resolve_path(p) for p in cfg.get(f"data.{community}.comments", [])]
    post_files = [Config.resolve_path(p) for p in cfg.get(f"data.{community}.posts", [])]

    use_sample = bool(cfg.get("data.use_sample", True))
    if not comment_files or not _exists_all(comment_files) or not _exists_all(post_files):
        if use_sample:
            logger.warning(
                "Raw data for '%s' not found; using synthetic sample "
                "(set data.use_sample: false to require real data).",
                community,
            )
            from src.data.make_sample import sample_community

            return sample_community(cfg, community)
        raise FileNotFoundError(
            f"Configured data for community '{community}' is missing and "
            f"data.use_sample is false. Expected: {comment_files + post_files}"
        )

    comments = pd.concat([read_csv(p) for p in comment_files], ignore_index=True)
    posts = add_text_column(pd.concat([read_csv(p) for p in post_files], ignore_index=True))
    logger.info(
        "Loaded community '%s': %d comments, %d posts.", community, len(comments), len(posts)
    )
    return {"posts": posts, "comments": comments}
