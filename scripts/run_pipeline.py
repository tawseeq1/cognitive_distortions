#!/usr/bin/env python3
"""End-to-end (and stage-selectable) pipeline runner.

WHY THIS FILE EXISTS
--------------------
The original analysis was spread across eight notebooks that had to be run by
hand, in the right order, with paths edited each time. This single entry point
reproduces every major output from config, and -- per the project brief -- lets
you run *only the part(s) you want* via ``--stage``.

USAGE
-----
    # Run the default stages (make_sample -> detect -> timeseries -> correlation
    # -> visualize) on whatever data is configured (synthetic sample by default):
    python scripts/run_pipeline.py

    # Run just one or two stages:
    python scripts/run_pipeline.py --stage detect
    python scripts/run_pipeline.py --stage timeseries --stage visualize

    # Run everything, including the heavy optional stages:
    python scripts/run_pipeline.py --all

    # Restrict to one community:
    python scripts/run_pipeline.py --community sgexams

Stages
------
    make_sample  Write the synthetic sample CSVs to data/raw/sample/.
    detect       Per-document distortion-count matrices + per-document correlation.
    timeseries   Weekly distortion time series (raw, normalised, z-scored).
    correlation  Pearson correlation matrices overall and by COVID period.
    visualize    Render the manuscript-style figures from the tables above.
    topic_model  Sentence-BERT + K-means topic model (needs sentence-transformers).
    sentiment    RoBERTa sentiment of distortion sentences (needs transformers+torch).

Each stage writes its artefacts under results/ and data/processed/ and caches
in-memory results so dependent stages can run together in one invocation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Make `import src...` work no matter where the script is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cognitive_distortions.detect import build_distortion_sets, detect_dataframe  # noqa: E402
from src.cognitive_distortions.timeseries import weekly_distortion_counts  # noqa: E402
from src.data.datasets import load_community  # noqa: E402
from src.data.make_sample import write_sample  # noqa: E402
from src.temporal_analysis.correlation import correlation_matrix, per_document_correlation  # noqa: E402
from src.temporal_analysis.timeseries_ops import (  # noqa: E402
    VOLUME_COL,
    normalize_by_volume,
    split_by_covid,
    zscore_frame,
)
from src.utils.config import Config, load_config  # noqa: E402
from src.utils.io import save_dataframe, save_figure  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.visualization import plots  # noqa: E402

logger = get_logger("pipeline")

ALL_STAGES = [
    "make_sample",
    "detect",
    "timeseries",
    "correlation",
    "visualize",
    "topic_model",
    "sentiment",
]


# ----------------------------------------------------------------- helpers
def _text_cols(cfg: Config, kind: str) -> List[str]:
    return list(cfg.get(f"data.columns.{kind}_text", ["body"]))


def _load_all(cfg: Config, communities: List[str]) -> Dict[str, dict]:
    """Load every requested community's posts and comments once."""
    return {c: load_community(cfg, c) for c in communities}


# ----------------------------------------------------------------- stages
def stage_make_sample(cfg: Config, ctx: dict) -> None:
    """Materialise the synthetic sample CSVs on disk (no-op for real data)."""
    if cfg.get("data.use_sample", True):
        write_sample(cfg)
    else:
        logger.info("data.use_sample is false; skipping synthetic sample generation.")


def stage_detect(cfg: Config, ctx: dict) -> None:
    """Build per-document distortion matrices and per-document correlations."""
    dsets = build_distortion_sets()
    processed = cfg.path("paths.data_processed")
    tables = cfg.path("paths.results_tables")
    ctx.setdefault("detection", {})
    for community, data in ctx["data"].items():
        det_comments = detect_dataframe(data["comments"], _text_cols(cfg, "comment"), dsets)
        det_posts = detect_dataframe(data["posts"], _text_cols(cfg, "post"), dsets)
        # Merge comment- and post-level rows into one per-document matrix, mirroring
        # corr_row.ipynb / correlation.ipynb which concatenated both before correlating.
        merged = pd.concat([det_comments, det_posts], ignore_index=True)
        save_dataframe(det_comments, processed / f"{community}_detect_comments.csv")
        save_dataframe(det_posts, processed / f"{community}_detect_posts.csv")
        per_doc_corr = per_document_correlation(merged)
        save_dataframe(per_doc_corr, tables / f"{community}_per_document_correlation.csv", index=True)
        ctx["detection"][community] = merged


def stage_timeseries(cfg: Config, ctx: dict) -> None:
    """Build weekly raw / normalised / z-scored distortion time series."""
    processed = cfg.path("paths.data_processed")
    scale = float(cfg.get("temporal.normalize_scale", 100))
    n_orders = cfg.get("temporal.n_orders", [1, 2, 3, 4, 5])
    ctx.setdefault("timeseries", {})
    for community, data in ctx["data"].items():
        # Comments dominate volume; build the series from comment bodies, which is
        # what the weekly-count notebooks did. Posts could be added analogously.
        ts = weekly_distortion_counts(
            data["comments"], _text_cols(cfg, "comment"),
            time_col=cfg.get("data.columns.time", "created_utc"), n_orders=n_orders,
        )
        norm = normalize_by_volume(ts, scale=scale)
        z = zscore_frame(norm)
        save_dataframe(ts, processed / f"{community}_weekly_raw.csv", index=True)
        save_dataframe(norm, processed / f"{community}_weekly_normalized.csv", index=True)
        save_dataframe(z, processed / f"{community}_weekly_zscore.csv", index=True)
        ctx["timeseries"][community] = {"raw": ts, "norm": norm, "z": z}


def stage_correlation(cfg: Config, ctx: dict) -> None:
    """Correlation matrices overall and split by COVID period (weekly series)."""
    if "timeseries" not in ctx:
        stage_timeseries(cfg, ctx)
    tables = cfg.path("paths.results_tables")
    cs = cfg.get("temporal.covid_start")
    ce = cfg.get("temporal.covid_end")
    ed = cfg.get("temporal.end_date")
    ctx.setdefault("correlation", {})
    for community, data in ctx["data"].items():
        z = ctx["timeseries"][community]["z"]
        overall = correlation_matrix(z)
        save_dataframe(overall, tables / f"{community}_correlation_overall.csv", index=True)
        # Per-period correlations require splitting the *documents* and rebuilding
        # weekly series per period; here we split the weekly z-series by week->utc.
        periods = split_by_covid(
            data["comments"], covid_start=cs, covid_end=ce, end_date=ed,
            time_col=cfg.get("data.columns.time", "created_utc"),
        )
        period_corrs = {}
        for name, subset in periods.items():
            if len(subset) < 2:
                continue
            sub_ts = weekly_distortion_counts(
                subset, _text_cols(cfg, "comment"),
                time_col=cfg.get("data.columns.time", "created_utc"),
            )
            sub_norm = normalize_by_volume(sub_ts, scale=float(cfg.get("temporal.normalize_scale", 100)))
            corr = correlation_matrix(zscore_frame(sub_norm))
            save_dataframe(corr, tables / f"{community}_correlation_{name}.csv", index=True)
            period_corrs[name] = corr
        ctx["correlation"][community] = {"overall": overall, **period_corrs}


def stage_visualize(cfg: Config, ctx: dict) -> None:
    """Render manuscript-style figures from the computed tables."""
    if "timeseries" not in ctx:
        stage_timeseries(cfg, ctx)
    if "correlation" not in ctx:
        stage_correlation(cfg, ctx)
    figdir = cfg.path("paths.results_figures")
    for community in ctx["data"]:
        z = ctx["timeseries"][community]["z"]
        raw = ctx["timeseries"][community]["raw"].drop(columns=[VOLUME_COL])
        save_figure(plots.timeseries_tiles(z, title=f"{community}: z-score time series", covid_marker=None),
                    figdir / f"{community}_timeseries_zscore.png")
        save_figure(plots.timeseries_tiles(raw, title=f"{community}: raw counts", covid_marker=None),
                    figdir / f"{community}_timeseries_raw.png")
        save_figure(plots.timeseries_overlay(z, title=f"{community}: all series overlay"),
                    figdir / f"{community}_timeseries_overlay.png")
        for name, corr in ctx["correlation"][community].items():
            save_figure(plots.correlation_heatmap(corr, title=f"{community}: correlation ({name})"),
                        figdir / f"{community}_correlation_{name}.png")


def stage_topic_model(cfg: Config, ctx: dict) -> None:
    """Topic-model the focus distortion category (heavy: needs sentence-transformers)."""
    from src.cognitive_distortions.target_words import DISTORTIONS
    from src.topic_modeling.embed_cluster import run_topic_model

    focus = cfg.get("topic_modeling.focus_category", "Emotional Reasoning")
    ngrams = DISTORTIONS[focus]
    k_range = range(int(cfg.get("topic_modeling.k_min", 10)), int(cfg.get("topic_modeling.k_max", 50)))
    # Merge all communities' text, as the research journal concluded was correct.
    texts: List[str] = []
    for data in ctx["data"].values():
        texts += data["comments"]["body"].dropna().astype(str).tolist()
        texts += data["posts"]["text"].dropna().astype(str).tolist()
    result = run_topic_model(
        texts, ngrams,
        model_name=cfg.get("topic_modeling.embedding_model"),
        k_range=k_range,
        context=cfg.get("topic_modeling.context", "sentence"),
        window=int(cfg.get("topic_modeling.window", 10)),
    )
    tables = cfg.path("paths.results_tables")
    figdir = cfg.path("paths.results_figures")
    save_dataframe(result["examples"], tables / f"topic_clusters_{focus.replace(' ', '_')}.csv")
    save_figure(
        plots.davies_bouldin_plot({focus: result["scores"]}, list(k_range)),
        figdir / f"davies_bouldin_{focus.replace(' ', '_')}.png",
    )
    save_figure(plots.cluster_frequency_bar(result["labels"]),
                figdir / f"cluster_frequency_{focus.replace(' ', '_')}.png")
    logger.info("Topic model for '%s': optimal_k=%d", focus, result["optimal_k"])


def stage_sentiment(cfg: Config, ctx: dict) -> None:
    """RoBERTa sentiment of focus-distortion sentences (heavy: needs transformers)."""
    from src.cognitive_distortions.target_words import DISTORTIONS
    from src.sentiment.roberta_sentiment import score_sentences, summarise_sentiment
    from src.topic_modeling.embed_cluster import sentences_with_distortion

    focus = cfg.get("topic_modeling.focus_category", "Emotional Reasoning")
    ngrams = DISTORTIONS[focus]
    texts: List[str] = []
    for data in ctx["data"].values():
        texts += data["comments"]["body"].dropna().astype(str).tolist()
    sentences = sentences_with_distortion(texts, ngrams)
    cap = cfg.get("sentiment.max_sentences")
    if cap:
        sentences = sentences[: int(cap)]
    scored = score_sentences(sentences, model_name=cfg.get("sentiment.model"),
                             batch_size=int(cfg.get("sentiment.batch_size", 32)))
    outputs = cfg.path("paths.results_outputs")
    save_dataframe(scored, outputs / f"sentiment_{focus.replace(' ', '_')}.csv")
    logger.info("Mean sentiment for '%s':\n%s", focus, summarise_sentiment(scored).to_string())


STAGE_FUNCS = {
    "make_sample": stage_make_sample,
    "detect": stage_detect,
    "timeseries": stage_timeseries,
    "correlation": stage_correlation,
    "visualize": stage_visualize,
    "topic_model": stage_topic_model,
    "sentiment": stage_sentiment,
}


# ----------------------------------------------------------------- CLI
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", action="append", choices=ALL_STAGES,
                   help="Run only this stage (repeatable). Default: experiment.default_stages.")
    p.add_argument("--all", action="store_true", help="Run every stage, including heavy ones.")
    p.add_argument("--community", action="append", choices=["sgexams", "teenagers"],
                   help="Restrict to this community (repeatable). Default: all configured.")
    p.add_argument("--config-dir", default=None, help="Override the configs/ directory.")
    p.add_argument("--log-level", default=None, help="Override log level (e.g. DEBUG).")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(config_dir=Path(args.config_dir) if args.config_dir else None)

    level = args.log_level or cfg.get("experiment.log_level", "INFO")
    get_logger("pipeline", level=getattr(logging, str(level).upper(), logging.INFO))

    if args.all:
        stages = ALL_STAGES
    elif args.stage:
        stages = args.stage
    else:
        stages = cfg.get("experiment.default_stages", ALL_STAGES[:5])

    communities = args.community or cfg.get("experiment.communities", ["sgexams", "teenagers"])

    logger.info("Pipeline start | stages=%s | communities=%s", stages, communities)
    ctx: dict = {"data": _load_all(cfg, communities)}
    for stage in stages:
        logger.info("=== stage: %s ===", stage)
        STAGE_FUNCS[stage](cfg, ctx)
    logger.info("Pipeline complete. Outputs under results/ and data/processed/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
