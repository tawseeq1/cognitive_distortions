#!/usr/bin/env python3
"""Regenerate the figures and tables reported in the manuscript.

WHY THIS FILE EXISTS
--------------------
``scripts/run_pipeline.py`` is the general engine; this script is the
*paper-facing* wrapper. It runs the exact stages that produce the manuscript's
figures and correlation tables, then writes a manifest
(``results/outputs/paper_results_manifest.csv``) that maps each generated file to
the corresponding figure/table in ``paper/``. This is what a reviewer or future
collaborator runs to reproduce the paper.

USAGE
-----
    python scripts/reproduce_paper_results.py            # core figures/tables
    python scripts/reproduce_paper_results.py --with-heavy  # + topic model + sentiment

NOTE ON DATA
------------
The real ~31M-document corpus is private and not shipped. With the default
config this runs on the synthetic sample, so the *shapes* of all outputs are
reproduced but the *numbers* are illustrative. Place the real CSVs under
data/raw/ and set ``data.use_sample: false`` in configs/data.yaml to reproduce
the actual published figures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts import run_pipeline as rp  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_dataframe  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("reproduce")

# Maps our generated figure stems -> the figure names referenced in paper/sections.
PAPER_FIGURE_MAP = [
    ("{c}_timeseries_raw.png", "figures/timeseries_raw_{c}.png", "Raw weekly distortion counts"),
    ("{c}_timeseries_zscore.png", "figures/timeseries_zscore_{c}.png", "Z-score normalised series"),
    ("{c}_timeseries_overlay.png", "figures/timeseries_multiple_{c}.png", "All 12 series overlaid"),
    ("{c}_correlation_before.png", "figures/correlation_before_{c}.png", "Correlation before COVID"),
    ("{c}_correlation_during.png", "figures/correlation_during_{c}.png", "Correlation during COVID"),
    ("{c}_correlation_after.png", "figures/correlation_after_{c}.png", "Correlation after COVID"),
    ("{c}_correlation_overall.png", "figures/correlation_all_{c}.png", "Overall correlation"),
]


def build_manifest(cfg, communities, with_heavy: bool) -> pd.DataFrame:
    figdir = cfg.path("paths.results_figures")
    rows = []
    for c in communities:
        for gen, paper, desc in PAPER_FIGURE_MAP:
            gen_path = figdir / gen.format(c=c)
            rows.append({
                "community": c,
                "generated_figure": str(gen_path.relative_to(REPO_ROOT)),
                "paper_figure": paper.format(c=c),
                "description": desc,
                "exists": gen_path.exists(),
            })
    if with_heavy:
        focus = cfg.get("topic_modeling.focus_category", "Emotional Reasoning").replace(" ", "_")
        for stem, desc in [
            (f"davies_bouldin_{focus}.png", "Davies-Bouldin cluster selection"),
            (f"cluster_frequency_{focus}.png", "Topic cluster frequency"),
        ]:
            p = figdir / stem
            rows.append({"community": "merged", "generated_figure": str(p.relative_to(REPO_ROOT)),
                         "paper_figure": f"figures/davies_bouldin_*.png", "description": desc,
                         "exists": p.exists()})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-heavy", action="store_true",
                        help="Also run topic modelling and sentiment (needs heavy deps).")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(config_dir=Path(args.config_dir) if args.config_dir else None)
    communities = cfg.get("experiment.communities", ["sgexams", "teenagers"])

    if cfg.get("data.use_sample", True):
        logger.warning(
            "Running on SYNTHETIC sample data: output shapes reproduce the paper, "
            "numbers are illustrative. Set data.use_sample=false with real CSVs for "
            "the published figures."
        )

    ctx = {"data": rp._load_all(cfg, communities)}
    stages = ["detect", "timeseries", "correlation", "visualize"]
    if args.with_heavy:
        stages += ["topic_model", "sentiment"]
    for stage in stages:
        logger.info("=== reproduce stage: %s ===", stage)
        rp.STAGE_FUNCS[stage](cfg, ctx)

    manifest = build_manifest(cfg, communities, args.with_heavy)
    out = cfg.path("paths.results_outputs") / "paper_results_manifest.csv"
    save_dataframe(manifest, out)
    missing = manifest.loc[~manifest["exists"], "generated_figure"].tolist()
    logger.info("Reproduced %d/%d paper artefacts.", int(manifest["exists"].sum()), len(manifest))
    if missing:
        logger.warning("Missing artefacts: %s", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
