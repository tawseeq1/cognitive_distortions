# Cognitive Distortions in Online Student Communities

Tracking **12 categories of cognitive distortions** in two Reddit communities —
**r/SGExams** (Singaporean students) and **r/teenagers** (North American teenagers) —
**before, during, and after COVID-19**, by adapting the population-scale linguistic
methodology of Bollen et al. (2021) to social-media text.

> Reproducible, refactored repository that de-duplicates and packages the logic
> originally lived across eight exploratory Colab notebooks. See
> `docs/repository_audit.md` for what changed and why.

---

## Project Overview

Cognitive distortions are systematic patterns of biased thinking (catastrophizing,
all-or-nothing thinking, emotional reasoning, …) central to depression and anxiety.
Bollen et al. (2021) showed they can be tracked at population scale via linguistic
markers in published text. This project extends that to **spontaneous social-media
text** (> 31M Reddit posts/comments, 2019–2024), at **weekly** resolution, to study
how an acute global stressor (COVID-19) affected distortion expression — and whether
it differed by region.

## Research Motivation

- Published-book n-grams (Bollen) lag real-time population mental health; social
  media is spontaneous and near-real-time.
- COVID-19 is a natural experiment in population-level stress.
- Adolescent mental health is deteriorating globally; r/SGExams vs r/teenagers
  contrasts a high-pressure academic culture with general adolescent discourse.

**Research questions:** (1) How did distortion expression change around COVID-19?
(2) Are there regional differences? (3) How do distortion categories correlate, and
does that change by period?

## Methodology

Dictionary-based detection (1–5 gram matching, social-media-adapted dictionary) →
weekly time series normalised by word volume and z-scored → spike detection and
12×12 correlation per COVID period → Sentence-BERT + KMeans topic modelling
(Davies-Bouldin optimal *k*) → RoBERTa sentiment validation. Full details in
`docs/pipeline.md` and `paper/sections/methods.tex`.

## Pipeline Diagram

See **`docs/pipeline.md`** for the full Mermaid diagram and stage-to-module map. In short:

```
Data → Cleaning → Preprocessing → Detection → Weekly Time Series
   → Temporal Analysis → Correlation ┐
   → Topic Modeling  ─────────────────┼→ Visualization → Results
   → Sentiment       ─────────────────┘
```

## Installation

```bash
# Option A: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `import src...` work everywhere

# Option B: conda
conda env create -f environment.yml
conda activate cogdist
pip install -e .
```

The **core** pipeline (data → detection → time series → correlation → visualisation)
needs only `numpy pandas matplotlib seaborn pyyaml`. Topic modelling adds
`scikit-learn sentence-transformers`; sentiment adds `transformers torch`.

## Reproducing Results

The raw corpus is **private and not shipped**. Out of the box the pipeline runs on a
deterministic **synthetic sample** so you can exercise every stage immediately:

```bash
# Default stages on the synthetic sample (no heavy deps required):
python scripts/run_pipeline.py

# Run only specific stages (repeatable flag):
python scripts/run_pipeline.py --stage detect --stage timeseries

# Everything, incl. topic modelling + sentiment (needs heavy deps):
python scripts/run_pipeline.py --all

# One community only:
python scripts/run_pipeline.py --community sgexams

# Regenerate the manuscript figures/tables + a mapping manifest:
python scripts/reproduce_paper_results.py
```

**To reproduce the *published* numbers:** place the real CSVs under `data/raw/`
(filenames in `configs/data.yaml`) and set `data.use_sample: false`, then rerun.

Outputs land in `results/figures/`, `results/tables/`, `results/outputs/`, and
intermediates in `data/processed/`. Nothing is hard-coded — edit `configs/*.yaml`.

## Published Results

The manuscript figures and correlation tables from the paper are included under
`results/` (no raw data is shipped):

| Output | Location |
|--------|----------|
| Time-series & correlation heatmaps | `results/figures/` |
| Correlation matrices (CSV) | `results/tables/` |
| Figure-to-manuscript mapping | `results/outputs/paper_results_manifest.csv` |

## Repository Structure

```
├── README.md, requirements.txt, environment.yml, setup.py
├── configs/            # YAML config (paths, data, temporal, topic, sentiment, experiment)
├── data/               # raw / interim / processed / metadata  (raw is git-ignored)
├── src/
│   ├── data/           # dataset loading + synthetic sample generator
│   ├── preprocessing/  # tokenisation + n-gram generation
│   ├── cognitive_distortions/  # dictionary, detection, weekly time series
│   ├── temporal_analysis/      # normalise, z-score, spikes, COVID split, correlation
│   ├── topic_modeling/ # SBERT + KMeans + Davies-Bouldin
│   ├── sentiment/      # RoBERTa sentiment
│   ├── visualization/  # all plotting
│   └── utils/          # config, logging, I/O
├── notebooks/          # thin notebooks that ONLY call src/ functions
├── scripts/            # run_pipeline.py (stage-selectable), reproduce_paper_results.py
├── tests/              # pytest suite for the core logic
├── results/            # figures / tables / outputs (committed); models git-ignored
└── docs/               # audit, pipeline, results, research journal, validation
```

## Citation

```bibtex
@misc{ahmad_cogdist_reddit,
  title  = {Cognitive Distortions in Online Student Communities:
            A Comparative Analysis of Singaporean and North American Teenagers
            Before and After COVID-19},
  author = {Ahmad, Tawseeq and Ali, Farhan},
  note   = {Singapore Management University},
  year   = {2024}
}
```
Builds on: Bollen et al. (2021), *Historical language records reveal a surge of
cognitive distortions in recent decades*, PNAS. Full references in `paper/references.bib`.

## Future Work

- Finish per-cluster **proportion-over-time** comparison (merged clustering → timestamps; see journal Week 45).
- **User-overlap** analysis during co-spiking distortions (planned Week 20).
- Validation against clinically labelled users; cross-platform and multilingual extension.

## Known Limitations

- **Raw data not distributed** (private; ~31M docs) — published numbers require the real CSVs.
- N-gram matching can't disambiguate quotation/sarcasm/meta-discussion (topic modelling only partly mitigates).
- Reddit selection + platform bias; **no causal** claims; English-only dictionary; temporal confounds (activity/moderation shifts). See `paper/sections/discussion.tex`.
- The dictionary preserves 3 original missing-comma concatenation artefacts for result fidelity (588 effective n-grams); see `docs/repository_audit.md`.
