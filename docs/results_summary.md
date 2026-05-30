<!--
WHY THIS FILE EXISTS (Step 7 of the brief)
A self-contained summary of the study's datasets, methods, metrics, figures and
findings, extracted from the manuscript in `paper/` (the authoritative results)
and cross-checked against the code in `src/`.
-->
# Results Summary

> **Scope note.** The quantitative findings below are taken from the manuscript
> (`paper/sections/`), which was produced on the **real ~31M-document corpus**.
> The shipped code reproduces these *figures and tables* from that data; with the
> default synthetic sample it reproduces their *shapes* only (numbers are
> illustrative). See `docs/final_validation_report.md`.

## Datasets

| Community | Subreddit | Population | Role |
|---|---|---|---|
| SGExams | r/SGExams | Singaporean students | High-pressure academic context (SE Asia) |
| Teenagers | r/teenagers | North American teenagers | General adolescent discourse |

- **Volume:** > 31 million posts + comments, ~2019–2024 (≈ 250 weeks).
- **Periods:** before COVID (< Mar 2020), during (2020–2021 acute phase), after (2022+).
- Teen comments sharded into 4 CSVs (processed separately, merged).

## Methods (as implemented in `src/`)

1. **Distortion dictionary** — 12 categories, 588 n-gram surface forms (1–5 grams) incl. social-media variants (abbreviations, phonetic spellings, letter substitutions). `src/cognitive_distortions/target_words.py`.
2. **Detection** — tokenise → generate 1–5 grams → exact-match against each category's set. `detect.py`.
3. **Weekly time series** — per-week counts + total unigram volume `N_j`. `timeseries.py`.
4. **Normalisation** — `X̄ = X / N` (× scale), then **z-score**. `temporal_analysis/timeseries_ops.py`.
5. **Spike detection** — > 1 SD above trailing 4-week mean.
6. **Correlation** — 12×12 Pearson, overall and per COVID period. `temporal_analysis/correlation.py`.
7. **Topic modeling** — Sentence-BERT (`all-mpnet-base-v2`) embeddings → KMeans → **Davies-Bouldin** optimal *k*. `topic_modeling/embed_cluster.py`.
8. **Sentiment** — RoBERTa (3-class) on distortion sentences as a validity check. `sentiment/roberta_sentiment.py`.

## Metrics

- **Pearson correlation coefficient** (inter-distortion co-occurrence).
- **Davies-Bouldin index** (cluster separability; lower = better) for choosing #topics.
- **Sentiment class probabilities** (negative/neutral/positive).

## Key figures (paper/figures → reproduced as results/figures)

| Manuscript figure | Reproduced file |
|---|---|
| `timeseries_raw_{community}.png` | `{community}_timeseries_raw.png` |
| `timeseries_zscore_{community}.png` | `{community}_timeseries_zscore.png` |
| `timeseries_multiple_{community}.png` | `{community}_timeseries_overlay.png` |
| `correlation_{before,during,after}_{community}.png` | `{community}_correlation_{before,during,after}.png` |
| `correlation_all_teenagers.png` | `teenagers_correlation_overall.png` |
| `davies_bouldin_{comparison,emotional}.png` | `davies_bouldin_Emotional_Reasoning.png` |

(See `results/outputs/paper_results_manifest.csv` produced by `reproduce_paper_results.py`.)

## Key findings (from the manuscript)

1. **COVID-19 impact is region-specific.** Both communities show changed distortion expression around COVID-19, but direction/persistence differ: r/teenagers elevated before *and* during; r/SGExams more variable (plausibly tied to exam-calendar disruption).
2. **Dichotomous reasoning dominates** absolute counts in both communities and all periods (high base rate of "always/never/should"), consistent with absolutist-language markers of distress (Al-Mosaiwi & Johnstone 2018) and Bollen et al. (2021). *(Reproduced qualitatively even on the synthetic sample.)*
3. **Correlation structure is not static.** In r/teenagers, inter-distortion correlations **weakened during** COVID and stayed weaker after; in r/SGExams they **strengthened after** COVID (more synchronised expression).
4. **Topic structure differs by community.** r/teenagers yields more distinct clusters (lower Davies-Bouldin) than r/SGExams at equal *k*; for emotional reasoning, DB decreases with *k*, stabilising ~400–500 clusters.
5. **Methodological contribution.** Higher temporal resolution (weekly vs Bollen's yearly) captures acute responses; a social-media-adapted dictionary extends Bollen's book-based n-grams.

## Conclusions

Social-media text analysis is a feasible, near-real-time lens on population mental-health indicators. Adapting Bollen et al. (2021) to Reddit reveals region-specific COVID-19 dynamics and changing co-occurrence structure among distortions, with implications for public-health surveillance and youth mental-health monitoring. Limitations: selection/platform bias, n-gram context ambiguity, no causal claims, English-only dictionary, temporal confounds (see `paper/sections/discussion.tex`).
