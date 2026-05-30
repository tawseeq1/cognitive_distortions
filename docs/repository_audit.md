<!--
WHY THIS FILE EXISTS (Step 1 of the reorganisation brief)
A complete, honest inventory of the *original* repository: every notebook,
script, dataset reference, plus duplicated logic, broken code, missing
dependencies and unused files. Nothing in the original repo was deleted or
modified; this only documents findings to justify the clean re-build.
-->
# Repository Audit

**Audited on:** 2026-05-30
**Original repo root:** `/Users/tawseeq/Documents/Projects/cognitive-distortions`
**Clean rebuild:** `research_repo_clean/` (original files left untouched)

---

## 1. Notebooks discovered (8)

| Notebook | Code cells | Purpose (inferred) |
|---|---|---|
| `comments_masked.ipynb` | 16 | Detect distortions in **r/teenagers comments**; per-comment counts; weekly time-series via `weeks = np.zeros(267)`; before/during/after split. |
| `posts_masked.ipynb` | 16 | Same as above for **posts** (title + body); weekly counts. |
| `corr_row.ipynb` | 6 | Concatenates the 4 teen comment shards + posts; builds per-document distortion matrices for **before/during/after**; writes CSVs (mostly to literal `'your_path'`). |
| `correlation.ipynb` | 11 | Reads per-document count CSVs, merges comments+posts, computes 12×12 Pearson correlation per period, plots heatmaps. |
| `plots_masked.ipynb` | 14 | Time-series figures: z-scores, raw counts, filtered "spikes"; correlation of normalised vs raw series; exports time-series CSVs. Relies on many undefined-in-notebook variables (`z_scores1..12`, `catastrophizing`, …). |
| `topic_modelling.ipynb` | 38 | SBERT (`all-mpnet-base-v2`) + KMeans + Davies-Bouldin for **emotional reasoning**, SGExams vs teenagers vs merged. |
| `topic_modelling_all.ipynb` | 64 | Same pipeline copy-pasted **12 times** (`process1..process12`), then refactored mid-notebook into `process_distortion`. Contains broken cells. |
| `topicmodelling_latest.ipynb` | 30 | Latest topic-model iteration: `tokenize_and_filter`, fixed-window context expansion, random + centroid example extraction, cluster-frequency bars, time-period assignment (incomplete final cell). |

## 2. Python scripts discovered (1)

| File | Purpose |
|---|---|
| `targetwords.py` | The cognitive-distortion dictionary: 12 `target_*` lists of n-gram surface forms (incl. social-media variants). Imported by nearly every notebook. |

## 3. Datasets referenced (all via absolute Google-Drive paths; **none committed**)

All under `/content/drive/MyDrive/Cognitive_Distortions_DrFarhanAli_Tawseeq/Dataset/`:

- `sgexams_comments_Oct2023.csv`, `sgexams_posts_Oct2023.csv` (r/SGExams)
- `teen_sampled_posts.csv`
- `teen_sampled_comments_1.csv` … `_4.csv` (4 shards; #3 and #4 read with `on_bad_lines='skip', engine='python'`)

Intermediate/result CSVs were also referenced under `.../Teenagers/EXCEL/` and `.../Results/teenagers/` (e.g. `teen_all_comments.csv`, `teen_*_corr_percomment.csv`, `teen_time_series_data_*.csv`).

**Other artefacts in the repo root:**
- `Bollen 2021 …​.pdf` — the source paper the methodology is adapted from.
- `Project_Progress_Tawseeq.docx` — weekly research journal (Weeks 1–45). Mined for `docs/research_journal_summary.md`.
- `Final Plots.docx` — image-only document (two text labels: "Sg_exams", "Teenagers").
- `ngramtypes.pkl` — a pickled pandas object. **Could not be loaded** in the audit environment (`object.__new__(BlockManager) is not safe`), i.e. it was pickled with an older pandas and is not portable. Likely a small table of n-gram-type counts; treated as a non-essential cached artefact.
- `paper/` — a complete LaTeX manuscript (abstract, introduction, methods, results, discussion, conclusion) + 18 figures + `references.bib`. This is the authoritative description of the intended pipeline and results.

## 4. Duplicated functionality (the main motivation for the rebuild)

| Logic | Duplicated in |
|---|---|
| `generate_ngrams` / n-gram building | `comments_masked`, `corr_row` (and inline in others) |
| `process_data` (per-doc distortion counts) | `comments_masked`, `posts_masked`, `corr_row` (3 variants) |
| `distortion_ngram_sets` construction | every detection notebook |
| `count_occurrences_for_target_1gram` weekly counter | `comments_masked`, `posts_masked` (identical) |
| `filter_sentences` / `tokenize_and_filter` | `topic_modelling`, `topic_modelling_all` (×3), `topicmodelling_latest` |
| `evaluate(_clustering)` Davies-Bouldin sweep | `topic_modelling_all` defines it **~5 times**; `process1..12` repeat it 12×. |
| KMeans + optimal-k selection | all three topic-model notebooks |
| Correlation + heatmap plotting | `correlation.ipynb`, `plots_masked.ipynb` |

→ In the clean repo each of these exists **exactly once** under `src/`.

## 5. Broken / buggy code found

- **`targetwords.py` missing commas** at three places (around the `he`/`she` mind-reading entries): adjacent string literals with no comma are silently concatenated by Python (e.g. `"he'll not believe" "he will not know"` → one joined string). These become dead entries that never match. The clean dictionary (`src/cognitive_distortions/target_words.py`) preserves them **verbatim for result fidelity** and documents them here. Net effect: 588 effective n-grams vs the "over 600" quoted in the manuscript.
- **`topic_modelling_all.ipynb`**:
  - `process2` references `scores2`/`optimal_k2` *before assignment* and encodes `filtered_sentences2` (undefined) — would raise `NameError`.
  - `process` (cell 47) encodes `filtered_sentences2` instead of `filtered_sentences`.
  - cell 65 iterates `distortion_categories.items()` but `distortion_categories` is a **list**, not a dict → `AttributeError`.
- **`topicmodelling_latest.ipynb`** final cell: `process_data` does `new_df.loc[len(assign_time_period)]` (length of a *function*) and reads a `'UTC'` column that the loaded frames call `created_utc` → broken; the time-period-proportion step was never finished (confirmed by the research journal, Weeks 43–45).
- **`plots_masked.ipynb`** depends on dozens of variables (`z_scores1..12`, `catastrophizing`, `filtered_time_series1..12`, `dates`) that are **never defined in the notebook** — they came from an out-of-band session, so the notebook is not runnable as-is.
- **Literal `'your_path'`** passed to `to_csv` in `comments_masked`, `posts_masked`, `corr_row` — would write a file literally named `your_path` or crash.

## 6. Missing dependencies / environment issues

- Notebooks import **`tensorflow`/`keras`** but never use them → dead imports (excluded from `requirements.txt`).
- `sentence_transformers` installed inline via `!pip install` (not declared).
- Hard dependency on **Google Colab** (`google.colab.sheets`, Drive mounts, absolute Drive paths) → not runnable off Colab.
- `punkt` downloaded ad-hoc in many cells.

## 7. Unused / non-reproducible files

- `ngramtypes.pkl` — unloadable, not referenced by any committed code path.
- `Final Plots.docx` — image-only; superseded by `paper/figures/`.
- Dead `tensorflow`/`keras` imports.

## 8. Recommendations (implemented in `research_repo_clean/`)

1. **De-duplicate** all detection / time-series / clustering / plotting logic into `src/` modules. ✅
2. **Remove hard-coded paths**; drive everything from `configs/*.yaml`. ✅
3. **Make it runnable without the private data** via a deterministic synthetic sample (`src/data/make_sample.py`) so new collaborators can execute the full pipeline. ✅
4. **Single entry point** `scripts/run_pipeline.py` with `--stage` selection + `scripts/reproduce_paper_results.py`. ✅
5. **Tests** for the core logic (`tests/`). ✅
6. **Honest dependency list** (drop TF/Keras; declare SBERT/transformers). ✅
7. **Preserve original results & files** — nothing deleted; the rebuild is additive. ✅
