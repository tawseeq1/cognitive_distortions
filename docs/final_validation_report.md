<!--
WHY THIS FILE EXISTS (Step 13 of the brief)
An honest validation record. It states exactly what was executed and verified in
the build environment, what could NOT be verified (and why), the assumptions
made, and the remaining reproducibility risks. Nothing here is claimed to work
unless it was actually run.
-->
# Final Validation Report

**Environment:** macOS (darwin), Python 3.13.11.
**Installed:** numpy 2.4.2, pandas 3.0.0, matplotlib 3.10.8, seaborn 0.13.2, PyYAML 6.0.3.
**NOT installed:** nltk, scikit-learn, sentence-transformers, transformers, torch, pytest, scipy.

## ✅ Commands tested and verified (exit 0)

| Command | Result |
|---|---|
| `python -m py_compile` over all `src/`, `scripts/`, `tests/` `.py` | **All compile**, no syntax errors. |
| `python scripts/run_pipeline.py` (default stages, both communities) | exit 0 → **14 figures**, **10 tables**, **10 processed CSVs**, **4 sample CSVs**. |
| `python scripts/run_pipeline.py --stage detect` / `--stage timeseries` / etc. | Individual stages run; dependent stages auto-run prerequisites. |
| `python scripts/reproduce_paper_results.py` | exit 0 → manifest with **14/14** paper artefacts present. |
| Test suite (31 tests, run via a minimal runner since pytest is absent) | **31 passed, 0 failed, 0 skipped**. |
| All 5 notebooks | Valid notebook JSON (executed logic is covered by the same `src/` functions the tests exercise). |

### What the verified run demonstrates
- Config loads and resolves all paths relative to repo root (no hard-coded paths).
- Synthetic-sample fallback works when raw data is absent (`data.use_sample: true`).
- Detection, weekly time-series, normalisation, z-scoring (mean ≈ 0 verified to ~1e-16),
  spike detection, COVID splitting, correlation, and all figure rendering work end-to-end.
- Qualitatively reproduces the paper's "**Dichotomous Reasoning dominates**" finding even on synthetic data.

## ⚠️ Could NOT be verified in this environment (documented, not claimed)

| Item | Why | Mitigation |
|---|---|---|
| **Topic-modelling stage** (`topic_model`) | `sentence-transformers` + `scikit-learn` not installed. Verified it **fails with a clear, actionable `ImportError`** rather than a cryptic crash. | Pure-Python parts (`filter_sentences`, `expand_around_ngram`, example tables) **are** unit-tested and pass. Install heavy deps to run. |
| **Sentiment stage** (`sentiment`) | `transformers` + `torch` not installed. Same clear-error behaviour verified. | Code path reviewed; needs heavy deps + model download to execute. |
| **NLTK tokenisation path** | `nltk` not installed → the **regex fallback** tokeniser is used (and tested). | Numbers on real data may differ slightly from NLTK's `punkt`; install `nltk` to match the original notebooks exactly. |
| **Real published numbers** | The ~31M-document corpus is **private and not present**. | Pipeline runs on synthetic data; drop real CSVs in `data/raw/` and set `data.use_sample: false`. |
| `ngramtypes.pkl` | Pickled with an older pandas; **unloadable** here (`BlockManager` error). | Treated as a non-essential cached artefact; not used by any code path. |

## Assumptions made

1. **Dictionary fidelity over correction.** The 3 missing-comma concatenation bugs in the original `targetwords.py` are **preserved verbatim** (588 effective n-grams) so detection counts match the original; they are dead entries that never match a real n-gram. Documented in `docs/repository_audit.md`.
2. **Comments drive the weekly series.** The weekly-count notebooks built series primarily from comment bodies; the pipeline does the same (posts feed the per-document correlation matrices). Posts can be added to the time series with the same function.
3. **COVID constants** (`1579478400 / 1594598400 / 1697372672`) are taken from the notebooks and centralised in `configs/temporal.yaml`.
4. **Sentiment model.** No finished sentiment notebook existed; we selected the public `cardiffnlp/twitter-roberta-base-sentiment-latest` (social-media RoBERTa) as a faithful realisation of the manuscript's described step.
5. **Per-period correlation** is computed by re-binning each period's documents into weekly series and correlating those (matches the "weekly counts" correlation the paper reports).

## Reproducibility risks

- **Heavy-dep stages unverified end-to-end here** (see table). They are written, compile, and have unit-tested pure-Python components, but their model-dependent execution was not run.
- **Tokeniser divergence:** regex fallback vs NLTK `punkt` can change counts marginally. Install `nltk` for exact parity.
- **Non-determinism in clustering:** mitigated by fixing `random_state=42`, but BLAS/KMeans `n_init` differences across versions can shift cluster assignments slightly.
- **Synthetic sample is not science:** it validates code paths only. Treat all sample-mode numbers as illustrative.
- **Library major versions** are very new (pandas 3.0, numpy 2.4); pinned lower bounds in `requirements.txt` are permissive — pin exact versions for a frozen reproduction.

## Outstanding / unfinished (inherited from the original work)

- Per-cluster **proportion-over-time** plotting (journal Week 45) — data plumbing exists in `src/topic_modeling`, final aggregation not wired into a stage.
- **User-overlap during co-spikes** (journal Week 20) — not implemented; needs a username column not present in the sample schema.

## Bottom line

The **core pipeline is fully reproducible and verified** on synthetic data in a
minimal environment (5 packages). The **topic-modelling and sentiment stages are
implemented and import-safe** but require the heavy optional dependencies and were
not executed here. The **published numerical results require the private corpus**,
which is not distributed. All of the above is reflected honestly in the code's
runtime warnings and in this report.
