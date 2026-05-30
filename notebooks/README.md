# Notebooks

These notebooks are intentionally **thin**: they only *call* functions from
`src/` and render figures/tables. No analysis logic is duplicated here — that is
the whole point of the refactor (compare with the original eight notebooks, which
each re-implemented detection, clustering and plotting).

| Notebook | What it does | Heavy deps? |
|---|---|---|
| `01_data_and_detection.ipynb` | Load a community, build the per-document distortion-count matrix, show category totals. | No |
| `02_temporal_analysis.ipynb` | Weekly time series → normalise → z-score → tile plot; spike detection. | No |
| `03_correlation_analysis.ipynb` | 12×12 correlation heatmap per COVID period. | No |
| `04_topic_modeling.ipynb` | SBERT + KMeans + Davies-Bouldin for the focus distortion. | Yes (`sentence-transformers`, `scikit-learn`) |
| `05_sentiment_analysis.ipynb` | RoBERTa sentiment of distortion sentences. | Yes (`transformers`, `torch`) |

Run them from the repo root or the `notebooks/` folder — the first cell adds the
repo root to `sys.path`. Switch `COMMUNITY = 'sgexams'` to `'teenagers'` to analyse
the other community. By default everything runs on the synthetic sample; set
`data.use_sample: false` in `configs/data.yaml` (with real CSVs in `data/raw/`) for
the real corpus.
