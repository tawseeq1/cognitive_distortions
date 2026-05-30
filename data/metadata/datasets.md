<!--
WHY THIS FILE EXISTS
Documents the schema and provenance of the (private) datasets so a new
collaborator knows exactly what to drop into data/raw/ to reproduce the real
results, and what the synthetic sample mimics.
-->
# Dataset Metadata

The raw Reddit corpus is **private and not committed** (~31M posts/comments).
Place the CSVs below under `data/raw/` and set `data.use_sample: false` in
`configs/data.yaml` to run on real data. Filenames are configured in `configs/data.yaml`.

## Communities

| Community | Subreddit | Files |
|---|---|---|
| sgexams | r/SGExams | `sgexams_posts_Oct2023.csv`, `sgexams_comments_Oct2023.csv` |
| teenagers | r/teenagers | `teen_sampled_posts.csv`, `teen_sampled_comments_1..4.csv` |

> The teen corpus was subsampled (~800k comments / 100k posts) and the comments
> split into four ~200k-row shards. Shards 3 and 4 contained malformed rows and
> are read with `on_bad_lines='skip', engine='python'` (handled in `src/utils/io.read_csv`).

## Expected columns

| Column | Type | Description |
|---|---|---|
| `created_utc` | int (UTC seconds) | Post/comment creation time. Drives weekly binning + COVID split. |
| `body` | str | Comment text / post body. |
| `title` | str | Post title (posts only). |
| `text` | str | Derived `title + "." + body` (added by `add_text_column`). |

## Time reference

- Study window ≈ 2019-01 → 2024-02 (~250 weeks). Week length = 604800 s.
- COVID windows (UTC seconds, configurable in `configs/temporal.yaml`):
  start `1579478400`, end `1594598400`, study end `1697372672`.

## Synthetic sample (default)

`src/data/make_sample.py` fabricates a deterministic dataset with the **same
schema**, seeded with real distortion n-grams spread across the three COVID
windows, sized so r/teenagers > r/SGExams. It exists only to exercise the code
paths end-to-end; it produces **no scientific result**.
