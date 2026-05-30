<!--
WHY THIS FILE EXISTS (Step 11 of the brief)
A chronological reconstruction of the research timeline, extracted from
`Project_Progress_Tawseeq.docx` (the weekly progress log, Weeks 1–45, with
advisor "FAComment" feedback from Dr Farhan Ali). The original doc is in reverse
order; this presents it forward. "FA" = faculty advisor.
-->
# Research Journal Summary (Weeks 1–45)

Source: `Project_Progress_Tawseeq.docx`. Advisor feedback is marked **FA**.

| Week(s) | Dates | Objective | Methods | Results | Key insights |
|---|---|---|---|---|---|
| **1** | 7–15 Dec | Upskill | Coursera: Improving DNNs, Structuring ML Projects, CNNs; NLP-with-TF playlist | — | Onboarding/background. |
| **2** | 16–20 Dec | Establish n-gram approach | `nltk.ngrams` on Reddit posts+comments; set up shared Google Sheet of Bollen target n-grams | Reported counts of 2-/3-/…-grams | **FA**: add variants (contractions/truncations) per target n-gram; time one 2-gram ("great but") to estimate total runtime. |
| **3** | 22–29 Dec | Incorporate time | Tokenise posts+comments, store with UTC | RAM exceeded on Colab | **FA**: "divide and conquer" — process pieces sequentially, save, clear cache. |
| **4–5** | 29 Dec–10 Jan | Time-aware detection | Store UTC per token-tuple; created `targetwords.py`; tested Dichotomous Reasoning | Resolved RAM + n-gram detection issues | Splitting comments across notebooks works. |
| **6** | 11–18 Jan | First time series | 250-week array (COVID = week 55); z-score normalisation on posts-only catastrophizing | Mean of z-score ≈ 0 (−1.1e-16, verified); visible pre/post-COVID change | **FA**: any pre/post-COVID change is interesting; plot total 1-grams; try emotional reasoning. |
| **7–8** | 19 Jan–2 Feb | Normalisation by volume | Normalise by weekly unigram count | Values underflow to ~0 (denominator huge) | **FA**: small numerator dominated by large denominator; reciprocal-of-volume looks similar. |
| **9–10** | 3–16 Feb | Smoothing | Moving average, exponential smoothing | Catastrophizing plot weak (few detections) | Comments dataset needed for stable signal. |
| **11–12** | 16 Feb–1 Mar | Social-media variants | Add ChatGPT-generated (filtered) variants to dictionary; compare `log` vs `/100` denominators | Catastrophizing plotted (posts+comments) | **FA**: variants look fine; use full variant list. |
| **13** | 2–9 Mar | Plotting conventions | Same y-axis, x-axis in years, overlay 4 series, raw counts; correlate normalised vs raw | Series look highly correlated | **FA**: drop moving average; use `/100`; suspect correlation is denominator-driven → plot raw counts. |
| **14** | 10–16 Mar | Volume overlay | Overlay `1 / weekly-unigram-count` (z-scored) | — | Check whether normalisation drives cross-category correlation. |
| **15** | 16–23 Mar | Scale to teenagers | Start r/teenagers on Azure; import LLM | r/teenagers ≈ 1.6M posts, 29.2M comments | Plan: subsample teen to ~SGExams size; LLM sentiment on n-gram neighbourhoods. |
| **16** | 24–31 Mar | Subsample + shard | Subsample 800k comments / 100k posts; split comments into 4×200k | Catastrophizing detections for posts | LLM import attempts (Llama/GPT4All) failed. |
| **17** | 1–8 Apr | Robust loading | Saved 2 distortions for posts + half comments; handled malformed CSV | Teen unigram counts very large → divide by 1000; dropped last 7 weeks (2024 spike) | **FA**: maybe `/100` like SGExams if activity small. |
| **18** | 8–15 Apr | More distortions + corr | 2 more distortions; 4-series overlay; correlation matrices (all weeks vs minus last 7) | — | **FA**: y-axis must be z-scored/centred at 0; reuse the Week-13 SGExams overlay style. |
| **19** | 16–29 Apr | (Exams) | — | — | Pause. |
| **20** | 30 Apr–7 May | Plan tiles, corr, spikes, user-overlap | Tile all 12 categories (shared axes); correlation for 3 periods; **filtered "spikes" = >1 SD vs trailing 4-week mean**; user-overlap of co-spiking distortions | Saving weekly detections for all distortions, both subreddits | Defined the spike methodology; planned Venn-style user overlap during co-spikes. |
| **21** | 8–15 May | Scale detection | More distortions plotted; tried to track usernames | Mind-reading ≈ 210+ n-grams ≈ 100+ min/notebook (~500 min total) | Username linkage not achieved; Colab limited to 1 notebook. |
| **22** | 16–23 May | Per-community plots | Plots for teenagers and SGExams | — | — |
| **23** | 24–31 May | Per-document counts | Upload per-comment/post count Excel sheets | Many zero categories | **FA**: many zeros look wrong vs raw weekly plots; need all-to-all correlation on per-document structure; keep n-gram indices to know co-occurrence within a doc. → Begin **topic modelling** of each distortion (start emotional reasoning): embed with SBERT, KMeans, Davies-Bouldin for optimal k. |
| **24** | 1–8 Jun | Fix SGExams + corr | Recomputed SGExams (max-UTC bug); time-series + 12×12 corr per period saved | Per-document corr ≈ 1/NaN (no variance) | **FA**: NaN ≠ 1; NaN from all-zero/constant series. |
| **25** | 9–14 Jun | Housekeeping | Results to Drive | — | — |
| **26** | 15–22 Jun | First topic-model results | SBERT embeddings for emotional reasoning | — | **FA**: define "embedding"; combine both subreddits and cluster together to compare proportions; ~97–98 is lowest Davies-Bouldin. |
| **27** | 23–30 Jun | Embeddings per community | NLTK sentence tokenizer; SBERT `all-mpnet-base-v2` | SGExams emb (607, 768); teen (239, 768); optimal k: SGExams 97, teen 98, merged 98 (DB 2.67) | Established embedding shapes + merged clustering. |
| **28–30** | 1–22 Jul | Merged analysis | Cluster merged dataset | optimal k: SGExams 25, teen 98, merged 41 | **FA**: use merged; for k=41 plot per-cluster freq (SGExams vs teen, sorted), sample 5 posts/cluster, ChatGPT topic labels. |
| **31–34** | 23 Jul–20 Aug | Algorithm comparison | KMeans vs spectral vs agglomerative; serenawjw/slenps pipeline | KMeans 97, spectral 143, agglomerative 149; a rerun gave k=10 | High variance in optimal-k; sweeping 10–150 for convergence. |
| **35** | 21–28 Aug | Wider k sweep | k = 10…500 for emotional reasoning (merged) | optimal k = **496**, min DB = **0.979** | Larger k keeps lowering DB. |
| **36–41** | 29 Aug–10 Oct | All 12 distortions | Davies-Bouldin optimal-k per category (merged) | Catastrophizing 47, emotional reasoning 49, fortune telling 46, labeling 47, magnification 48, mental filtering 49, mindreading 45, overgeneralizing 44, personalizing 29, should 43 | Per-category optimal cluster counts. |
| **42** | 10–17 Oct | Example spreadsheets | Simple sentence tokenizer + fixed-window (x=10) tokenizer; 5 random + 5 centroid examples/cluster | emotional reasoning optimal k=49 (k=999 if range→1k) | **FA**: provide random+centroid examples = target ±x words; flexible x; focus emotional reasoning. |
| **43** | 10–17 Oct | Topic labelling | Named topics from examples | Counts matched | **FA**: labels roughly right but ordering off; you *can* split clusters by time using timestamps — reviewers will ask. |
| **44** | 18–25 Oct | (Interview) | — | — | Pause. |
| **45** | 26 Oct–2 Nov | Proportions over time | Re-labelled clusters; quantify cluster proportions before/during/after | Count sums verified consistent | **FA**: cluster **once on merged data**, then quantify per-cluster proportions per period/subreddit (proportions, not raw counts, due to size differences) — apples-to-apples comparison. |

## Narrative arc

1. **Weeks 1–6** — Build the n-gram detector and get time onto the x-axis (fighting Colab RAM via dataset sharding).
2. **Weeks 7–14** — Wrestle with normalisation (volume denominator dominates); settle on `/100` and z-scoring; adopt consistent plotting.
3. **Weeks 15–22** — Scale from r/SGExams to the much larger r/teenagers (subsample + 4-shard split); per-community plots.
4. **Weeks 20–24** — Define **spike detection** (>1 SD over trailing 4 weeks), per-period **correlation**, and discover the per-document-correlation pitfall (NaN from zero-variance rows).
5. **Weeks 23–45** — Pivot to **topic modelling** (SBERT + KMeans + Davies-Bouldin), converging on the methodology of clustering the *merged* corpus once and comparing cluster **proportions** across time and community, with ChatGPT-assisted topic labels. Focus distortion: **emotional reasoning**.

## Unfinished / open threads (per the journal)

- Per-cluster **proportion over time/subreddit** plotting was still being finalised at Week 45 (the incomplete final cell in `topicmodelling_latest.ipynb`). `src/topic_modeling/embed_cluster.py` provides the cluster labels + timestamps needed to complete this.
- **Username overlap** during co-spiking distortions (Week 20 plan) was never achieved (Week 21: username tracking failed).
- **LLM sentiment** on n-gram neighbourhoods (Weeks 15–16) was planned but not completed in any notebook; `src/sentiment/roberta_sentiment.py` is the clean implementation of that intent.
