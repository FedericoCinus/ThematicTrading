"""Generate notebooks/1.0-bertopic-small-sample.ipynb from source cells.

Keeping the notebook in a generator makes it easy to diff/regenerate. Run:
    uv run python scripts/make_bertopic_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "1.0-bertopic-small-sample.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md(r"""
# BERTopic on a small Bloomberg sample — offline vs online

**Goal:** a first, fast playground for *unsupervised theme discovery* on Bloomberg
headlines. We deliberately work on a **tiny** corpus (≈100 headlines/year × 10
years ≈ 1,000 docs) so every step runs in seconds and we can build intuition for:

1. **Offline BERTopic** — fit once on the whole corpus (embed → UMAP → HDBSCAN → c-TF-IDF).
2. **Online / incremental BERTopic** — `partial_fit` year-by-year with `IncrementalPCA`
   + `MiniBatchKMeans` + `OnlineCountVectorizer`, the way you'd run on a stream.

> **Data:** raw news live in the sibling *Theme-Detection* project and are reached
> through the symlink `data/raw/bloomberg/` — nothing heavy is copied here.
> The small sample is built once by `scripts/build_sample.py` and cached to parquet.
"""),
    code(r"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

PER_YEAR = 10000
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / f"sample_{PER_YEAR}_per_year.parquet"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, general-purpose (384-D)
RANDOM_SEED = 42

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("device:", DEVICE)
"""),
    md(r"""
## 1. Load the cached sample

If the parquet doesn't exist yet, build it once (≈30s for all 10 years). `.xz` has no
random access and the files are ~19 M headlines/year, so to stay fast and within a
small RAM budget the builder decompresses only a **prefix chunk** of each year
(≈ first 10 days) and samples from it. Tradeoff: the sample is drawn from the *start*
of each year, not spread evenly across all 12 months — fine for a first playground.

```bash
uv run python scripts/build_sample.py --per-year 100
```
"""),
    code(r"""
assert SAMPLE_PATH.exists(), (
    f"{SAMPLE_PATH} missing — run:\n"
    f"    uv run python scripts/build_sample.py --per-year {PER_YEAR}"
)

corpus = pd.read_parquet(SAMPLE_PATH)
# Drop exact-duplicate headlines (wires repost the same story); keep first occurrence.
corpus = corpus.drop_duplicates(subset="Headline").reset_index(drop=True)

docs = corpus["Headline"].tolist()
timestamps = corpus["CaptureTime"].tolist()
years = corpus["year"].tolist()

print(f"{len(docs):,} unique headlines, {corpus['year'].nunique()} years")
display(corpus.groupby("year").size().rename("headlines").to_frame().T)
corpus.head()
"""),
    md(r"""
## 2. Embed the headlines once

Both the offline and online runs reuse the same sentence embeddings, so we compute
them a single time. `all-MiniLM-L6-v2` is a small 384-D general-purpose model — fine
for a first look; swap in a finance-tuned model later to compare.
"""),
    code(r"""
embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
embeddings = embedder.encode(docs, batch_size=64, show_progress_bar=True)
# float64: the online path (IncrementalPCA → MiniBatchKMeans) needs doubles; the
# offline path is happy with either. Tiny array, so the cast is essentially free.
embeddings = np.asarray(embeddings, dtype="float64")
print("embeddings:", embeddings.shape, embeddings.dtype)
"""),
    md(r"""
## 3. Offline BERTopic

Fit the full pipeline once. With only ~1,000 docs the default HDBSCAN settings would
mark almost everything as an outlier, so we shrink the neighbourhood / cluster sizes
to small-corpus-friendly values:

- `UMAP(n_neighbors=15, n_components=5)` — reduce embeddings before clustering.
- `HDBSCAN(min_cluster_size=10)` — a "topic" needs ≥10 headlines.
- `CountVectorizer(stop_words="english", ngram_range=(1,2))` — drives the c-TF-IDF
  keywords that *describe* each topic.

Topic `-1` is the outlier bucket.
"""),
    code(r"""
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=RANDOM_SEED
)
hdbscan_model = HDBSCAN(
    min_cluster_size=10, min_samples=5, metric="euclidean",
    cluster_selection_method="eom", prediction_data=True,
)
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

topic_model = BERTopic(
    embedding_model=embedder,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True,
)
topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

info = topic_model.get_topic_info()
print(f"{(info['Topic'] != -1).sum()} topics + outliers; "
      f"{info.loc[info['Topic'] == -1, 'Count'].sum()} / {len(docs)} docs are outliers")
info.head(20)
"""),
    md("### Inspect a few topics — top keywords and representative headlines"),
    code(r"""
for topic_id in info.loc[info["Topic"] != -1, "Topic"].head(6):
    words = ", ".join(w for w, _ in topic_model.get_topic(topic_id)[:8])
    print(f"\n── Topic {topic_id} ── {words}")
    for doc in topic_model.get_representative_docs(topic_id)[:3]:
        print("   •", doc[:120])
"""),
    md(r"""
### Visualise

`visualize_barchart` shows the defining keywords per topic; `visualize_topics_over_time`
uses the `CaptureTime` stamps to show how each theme's headline volume moves across the
10 years — a first hint at *emerging* vs *fading* themes.
"""),
    code(r"""
topic_model.visualize_barchart(top_n_topics=8, n_words=8)
"""),
    code(r"""
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=6)
"""),
    md(r"""
## 4. Online / incremental BERTopic

On a real feed you can't refit on the whole history every day. The online recipe
swaps the three stateful stages for incremental ones and calls `partial_fit` on each
batch — here, one batch **per year**:

| stage        | offline                | online (incremental)        |
|--------------|------------------------|-----------------------------|
| dim. reduce  | `UMAP`                 | `IncrementalPCA`            |
| cluster      | `HDBSCAN`              | `MiniBatchKMeans` (fixed k) |
| keywords     | `CountVectorizer`      | `OnlineCountVectorizer`     |

Trade-off: you must fix the number of clusters up front (`MiniBatchKMeans`), and the
keyword vocabulary updates with a `decay` so stale terms fade. This is the path to
explore for *streaming* theme discovery.
"""),
    code(r"""
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

online_umap = IncrementalPCA(n_components=5)
online_cluster = MiniBatchKMeans(n_clusters=10, random_state=RANDOM_SEED)
online_vectorizer = OnlineCountVectorizer(stop_words="english", decay=0.01)

online_model = BERTopic(
    umap_model=online_umap,
    hdbscan_model=online_cluster,
    vectorizer_model=online_vectorizer,
    verbose=False,
)

# Feed one year at a time, in chronological order — simulating a stream.
for year in sorted(corpus["year"].unique()):
    mask = corpus["year"].values == year
    online_model.partial_fit(
        [d for d, m in zip(docs, mask) if m],
        embeddings=embeddings[mask],
    )
    n_topics = len(set(online_model.topics_)) - (1 if -1 in online_model.topics_ else 0)
    print(f"after {year}: {n_topics} topics, {len(online_model.topics_):,} docs seen")

online_model.get_topic_info().head(12)
"""),
    md(r"""
> **Note** `partial_fit` keeps `online_model.topics_` for the *most recent* batch only.
> To track topics across the whole stream, accumulate the returned topics yourself, or
> look at `BERTopic.merge_models` for combining independently-fit models incrementally.

## 5. Where to go next

- **Compare embedders:** finance-tuned (e.g. `FinLang/finance-embeddings-investopedia`)
  vs MiniLM on the same sample.
- **Scale up gradually:** 100 → 1,000 → 10,000 per year (`--per-year`) and watch how
  topic granularity and outlier rate change.
- **Tune granularity:** `min_cluster_size` / `nr_topics` (offline) and `n_clusters`
  (online) control how broad vs fine the themes are.
- **Emergence:** lean on `topics_over_time` to flag themes whose volume accelerates —
  the bridge to the `bertrend`-style experiments in the `0.x` notebooks.
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
