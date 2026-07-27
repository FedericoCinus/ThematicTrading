"""Generate notebooks/1.2-bertopic-hierarchical-soft.ipynb from source cells.

Companion to make_bertopic_notebook.py / make_soft_clustering_notebook.py — same
generator pattern. Builds a *genuine hierarchy* (dendrogram over the ~1.4k topics, not
over the 100k docs) and derives soft per-document weights from it. Run:
    uv run python scripts/make_hierarchical_soft_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "1.2-bertopic-hierarchical-soft.ipynb"


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
# BERTopic — hierarchy-based soft clustering (scales to 100k)

`1.1` got soft weights from HDBSCAN's `calculate_probabilities=True`, which is slow and
memory-heavy, so it only ran on 10k docs. The heaviness, though, is **specific to
HDBSCAN's algorithm** (`all_points_membership_vectors`), *not* to soft clustering in
general — we're just working with vectors.

Here we build a **genuine hierarchy** and read soft weights off it, cheaply, on the full
**100k-doc** corpus. The key trick: cluster the **~1,400 topics** into a dendrogram (a
1,400² linkage — milliseconds), *not* the 100k documents.

> ⚠️ **Why not a dendrogram over the documents directly?** Agglomerative clustering
> needs an O(n²) distance matrix — at 100k docs that's **~80 GB**. That's the single
> heaviest thing you could do here. The output weight matrix is also a floor: `n×k`
> floats ≈ 1.15 GB at 100k×1439 in float64 → we keep cuts small and store **float32**.

**Recipe:** fit topics → linkage over topic-centroid embeddings → cut at `K`
super-themes → soft weight of each doc = `softmax(cosine(doc, super-centroid))`.
One matmul, batchable, tens of MB.
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
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import softmax

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

PER_YEAR = 10000  # full 100k-doc corpus — the hierarchy math is cheap regardless
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / f"sample_{PER_YEAR}_per_year.parquet"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

CUT_LEVELS = [10, 25, 50]  # dendrogram cuts → # of super-themes
PRIMARY_K = 25             # the cut we save in detail
TEMP = 0.1                 # softmax temperature on cosine sims (smaller = sharper)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("device:", DEVICE)
"""),
    md(r"""
## 1. Load sample & embed

Same 100k corpus as `1.0`. Build it first if missing:

```bash
uv run python scripts/build_sample.py --per-year 10000
```
"""),
    code(r"""
assert SAMPLE_PATH.exists(), (
    f"{SAMPLE_PATH} missing — run: uv run python scripts/build_sample.py --per-year {PER_YEAR}"
)

corpus = pd.read_parquet(SAMPLE_PATH).drop_duplicates(subset="Headline").reset_index(drop=True)
docs = corpus["Headline"].tolist()

embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
embeddings = np.asarray(embedder.encode(docs, batch_size=64, show_progress_bar=True), dtype="float32")
print(f"{len(docs):,} docs, embeddings {embeddings.shape} {embeddings.dtype}")
"""),
    md(r"""
## 2. Fit base topics (offline)

Same config as `1.0` — this is the only expensive step (~9 min, ~1.5 GB on 100k). We do
**not** need `calculate_probabilities`; the soft weights come later from the hierarchy.
After fitting, `topic_model.topic_embeddings_` holds one centroid vector per topic.
"""),
    code(r"""
topic_model = BERTopic(
    embedding_model=embedder,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=RANDOM_SEED),
    hdbscan_model=HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                          cluster_selection_method="eom", prediction_data=True),
    vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2)),
    verbose=True,
)
topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
topics = np.asarray(topics)

info = topic_model.get_topic_info()
n_topics = int((info["Topic"] != -1).sum())
print(f"{n_topics} base topics + outliers; topic_embeddings_ {np.asarray(topic_model.topic_embeddings_).shape}")
"""),
    md(r"""
## 3. Build the topic dendrogram

`topic_embeddings_` is row-aligned with the **sorted** topic ids (topic `-1` first). We
drop `-1` and run a **ward** `linkage` over the remaining ~1,400 topic centroids (on
L2-normalized vectors, so it's cosine geometry but with *balanced* clusters — average/
single linkage chains into one giant blob in high-dim embedding space). The same
`linkage_function` is handed to BERTopic's `hierarchical_topics` / `visualize_hierarchy`,
so the picture and the soft-weight math use *one* consistent hierarchy.
"""),
    code(r"""
topic_ids = sorted(info["Topic"])                       # [-1, 0, 1, ...]
te = np.asarray(topic_model.topic_embeddings_)          # aligned with topic_ids
real_rows = [i for i, t in enumerate(topic_ids) if t != -1]
real_ids = [topic_ids[i] for i in real_rows]
te_real = te[real_rows]
sizes = info.set_index("Topic")["Count"]

def topic_linkage(x):
    # ward gives *balanced* clusters (average/single linkage chains into one giant blob in
    # high-dim embedding space). Handle both call conventions: we pass 2-D topic vectors
    # (-> ward on L2-normalized = cosine geometry); BERTopic's viz passes a 1-D condensed
    # distance matrix (-> feed straight to ward).
    x = np.asarray(x)
    if x.ndim == 1:
        return linkage(x, method="ward")
    return linkage(normalize(x), method="ward", metric="euclidean")

Z = topic_linkage(te_real)
print(f"linkage over {len(real_ids)} topic centroids -> Z {Z.shape}")
"""),
    md(r"""
### Hover-rich names for every internal node

BERTopic draws an internal node's hover label from `hier_df['Parent_Name']`. By default
that's a stopword-heavy underscore string; we **overwrite it** with the *pooled top
words* of the topics beneath the node plus its size (`N topics, M docs`). Now hovering
any merge point answers "what cluster would form if I cut here?".
"""),
    code(r"""
hier_df = topic_model.hierarchical_topics(docs, use_ctfidf=False, linkage_function=topic_linkage)

def clean_name(topic_list):
    # pooled top c-TF-IDF words across the topics under a node, + cluster size
    pool = {}
    for tid in topic_list:
        for w, sc in topic_model.get_topic(tid)[:5]:
            if w:
                pool[w] = pool.get(w, 0.0) + sc
    top = sorted(pool, key=pool.get, reverse=True)[:6]
    ndocs = int(sizes.reindex(topic_list).fillna(0).sum())
    return f"{' · '.join(top)}  ({len(topic_list)} topics, {ndocs:,} docs)"

hier_df["Parent_Name"] = hier_df["Topics"].apply(clean_name)

# Inline preview: cap leaves so it's readable in the notebook. Hover the merge points.
fig_hier = topic_model.visualize_hierarchy(
    hierarchical_topics=hier_df, use_ctfidf=False, linkage_function=topic_linkage,
    top_n_topics=80,
)
fig_hier
"""),
    md(r"""
### A prettier view: nested theme icicle

A 1,400-leaf dendrogram is hard to read. The same hierarchy is far nicer as a **nested
icicle**: cut the tree at a few resolutions (coarse → fine), nest them (cuts of one
linkage are always nested), size each block by article count, colour by coarse theme,
and label with pooled top words. **Click any block to zoom in; hover for the full name.**
"""),
    code(r"""
import plotly.graph_objects as go
import plotly.colors as pc

K_LEVELS_VIZ = [k for k in (8, 25, 60) if k < len(real_ids)]   # coarse -> fine
lev = {K: fcluster(Z, t=K, criterion="maxclust") for K in K_LEVELS_VIZ}

def words_for(members, n=5):
    pool = {}
    for tid in members:
        for w, sc in topic_model.get_topic(tid)[:5]:
            if w:
                pool[w] = pool.get(w, 0.0) + sc
    return sorted(pool, key=pool.get, reverse=True)[:n]

# Build nested nodes: ROOT -> coarse -> mid -> fine, members accumulated bottom-up.
nodes = {}
def add(nid, parent, coarse, members):
    nd = nodes.setdefault(nid, {"parent": parent, "coarse": coarse, "members": set()})
    nd["members"].update(members)

for idx, tid in enumerate(real_ids):
    coarse = int(lev[K_LEVELS_VIZ[0]][idx])
    path = ["ROOT"]
    for K in K_LEVELS_VIZ:
        path.append(path[-1] + f"|K{K}:{lev[K][idx]}")
    for d in range(1, len(path)):
        add(path[d], path[d - 1], coarse, [tid])   # parent of the coarse level is "ROOT"
add("ROOT", "", -1, real_ids)

palette = pc.qualitative.Set3 + pc.qualitative.Pastel1 + pc.qualitative.Pastel2
ids, labels, parents, values, hover, colors = [], [], [], [], [], []
for nid, nd in nodes.items():
    mem = list(nd["members"])
    val = int(sizes.reindex(mem).fillna(0).sum())
    ids.append(nid); parents.append(nd["parent"]); values.append(val)
    labels.append("ALL THEMES" if nid == "ROOT" else " · ".join(words_for(mem, 3)))
    hover.append(f"<b>{' · '.join(words_for(mem, 6))}</b><br>{len(mem)} topics · {val:,} docs")
    colors.append("#efefef" if nid == "ROOT" else palette[nd["coarse"] % len(palette)])

fig_ice = go.Figure(go.Icicle(
    ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
    customdata=hover, hovertemplate="%{customdata}<extra></extra>",
    marker=dict(colors=colors, line=dict(color="white", width=1)),
    tiling=dict(orientation="h"), sort=True,
))
fig_ice.update_layout(
    margin=dict(t=46, l=8, r=8, b=8), height=720, width=1150,
    title="Bloomberg theme hierarchy  ·  click a block to zoom, hover for the cluster name",
    font=dict(size=12),
)
print(f"icicle: {len(ids)} nodes over levels {K_LEVELS_VIZ}")
fig_ice
"""),
    md(r"""
## 4. Cut the tree → super-themes

`fcluster(..., 'maxclust')` cuts the dendrogram into `K` groups. Each super-theme's
embedding is the **size-weighted mean** of its member topic centroids (bigger topics
pull harder), and its keywords are pooled from the member topics' top c-TF-IDF words.
"""),
    code(r"""
def build_supers(K):
    # returns (tid->super, super_centroids[K,d], super_keywords[K])
    lab = fcluster(Z, t=K, criterion="maxclust") - 1            # 0..K-1 per real topic
    tid_to_super = {tid: int(s) for tid, s in zip(real_ids, lab)}

    cent = np.zeros((K, te_real.shape[1]), dtype="float32")
    for ridx, tid in enumerate(real_ids):
        cent[tid_to_super[tid]] += te_real[ridx] * float(sizes[tid])
    cent = normalize(cent)

    words = []
    for s in range(K):
        members = [tid for tid in real_ids if tid_to_super[tid] == s]
        pool = {}
        for tid in members:
            for w, sc in topic_model.get_topic(tid)[:5]:
                pool[w] = pool.get(w, 0.0) + sc
        top = sorted(pool, key=pool.get, reverse=True)[:8]
        words.append(", ".join(top))
    return tid_to_super, cent, words

tid_to_super, super_cent, super_words = build_supers(PRIMARY_K)
print(f"K={PRIMARY_K} super-themes:")
for s, w in enumerate(super_words):
    n = sum(v == s for v in tid_to_super.values())
    print(f"  S{s:2d} ({n:2d} topics): {w}")
"""),
    md(r"""
## 5. Soft per-document weights

The whole point: `softmax(cosine(doc, super-centroid) / TEMP)` → a `(n_docs × K)` matrix
where every row is a probability distribution over the `K` super-themes. One matmul,
float32, ~10 MB. Every doc gets weights — including the HDBSCAN outliers (topic `-1`),
which simply end up with a flatter distribution.
"""),
    code(r"""
sims = normalize(embeddings) @ super_cent.T          # (n_docs, K) cosine
W = softmax(sims / TEMP, axis=1).astype("float32")
print(f"W {W.shape} {W.dtype}  rows sum~1: {np.allclose(W.sum(1), 1, atol=1e-4)}  ~{W.nbytes/1e6:.0f} MB")

def top_supers(i, n=3):
    order = np.argsort(W[i])[::-1][:n]
    return ", ".join(f"S{s}={W[i, s]:.2f}" for s in order)

# Most ambiguous headlines (highest 2nd-best super-theme weight).
second = np.sort(W, axis=1)[:, -2]
amb = pd.DataFrame({
    "headline": [d[:70] for d in docs],
    "hard_topic": topics,
    "top_super_themes": [top_supers(i) for i in range(len(docs))],
    "second_weight": second.round(3),
}).sort_values("second_weight", ascending=False)
amb.head(12)
"""),
    code(r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(W.max(axis=1), bins=40)
ax[0].set(title=f"Max super-theme weight per doc (K={PRIMARY_K})", xlabel="weight", ylabel="docs")
ax[1].hist((W > 0.1).sum(axis=1), bins=range(0, PRIMARY_K + 1))
ax[1].set(title="# super-themes with weight > 0.1", xlabel="n super-themes")
plt.tight_layout()
"""),
    md(r"""
## 6. Multi-resolution view

Because it's so cheap, we can read weights at several cuts at once — a coarse→fine soft
profile per document. Here: the dominant super-theme each doc lands in at each `K`.
"""),
    code(r"""
multi = {"headline": [d[:60] for d in docs]}
for K in CUT_LEVELS:
    _, cent_k, _ = build_supers(K)
    Wk = softmax((normalize(embeddings) @ cent_k.T) / TEMP, axis=1)
    multi[f"top_super@K{K}"] = Wk.argmax(1)
    multi[f"conf@K{K}"] = Wk.max(1).round(2)
pd.DataFrame(multi).head(10)
"""),
    md(r"""
## 7. Save the soft assignments
"""),
    code(r"""
OUT = PROJECT_ROOT / "data" / "processed" / f"hier_soft_{PER_YEAR}_per_year_K{PRIMARY_K}.parquet"
soft_df = pd.concat([
    corpus[["Headline", "CaptureTime", "year"]].reset_index(drop=True),
    pd.Series(W.argmax(1), name=f"super_{PRIMARY_K}"),
    pd.DataFrame(W, columns=[f"super_{s}" for s in range(PRIMARY_K)]),
], axis=1)
soft_df.to_parquet(OUT, index=False)
print(f"wrote {soft_df.shape} -> {OUT.relative_to(PROJECT_ROOT)}")

# Export the FULL hover-rich dendrogram as standalone HTML (viewable without Jupyter).
# Height scales with #topics so every leaf/merge is reachable; hover any node for its name.
fig_full = topic_model.visualize_hierarchy(
    hierarchical_topics=hier_df, use_ctfidf=False, linkage_function=topic_linkage,
    width=1200, height=max(600, 14 * n_topics),
)
out_dir = PROJECT_ROOT / "notebooks" / "output"
out_dir.mkdir(parents=True, exist_ok=True)

# Pretty primary view: the nested theme icicle (compact, zoomable).
out_ice = out_dir / "bertopic_theme_icicle.html"
fig_ice.write_html(str(out_ice), include_plotlyjs="cdn")
print("icicle ->", out_ice.relative_to(PROJECT_ROOT))

# Full classic dendrogram too (tall; hover any node for its name).
out_html = out_dir / "bertopic_hierarchy.html"
fig_full.write_html(str(out_html), include_plotlyjs="cdn")
print(f"full dendrogram ({n_topics} leaves, {fig_full.layout.height}px) -> {out_html.relative_to(PROJECT_ROOT)}")
"""),
    md(r"""
## Notes

- **Cheap because the hierarchy is over ~1,400 topics, not 100k docs.** A doc-level
  dendrogram would need an O(n²) ≈ 80 GB distance matrix — avoid it.
- **Rows are a softmax simplex** (sum to 1), unlike `1.1`'s HDBSCAN memberships. Tune
  `TEMP` for sharper/softer assignments; tune `CUT_LEVELS`/`PRIMARY_K` for theme breadth.
- **Output floor:** the weight matrix is `n×K` floats — fine here (small `K`, float32).
  For the full `n×1439` resolution, store float32 or keep only top-k per row (sparse).
- **Outliers** (`hard_topic == -1`) still receive weights; a low `max` weight row means
  "no dominant super-theme" — a useful filter.
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
