"""Sample the 50 real 2024 headlines of notebook 0.44: 25 at random, 25 by keyword coverage.

    python scripts/sample_044.py

Reads only the 2024 row group of news_corpus.parquet (two columns), drops exact duplicates and the
150 headlines of 0.42, then draws with seed 44. The coverage half picks, per predicate, random
headlines matching a keyword pattern fixed here before any candidate was read; nothing is replaced
after being seen, and no model takes part in the choice. Writes gold/0.44/real_50.jsonl + manifest.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parent.parent
CORPUS = CODE / "data" / "processed" / "news_corpus.parquet"
OUT = CODE / "notebooks" / "gold" / "0.44"
SEED, N_RANDOM = 44, 25

# predicate -> (pattern, how many). Patterns are cues for the predicate, not proof of it.
COVERAGE = {
    "ACTIVE_IN": (r"\b(Maker|Startup|Start-Up|Miner|Producer|Developer|Operator)\b", 4),
    "OFFERS": (r"\b(Launches|Unveils|Introduces|Rolls Out|Debuts)\b", 4),
    "INVESTS_IN": (r"\b(Invests?|Stake Rises|Takes Stake|Minority Stake)\b", 3),
    "ACQUIRES": (r"\b(to Buy|to Acquire|Acquires|Takeover)\b", 3),
    "PARTNERS_WITH": (r"\b(Partners? With|Ties? Up|Joint Venture|Alliance|Teams Up)\b", 4),
    "SUPPLIES_TO": (r"\b(to Supply|Supplies|Supply Deal|Supply Contract|Wins .*(Order|Contract))\b", 4),
    "IMPACTS": (r"\b(Lifts|Boosts|Hurts|Weighs on|Hits)\b|\bon [A-Z][\w-]* (Demand|Prices?|Costs?|Tariffs?)\b", 3),
}


def key(h):
    return re.sub(r"\s+", " ", str(h)).strip().lower()


def main():
    f = pq.ParquetFile(CORPUS)
    groups = [i for i in range(f.metadata.num_row_groups)
              if f.metadata.row_group(i).column(1).statistics.min.year == 2024]
    df = f.read_row_groups(groups, columns=["Headline", "date"]).to_pandas()
    df = df[df["date"].dt.year == 2024].dropna(subset=["Headline"])
    df["key"] = df["Headline"].map(key)
    df = df.sort_values("date").drop_duplicates("key")          # earliest publication of each wording
    seen_042 = {key(x["headline"]) for x in json.loads((CODE / "notebooks/gold/candidates_2024.json").read_text())}
    df = df[~df["key"].isin(seen_042)].reset_index(drop=True)

    rng = random.Random(SEED)
    picked = []
    for i in rng.sample(range(len(df)), N_RANDOM):
        picked.append((i, "random", None))
    taken = {i for i, _, _ in picked}
    for pred, (pattern, n) in COVERAGE.items():
        pool = [i for i in df.index[df["Headline"].str.contains(pattern, regex=True)] if i not in taken]
        for i in rng.sample(pool, n):
            picked.append((i, "coverage", pred)); taken.add(i)

    rows = []
    for k, (i, stratum, pred) in enumerate(picked):
        h, d = df.at[i, "Headline"], df.at[i, "date"]
        rows.append(dict(id=f"r{k:02d}", stratum=stratum, cue=pred, date=str(d.date()),
                         published=str(d), headline=h, sha1=hashlib.sha1(h.encode()).hexdigest()[:12]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_50.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = dict(corpus=str(CORPUS.relative_to(CODE)), corpus_rows_2024_distinct_after_exclusion=len(df),
                    seed=SEED, n_random=N_RANDOM, coverage={p: dict(pattern=pt, n=n) for p, (pt, n) in COVERAGE.items()},
                    excluded="150 headlines of 0.42 (gold/candidates_2024.json)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(rows), "headlines;", len(df), "distinct 2024 candidates")


if __name__ == "__main__":
    main()
