"""Sample the real headlines used for the example graph of notebook 0.45.

    python scripts/sample_045.py        # -> data/processed/news_045_sample.jsonl

Two groups of 2023 Bloomberg headlines, picked with keywords only to get a readable example: artificial
intelligence and obesity drugs. In real theme detection no theme name is used; here the keywords just
make the demo show two groups forming. Per group: 16 random headlines, plus 3 with a deal cue and 2 with
a supply cue; two more are chosen by hand (EXTRA) so that every edge type and modality appears.
Flash headlines ("*...") and very short ones are skipped. Reads only the 2023 row group, two columns.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parent.parent
CORPUS = CODE / "data" / "processed" / "news_corpus.parquet"
OUT = CODE / "data" / "processed" / "news_045_sample.jsonl"
SEED = 45
GROUPS = {
    "intelligenza artificiale": r"\bAI\b|Artificial Intelligence|ChatGPT|Generative|OpenAI",
    "farmaci per l'obesità": r"Weight[- ]Loss|Obesity|Ozempic|Wegovy|Mounjaro|GLP-1|Zepbound",
}
CUES = {"deal": (r"\b(to Buy|Buys|to Acquire|Acquires|Stake|Invests?|Investment)\b", 3),
        "supply": (r"\b(Supply|Supplies|Order|Contract)\b", 2)}
N_RANDOM = 16
# chosen by hand after a keyword search, to show the last edge type and the last modality
EXTRA = [("Samsung Signs Deal With Nvidia to Supply HBM3: Daily", "intelligenza artificiale", "scelta: SUPPLIES_TO"),
         ("Adobe and Figma Terminate $20 Billion Deal After Regulator Clash", "altro", "scelta: modalità negato")]


def main():
    f = pq.ParquetFile(CORPUS)
    groups = [i for i in range(f.metadata.num_row_groups) if f.metadata.row_group(i).column(1).statistics.min.year == 2023]
    df = f.read_row_groups(groups, columns=["Headline", "date"]).to_pandas()
    df = df[(df["date"].dt.year == 2023) & ~df["Headline"].str.startswith("*") & (df["Headline"].str.split().str.len() >= 6)]
    df = df.drop_duplicates("Headline").reset_index(drop=True)
    rng = random.Random(SEED)
    rows, taken = [], set()
    for group, pattern in GROUPS.items():
        pool = df.index[df["Headline"].str.contains(pattern, regex=True)].tolist()
        picks = [(i, "a caso") for i in rng.sample(pool, N_RANDOM)]
        taken |= {i for i, _ in picks}
        for cue, (cpat, n) in CUES.items():
            sub = [i for i in pool if i not in taken and re.search(cpat, df.at[i, "Headline"])]
            for i in rng.sample(sub, min(n, len(sub))):
                picks.append((i, cue)); taken.add(i)
        for i, how in picks:
            rows.append(dict(group=group, how=how, date=str(df.at[i, "date"].date()), headline=df.at[i, "Headline"]))
    for headline, group, how in EXTRA:
        i = df.index[df["Headline"] == headline][0]
        rows.append(dict(group=group, how=how, date=str(df.at[i, "date"].date()), headline=headline))
    rows.sort(key=lambda r: r["date"])
    for k, r in enumerate(rows):
        r["id"] = f"n{k:02d}"
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "headlines ->", OUT.relative_to(CODE))


if __name__ == "__main__":
    main()
