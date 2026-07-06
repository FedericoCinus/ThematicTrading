"""Preprocess the raw Bloomberg feed capture into ONE clean corpus file.

Dating rule (decided in notebook 0.25 — raw-feed replay artifact):

  1. A headline's date is the timestamp of its earliest *publication* message
     (``ADD_1STPASS`` or ``ADD_STORY``).
  2. ``UPDATE_ATTRIBUTE`` messages (tag edits that re-transmit old stories, 72% of
     the feed) are ignored — they carry edit times, not publication times, and bulk
     re-tag days otherwise inject fake news bursts (proof in 0.25).

File layout produced:

  data/raw/raw_news_{year}.csv.xz            raw feed capture (input, immutable)
  notebooks/output/tmp/terms_{year}.parquet  per-year build cache (disposable)
  notebooks/output/news_corpus.parquet       THE corpus: all years, cross-year deduped
  notebooks/output/news_corpus_manifest.json build record (rule version, rows, times)

Analyses must read ONLY news_corpus.parquet; output/tmp/ exists so that single years
can be rebuilt incrementally and is safe to delete (regenerable from raw).

Per-year term extraction (wires, prefix stripping, >=4-word filter, unigram/bigram
terms) is identical to the legacy 0.24 builder, so the schema is unchanged.
Downstream notebooks should read ONLY news_corpus.parquet.

Usage:
    python scripts/preprocess_news.py              # build whatever is missing + merge
    python scripts/preprocess_news.py --force      # rebuild everything from raw
    python scripts/preprocess_news.py 2013         # one year (no merge)
"""
import argparse
import json
import lzma
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]                  # code/
sys.path.insert(0, str(_ROOT))
from theme_detect import WIRES, _extract, _strip_prefix      # noqa: E402  (identical extraction)

RAW_DIR = _ROOT / "data" / "raw"
OUT_DIR = _ROOT / "notebooks" / "output"
TMP_DIR = OUT_DIR / "tmp"                                    # per-year build caches (disposable)
CORPUS = OUT_DIR / "news_corpus.parquet"
MANIFEST = OUT_DIR / "news_corpus_manifest.json"
RULE = "add-event-v1"                                        # bump when the dating rule changes
ADD_EVENTS = ["ADD_1STPASS", "ADD_STORY"]                    # publication messages (0.25 §3)


def _load_manifest() -> dict:
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        if m.get("rule") == RULE:
            return m
    return {"rule": RULE, "years": {}, "corpus": None}


def _save_manifest(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=2))


def raw_years() -> list[int]:
    return sorted(int(p.stem.split("_")[-1].split(".")[0]) for p in RAW_DIR.glob("raw_news_*.csv.xz"))


def build_year(yr: int, force: bool = False) -> Path:
    """Build tmp/terms_{yr}.parquet from raw with ADD-event dating; record it in the manifest."""
    TMP_DIR.mkdir(exist_ok=True)
    out = TMP_DIR / f"terms_{yr}.parquet"
    m = _load_manifest()
    if not force and str(yr) in m["years"] and out.exists():
        print(f"{yr}: certified {RULE} build exists — skipping")
        return out
    raw = RAW_DIR / f"raw_news_{yr}.csv.xz"
    if not raw.exists():
        raise FileNotFoundError(f"no raw file for {yr}: {raw}")

    t0 = time.time()
    with lzma.open(raw, "rb") as f:
        d = (pl.scan_csv(f, infer_schema_length=10_000)
               .select(["Headline", "CaptureTime", "WireName", "Event"])
               .filter(pl.col("WireName").is_in(WIRES) & pl.col("Headline").is_not_null())
               .collect())
    n_msgs, n_headlines = len(d), d["Headline"].n_unique()

    # publications only: date := earliest ADD message per headline
    d = (d.filter(pl.col("Event").is_in(ADD_EVENTS))
           .with_columns(pl.col("CaptureTime").str.to_datetime(time_zone="UTC", strict=False))
           .drop_nulls("CaptureTime")
           .sort("CaptureTime")
           .unique(subset="Headline", keep="first"))
    n_added = len(d)

    df = d.to_pandas()
    df["date"] = pd.to_datetime(df["CaptureTime"]).dt.tz_localize(None)
    df = df[df.date.dt.year == yr]                           # ADDs from other years belong to those files
    df["Headline"] = df["Headline"].map(_strip_prefix)
    df = df[df.Headline.str.split().map(len) >= 4]
    df = df.sort_values("date").reset_index(drop=True)
    df["terms"] = [_extract(h) for h in df["Headline"]]
    df[["Headline", "date", "terms"]].to_parquet(out, index=False)

    dropped = n_headlines - n_added
    m = _load_manifest()                                     # re-read: parallel-safe enough for our use
    m["years"][str(yr)] = {"rows": len(df), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    m["corpus"] = None                                       # corpus is now stale
    _save_manifest(m)
    print(f"{yr}: {n_msgs:,} messages -> {n_added:,} published headlines "
          f"({dropped / n_headlines:.1%} never seen as ADD, dropped) "
          f"-> {len(df):,} rows -> {out.name}  [{time.time() - t0:.0f}s]")
    return out


def merge() -> Path:
    """Concat all certified years (ascending) into news_corpus.parquet, cross-year deduped."""
    m = _load_manifest()
    years = raw_years()
    missing = [y for y in years if str(y) not in m["years"]]
    if missing:
        raise RuntimeError(f"cannot merge — years not yet built with {RULE}: {missing}")

    writer, seen, total = None, set(), 0
    for yr in years:
        df = pd.read_parquet(TMP_DIR / f"terms_{yr}.parquet")
        hs = df.Headline.map(hash)                           # first publication wins across years
        keep = ~hs.isin(seen)
        seen.update(hs[keep])
        df = df[keep].sort_values("date")
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(CORPUS, tbl.schema)
        writer.write_table(tbl)
        total += len(df)
        print(f"  {yr}: +{len(df):,} rows ({(~keep).sum():,} cross-year duplicates dropped)")
    writer.close()
    m = _load_manifest()
    m["corpus"] = {"rows": total, "years": years, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_manifest(m)
    print(f"corpus: {total:,} rows, {years[0]}-{years[-1]} -> {CORPUS.name}")
    return CORPUS


def ensure(force: bool = False) -> Path:
    """Idempotent entry point: build any missing/uncertified year, merge if stale, return corpus path."""
    for yr in raw_years():
        build_year(yr, force=force)
    if _load_manifest()["corpus"] is None or not CORPUS.exists():
        merge()
    else:
        print(f"corpus up to date: {CORPUS.name}")
    return CORPUS


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("years", nargs="*", type=int,
                    help="build specific year(s) only; default: ensure everything + merge")
    ap.add_argument("--force", action="store_true", help="rebuild even if certified")
    a = ap.parse_args()
    if a.years:
        for yr in (a.years if len(a.years) == 1 else range(min(a.years), max(a.years) + 1)):
            build_year(yr, force=a.force)
    else:
        ensure(force=a.force)


if __name__ == "__main__":
    main()
