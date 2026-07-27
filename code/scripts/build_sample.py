"""Build a small Bloomberg headline sample for BERTopic play — fast & low memory.

The raw news live as one ``.xz``-compressed CSV per year under
``data/raw/bloomberg/`` (a symlink to the Theme-Detection project — ~1 GB each,
~19 M headlines/year, never copied here).

``.xz`` has no random access and the files are time-sorted, so a *fair* sample across
all 12 months would mean decompressing the entire year (slow, and easy to blow past a
small RAM budget). We don't need that for a first playground: instead we decompress
only a **prefix chunk** of each year. ``xz`` stops as soon as ``head`` has taken its
lines, so this is sub-second and holds only the chunk (~80 MB) in memory — one year at
a time. We then randomly sample ``--per-year`` headlines from the chunk.

Tradeoff (accepted on purpose): the sample is drawn from the *start* of each year
(roughly the first few weeks), not spread evenly across it. Good enough to play with
BERTopic across 10 years; revisit if you need a calendar-fair sample.

Usage:
    uv run python scripts/build_sample.py                          # 100/year, all years
    uv run python scripts/build_sample.py --per-year 200 --years 2023 2024 2025
"""

from __future__ import annotations

import argparse
import io
import subprocess
import time
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bloomberg"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

KEEP_COLS = ["Headline", "CaptureTime", "WireName", "DerivedTopicsId"]
PREFIX_ROWS = 500_000  # decompress only this many lines/year (~first ~10 days, keeps RAM ~1 GB)


def read_prefix(path: Path, prefix_rows: int) -> bytes:
    """Decompress only the first ``prefix_rows`` lines of a .xz file (xz stops early)."""
    xz = subprocess.Popen(["xz", "-dc", str(path)], stdout=subprocess.PIPE)
    head = subprocess.Popen(
        ["head", "-n", str(prefix_rows)], stdin=xz.stdout, stdout=subprocess.PIPE
    )
    xz.stdout.close()  # head closing the pipe gives xz a SIGPIPE → it stops decompressing
    out, _ = head.communicate()
    xz.wait()
    return out


def sample_year(path: Path, per_year: int, prefix_rows: int, seed: int) -> pl.DataFrame:
    raw = read_prefix(path, prefix_rows)
    df = (
        pl.read_csv(io.BytesIO(raw), columns=KEEP_COLS, infer_schema_length=0)
        .filter(pl.col("Headline").is_not_null() & (pl.col("Headline").str.len_chars() > 0))
        .unique(subset="Headline", keep="first")  # wires repost the same story
    )
    n = min(per_year, df.height)
    return df.sample(n=n, seed=seed, shuffle=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-year", type=int, default=100, help="headlines sampled per year")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2026)))
    ap.add_argument("--prefix-rows", type=int, default=PREFIX_ROWS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_path = args.out or OUT_DIR / f"sample_{args.per_year}_per_year.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in args.years:
        path = RAW_DIR / f"raw_news_{year}.csv.xz"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        t = time.time()
        df = sample_year(path, args.per_year, args.prefix_rows, args.seed + year).with_columns(
            pl.lit(year).alias("year")
        )
        frames.append(df)
        print(f"[{year}] kept {df.height:3d} headlines in {time.time() - t:.1f}s")

    corpus = (
        pl.concat(frames)
        .with_columns(pl.col("CaptureTime").str.to_datetime(time_zone="UTC", strict=False))
        .sort("CaptureTime")
    )
    corpus.write_parquet(out_path)

    try:
        shown = out_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out_path
    print(f"\nWrote {corpus.height:,} rows → {shown}")
    print(corpus.head())


if __name__ == "__main__":
    main()
