# Data setup — what to get, in what order

Audited 2026-09-08. Written for someone cloning this repo onto a machine with none of
Alberto's local data.

**The repo contains no data at all.** `.gitignore` excludes `data/`, `output/`,
`notebooks/output/`, `*.parquet`, `*.png`, `*.pdf` and `docs/`. A fresh clone gives you
the Python modules, the scripts and the notebooks — nothing else. Everything below has to
be fetched, rebuilt, or handed to you.

---

## 0 · Environment

```bash
cd code
uv sync                     # Python >= 3.12, creates code/.venv
echo 'SEC_USER_AGENT=Your Name you@example.com' > code/.env
```

`SEC_USER_AGENT` is not optional: SEC EDGAR returns 403 without an identifying
User-Agent, and both the basket build and the point-in-time market cap depend on it.

Notebooks must run on `code/.venv`, not the system/Anaconda Python.

---

## 1 · The four tiers of input

### Tier A — regenerates itself from the network. Nothing to do.

| what | where it lands | size here | source |
|---|---|---|---|
| daily adjusted prices | `code/data/raw/prices/*.parquet` | 53 MB, 691 tickers | yfinance, on first call |
| 13-week T-bill (cash rate) | `code/data/raw/prices/_irx.parquet` | small | yfinance `^IRX` |
| OHLCV + company info | `code/data/raw/ohlcv/`, `_info.json` | 5.6 MB, 31 tickers | yfinance |
| EDGAR filings / full-text / index | `code/data/raw/edgar/` | 209 MB | SEC, needs `SEC_USER_AGENT` |
| point-in-time share counts | `code/data/raw/prices/_market_caps.json` | small | yfinance, falls back to SEC XBRL |

These are all caches. Delete any of them and the code refetches. Expect the first EDGAR
build to take a while and to hit transient 500s — the module retries.

### Tier B — small, must be handed over (or rebuilt via Tier C)

**60 KB total.** These five files are what stand between a fresh clone and a working
backtest:

```
code/data/processed/theme_basket_genai_2023-01-17.parquet            17.3 KB
code/data/processed/theme_basket_genai_2023-01-17_manifest.json       1.3 KB
code/data/processed/theme_basket_quantum_2022-09-30.parquet          19.5 KB
code/data/processed/theme_basket_quantum_2022-09-30_manifest.json     1.4 KB
code/data/processed/theme_exit_weekly_hits.parquet                    8.2 KB
```

The baskets are rebuildable from EDGAR (Tier A) given the vocabularies, which are stored
inside the manifests. The weekly-hits file is **not** rebuildable without Tier C.

### Tier C — the news corpus. Not reproducible. Get a copy.

```
code/notebooks/output/news_corpus.parquet     1.8 GB, 22,626,656 rows, 2010-01-01 .. 2025-12-31
```

Columns: `Headline`, `date`, `terms` (list of extracted n-gram terms).

This is the input to theme discovery (box 1–3), the hotness/entry signal (box 4) and the
news-based exit signal. **It can no longer be rebuilt, on any machine, including
Alberto's:**

- `code/scripts/preprocess_news.py` builds it from `code/data/raw/raw_news_{year}.csv.xz`.
- Those raw files are gone. They lived in a sibling project via
  `code/data/raw/bloomberg -> ~/Documents/Intesa/Theme-Detection/data/raw/bloomberg`,
  **and that symlink is broken** — the target directory no longer exists.
- `code/data/processed/tmp/terms_*.parquet` (the per-year cache) is also gone.

So the 1.8 GB file is the only surviving copy of this data. Treat it as an irreplaceable
artifact: copy it, don't move it, and back it up.

Note also that its location disagrees with the documented layout — notebook 0.26 specifies
`data/processed/news_corpus.parquet`, and every consumer actually reads
`code/notebooks/output/news_corpus.parquet`.

### Tier D — hand-supplied, no reproduction path in the repo

```
code/data/raw/tickers/ticker_list_US.csv      141 KB, 8,944 rows
```

A Bloomberg-style export, one line per name (`A US Equity`). Nothing in the repo generates
it. It defines the universe the basket builder searches, so it must be supplied.

---

## 2 · What you can run at each level

| you have | you can run |
|---|---|
| Tier A only | nothing theme-specific — no baskets to test |
| **A + B + D** | **0.38 (monthly backtest), 0.39 (exit signal), `deck11/make_figs.py`** |
| A + B + C + D | everything, including new themes and new date ranges |

**Verified end to end on 2026-09-08:** with the corpus renamed out of the way,
`docs/Technical meetings/deck11/make_figs.py` rebuilt all four of its artifacts
(`level.png`, `curves.png`, `numbers.tex`, `names.tex`) with no error. The news signal came
entirely from the 8 KB `theme_exit_weekly_hits.parquet` cache.

**The one catch:** that cache is keyed by *exact date range*. The stored key is
`{theme}|2018-01-01|2025-12-31`. Ask `theme_exit.weekly_hits` for any other range and it
falls through to the corpus and raises `FileNotFoundError`. So a colleague can reproduce
today's results exactly, but cannot extend the window or add a theme without Tier C.

---

## 3 · Bootstrap order

```
1.  uv sync                                   # code/.venv
2.  write code/.env with SEC_USER_AGENT
3.  drop in ticker_list_US.csv                 (Tier D)
4.  drop in the 5-file processed bundle        (Tier B, 60 KB)
5.  run code/notebooks/0.38-monthly-backtest.ipynb
        -> pulls Tier A from the network on demand (slow the first time)
6.  run code/notebooks/0.39-exit-signal.ipynb
7.  cd "docs/Technical meetings/deck11" && ../../../code/.venv/bin/python make_figs.py
        && pdflatex deck.tex
```

Steps 3 and 4 are the whole blocker. Everything else either ships in git or downloads
itself.

---

## 4 · Gotchas found in this audit

- **`docs/` is gitignored**, so no deck — source or PDF — is under version control. Two
  frames have already been lost this way with no history to recover them.
- **`theme_exit.weekly_hits(corpus=...)` binds its default at import time.** Reassigning
  `theme_exit.CORPUS` after import does nothing; pass `corpus=` explicitly.
- **`weekly_hits` writes to the shared cache by default.** Calling it with an exploratory
  date range silently adds a column to `theme_exit_weekly_hits.parquet`. Pass `cache=False`
  when probing.
- **A market cap of `None` means "unknown", not "fails the filter".** `market_cap_asof`
  returns `None` when a ticker cannot be resolved; 0.38/0.39 drop those names and print
  them. Don't let that quietly become "excluded for being small".
- **Yahoo back-adjusts price *and* volume for later splits.** `theme_features` corrects
  both via `split_factor`; anything new that touches raw Yahoo data must do the same.

---

## 5 · Recommendation

Commit the Tier B bundle (60 KB) and `ticker_list_US.csv` (141 KB) — 200 KB total against
a `.gitignore` that currently blocks them. That turns "ask Alberto for files" into `git
clone`, and it is the single highest-value change available here.

Then put `docs/` under version control, and get a second copy of the 1.8 GB corpus onto
different hardware.
