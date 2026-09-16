"""The pipeline, end to end: news in, a portfolio and its performance out.

Run it with `run(asof)`. Everything else in here is the map.
"""
from __future__ import annotations

import importlib
import math
import os

import polars as pl

import artifacts
import config
import progress

# ======================================================================================
# WHERE THE DATA COMES FROM, AND HOW IT IS REACHED
#
#   news feed        data/raw/raw_news_{year}.csv.xz          on disk · 16 xz files, ~10 GB
#                    A Bloomberg feed capture, one row per MESSAGE. Immutable: it is the only
#                    thing here that cannot be regenerated, and nothing writes to it.
#
#   company names    www.sec.gov/files/company_tickers.json   HTTP · needs SEC_USER_AGENT
#                    Ticker, name and CIK of every current SEC registrant (~10,400). Cached in
#                    data/raw/edgar/. Current registrants only — a company delisted before today
#                    is absent, which is survivorship bias in every basket built from it.
#
#   SEC filings      efts.sec.gov full-text search            HTTP · needs SEC_USER_AGENT
#                    Which filings contain a phrase, by filing date. Not used by the stages below
#                    yet; it is the second mapping variant, and it answers a question names cannot.
#
#   prices           Yahoo, through yfinance                  HTTP · cached in data/raw/prices/
#
#   language model   OPENAI_BASE_URL                          HTTP · needs OPENAI_API_KEY
#                    Two uses, both in detection: removing clusters whose headlines show they are
#                    not a theme, and keeping only the vocabulary words that name something.
#
# The SEC refuses anonymous requests, so SEC_USER_AGENT must carry a real contact:
#   SEC_USER_AGENT=Name Surname you@example.com     in code/.env
# ======================================================================================

# ======================================================================================
# THE PIPELINE
#
#   data/raw/raw_news_*.csv.xz
#        │  0_preprocessing/first_publication.py      build_corpus()
#        │     three filters on three columns: keep Bloomberg's wires, keep the messages that
#        │     announce a publication, keep one row per headline ever
#        ▼
#   CORPUS  [Headline, date]                          22.8 M rows · data/processed/
#        │     the words as the wire published them — no stripping, no tokenizing, because
#        │     each method needs something different from a headline
#        │
#        │  1_detectors/graph_anomaly/detect.py       detect(corpus, asof)
#        │     four gates: a term nobody used before · co-occurring widely · lasting, peaking
#        │     and bridging · and a model removing what is plainly not a theme
#        ▼
#   THEMES  [theme, week, vocab, vocab_wide]          a theme is a set of words — two lists,
#        │                                            `vocab` without the words too common to count
#        │                                            a theme by, `vocab_wide` with them
#        │
#        │  2_monitors/monitor.py                     monitor(corpus, themes, asof)
#        │     weekly headline counts on a zero-filled grid, then two halves that do not
#        │     share a measure: entry/ enters on ACCELERATION, exit/ leaves on LEVEL
#        ▼
#   SIGNAL  [theme, week, score, position]            the trading rule
#        │     entry_dates(signal) gives each theme its as-of: nothing downstream may read
#        │     anything published after it
#        │
#        │  3_baskets/uniform_method/basket.py                      weights(themes, universe)
#        │     a model turns each of `vocab_wide`'s brands into the company that owns it,
#        │     and that company is looked up in the SEC register — one holding per cik
#        ▼
#   WEIGHTS [theme, ticker, weight]                   the portfolio
#        │
#        │  4_backtesting/rebalance.py                backtest(weights, signal, prices)
#        │     hold while `position` is True, flat otherwise, trading the first price
#        │     date after the week boundary — run(asof, until=...) only
#        ▼
#   PERFORMANCE  [theme, total_return, ann_return, ann_vol, sharpe, max_dd, bench_*]
#   CURVES       [theme, date, level, bench]
#
#   Every stage returns its canonical frame and a diagnostics frame on the same index. Which
#   method runs at each stage, and how it is tuned, is config.yaml — and the three method names
#   together are the run id that names every file a run writes.
# ======================================================================================

# ======================================================================================
# ONE THEME ALL THE WAY THROUGH — genAI, run at asof = 2023-01-02
# ======================================================================================
#   Every number below is measured, and `tests/test_pipeline.py` fails when one stops holding.
#
#   DETECT       five weeks of headlines, twenty-four months of baseline behind them
#        │
#        │   gate 1  `chatgpt` appears, and never appeared in the baseline    → candidate
#        │   gate 2  it shares headlines with enough different words          → caught
#        │   gate 3  it lasts, it peaks, and its partners barely know
#        │           each other — it bridges separate worlds                  → promoted
#        │   gate 4  a model reads 8 of its headlines: not one company
#        │           announcing its own news                                  → kept
#        ▼
#   VOCABULARY   a second pass, over ALL history up to the as-of date, not over the week:
#        │       what does `chatgpt` share a headline with?
#        │
#        │       30 raw partners      olson · parmy olson · way · peril · asia league ·
#        │            │               venture capital · google · iphone · reuters · openai · …
#        │            │  ENTITIES, a model: keep only what NAMES something
#        │            ▼               `parmy olson` was the columnist's byline, `reuters` the wire
#        │       5 words             chatgpt · iphone · google ·
#        │            │              microsoft-backed chatgpt · openai
#        │            │  max_doc_freq = 0.0001: drop what is in the news anyway
#        │            ▼
#        ├──▶  vocab        chatgpt · microsoft-backed chatgpt · openai
#        └──▶  vocab_wide   the same, plus iphone · google
#
#   MONITOR      reads `vocab` — weekly headline counts, z-scored against the 12 weeks before
#        │
#        │        0  0  0  0  0  2  4  2  0  11        score 7.60   → IN on 2023-01-02
#        │
#        │       on `vocab_wide` the same eleven headlines score −0.39 and it never enters:
#        │       `google` and `iphone` carry 20-50 headlines a week of their own, and the
#        │       deviation the score divides by goes from about 1 to about 13
#        ▼
#   BASKET       reads `vocab_wide` — because only the big names can be bought
#        │
#        │       chatgpt                   → openai            → not listed
#        │       iphone                    → apple             → aapl
#        │       google                    → alphabet          → googl
#        │       microsoft-backed chatgpt  → openai, microsoft → msft
#        ▼
#   WEIGHTS      aapl · googl · msft, a third each
#
#   The two surprises are both real. OPENAI, the centre of the theme, is absent because it is not
#   listed. And MSFT does not arrive from the word `microsoft`, which is too common for either
#   list, but from a model reading a company out of the bigram `microsoft-backed chatgpt`.
# ======================================================================================

# ======================================================================================
# THE METHODS — what `method:` in config.yaml selects
#
#   stage     method          module                             what it is
#   ─────     ──────          ──────                             ──────────
#   detect    graph_anomaly   1_detectors.graph_anomaly.detect   a new word bridging worlds
#   monitor   —               2_monitors.monitor                 fixed: it counts, it does not decide
#   basket    uniform         3_baskets.uniform_method.basket    the theme's words vs COMPANY NAMES
#             filings         3_baskets.filings_method.basket    vs 10-K/10-Q TEXT — needs the SEC
#   backtest  rebalance       4_backtesting.rebalance            fixed, pure math
#
#   The monitor's two halves are chosen inside its own block and resolved by the stage itself,
#   because they live inside its folder: monitor.entry.method -> 2_monitors/entry/<method>.py
#                                        monitor.exit.method  -> 2_monitors/exit/<method>.py
#   Both names reach the run id, so `rolling_z+news_level` says which pair traded.
#
#   The short name is what the experiment file sets and what the run id carries; the long one is
#   where it lives. This table is the only place one becomes the other, so a method folder can be
#   renamed without touching the run ids already on disk.
# ======================================================================================
FIXED = {"monitor": "2_monitors.monitor"}       # stages with nothing to choose

METHODS = {
    "detect": {"graph_anomaly": "1_detectors.graph_anomaly.detect"},
    "basket": {"uniform": "3_baskets.uniform_method.basket", "filings": "3_baskets.filings_method.basket"},
    "backtest": {"rebalance": "4_backtesting.rebalance"},
}

NEEDED = {
    "OPENAI_API_KEY": "detection: the gate that removes non-themes, and the vocabulary filter\n                         baskets: translating a theme's brands into the companies that own them",
    "SEC_USER_AGENT": "baskets: the SEC refuses anonymous requests for the company list",
}


def check_environment(verbose: bool = True) -> bool:
    """What must be in place before a run, checked before anything slow starts.

    INPUT   verbose   print the report
    OUTPUT  True when the pipeline can run end to end

    Secrets are reported as present or missing, never printed.
    """
    config.load_env()
    ok = True

    if verbose:
        print("environment")
    for name, why in NEEDED.items():
        here = bool(os.environ.get(name))
        ok &= here
        if verbose:
            print(f"   {'OK ' if here else 'MISSING'}  {name:16} {why}")

    corpus = config.CORPUS.exists()
    years = len(list(config.FEED.glob("raw_news_*.csv.xz")))
    ok &= corpus or years > 0
    if verbose:
        print(f"   {'OK ' if corpus else 'absent'}  {'corpus':16} {config.CORPUS}")
        print(f"   {'OK ' if years else 'MISSING'}  {'raw captures':16} {years} years in {config.FEED}")
        print(f"   run id {config.run_id()}")
        if not ok:
            print("   -> add what is missing to code/.env; a run without it fails partway through")
    return bool(ok)


def run(asof: str, themes: pl.DataFrame | None = None, until: str | None = None,
        save: bool = True) -> dict:
    """The stages in order, for one as-of date.

    INPUT   asof     the day the pipeline pretends it is; detection and the vocabulary never read
                     past it, which is what keeps the whole chain point-in-time
            themes   skip detection and monitor these instead — for iterating on later stages
            until    run the monitor forward to this date and backtest what it says. Leave it out
                     and the run stops at the portfolio, which is what a live Monday morning wants.
            save     write every stage to data/processed/{stage}/{run_id}. False leaves no trace.
    OUTPUT  every frame produced, so any link of the chain can be inspected

    `until` is the only place the future is allowed in, and it is allowed because the monitor is
    causal: each week is scored against the weeks before it, so running to a later date replays
    decisions rather than revealing them. Detection and the basket still stop at `asof`.
    """
    if not check_environment(verbose=False):
        raise RuntimeError("environment incomplete — run check_environment() to see what is missing")

    detect, monitor, basket = (stage("detect"), stage("monitor"), stage("basket"))

    corpus = pl.scan_parquet(config.CORPUS)
    out = {}
    if themes is None:
        with progress.step("detect", detail=f"as of {asof}"):
            out["themes"], out["theme_diagnostics"], out["rejects"] = detect.detect(
                corpus, asof, **params("detect"))
    else:
        out["themes"] = themes

    if save:
        artifacts.save("themes", out["themes"], out.get("theme_diagnostics"),
                       asof=asof, **params("detect"))

    out["signal"], out["signal_diagnostics"] = monitor.monitor(
        corpus, out["themes"], until or asof, **monitorparams(out["themes"], asof, until))
    out["asof"] = monitor.entry_dates(out["signal"])
    if save:
        artifacts.save("signal", out["signal"], out["signal_diagnostics"],
                       asof=asof, until=until, **monitorparams(out["themes"], asof, until))

    with progress.step("basket", detail=config.params("basket")["method"]):
        out["weights"], out["weight_diagnostics"] = basket.weights(
            out["themes"], **params("basket"))
    if save:
        artifacts.save("weights", out["weights"], out["weight_diagnostics"],
                       asof=asof, **params("basket"))

    if until:
        prices = importlib.import_module("4_backtesting.prices")
        rebalance = importlib.import_module("4_backtesting.rebalance")
        tunables = params("backtest")
        symbols = sorted({t.upper() for t in out["weights"]["ticker"]} | {tunables["benchmark"]})
        with progress.step("backtest", detail=f"{len(symbols)} tickers"):
            quotes = prices.load(symbols, max_age_days=tunables.pop("price_max_age_days"))
            out["performance"], out["curves"] = rebalance.backtest(
                out["weights"], out["signal"], quotes, **tunables)
        if save:
            artifacts.save("backtest", out["performance"], out["curves"],
                           asof=asof, until=until, **tunables)
    return out


def stage(name: str):
    """The module configured for one stage.

    INPUT   name   detect · monitor · basket · backtest
    OUTPUT  the imported module, which exposes that stage's function

    Raises when config.yaml names a method the table does not know, rather than importing
    something with a similar name.
    """
    if name in FIXED:
        return importlib.import_module(FIXED[name])
    method = config.params(name).get("method")
    known = METHODS[name]
    if method not in known:
        raise KeyError(f"the experiment sets {name}.method = {method!r}; "
                       f"known methods are {sorted(known)}")
    return importlib.import_module(known[method])


def params(stage: str) -> dict:
    """The stage's tunables from config.yaml, without the name of the method itself."""
    return {k: v for k, v in config.params(stage).items() if k != "method"}


def monitorparams(themes: pl.DataFrame, asof: str, until: str | None) -> dict:
    """The monitor's tunables, with a window long enough to contain the themes it is given.

    INPUT   themes   the detected themes, for their emergence weeks
            asof     the detection date
            until    the date the monitor is run forward to, or None
    OUTPUT  the config values, with `lookback_months` widened when running forward

    `lookback_months` is measured back from the monitor's own end date, so running forward to
    `until` would walk the window past the theme's birth and score nothing. Widening it keeps the
    configured value meaning what it says: how much quiet history to hold a theme up against.
    """
    tunables = params("monitor")
    if until and len(themes):
        gap = (pl.Series([until]).str.to_datetime().item() - themes["week"].min()).days
        tunables["lookback_months"] += math.ceil(gap / 30.44)
    return tunables
