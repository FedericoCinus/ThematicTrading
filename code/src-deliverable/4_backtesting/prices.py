"""Stage 5's only source — adjusted closes, one cached file per ticker.

    load(tickers) -> DataFrame [date × ticker]

The only fetched input in the pipeline, hence the only one that can change under a finished run.
The cache is what makes a run repeatable.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import config
from importlib import import_module

progress = import_module("progress")

# ======================================================================================
# THE SOURCE — where a price comes from
#
#   IN     tickers                                 symbols, Yahoo spelling
#   OUT    DataFrame [date × ticker]               one column per ticker THAT HAS PRICES
#
#   source   Yahoo Finance, through yfinance        auto_adjust=True, period="max"
#            adjusted closes — splits and dividends folded in, so no corporate action
#            shows up as a crash
#   cache    data/raw/prices/{TICKER}.parquet       [date, close] · refetched after max_age_days
#   needs    nothing — no key, no user agent
#
#   unknown ticker   cached EMPTY, not left missing, so delisted names, renamings and non-US
#                    lines are not re-queried every run
#                    → absent from the columns → dropped by the backtest → survivorship bias
# ======================================================================================

CACHE = config.PRICES


def download(tickers: list[str], *, force: bool = False, max_age_days: float = 7,
             chunk: int = 40, report: bool = True) -> None:
    """Fill the cache for these tickers, skipping the ones already fresh.

    INPUT   tickers        symbols, Yahoo spelling
            force          re-fetch even what is fresh
            max_age_days   a cached file older than this is refetched
            chunk          how many symbols per Yahoo request
            report         print progress
    OUTPUT  nothing — data/raw/prices/{TICKER}.parquet is written, empty when Yahoo has nothing
    """
    import yfinance as yf

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [t for t in dict.fromkeys(tickers) if force or _stale(_path(t), max_age_days)]
    if not todo:
        return
    if report:
        print(f"   prices: fetching {len(todo)} of {len(set(tickers))} tickers")

    batches = list(range(0, len(todo), chunk))
    for start in progress.each(batches, "downloading prices", on=report, unit="batch"):
        batch = todo[start:start + chunk]
        raw = yf.download(batch, period="max", auto_adjust=True, progress=False,
                          threads=True, group_by="column")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            close.columns = batch
        for ticker in batch:
            series = close[ticker].dropna() if ticker in close.columns else pd.Series(dtype=float)
            pd.DataFrame({"date": pd.to_datetime(series.index).tz_localize(None),
                          "close": series.to_numpy()}).to_parquet(_path(ticker), index=False)
        if report:
            got = sum(1 for t in batch if len(pd.read_parquet(_path(t))))
            print(f"      {start + len(batch):>4}/{len(todo)}   {got} of {len(batch)} had data")


def load(tickers: list[str], *, fetch: bool = True, max_age_days: float = 7,
         report: bool = True) -> pd.DataFrame:
    """The price matrix the backtest works on.

    INPUT   tickers        symbols to load
            fetch          download what is missing or stale first; False reads the cache only
            max_age_days   passed to `download`
            report         print progress
    OUTPUT  DataFrame indexed by date, one column per ticker that HAS prices, sorted by date.
            A ticker Yahoo knows nothing about is simply absent — check the columns, never
            assume a requested symbol is there.

    pandas and not polars, alone in this package: what follows is matrix arithmetic on a date
    index — ffill, cummax, pct_change — and the stage boundary either side stays polars.
    """
    tickers = list(dict.fromkeys(tickers))
    if fetch:
        download(tickers, max_age_days=max_age_days, report=report)
    columns = {}
    for ticker in tickers:
        path = _path(ticker)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if len(frame):
            columns[ticker] = frame.set_index("date")["close"]
    return pd.DataFrame(columns).sort_index() if columns else pd.DataFrame()


def _path(ticker: str) -> Path:
    return CACHE / f"{ticker.upper()}.parquet"


def _stale(path: Path, max_age_days: float) -> bool:
    return not path.exists() or (time.time() - path.stat().st_mtime) / 86400 > max_age_days
