"""Price backtest for theme baskets: equal-weight buy-and-hold vs a benchmark.

Companion to `theme_basket.py`. Takes a basket parquet (or any ticker list),
buys the top-N names in equal weight at an information date, holds, and compares
the curve to SPY over a set of horizons.

Prices come from Yahoo (split/dividend adjusted closes) and are cached one
parquet per ticker under `data/raw/prices/`, so reruns are offline.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
PRICE_DIR = _ROOT / "data" / "raw" / "prices"
PROC_DIR = _ROOT / "data" / "processed"

BENCHMARK = "SPY"          # total-return proxy for the S&P 500 (adjusted closes)
MAX_AGE_DAYS = 7           # refresh a cached series after a week
HORIZONS = [1, 3, 6, 12, 24, 36]   # months


# ---------------------------------------------------------------- price cache
def _cache_path(ticker: str) -> Path:
    return PRICE_DIR / f"{ticker}.parquet"


def _is_stale(path: Path, max_age_days: float) -> bool:
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) / 86400 > max_age_days


def download_prices(tickers, *, force: bool = False, max_age_days: float = MAX_AGE_DAYS,
                    chunk: int = 40, verbose: bool = True) -> None:
    """Fetch full-history adjusted closes for `tickers` into the parquet cache.

    Tickers Yahoo has no data for (delisted, renamed, non-US listings) are cached
    as an empty frame so we do not re-query them on every run.
    """
    import yfinance as yf

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in dict.fromkeys(tickers) if force or _is_stale(_cache_path(t), max_age_days)]
    if not todo:
        return
    if verbose:
        print(f"downloading {len(todo)} tickers from Yahoo ...")

    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        raw = yf.download(batch, period="max", auto_adjust=True, progress=False,
                          threads=True, group_by="column")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            close.columns = batch
        for t in batch:
            s = close[t].dropna() if t in close.columns else pd.Series(dtype="float64")
            df = pd.DataFrame({"date": pd.to_datetime(s.index).tz_localize(None),
                               "close": s.to_numpy()}) if len(s) else \
                pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                              "close": pd.Series(dtype="float64")})
            df.to_parquet(_cache_path(t), index=False)
        if verbose:
            got = sum(1 for t in batch if _cache_path(t).stat().st_size and
                      len(pd.read_parquet(_cache_path(t))))
            print(f"  [{i + len(batch):>4}/{len(todo)}] {got}/{len(batch)} with data")


def load_prices(tickers, *, download: bool = True, force: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """Wide frame of adjusted closes (index=date, columns=ticker); empty series dropped."""
    tickers = list(dict.fromkeys(tickers))
    if download:
        download_prices(tickers, force=force, verbose=verbose)
    cols = {}
    for t in tickers:
        p = _cache_path(t)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if len(df):
            cols[t] = df.set_index("date")["close"]
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


# ---------------------------------------------------------------- portfolio math
def equal_weight_curve(prices: pd.DataFrame, tickers, start, end=None):
    """Equal-weight buy-and-hold curve normalised to 1.0 at the first trading day >= start.

    Returns (curve, held, dropped). A name is dropped when it has no price at t0
    (not listed yet / no Yahoo history); one that stops trading later is carried
    at its last close, which is the optimistic reading of a delisting.
    """
    start, end = pd.Timestamp(start), (pd.Timestamp(end) if end is not None else prices.index.max())
    cols = [t for t in tickers if t in prices.columns]
    px = prices.loc[start:end, cols].ffill()
    if px.empty:
        return pd.Series(dtype="float64"), [], list(tickers)
    held = [t for t in cols if pd.notna(px[t].iloc[0])]
    dropped = [t for t in tickers if t not in held]
    if not held:
        return pd.Series(dtype="float64"), [], dropped
    rel = px[held] / px[held].iloc[0]
    return rel.mean(axis=1), held, dropped


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def summarise(curve: pd.Series) -> dict:
    """Total return, annualised return/vol and max drawdown for one curve."""
    if curve.empty:
        return {}
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = float(curve.iloc[-1] - 1)
    ret = curve.pct_change().dropna()
    return {"total_return": total,
            "ann_return": (1 + total) ** (1 / years) - 1 if years > 0 else float("nan"),
            "ann_vol": float(ret.std() * (252 ** 0.5)),
            "max_drawdown": max_drawdown(curve),
            "years": years}


def excess_curve(prices: pd.DataFrame, tickers, start, *, benchmark: str = BENCHMARK,
                 months: float | None = None) -> pd.Series:
    """Cumulative excess of the equal-weight basket over the benchmark.

    Indexed by *months since entry* rather than by date, so portfolios entered on
    different information dates can be plotted on the same axis.
    """
    basket, _, _ = equal_weight_curve(prices, tickers, start)
    bench, _, _ = equal_weight_curve(prices, [benchmark], start)
    if basket.empty or bench.empty:
        return pd.Series(dtype="float64")
    idx = basket.index.intersection(bench.index)
    if months is not None:
        idx = idx[idx <= pd.Timestamp(start) + pd.DateOffset(months=months)]
    out = basket.loc[idx] - bench.loc[idx]
    out.index = (idx - pd.Timestamp(start)).days / 30.44
    out.index.name = "months since entry"
    return out


def horizon_table(prices: pd.DataFrame, tickers, start, *, benchmark: str = BENCHMARK,
                  horizons=HORIZONS, label: str = "basket") -> pd.DataFrame:
    """Basket vs benchmark total return at each horizon, plus the per-name hit rate."""
    start = pd.Timestamp(start)
    bcurve, held, _ = equal_weight_curve(prices, tickers, start)
    mcurve, _, _ = equal_weight_curve(prices, [benchmark], start)
    if bcurve.empty or mcurve.empty:
        return pd.DataFrame()

    last = min(bcurve.index[-1], mcurve.index[-1])
    rows = []
    for m in list(horizons) + ["all"]:
        stop = last if m == "all" else start + pd.DateOffset(months=m)
        if stop > last:
            continue
        b = bcurve.loc[:stop].iloc[-1] - 1
        s = mcurve.loc[:stop].iloc[-1] - 1
        names = prices.loc[start:stop, held].ffill()
        per_name = names.iloc[-1] / names.iloc[0] - 1
        rows.append({"horizon": f"{m}m" if m != "all" else f"all ({(stop - start).days / 365.25:.1f}y)",
                     label: b, benchmark: s, "excess": b - s,
                     "hit_rate": float((per_name > s).mean()), "n": len(held)})
    return pd.DataFrame(rows).set_index("horizon")


def basket_tickers(name: str, top_n: int = 10, *, proc_dir: Path = PROC_DIR) -> list[str]:
    """Top-N tickers from a `theme_basket_{name}.parquet` output."""
    return pd.read_parquet(proc_dir / f"theme_basket_{name}.parquet").head(top_n)["ticker"].tolist()
