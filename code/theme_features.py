"""Point-in-time stock characteristics for a theme basket.

The backtest only ever asked one question of a name: what did it return?  A trading
desk asks a dozen more before it will hold it -- how big is it, can I get in and out,
what does it cost to borrow, is it even in my universe.  This module answers the ones
that can be reconstructed from public data *as they stood on a given date*, and names
the ones that cannot (see `MISSING`).

Everything here is as-of: trailing windows end on `date`, share counts are the last
value *filed* on or before it, and nothing reads a row the market had not printed yet.
The one deliberate exception is `still_quoted` / `float_pct_now`, which are marked
`(today)` because no free source keeps history for them -- they are diagnostics about
the data, not inputs to a rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import theme_backtest as bt

_ROOT = Path(__file__).resolve().parent
OHLCV_DIR = _ROOT / "data" / "raw" / "ohlcv"
INFO_CACHE = OHLCV_DIR / "_info.json"
BENCH = "SPY"

# windows, in trading days
W_LIQ, W_VOL, W_BETA, W_MOM = 60, 60, 250, 250
# a $10m clip is the unit the desk thinks in; 20% of a day's volume is the usual cap
CLIP_MUSD, PARTICIPATION = 10.0, 0.20

# what public data cannot give us, and who has to
MISSING = [
    ("free float (point-in-time)", "float changes with lock-ups, insider and strategic stakes; "
     "Yahoo publishes only today's figure", "Bloomberg free-float history"),
    ("index membership", "Russell 3000/2000, S&P 1500 on the date -- the usual definition of "
     "'investable'", "index constituent history"),
    ("borrow availability and fee", "a name can be uninvestable long-only anyway, but the short "
     "leg and any hedge depend on it", "prime broker / securities-lending screen"),
    ("short interest, days to cover", "crowding; the 2024 quantum move happened partly on it",
     "exchange short-interest file"),
    ("bid-ask spread and depth", "the single biggest cost term for a $200m-cap name; daily bars "
     "cannot show it", "intraday quotes (TAQ / Bloomberg)"),
    ("primary vs consolidated vol.", "Yahoo's volume is consolidated; the tradable share is "
     "smaller", "venue-level volume"),
    ("listed options", "whether the exposure can be taken or hedged synthetically",
     "options chain history"),
    ("restricted / compliance list", "internal, and it overrides everything above",
     "Intesa compliance"),
]


# ---------------------------------------------------------------- data access
def _path(ticker: str) -> Path:
    return OHLCV_DIR / f"{ticker}.parquet"


def download_ohlcv(tickers, *, force: bool = False, chunk: int = 20, verbose: bool = True) -> None:
    """Cache full-history *unadjusted* OHLCV + adjusted close per ticker.

    `bt.load_prices` keeps adjusted closes only.  Volume and the traded price level
    (is it a $2 stock?) need the unadjusted bars, so they get their own cache.
    """
    import yfinance as yf

    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in dict.fromkeys(tickers) if force or not _path(t).exists()]
    if not todo:
        return
    if verbose:
        print(f"downloading OHLCV for {len(todo)} tickers ...")
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        raw = yf.download(batch, period="max", auto_adjust=False, progress=False,
                          threads=True, group_by="ticker")
        for t in batch:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(how="all")
            except Exception:
                d = pd.DataFrame()
            if len(d):
                out = pd.DataFrame({
                    "date": pd.to_datetime(d.index).tz_localize(None),
                    "close": d["Close"].to_numpy(),
                    "adjclose": d["Adj Close"].to_numpy() if "Adj Close" in d else d["Close"].to_numpy(),
                    "high": d["High"].to_numpy(), "low": d["Low"].to_numpy(),
                    "volume": d["Volume"].to_numpy()})
            else:
                out = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                                    "close": pd.Series(dtype="float64"),
                                    "adjclose": pd.Series(dtype="float64"),
                                    "high": pd.Series(dtype="float64"),
                                    "low": pd.Series(dtype="float64"),
                                    "volume": pd.Series(dtype="float64")})
            out.to_parquet(_path(t), index=False)
        if verbose:
            print(f"  [{i + len(batch):>3}/{len(todo)}]")


def load_ohlcv(ticker: str) -> pd.DataFrame:
    p = _path(ticker)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p).set_index("date").sort_index()


def info(ticker: str, *, cache: bool = True) -> dict:
    """Static descriptors (exchange, sector, float, ...) -- current, not point-in-time."""
    store = json.loads(INFO_CACHE.read_text()) if cache and INFO_CACHE.exists() else {}
    if ticker in store:
        return store[ticker]
    import yfinance as yf
    keys = ["longName", "shortName", "exchange", "fullExchangeName", "quoteType", "sector",
            "industry", "country", "currency", "floatShares", "sharesOutstanding",
            "marketCap", "trailingAnnualDividendYield"]
    try:
        raw = yf.Ticker(ticker).info or {}
    except Exception:
        raw = {}
    out = {k: raw.get(k) for k in keys}
    if cache:
        store[ticker] = out
        INFO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        INFO_CACHE.write_text(json.dumps(store, indent=1))
    return out


# ---------------------------------------------------------------- the feature sheet
def _trailing(s: pd.Series, date, n: int) -> pd.Series:
    return s[s.index <= pd.Timestamp(date)].tail(n)


def split_factor(ticker: str, date, *, cache: dict | None = None) -> float:
    """Product of every split ratio *after* `date` (1.0 if none).

    Yahoo back-adjusts both price and volume for later splits, so a series read at face
    value is in today's share units, not that day's.  price_then = close x factor and
    shares_then = volume / factor put them back.  A 10-for-1 gives 10; a 1-for-80 reverse
    gives 1/80.  Dollar volume is invariant and needs no correction.
    """
    import yfinance as yf

    if cache is not None and ticker in cache:
        sp = cache[ticker]
    else:
        try:
            sp = yf.Ticker(ticker).splits
        except Exception:
            sp = pd.Series(dtype="float64")
        if cache is not None:
            cache[ticker] = sp
    if sp is None or not len(sp):
        return 1.0
    idx = sp.index.tz_localize(None) if sp.index.tz is not None else sp.index
    return float(sp[idx > pd.Timestamp(date)].prod()) or 1.0


def features_asof(ticker: str, date, prices: pd.DataFrame | None = None) -> dict:
    """Every characteristic of `ticker` we can reconstruct as of `date`."""
    date = pd.Timestamp(date)
    d = load_ohlcv(ticker)
    meta = info(ticker)
    row: dict = {"ticker": ticker,
                 "name": meta.get("longName") or meta.get("shortName") or "",
                 "exchange": meta.get("fullExchangeName") or meta.get("exchange"),
                 "type": meta.get("quoteType"), "sector": meta.get("sector"),
                 "industry": meta.get("industry"), "country": meta.get("country"),
                 "currency": meta.get("currency")}
    if d.empty:
        return {**row, "note": "no price history"}

    hist = d[d.index <= date]
    if hist.empty:
        return {**row, "note": f"not quoted on {date.date()}"}

    px, adj, vol = hist["close"], hist["adjclose"], hist["volume"]
    f = split_factor(ticker, date)
    row["split_factor"] = round(f, 6)
    row["price"] = round(float(px.iloc[-1]) * f, 2)      # the level actually quoted that day
    # a factor this extreme means the series has been re-based past the point of trust
    row["adj_suspect"] = bool(f > 100 or f < 0.01)
    row["first_quote"] = d.index[0].date().isoformat()
    row["age_yrs"] = round((date - d.index[0]).days / 365.25, 1)
    row["last_quote"] = d.index[-1].date().isoformat()
    row["still_quoted"] = bool(d.index[-1] > date + pd.Timedelta(days=365))

    # --- size, point in time
    cap = bt.market_cap_asof(ticker, date, prices) if prices is not None else None
    row["cap_musd"] = None if cap is None else round(cap, 1)
    fs, so = meta.get("floatShares"), meta.get("sharesOutstanding")
    row["float_pct_now"] = round(100 * fs / so, 1) if fs and so else None
    row["float_cap_musd"] = (None if cap is None or row["float_pct_now"] is None
                             else round(cap * row["float_pct_now"] / 100, 1))

    # --- liquidity
    v = _trailing(vol, date, W_LIQ)
    p = _trailing(px, date, W_LIQ)
    dollar = (v * p) / 1e6                       # $m traded per day
    row["adv_sh_k"] = round(float(v.mean()) / f / 1e3, 1) if len(v) else None
    row["addv_musd"] = round(float(dollar.mean()), 2) if len(dollar) else None
    row["addv_med_musd"] = round(float(dollar.median()), 2) if len(dollar) else None
    row["zero_vol_days"] = int((v == 0).sum()) if len(v) else None
    row["turnover_pct"] = (round(100 * row["addv_musd"] / cap, 2)
                           if cap and row["addv_musd"] else None)
    # days to build a CLIP_MUSD position at PARTICIPATION of a normal day
    row["days_to_clip"] = (round(CLIP_MUSD / (PARTICIPATION * row["addv_med_musd"]), 1)
                           if row["addv_med_musd"] else None)
    # what a name this liquid can carry, if a position may take 5 days to build
    row["capacity_musd"] = (round(5 * PARTICIPATION * row["addv_med_musd"], 3)
                            if row["addv_med_musd"] else None)
    # a spread proxy -- Corwin-Schultz needs quotes, so use the crude high-low range
    rng = (_trailing(hist["high"], date, W_LIQ) - _trailing(hist["low"], date, W_LIQ)) / p
    row["hl_range_pct"] = round(100 * float(rng.mean()), 2) if len(rng) else None

    # --- risk, from adjusted closes
    r = _trailing(adj, date, W_VOL + 1).pct_change(fill_method=None).dropna()
    row["vol_ann_pct"] = round(100 * float(r.std()) * np.sqrt(252), 1) if len(r) > 5 else None
    rl = _trailing(adj, date, W_MOM + 1)
    row["mom_12m_pct"] = (round(100 * (float(rl.iloc[-1]) / float(rl.iloc[0]) - 1), 1)
                          if len(rl) > 20 else None)
    row["max_1d_pct"] = (round(100 * float(_trailing(adj, date, W_BETA + 1)
                                           .pct_change(fill_method=None).abs().max()), 1)
                         if len(rl) > 20 else None)
    if prices is not None and BENCH in prices.columns:
        rb = _trailing(prices[BENCH].dropna(), date, W_BETA + 1).pct_change(fill_method=None)
        ra = _trailing(adj, date, W_BETA + 1).pct_change(fill_method=None)
        j = pd.concat([ra, rb], axis=1, join="inner").dropna()
        row["beta"] = (round(float(j.cov().iloc[0, 1] / j.iloc[:, 1].var()), 2)
                       if len(j) > 60 else None)

    # --- tradability flags
    row["penny"] = row["price"] < 5
    row["microcap"] = bool(cap is not None and cap < 200)
    row["illiquid"] = bool(row["addv_med_musd"] is not None and row["addv_med_musd"] < 1)
    row["young"] = row["age_yrs"] < 1
    row["div_yield_pct"] = (round(100 * meta["trailingAnnualDividendYield"], 2)
                            if meta.get("trailingAnnualDividendYield") else None)
    return row


def basket_sheet(tickers, date, prices: pd.DataFrame | None = None,
                 *, download: bool = True, verbose: bool = True) -> pd.DataFrame:
    """`features_asof` for every ticker, as one frame."""
    tickers = list(dict.fromkeys(tickers))
    if download:
        download_ohlcv(tickers, verbose=verbose)
    return pd.DataFrame([features_asof(t, date, prices) for t in tickers])


def screen(sheet: pd.DataFrame, *, min_cap: float = 200.0, min_addv: float = 1.0,
           min_price: float = 5.0, min_age: float = 1.0) -> pd.DataFrame:
    """Successive-filter waterfall: how many names each rule removes, in order."""
    rules = [("all names", lambda d: d.index == d.index),
             (f"cap >= ${min_cap:.0f}m", lambda d: d.cap_musd.fillna(-1) >= min_cap),
             (f"ADDV >= ${min_addv:.0f}m/day", lambda d: d.addv_med_musd.fillna(-1) >= min_addv),
             (f"price >= ${min_price:.0f}", lambda d: d.price.fillna(0) >= min_price),
             (f"age >= {min_age:.0f}y", lambda d: d.age_yrs.fillna(0) >= min_age)]
    live, rows = sheet.copy(), []
    for label, rule in rules:
        live = live[rule(live)]
        rows.append({"rule": label, "kept": len(live),
                     "names": ", ".join(live.ticker.tolist())})
    return pd.DataFrame(rows)
