"""theme_exit — when to close a theme position.

Box 4 (Hotness) gives the entry date. This module is its mirror: three rules that
answer "is this position still worth holding?", each evaluated only on the monthly
rebalance days the backtest already stops on, using only information available on
that day.

    news      the theme's news flow has fallen back to its pre-entry normal
    stop      the basket is more than X% below its own running peak
    horizon   a fixed number of months has passed (the baseline the others must beat)

A rule returns the rebalance day the position is closed on, or None if it never
fires. `apply_exit` then rewrites the period-return series: every period from the
exit day onward earns cash instead of the basket, so an exited run and a held run
stay directly comparable — same dates, same number of periods.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import theme_backtest as bt

_ROOT = Path(__file__).resolve().parent.parent          # code/, one above scripts/
PROC_DIR = _ROOT / "data" / "processed"
CORPUS = _ROOT / "notebooks" / "output" / "news_corpus.parquet"
NEWS_CACHE = PROC_DIR / "theme_exit_weekly_hits.parquet"

ROLL_WEEKS = 12      # trailing window the news level is measured over
BASE_WEEKS = 52      # pre-entry window that defines "normal" for a theme
N_CONSEC = 4         # weeks the news rule must stay below the line before it fires
DRAWDOWN = 0.20      # stop-loss distance from the running peak
HORIZON = 12         # months for the fixed-horizon baseline


# ======================================================================================
# 1 · The theme's news flow
# ======================================================================================
def theme_vocab(name: str, *, proc_dir: Path = PROC_DIR) -> list[str]:
    """The basket's own search vocabulary, minus the terms its manifest already flagged
    as generic. Matching is on the exact phrases (the corpus carries n-gram terms) —
    splitting them into unigrams drags in 'key', 'error', 'jpmorgan' and drowns the
    signal in off-theme news."""
    m = json.loads((proc_dir / f"theme_basket_{name}_manifest.json").read_text())
    generic = set(m.get("generic_terms", []))
    return [t for t in m["vocab"] if t not in generic]


def weekly_hits(name: str, date_from, date_to, *, corpus: Path = CORPUS,
                cache: bool = True) -> pd.Series:
    """Headlines per week mentioning any of the theme's terms. Cached — the scan is
    over 22m rows."""
    key = f"{name}|{pd.Timestamp(date_from).date()}|{pd.Timestamp(date_to).date()}"
    if cache and NEWS_CACHE.exists():
        df = pd.read_parquet(NEWS_CACHE)
        if key in df.columns:
            return df[key].dropna().astype(int)
    from theme_monitor import weekly_theme_hits_from_parquet
    s = weekly_theme_hits_from_parquet(corpus, theme_vocab(name), date_from, date_to)
    if cache:
        df = pd.read_parquet(NEWS_CACHE) if NEWS_CACHE.exists() else pd.DataFrame()
        df[key] = s
        df.to_parquet(NEWS_CACHE)
    return s


def news_level(hits: pd.Series, entry, *, roll: int = ROLL_WEEKS,
               base_weeks: int = BASE_WEEKS) -> pd.Series:
    """Theme loudness as a multiple of its pre-entry normal: the trailing `roll`-week
    mean of weekly hits divided by the mean over the `base_weeks` before entry.

    Deliberately *not* the rolling z-score used at entry. That measures acceleration,
    so its own window catches up with any sustained level within `roll` weeks and it
    returns to zero while the theme is still loud — it can detect an emergence but not
    the end of one. A ratio to a fixed pre-entry baseline is a level, which is what
    "the story is over" actually means. 1.0 = back to normal.
    """
    entry = pd.Timestamp(entry)
    window = hits[(hits.index < entry) & (hits.index >= entry - pd.Timedelta(weeks=base_weeks))]
    base = float(window.mean())
    if not np.isfinite(base) or base <= 0:
        return pd.Series(np.nan, index=hits.index)
    return hits.rolling(roll).mean() / base


# ======================================================================================
# 2 · The three rules — each returns the rebalance day the position is closed on
# ======================================================================================
def news_exit(level: pd.Series, days, *, threshold: float = 1.0,
              n_consecutive: int = N_CONSEC, arm: float | None = None) -> pd.Timestamp | None:
    """First rebalance day on which the news level has been below `threshold` for
    `n_consecutive` straight weeks — but only once the theme has actually got loud
    first (level >= `arm`, default `threshold`).

    The arming step is not a tuning knob, it is what makes the rule mean anything.
    At the entry date the 12-week trailing mean is still mostly pre-theme, so the
    level sits at ~1.0 and an unarmed rule closes the position a month after opening
    it, on both themes, every time. "The story is over" presupposes it began.
    """
    arm = threshold if arm is None else arm
    entry = pd.Timestamp(days[0])
    post = level[level.index >= entry]
    up = post[post >= arm]
    if not len(up):
        return None                          # never got loud -> nothing to close
    fired = (level < threshold).rolling(n_consecutive).sum() == n_consecutive
    weeks = fired[fired & (fired.index > up.index[0])].index
    if not len(weeks):
        return None
    for d in days[1:]:                       # never on the entry day itself
        if (weeks <= d).any():
            return d
    return None


def stop_exit(curve: pd.Series, days, *, drawdown: float = DRAWDOWN) -> pd.Timestamp | None:
    """First rebalance day the basket is more than `drawdown` below its running peak.
    Checked monthly like everything else, not intra-month: we only look when we are
    standing at the desk anyway."""
    peak = curve.cummax()
    for d in days[1:]:
        c = curve[curve.index <= d]
        if len(c) and float(c.iloc[-1]) < float(peak[peak.index <= d].iloc[-1]) * (1 - drawdown):
            return d
    return None


def horizon_exit(days, *, months: int = HORIZON) -> pd.Timestamp | None:
    """Close after a fixed number of months, whatever the theme is doing."""
    cutoff = pd.Timestamp(days[0]) + pd.DateOffset(months=months)
    later = [d for d in days[1:] if d >= cutoff]
    return later[0] if later else None


# ======================================================================================
# 3 · Applying an exit to a backtest
# ======================================================================================
def _cash_by_day(days) -> dict:
    """Cash return of each holding period, keyed by the day the period starts —
    positional alignment breaks whenever a period is dropped for want of prices."""
    days = list(days)
    return {pd.Timestamp(d): float(v)
            for d, v in zip(days[:-1], bt.cash_returns(days))}


def apply_exit(r: pd.Series, days, exit_day) -> pd.Series:
    """Period returns with the position closed on `exit_day`: every period starting on
    or after it earns the cash rate instead of the basket. Same index, same length —
    so an exited run and a held run are compared over identical dates."""
    out = r.copy()
    if exit_day is None:
        return out
    cash = _cash_by_day(days)
    for i, d in enumerate(r.index):
        d = pd.Timestamp(d)
        if d >= pd.Timestamp(exit_day):
            out.iloc[i] = float(cash.get(d, 0.0))
    return out


def exit_stats(r: pd.Series, days, months: int = 1) -> dict:
    """The same numbers the backtest already reports, computed from a period-return
    series (so it works identically on held and exited runs)."""
    sd = float(r.std(ddof=1))
    cash = _cash_by_day(days)
    ex = r.to_numpy() - np.array([cash.get(pd.Timestamp(d), 0.0) for d in r.index])
    sd_ex = float(np.std(ex, ddof=1))
    curve = (1 + r).cumprod()
    # a run that is entirely in cash has no excess return and no risk: its Sharpe is
    # 0/0, and the ratio of two rounding errors is not a number worth printing.
    sharpe = (float(np.mean(ex) / sd_ex * (12 / months) ** 0.5)
              if sd_ex > 1e-6 else np.nan)
    return {"periods": len(r),
            "compounded": float(curve.iloc[-1] - 1) if len(curve) else np.nan,
            "std": sd,
            "ret/risk": float(r.sum() / sd) if sd > 0 else np.nan,
            "Sharpe ann": sharpe,
            "max DD": bt.max_drawdown(curve)}


# ======================================================================================
# 4 · Re-entry — the same rule allowed to flip back on
# ======================================================================================
def news_state(level: pd.Series, days, *, threshold: float = 1.0,
               n_consecutive: int = N_CONSEC, arm: float | None = None,
               reenter: bool = True) -> pd.Series:
    """Hold-or-cash decision at every rebalance day, from the news level alone.

    Out when the level has been below `threshold` for `n_consecutive` weeks; back in
    when it has been above for `n_consecutive` weeks. With `reenter=False` the first
    exit is final — that is the v1 rule, and the difference between the two is the
    price of the simplification.

    Only weeks on or before each rebalance day are read, so the state on day d uses
    only news published by d.
    """
    arm = threshold if arm is None else arm
    entry = pd.Timestamp(days[0])
    post = level[level.index >= entry]
    up = post[post >= arm]
    armed_from = up.index[0] if len(up) else None

    lo = (level < threshold).rolling(n_consecutive).sum() == n_consecutive
    hi = (level >= threshold).rolling(n_consecutive).sum() == n_consecutive

    # walk the weeks, not the rebalance days: a below-the-line spell that opens and
    # closes inside one month must still flip the state, exactly as `news_exit` sees it.
    held, by_week = True, {}
    for w in level.index:
        if armed_from is not None and w > armed_from:
            if held and bool(lo.get(w, False)):
                held = False
            elif not held and reenter and bool(hi.get(w, False)):
                held = True
        by_week[w] = held

    state = {}
    for d in days[:-1]:
        d = pd.Timestamp(d)
        past = level.index[level.index <= d]
        state[d] = by_week[past[-1]] if len(past) else True
    return pd.Series(state, name="held")


def apply_state(r: pd.Series, days, state: pd.Series) -> pd.Series:
    """Period returns with the basket held only where `state` is True, cash elsewhere."""
    cash = _cash_by_day(days)
    out = r.copy()
    for i, d in enumerate(r.index):
        d = pd.Timestamp(d)
        if not bool(state.get(d, True)):
            out.iloc[i] = float(cash.get(d, 0.0))
    return out
