"""Stage 5 — mark a theme's basket to market for the days the monitor says to hold it.

    backtest(weights, signal, prices) -> performance [theme, ...] · curves [theme, date, ...]

Arithmetic only: nothing is fetched and nothing is decided here.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl

# ======================================================================================
# THE BACKTEST — a basket and a rule in, a track record out
#
#   IN    weights      [theme, ticker, weight]        3_baskets
#   IN    signal       [theme, week, position]        2_monitors
#   IN    prices       [date × ticker]                prices.py, benchmark column included
#   OUT   performance  [theme, total_return, ann_return, ann_vol, sharpe, max_dd, bench_*]
#   OUT   curves       [theme, date, level, bench]
#
#   HOW
#     signal   ──▶  segments()  ──▶  curve()      ──▶  summarise()  ──▶  performance
#     weights                        level/day                           curves
#     prices
#
#   segments()   a run of True weeks is one stretch. The monitor gives exactly one per theme;
#                this reads whatever the column says, so a rule that gave more would still work.
#       week       01-02   01-09   01-16   01-23   01-30
#       position   True    True    True    False   False
#                  └───────────────────┘   held, then closed for good
#
#   curve()      inside a stretch, one period per rebalance_months:
#       drop      tickers with no price on day one
#       normalise the remaining weights to sum to 1
#       level     Σ weightᵢ × priceᵢ(t) / priceᵢ(t₀)
#       compound  period after period, stretch after stretch
#       flat      out of the market, the level is carried unchanged
#
#   summarise()  total_return  level(end) − 1
#                ann_return    (1 + total) ^ (1/years) − 1
#                ann_vol       std(daily returns) × √252
#                sharpe        ann_return / ann_vol
#                max_dd        min(level / cummax(level) − 1)
#
#   trade on     the first price date AFTER the week, never the same bar
#   window       first entry  →  last week in the signal, NOT the last price available
#
#   NOT MODELLED
#     transaction costs      none
#     risk-free rate         0
#     a name with no price   dropped, its weight goes to the others
#     a name that stops      carried at its last close — optimistic
#     the universe           today's registrants — survivorship bias
# ======================================================================================


def segments(signal: pl.DataFrame) -> dict[str, list[tuple[datetime, datetime | None]]]:
    """The stretches the monitor says to be invested for.

    INPUT   signal   [theme, week, position] — one row per theme per week, `position` the rule
    OUTPUT  {theme: [(first week in, first week out)]} — `out` is None when the position is still
            open when the signal ends, which means hold to the last price

    A theme may be entered, left and entered again: each run of True weeks is its own stretch.
    """
    out: dict[str, list[tuple[datetime, datetime | None]]] = {}
    for theme, rows in signal.sort("week").group_by("theme", maintain_order=True):
        name = theme[0] if isinstance(theme, tuple) else theme
        spans, opened = [], None
        for week, held in zip(rows["week"], rows["position"]):
            if held and opened is None:
                opened = week
            elif not held and opened is not None:
                spans.append((opened, week))
                opened = None
        if opened is not None:
            spans.append((opened, None))
        if spans:
            out[name] = spans
    return out


def curve(prices: pd.DataFrame, weights: dict[str, float], start, end,
          *, rebalance_months: int = 3) -> pd.Series:
    """One holding stretch, marked to market day by day.

    INPUT   prices            [date × ticker] adjusted closes
            weights           {ticker: weight} — the basket's target, any positive scale
            start, end        the first and last price date held
            rebalance_months  how often drifted weights are put back to target
    OUTPUT  a Series of levels indexed by date, 1.0 on `start`; empty when nothing is priced

    Each period is priced from its own first day; the pieces compound. The boundary day opens the
    next period instead of closing the previous one, so no day is counted twice or lost.
    """
    days = _rebalance_days(prices.index, start, end, rebalance_months)
    level, parts = 1.0, []
    for opened, closed in zip(days[:-1], days[1:]):
        piece = _stretch(prices, weights, opened, closed)
        if piece.empty:
            continue
        piece = piece * level
        level = float(piece.iloc[-1])
        parts.append(piece if closed == days[-1] else piece.iloc[:-1])
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def summarise(level: pd.Series) -> dict[str, float]:
    """What one curve is worth, in the five numbers that describe it.

    INPUT   level   a curve of levels indexed by date, starting at 1.0
    OUTPUT  {total_return, ann_return, ann_vol, sharpe, max_dd} — all as fractions, not percent

    `sharpe` is ann_return / ann_vol with a risk-free rate of zero, so it is consistent with the
    two numbers printed beside it rather than being a third convention.
    """
    if len(level) < 2:
        return {k: float("nan") for k in ("total_return", "ann_return", "ann_vol", "sharpe", "max_dd")}
    years = (level.index[-1] - level.index[0]).days / 365.25
    total = float(level.iloc[-1]) - 1.0
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = float(level.pct_change().dropna().std() * 252 ** 0.5)
    return {"total_return": total, "ann_return": ann, "ann_vol": vol,
            "sharpe": ann / vol if vol > 0 else float("nan"),
            "max_dd": float((level / level.cummax() - 1).min())}


def backtest(weights: pl.DataFrame, signal: pl.DataFrame, prices: pd.DataFrame,
             *, benchmark: str = "SPY", rebalance_months: int = 3,
             report: bool = True, **_) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold each theme's basket when the monitor says to, and say what that was worth.

    INPUT   weights           [theme, ticker, weight] from stage 4
            signal            [theme, week, position] from stage 3
            prices            [date × ticker] from prices.load, benchmark column included
            benchmark         the symbol every theme is measured against, over its own window
            rebalance_months  how often drifted weights are put back to target
    OUTPUT  performance   [theme, ...summarise..., bench_...] — one row per theme entered
            curves        [theme, date, level, bench] — both rebased to 1.0 on the entry date

    Window: first entry → last week in the signal. Not the last price available — flat days the
    monitor never ruled on would annualise the return over a longer life than the strategy had.
    """
    # upper-cased because the register hands tickers over in lower case and Yahoo names its
    # columns in upper; the join between the two stages happens here and nowhere else
    basket = {(theme[0] if isinstance(theme, tuple) else theme):
              {t.upper(): w for t, w in zip(rows["ticker"], rows["weight"])}
              for theme, rows in weights.group_by("theme")}
    rows, paths = [], []

    horizon = _day_after(prices.index, signal["week"].max()) or prices.index[-1]

    for theme, spans in segments(signal).items():
        if theme not in basket:
            continue
        level, pieces = 1.0, []
        for opened, closed in spans:
            first = _day_after(prices.index, opened)
            last = _day_after(prices.index, closed) if closed else horizon
            if first is None or last is None or first >= last:
                continue
            piece = curve(prices, basket[theme], first, last,
                          rebalance_months=rebalance_months) * level
            if piece.empty:
                continue
            level = float(piece.iloc[-1])
            pieces.append(piece)
        if not pieces:
            if report:
                print(f"   {theme[:34]:36} no priced days")
            continue

        held = pd.concat(pieces)
        held = held[~held.index.duplicated(keep="last")]
        # flat stretches: carry the level forward, so the curve covers every day of the window and
        # is comparable with the benchmark's. It stops where the SIGNAL stops, not where the price
        # history does — days past the last week the monitor ruled on are days with no strategy.
        theme_curve = held.reindex(prices.loc[held.index[0]:horizon].index).ffill()
        bench = _stretch(prices, {benchmark: 1.0}, theme_curve.index[0], theme_curve.index[-1])

        stats, bench_stats = summarise(theme_curve), summarise(bench)
        rows.append({"theme": theme} | stats
                    | {f"bench_{k}": v for k, v in bench_stats.items()})
        paths += [{"theme": theme, "date": d, "level": float(v),
                   "bench": float(bench.get(d, float("nan")))}
                  for d, v in theme_curve.items()]
        if report:
            print(f"   {theme[:34]:36} {stats['total_return']:+7.1%}  vs {benchmark} "
                  f"{bench_stats['total_return']:+7.1%}   sharpe {stats['sharpe']:.2f}")

    return pl.DataFrame(rows, schema=_PERFORMANCE), pl.DataFrame(paths, schema=_CURVES)


def _stretch(prices: pd.DataFrame, weights: dict[str, float], start, end) -> pd.Series:
    """Buy and hold at fixed target weights, 1.0 on the first day.

    INPUT   prices, weights, start, end   as in `curve`
    OUTPUT  a Series of levels; empty when no name has a price on the first day

    Weights are renormalised over the names actually priced on day one, so a name Yahoo has
    nothing for hands its weight to the others rather than quietly leaving the book part in cash.
    """
    known = [t for t in weights if t in prices.columns]
    window = prices.loc[start:end, known].ffill()
    if window.empty:
        return pd.Series(dtype=float)
    live = [t for t in known if pd.notna(window[t].iloc[0])]
    if not live:
        return pd.Series(dtype=float)
    share = pd.Series({t: float(weights[t]) for t in live})
    share = share / share.sum()
    return (window[live] / window[live].iloc[0] * share).sum(axis=1)


def _rebalance_days(index: pd.DatetimeIndex, start, end, months: int) -> list:
    """The trading days the weights are reset on: every `months` from `start`, plus `end`."""
    grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=pd.DateOffset(months=months))
    grid = grid.append(pd.DatetimeIndex([pd.Timestamp(end)])) if grid[-1] < pd.Timestamp(end) else grid
    days = [d for d in (_day_after(index, g, inclusive=True) for g in grid) if d is not None]
    return list(dict.fromkeys(days))


def _day_after(index: pd.DatetimeIndex, when, *, inclusive: bool = False):
    """The first price date at or after `when` — the day an order placed on `when` would fill."""
    position = index.searchsorted(pd.Timestamp(when), side="left" if inclusive else "right")
    return index[position] if position < len(index) else None


_STATS = {"total_return": pl.Float64, "ann_return": pl.Float64, "ann_vol": pl.Float64,
          "sharpe": pl.Float64, "max_dd": pl.Float64}
_PERFORMANCE = {"theme": pl.String} | _STATS | {f"bench_{k}": v for k, v in _STATS.items()}
_CURVES = {"theme": pl.String, "date": pl.Datetime, "level": pl.Float64, "bench": pl.Float64}
