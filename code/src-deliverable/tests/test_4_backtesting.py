"""4_backtesting — marking a theme's basket to market while the monitor says to hold it."""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

bt = importlib.import_module("4_backtesting.rebalance")

DAYS = pd.date_range("2023-01-02", "2024-12-31", freq="B")
PRICES = pd.DataFrame({"UP": np.linspace(100, 300, len(DAYS)),      # trends, doubles and a half
                       "FLAT": np.full(len(DAYS), 50.0),            # never moves
                       "SPY": np.linspace(100, 150, len(DAYS))},    # the benchmark
                      index=DAYS)
BORN = datetime(2023, 1, 2)


def _signal(pattern: str) -> pl.DataFrame:
    """One character per week from BORN: `#` in, `.` out."""
    return pl.DataFrame([{"theme": "t", "week": BORN + timedelta(weeks=i),
                          "score": 0.0, "position": c == "#"}
                         for i, c in enumerate(pattern)],
                        schema={"theme": pl.String, "week": pl.Datetime,
                                "score": pl.Float64, "position": pl.Boolean})


WEIGHTS = pl.DataFrame([{"theme": "t", "ticker": "up", "weight": 0.5},
                        {"theme": "t", "ticker": "flat", "weight": 0.5}])


def segments_read_the_column_as_written():
    """each run of True weeks is its own stretch, and one still open at the end says so

    The monitor gives exactly one stretch per theme. This reads whatever it is handed, so a rule
    that gave more would still be backtested correctly rather than silently truncated.
    """
    spans = bt.segments(_signal("..##..#"))["t"]
    want = [(BORN + timedelta(weeks=2), BORN + timedelta(weeks=4)),
            (BORN + timedelta(weeks=6), None)]
    return f"got {spans}" if spans != want else None


def trading_is_never_same_bar():
    """an order dated on a price date fills on the NEXT one, so no week is entered on its own close"""
    on_a_trading_day = DAYS[10]
    after = bt._day_after(DAYS, on_a_trading_day)
    if after != DAYS[11]:
        return f"filled on {after}, expected the following day {DAYS[11]}"
    _, curves = bt.backtest(WEIGHTS, _signal("#####"), PRICES, report=False)
    first = curves["date"].min()
    return None if first > BORN else f"the curve opens on {first}, the week boundary itself"


def a_name_without_a_price_hands_its_weight_over():
    """the book stays fully invested: a missing ticker does not leave half of it in cash"""
    got = bt.curve(PRICES, {"UP": 0.5, "NOTLISTED": 0.5}, DAYS[0], DAYS[-1], rebalance_months=120)
    alone = PRICES["UP"].iloc[-1] / PRICES["UP"].iloc[0]
    return None if abs(float(got.iloc[-1]) - alone) < 1e-9 else f"ended at {got.iloc[-1]}, expected {alone}"


def one_name_is_its_own_price():
    """a basket of one is the price ratio itself — the arithmetic adds nothing of its own"""
    got = bt.curve(PRICES, {"UP": 1.0}, DAYS[0], DAYS[-1], rebalance_months=3)
    want = PRICES["UP"] / PRICES["UP"].iloc[0]
    return None if np.allclose(got.values, want.values) else "the curve is not the price ratio"


def rebalancing_sells_the_winner():
    """resetting to target takes money off the name that ran — it must cost in a trending market"""
    often = float(bt.curve(PRICES, {"UP": .5, "FLAT": .5}, DAYS[0], DAYS[-1], rebalance_months=3).iloc[-1])
    never = float(bt.curve(PRICES, {"UP": .5, "FLAT": .5}, DAYS[0], DAYS[-1], rebalance_months=120).iloc[-1])
    return None if often < never else f"rebalanced {often:.4f} did not lag buy-and-hold {never:.4f}"


def out_of_the_market_means_flat():
    """while `position` is False the level does not move — not a single day of return"""
    _, curves = bt.backtest(WEIGHTS, _signal("##......##"), PRICES, report=False)
    gap = curves.filter((pl.col("date") > BORN + timedelta(weeks=3))
                        & (pl.col("date") < BORN + timedelta(weeks=7)))
    if gap.is_empty():
        return "no days fell inside the gap, so nothing was tested"
    return None if gap["level"].n_unique() == 1 else f"{gap['level'].n_unique()} levels while flat"


def the_window_stops_where_the_signal_does():
    """days the monitor never ruled on are not days the strategy existed"""
    _, curves = bt.backtest(WEIGHTS, _signal("####"), PRICES, report=False)
    last_week = BORN + timedelta(weeks=3)
    end = curves["date"].max()
    if end > last_week + timedelta(days=7):
        return f"the curve runs to {end}, past the last week ruled on ({last_week})"
    return None if end < DAYS[-1].to_pydatetime() else "the curve ran to the last price available"


def the_statistics_agree_with_the_curve():
    """total, annualised and sharpe are three readings of one curve, not three conventions"""
    level = pd.Series(np.linspace(1.0, 2.0, len(DAYS)), index=DAYS)
    s = bt.summarise(level)
    years = (DAYS[-1] - DAYS[0]).days / 365.25
    if abs(s["total_return"] - 1.0) > 1e-9:
        return f"total_return {s['total_return']}"
    if abs(s["ann_return"] - (2.0 ** (1 / years) - 1)) > 1e-9:
        return f"ann_return {s['ann_return']} does not compound to the total"
    if abs(s["sharpe"] - s["ann_return"] / s["ann_vol"]) > 1e-9:
        return "sharpe is not ann_return / ann_vol"
    return None if s["max_dd"] == 0.0 else f"a curve that only rises drew down {s['max_dd']}"


CHECKS = [segments_read_the_column_as_written, trading_is_never_same_bar,
          a_name_without_a_price_hands_its_weight_over, one_name_is_its_own_price,
          rebalancing_sells_the_winner, out_of_the_market_means_flat,
          the_window_stops_where_the_signal_does, the_statistics_agree_with_the_curve]
