"""Stage 3 — watch a theme's volume and decide, week by week, whether to be invested in it.

    monitor(corpus, themes, asof) -> signal [theme, week, score, position] · diagnostics

The stage counts; two swappable halves decide. `entry/` says when to open a position and `exit/`
says when to close it, and they do not share a measure.
"""
from __future__ import annotations

import importlib
from collections import defaultdict
from datetime import datetime, timedelta

import polars as pl

tokenizer = importlib.import_module("1_detectors.graph_anomaly.tokenizer")
progress = importlib.import_module("progress")

# ======================================================================================
# FLOW — one run of the stage
#
#   corpus + themes
#        |   COUNT      one pass: how many headlines carried any of the theme's words, per week
#        v
#   a contiguous weekly grid, zero-filled — including every week before the theme existed
#        |
#        |   entry/<method>.week(counts, born)    -> the week to open on, and its series
#        |   exit/<method>.week(counts, entry)    -> the week to close on, and its series
#        v
#   signal [theme, week, score, position] · diagnostics [hits, score, level]
#
#   ONE ENTRY AND ONE EXIT PER THEME. `position` is one unbroken run of True, from the entry week
#   to the week before the exit. A theme is a thing that happens once; a rule that reopens is not
#   trading the theme any more, it is trading a score crossing its own threshold.
#
#   THE TWO HALVES MEASURE DIFFERENT THINGS, and that is the point.
#     entry   ACCELERATION — is this week unusual against the weeks just before it
#     exit    LEVEL        — is the flow still above what it was before we bought
#   An acceleration measure cannot find an end: its window catches up with any sustained level and
#   returns to zero while the theme is still running. Using it for both held genAI two weeks.
#
#   The zeros are load-bearing. A theme is born loud, so its own weeks are all large; measured
#   against those alone the arrival is unremarkable. Measured against the quiet months before it,
#   it is the whole signal. That is why the window reaches back well before the emergence week.
# ======================================================================================





def weeks_between(start: datetime, end: datetime) -> list[datetime]:
    """Every Monday from `start` to `end`.

    INPUT   start, end   the ends of the window, any weekday
    OUTPUT  list of Mondays, contiguous — a week with no news becomes a zero, never a gap
    """
    first = start - timedelta(days=start.weekday())
    return [first + timedelta(weeks=i) for i in range((end - first).days // 7 + 1)]


# ======================================================================================
# COUNT — how loud each theme is, week by week
# ======================================================================================
def weekly_hits(corpus, themes: pl.DataFrame, weeks: list[datetime],
                report: bool = False) -> dict[str, list[int]]:
    """How loud each theme is, week by week.

    INPUT   corpus   the canonical corpus, [Headline, date], lazy or eager
            themes   the detector's frame: [theme, week, vocab]
            weeks    the Monday grid from `weeks_between`
    OUTPUT  {theme: [count for each week]}, one list per theme, aligned to `weeks`

    One pass over the corpus for every theme at once, through an inverted word -> themes map:
    scanning once per theme would mean re-reading millions of headlines for each.

    A headline counts once for a theme however many of its words it carries — the question is how
    many stories are about the theme, not how many words they used.
    """
    owners = defaultdict(set)
    for theme, vocab in zip(themes["theme"], themes["vocab"]):
        for word in vocab:
            owners[word].add(theme)

    index = {week: i for i, week in enumerate(weeks)}
    hits = {theme: [0] * len(weeks) for theme in themes["theme"]}
    window = corpus.lazy().filter(pl.col("date").is_between(weeks[0], weeks[-1] + timedelta(days=7))).collect()

    for headline, date in progress.each(list(zip(window["Headline"].to_list(),
                                                 window["date"].to_list())),
                                        "counting headlines", on=report, unit="hl"):
        monday = datetime(date.year, date.month, date.day) - timedelta(days=date.weekday())
        if monday not in index:
            continue
        touched = set()
        for word in set(tokenizer.extract_terms(headline)) & owners.keys():
            touched |= owners[word]
        for theme in touched:
            hits[theme][index[monday]] += 1
    return hits


# ======================================================================================
# THE STAGE — corpus and themes in, a trading rule out
# ======================================================================================
def monitor(corpus, themes: pl.DataFrame, asof: str, *, entry: dict | None = None,
            exit: dict | None = None, lookback_months: int = 30,
            report: bool = True, **_) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score every theme up to `asof` and say when to hold it.

    INPUT   corpus            the canonical corpus, [Headline, date]
            themes            the detector's frame: [theme, week, vocab]
            asof              the day we are pretending it is; the window is derived from it
            entry             {method, ...} — which module in entry/ decides, and how
            exit              {method, ...} — which module in exit/ decides, and how
            lookback_months   how far back the weekly grid reaches from `asof`
    OUTPUT  signal        [theme, week, score, position] — the rule the backtest follows
            diagnostics   same index: [hits, score, level] — what each decision was made of

    One date, like the detector, and for the same reason: `asof` is the day we are pretending it is,
    and nothing published after it may reach the result. The window is `[asof - lookback_months,
    asof]` — derived, not given, so no caller can widen it past what was knowable. The lookback needs
    to reach well before the theme was born, because the quiet months before it are what the entry
    score is measured against, and the year before it is what the exit calls normal.
    """
    entry = dict(entry or {"method": "rolling_z"})
    exit = dict(exit or {"method": "news_level"})
    opens = importlib.import_module(f"2_monitors.entry.{entry.pop('method')}")
    closes = importlib.import_module(f"2_monitors.exit.{exit.pop('method')}")

    asof = _date(asof)
    weeks = weeks_between(asof - timedelta(days=round(lookback_months * 30.44)), asof)
    hits = weekly_hits(corpus, themes, weeks, report)
    born = {theme: _born(weeks, week) for theme, week in zip(themes["theme"], themes["week"])}

    rows, notes = [], []
    for theme, counts in hits.items():
        opened, scores = opens.week(counts, born[theme], **entry)
        closed, level = (closes.week(counts, opened, **exit) if opened is not None
                         else (None, [None] * len(weeks)))
        for i, week in enumerate(weeks):
            held = opened is not None and opened <= i and (closed is None or i < closed)
            rows.append({"theme": theme, "week": week, "score": scores[i], "position": held})
            notes.append({"theme": theme, "week": week, "hits": counts[i],
                           "score": scores[i], "level": level[i]})

    signal = pl.DataFrame(rows, schema=_SIGNAL)
    if report:
        _report(themes, hits, signal, weeks, asof)
    return signal, pl.DataFrame(notes, schema=_NOTES)


def entry_dates(signal: pl.DataFrame) -> pl.DataFrame:
    """The day each theme was first held — the date everything downstream must stop reading at.

    INPUT   signal   the frame `monitor` returns
    OUTPUT  [theme, asof]: the first week `position` is true, or null for a theme never entered

    This is the point-in-time boundary and it is easy to lose. Exposure searches company filings for
    a theme's words; read filings published after we would have bought and the companies are telling
    us about a theme we already own. For genAI that boundary is 2023-01-02, not the run's own as-of
    of 2023-05-18 — four months of filings apart.
    """
    return (signal.filter("position").group_by("theme").agg(pl.col("week").min().alias("asof"))
            .join(signal.select("theme").unique(), on="theme", how="right").sort("theme"))


def _date(value) -> datetime:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _born(weeks: list[datetime], week: datetime) -> int:
    """Index of the theme's emergence week in the grid; 0 if the grid starts after it."""
    return next((i for i, w in enumerate(weeks) if w >= week), len(weeks))


_SIGNAL = {"theme": pl.String, "week": pl.Datetime, "score": pl.Float64, "position": pl.Boolean}
_NOTES = {"theme": pl.String, "week": pl.Datetime, "hits": pl.Int64,
           "score": pl.Float64, "level": pl.Float64}


def _report(themes: pl.DataFrame, hits: dict, signal: pl.DataFrame, weeks: list[datetime],
            asof: datetime) -> None:
    """One line per theme: how loud it got, and when we would have been in it."""
    print(f"   asof {asof.date()}   window {weeks[0].date()} .. {weeks[-1].date()}"
          f"   {len(weeks)} weeks, {len(themes)} themes")
    for theme in themes["theme"]:
        held = signal.filter((pl.col("theme") == theme) & pl.col("position"))["week"]
        counts = hits[theme]
        when = f"in from {str(held[0])[:10]}, {len(held)} weeks" if len(held) else "never entered"
        print(f"   {theme[:34]:36} peak {max(counts):>4} headlines/week   {when}")
