"""Stage 1 — the Bloomberg feed capture becomes the one canonical news corpus.

Every row of the raw feed is a *message*, not a story. One publication already arrives as two of them
(ADD_1STPASS then ADD_STORY, a fraction of a second apart), and the wire then re-transmits the
headline every time an attribute changes. Dating a headline by whichever message you happen to see
therefore invents bursts of news that never happened.

How much of the capture is re-transmission changes completely over the years, so do not trust a
single figure for it — on the wires we keep, publication messages are 100% of 2010, 17% of 2018 and
33% of 2025. The rule below is what makes those years comparable at all.

This stage repairs exactly that and nothing else. It does not strip, shorten, tokenize or thin the
headlines: whatever the wire published, in the words it published it, is what comes out — because
a method that reads words, one that reads entities and one that embeds sentences all need to see
something different, and only they know what. The corpus is the raw material, not a preparation.

    output: [Headline, date] — one row per publication, dated by its first message
"""
from __future__ import annotations

import hashlib
import json
import lzma
import time
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config

# ======================================================================================
# FLOW — one year through the stage, with the real numbers of the 2010 capture
#
#   data/raw/raw_news_2010.csv.xz
#        4,835,507 ROWS, AND EACH ROW IS A MESSAGE — NOT A STORY
#            |
#            |   KEEP THE BLOOMBERG WIRES          drop web, blogs, EDGAR feed
#            |   KEEP THE "PUBLISHED" MESSAGES     drop re-transmissions
#            |   ONE ROW PER HEADLINE, EVER        a title said again is the same story
#            |   DATE IT BY ITS FIRST MESSAGE
#            v
#   data/processed/tmp/terms_2010.parquet          <- build_year(), so this never runs twice
#            |
#            |   CONCATENATE ALL 16 YEARS
#            v
#   data/processed/news_corpus.parquet             <- build_corpus()
#
#   Titles match character for character, prefix included — which is why the corpus keeps the wire's
#   own words. "Daniel Yergin: Surveillance With Prewitt and Keene" and "HSBC's King: Surveillance..."
#   are two stories with two guests; a bulletin re-issued verbatim for a decade is one.
# ======================================================================================

# What the raw capture's columns mean:
#
#   CaptureTime   when this MESSAGE was transmitted — not when the story was published
#   Headline      the headline text as the story record read at message time
#   Event         why the message was sent:
#                   ADD_1STPASS       first publication, the instant flash before the article exists
#                   ADD_STORY         the full story record added to the wire
#                   UPDATE_ATTRIBUTE  the story's tags were edited and the record re-transmitted,
#                                     seconds or YEARS after publication — the replay artifact
#   Version       revision state of the story text: ORIGINAL / UPDATE / CORRECTION. Orthogonal to
#                 Event, and useless for telling a publication from a replay: filtering on it leaves
#                 44% of headlines spread over several days, exactly as filtering on nothing does.
#   WireName      which wire carried it (see WIRES below)
#   DerivedTickersId, AssignedTickersId, DerivedTopicsId, AssignedTopicsId, LanguageString
#                 Bloomberg's own tagging, unused here — a later method may want it
RULE = "first-publication-v3"               # bump when the rule changes; invalidates every cached year

# Bloomberg's own editorial wires. Everything else in the capture is aggregated web content, blogs,
# third-party wires and machine feeds (EDGAR filings), which report no news of their own. The mix
# shifts hard over time: the 2010 capture carries 3 wires and this keeps 91% of its messages, the
# 2025 capture carries 189 and this keeps 32%. Codes are from data/raw/wire mapping.xlsx.
WIRES = ["BN", "BFW", "BBO"]                # Bloomberg News, First Word, Opinion

# `Event` values meaning "this story is being published"; everything else re-transmits an existing one.
# The 2010 capture holds nothing but these two — UPDATE_ATTRIBUTE appears only in later years, where
# it grows to dominate, so this filter removes nothing early on and most of the file later.
ADD_EVENTS = ["ADD_1STPASS", "ADD_STORY"]

TMP = config.PROC / "tmp"                   # per-year builds, disposable


# ======================================================================================
# THE THREE FILTERS — the whole method. Messages in, stories out. Pure: no disk, no config.
#
#   keep_wires        on WireName   drop everyone who is not Bloomberg
#   keep_published    on Event      drop re-transmissions of stories we keep
#   first_occurrence  on Headline   drop a title already published, verbatim, before
# ======================================================================================
def keep_wires(messages: pl.DataFrame, report: bool = True) -> pl.DataFrame:
    """Keep the wires that report news of their own.

    INPUT   messages   raw feed rows: [Headline, CaptureTime, WireName, Event]
    OUTPUT  the same columns, only the rows on WIRES that carry a headline
    """
    ok = pl.col("WireName").is_in(WIRES) & pl.col("Headline").is_not_null()
    kept = messages.filter(ok)
    _step(report, "keep the Bloomberg wires", messages, kept,
          _example(messages.filter(~ok), "WireName"))
    return kept


def keep_published(messages: pl.DataFrame, report: bool = True) -> pl.DataFrame:
    """Drop re-transmissions, keeping only the messages that announce a publication.

    INPUT   messages   feed rows, wires already filtered
    OUTPUT  the same columns, only the rows whose Event announces a publication

    This is the whole point of the stage, and it is worth knowing what it costs and why it is still
    right. In 2018 it discards 85% of the rows, and — more alarming at first sight — 46% of the
    *distinct headlines*, which never carry an ADD event at all. Those look like real news being
    thrown away:

        Vale Quarterly Profit Slides 66% on Lower Prices, Provisions
        Liberty Interactive in Pact to Buy General Communication

    They are real news, but not of 2018. Look them up in the finished corpus and they date from
    2012-10-24 and 2017-04-04. They re-enter the 2018 capture because someone edited the story's
    tags years later, and the re-transmission carries a *fresh* CaptureTime. Keeping them would date
    a 2012 story as 2018 news — a bulk re-tagging day would read as a one-day burst of news about old
    events, which is precisely the false positive an emergence detector must not swallow.

    So a headline whose earliest message is a re-transmission was published before this capture, or
    in the year where its own ADD lives, and belongs to that year and not this one. The figures above
    are measured on the 2018 capture; look any of those headlines up in the corpus to see it.
    """
    kept = messages.filter(pl.col("Event").is_in(ADD_EVENTS))
    _step(report, "drop re-transmissions", messages, kept,
          _example(messages.filter(~pl.col("Event").is_in(ADD_EVENTS)), "Event"))
    return kept


def first_occurrence(messages: pl.DataFrame, report: bool = True) -> pl.DataFrame:
    """Collapse messages into stories.

    INPUT   messages   publication messages
    OUTPUT  one row per distinct headline, carrying the CaptureTime of its earliest message —
            this is where a row stops being a message and becomes a story

    Exactly as written: two rows are the same story when their raw text matches character for
    character, prefix included. That is why the corpus keeps the wire's own words — `Daniel Yergin:
    Surveillance With Prewitt and Keene` and `HSBC's King: Surveillance...` are different titles and
    both stay, while a bulletin re-issued verbatim for a decade collapses to its first issue.
    `Five Things You Need to Know to Start Your Day` is 3,322 rows and 23 distinct titles.

    Applied within a year here, and again across years in `build_corpus` — a story does not become
    new by crossing a New Year.
    """
    dated = (messages
             .with_columns(pl.col("CaptureTime").str.to_datetime(time_zone="UTC", strict=False))
             .drop_nulls("CaptureTime")
             .sort("CaptureTime"))
    once = dated.unique(subset="Headline", keep="first", maintain_order=True)
    _step(report, "one row per headline, ever", dated, once, _collapsed(dated))
    return once


# ======================================================================================
# WHAT THE DATA IS — read the capture before filtering it
# ======================================================================================
# What each Event value means — this is the column the second filter turns on
EVENT_MEANING = {
    "ADD_1STPASS":      "first flash of the headline, before the article body exists",
    "ADD_STORY":        "the full story record is added to the wire",
    "UPDATE_ATTRIBUTE": "the story's tags were edited and the record re-sent — seconds or YEARS later",
}


def describe(messages: pl.DataFrame) -> None:
    """Read the capture before filtering it.

    INPUT   messages   raw feed rows
    OUTPUT  nothing — prints the columns, the wires and the event types with their meanings
    """
    print(f"raw dataset — {len(messages):,} rows, and a row is a MESSAGE the wire sent, not a story")
    print(f"   columns   {' · '.join(messages.columns)}")
    print( "             Headline    the headline as the story record read at message time")
    print( "             CaptureTime when this message was transmitted — not when the story was published")

    names = _wire_names()
    wires = messages["WireName"].value_counts(sort=True)
    print(f"\n   WireName  which wire carried it — {messages['WireName'].n_unique()} in this capture, {len(WIRES)} kept")
    for row in wires.head(8).to_dicts():
        code = row["WireName"]
        print(f"      {'keep' if code in WIRES else '    '}  {code:<5} {row['count']:>11,}  {names.get(code, '')}")
    if len(wires) > 8:
        print(f"            ... and {len(wires) - 8} more, all dropped")

    print(f"\n   Event     why the wire sent this message — the column this stage turns on")
    for row in messages["Event"].value_counts(sort=True).to_dicts():
        event = row["Event"]
        print(f"      {'keep' if event in ADD_EVENTS else '    '}  {event:<17} {row['count']:>11,}")
        print(f"              {EVENT_MEANING.get(event, '')}")


def _wire_names() -> dict:
    """Wire code -> name, from data/raw/wire mapping.xlsx. Empty if the file is not there."""
    try:
        table = pd.read_excel(config.FEED / "wire mapping.xlsx")
        table.columns = ["code", "name"]
        return dict(zip(table.code.astype(str).str.strip(), table.name.astype(str).str.strip()))
    except Exception:
        return {}


# ======================================================================================
# REPORTING — what each filter removed, and one row it removed
# ======================================================================================
def _step(report: bool, what: str, before: pl.DataFrame, after: pl.DataFrame, example: str = "") -> None:
    """One line per step: how many rows survived it, and one row that did not."""
    if not report:
        return
    n0, n1 = len(before), len(after)
    print(f"   {what:30} {n0:>10,} -> {n1:>10,}   {(n1 - n0) / max(n0, 1):+6.1%}")
    if example:
        print(f"      {example}")


def _example(dropped: pl.DataFrame, because: str) -> str:
    """One headline this step threw away, and the value that condemned it."""
    if not len(dropped):
        return ""
    row = dropped.row(0, named=True)
    return f'dropped e.g. [{row[because]}] "{str(row["Headline"])[:64]}"'


def _collapsed(dated: pl.DataFrame) -> str:
    """One headline that arrived more than once, with the first two times — the rows this step merges."""
    repeats = dated.filter(pl.col("Headline").is_duplicated())
    if not len(repeats):
        return ""
    head = repeats.row(0, named=True)["Headline"]
    times = (repeats.filter(pl.col("Headline") == head)
                    .head(2)["CaptureTime"].dt.replace_time_zone(None)
                    .dt.strftime("%Y-%m-%d %H:%M:%S").to_list())
    return f'merged e.g. "{str(head)[:46]}" sent at {" and ".join(times)}'


# ======================================================================================
# BUILD THE CORPUS — the filters run over every year, cached, then joined
# ======================================================================================
def raw_years() -> list[int]:
    """INPUT nothing · OUTPUT the years that have a raw capture on disk, ascending."""
    return sorted(int(p.stem.split("_")[-1].split(".")[0])
                  for p in config.FEED.glob("raw_news_*.csv.xz"))


def load_raw(years: int | list[int]) -> pl.DataFrame:
    """Read raw captures off disk.

    INPUT   years   one year, or several to concatenate
    OUTPUT  every message of those years: [Headline, CaptureTime, WireName, Event]

    One row per message. A year is 5-18M of them, so ask for what you mean to look at.
    """
    frames = []
    for year in ([years] if isinstance(years, int) else years):
        path = config.FEED / f"raw_news_{year}.csv.xz"
        if not path.exists():
            raise FileNotFoundError(f"no raw capture for {year}: {path}")
        with lzma.open(path, "rb") as f:
            frames.append(pl.scan_csv(f, infer_schema_length=10_000).select(
                ["Headline", "CaptureTime", "WireName", "Event"]).collect())
    return pl.concat(frames)


def build_year(year: int, force: bool = False) -> Path:
    """Turn one year of messages into one year of the corpus, cached.

    INPUT   year, force   the year to build; `force` rebuilds one already certified
    OUTPUT  the path of tmp/terms_{year}.parquet — [Headline, date], dated and deduplicated

    That is the only reason this function exists. A raw capture is up to 1 GB compressed and there
    are sixteen of them, so each year is processed once into `tmp/terms_{year}.parquet` and read
    from there afterwards. A year already recorded in the manifest is skipped, unless `force`.

    Headlines dated outside `year` are dropped: they belong to that year's own capture, where their
    first message actually lives.
    """
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"terms_{year}.parquet"
    if not force and str(year) in _manifest()["years"] and out.exists():
        print(f"{year}: certified {RULE} build exists — skipping")
        return out

    t0 = time.time()
    messages = load_raw(year)
    describe(messages)
    df = first_occurrence(keep_published(keep_wires(messages))).to_pandas()
    df["date"] = pd.to_datetime(df["CaptureTime"]).dt.tz_localize(None)
    df = df[df.date.dt.year == year]
    df = df.sort_values(["date", "Headline"]).reset_index(drop=True)   # tie-break: same-instant rows
    df[["Headline", "date"]].to_parquet(out, index=False)

    m = _manifest()
    m["years"][str(year)] = {"rows": len(df), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    m["corpus"] = None                                       # the merged corpus no longer matches
    config.CORPUS_MANIFEST.write_text(json.dumps(m, indent=2))
    print(f"{year}: {len(messages):,} messages -> {len(df):,} rows  [{time.time() - t0:.0f}s]")
    return out


def build_corpus(years: list[int] | None = None, force: bool = False) -> Path:
    """The stage: build whatever is missing, then join the years into the one corpus.

    INPUT   years, force   which years to cover; default is every capture on disk
    OUTPUT  the path of news_corpus.parquet — [Headline, date], sorted, globally deduplicated

    Call this and nothing else — it is safe to re-run, since a year already built is skipped and the
    join is redone only when some year changed under it. A title already published in an earlier year
    is dropped: it is the same story, not a new one.
    """
    years = years or raw_years()
    for year in years:
        build_year(year, force=force)
    if _manifest()["corpus"] is not None and config.CORPUS.exists():
        print(f"corpus up to date: {config.CORPUS.name}")
        return config.CORPUS

    drop = set(config.params("preprocessing").get("drop_titles") or [])
    writer, seen, total = None, set(), 0
    for year in years:
        df = pd.read_parquet(TMP / f"terms_{year}.parquet")
        keep = ~_fingerprint(df.Headline).isin(seen)          # a title already published is not new
        seen.update(_fingerprint(df.Headline[keep]))
        df = df[keep & ~df.Headline.isin(drop)]
        table = pa.Table.from_pandas(df, preserve_index=False)
        writer = writer or pq.ParquetWriter(config.CORPUS, table.schema)
        writer.write_table(table)
        total += len(df)
        print(f"  {year}: +{len(df):,} rows ({(~keep).sum():,} already published in an earlier year)")
    writer.close()

    m = _manifest()
    m["corpus"] = {"rows": total, "years": years, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    config.CORPUS_MANIFEST.write_text(json.dumps(m, indent=2))
    print(f"corpus: {total:,} rows, {years[0]}-{years[-1]} -> {config.CORPUS.name}")
    return config.CORPUS


# ======================================================================================
# FOR THE HUMAN — look at what the filters did
# ======================================================================================
def volumes(freq: str = "1w", **frames: pl.DataFrame):
    """Message volume over time for each step, so a filter's effect is visible *when* it happened.

    INPUT   freq     the bucket, any polars duration
            frames   the named frames to draw, in order, each with a CaptureTime column
    OUTPUT  a plotly figure; in a notebook just let the cell display it

        volumes(raw=raw, wires=wires, published=published, dated=dated)

    A total says a step removed 83%; a curve says whether it removed a steady share or one bad week.
    That difference is the whole reason this stage exists — a re-tagging day looks like a news burst,
    and it shows up here as a spike in `wires` that is absent from `published`.

    Returns a plotly figure; in a notebook just let the cell display it.
    """
    import math

    import plotly.graph_objects as go

    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # fixed order, never cycled
    fig = go.Figure()
    for (name, frame), colour in zip(frames.items(), colours):
        when = frame["CaptureTime"]
        if when.dtype == pl.String:
            when = when.str.to_datetime(time_zone="UTC", strict=False)
        series = (frame.with_columns(when.alias("_t")).drop_nulls("_t").sort("_t")
                       .group_by_dynamic("_t", every=freq).agg(pl.len().alias("n")))
        series = series.head(len(series) - 1)          # the last bin is a part-week and would read as a collapse
        x, y = series["_t"].to_list(), series["n"].to_list()
        fig.add_scatter(x=x, y=y, name=name, mode="lines",
                        line=dict(color=colour, width=2), hovertemplate="%{y:,} messages<extra></extra>")
        if x:                                                   # direct label, so colour is never the only cue
            fig.add_annotation(x=x[-1], y=math.log10(max(y[-1], 1)),   # log axis: annotations sit in log10 units
                               text=f" {name}", showarrow=False, xanchor="left",
                               font=dict(color=colour, size=11))

    fig.update_layout(template="plotly_white", hovermode="x unified",
                      margin=dict(l=70, r=95, t=80, b=40), height=400,
                      title=dict(text=f"Messages per {freq}, before and after each step",
                                 subtitle=dict(text="log scale: the steps differ by more than an order of magnitude"),
                                 font=dict(size=15), y=0.94),
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)))
    fig.update_yaxes(title=None, type="log", gridcolor="#eceae2", zeroline=False,
                     tickformat=",", dtick=1)
    fig.update_xaxes(title=None, gridcolor="#eceae2", showspikes=True, spikemode="across",
                     spikethickness=1, spikecolor="#b5b3a8")
    return fig


def _fingerprint(headlines: pd.Series) -> pd.Series:
    """A stable 64-bit digest per headline, to spot one already published in an earlier year.

    Python's `hash` is randomized per process, so the same inputs produced a different corpus on
    every run; blake2b does not.
    """
    return headlines.map(lambda h: hashlib.blake2b(h.encode(), digest_size=8).digest())


def _manifest() -> dict:
    """The build record kept next to the corpus: which years are done, and whether the corpus is current.

    It is what lets a re-run be cheap. `build_year` skips a year listed in "years"; `build_corpus`
    skips the concatenation while "corpus" holds a record. Building any year sets "corpus" back to
    null, because the merged file no longer reflects its parts.

    A record written under a different RULE is thrown away whole, not migrated: every cached year
    behind it was built by a rule we no longer use, so all of them must be rebuilt.
    """
    if config.CORPUS_MANIFEST.exists():
        m = json.loads(config.CORPUS_MANIFEST.read_text())
        if m.get("rule") == RULE:
            return m
    return {"rule": RULE, "years": {}, "corpus": None}
