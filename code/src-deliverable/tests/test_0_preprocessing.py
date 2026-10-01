"""0_preprocessing — every published headline, in the words the wire published it, with its true date."""
from __future__ import annotations

import importlib

import polars as pl
import pyarrow.parquet as pq

import config

pub = importlib.import_module("0_preprocessing.first_publication")

SAMPLE = 200_000


def one_row_per_headline_per_day():
    """the steps collapse every message of one story, tomorrow's verbatim repeat included"""
    messages = pl.DataFrame({
        "Headline":    ["Alpha rises", "Alpha rises", "Alpha rises", "Alpha rises", "Gamma flat"],
        "CaptureTime": ["2024-01-01 09:00:00.0", "2024-01-01 09:00:00.4",   # one publication, two messages
                        "2024-01-01 14:00:00.0",                            # re-transmission, same day
                        "2024-01-02 09:00:00.0",                            # published again tomorrow
                        "2024-01-06 08:00:00.0"],
        "WireName":    ["BN", "BN", "BN", "BN", "PRN"],
        "Event":       ["ADD_1STPASS", "ADD_STORY", "UPDATE_ATTRIBUTE", "ADD_STORY", "ADD_STORY"],
    })
    out = pub.first_occurrence(pub.keep_published(pub.keep_wires(messages, report=False), report=False), report=False)
    if out["Headline"].to_list() != ["Alpha rises"]:
        return f"kept {out['Headline'].to_list()}, expected one row for Alpha"
    when = str(out["CaptureTime"][0])
    return None if when.startswith("2024-01-01 09:00:00") else f"dated {when}, expected the first message"


def headlines_are_untouched():
    """the corpus keeps the words the wire published, prefixes and all"""
    rows = next(pq.ParquetFile(config.CORPUS).iter_batches(batch_size=SAMPLE)).to_pylist()
    with_prefix = sum(":" in r["Headline"][:32] for r in rows)
    return None if with_prefix else "no headline carries a prefix — they look stripped, which loses the speaker"


def corpus_shape():
    """the corpus is [Headline, date] and sorted by date"""
    f = pq.ParquetFile(config.CORPUS)
    if f.schema_arrow.names != ["Headline", "date"]:
        return f"columns are {f.schema_arrow.names} — built by an older rule, rebuild needed"
    head = next(f.iter_batches(batch_size=SAMPLE)).to_pandas()
    return None if head.date.is_monotonic_increasing else "dates are not ascending"


def each_title_appears_once():
    """a title published verbatim again is not a new story, so it is in the corpus exactly once"""
    head = next(pq.ParquetFile(config.CORPUS).iter_batches(batch_size=SAMPLE)).to_pandas()
    repeated = head.Headline.duplicated().sum()
    return f"{repeated} titles appear more than once" if repeated else None


def wire_selection_is_honoured():
    """preprocessing.wires picks the wires: a list keeps those, `all` keeps every one"""
    messages = pl.DataFrame({
        "Headline":    ["Bloomberg says", "Web says", "Blog says", "Issuer says", "State says",
                        "Machine says", "Release says", None],
        "CaptureTime": ["2024-01-01 09:00:00.0"] * 8,
        "WireName":    ["BN", "NS1", "BLG", "CO1", "GO9", "EDG", "PRN", "BN"],
        "Event":       ["ADD_STORY"] * 8,
    })
    press = ["Bloomberg says", "Web says", "Blog says"]
    cases = {
        "an explicit list": (["BN", "NS1"], ["Bloomberg says", "Web says"]),
        "every wire (None)": (None, [h for h in messages["Headline"] if h]),
        "the Bloomberg three": (pub.BLOOMBERG_WIRES, ["Bloomberg says"]),
        "the newsrooms": ("press", press),     # no company, government, machine or press release
    }
    for what, (keep, expected) in cases.items():
        got = pub.keep_wires(messages, report=False, keep=keep)["Headline"].to_list()
        if got != expected:
            return f"{what}: kept {got}, expected {expected}"
    return None                                  # the None headline is dropped by every selection


def press_excludes_only_what_no_newsroom_wrote():
    """`press` is defined by exclusion, so a masthead it has never seen is kept, not dropped"""
    unseen = ["ZZZ", "NYT", "NS9", "FM4", "WE7", "BBC"]      # codes the rule must never have to know
    if not all(pub.is_press(c) for c in unseen):
        return f"dropped one of {unseen} — an allow-list crept in"
    for code in ("EDG", "PRN", "BUS", "PZM", "DBF", "CO1", "CO8", "GO1", "GO9"):
        if pub.is_press(code):
            return f"kept {code}, which no newsroom wrote"
    if not (pub.is_press("COMEX") and pub.is_press("GOAL")):  # CO/GO prefix, but not CO<digit>
        return "the CO/GO prefix rule is eating codes that merely start with those letters"
    return None


def wire_selection_names_the_corpus():
    """two wire selections write two corpora, so neither silently overwrites the other"""
    if config.WIRE_SUFFIX != config._wire_suffix(config.WIRES):
        return "config.WIRE_SUFFIX does not match the configured selection"
    if config._wire_suffix("bloomberg") or config._wire_suffix(sorted(config.BLOOMBERG_WIRES)):
        return "the default selection must keep the original file names, so nothing rebuilds"
    distinct = {config._wire_suffix(w) for w in ("bloomberg", "press", "all", ["BN", "NS1"])}
    return None if len(distinct) == 4 else f"selections collide on {distinct}"


CHECKS = [one_row_per_headline_per_day, headlines_are_untouched, corpus_shape, each_title_appears_once,
          wire_selection_is_honoured, press_excludes_only_what_no_newsroom_wrote,
          wire_selection_names_the_corpus]
