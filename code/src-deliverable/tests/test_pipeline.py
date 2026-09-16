"""pipeline — the worked theme in pipeline.py, held true.

Every check here asserts one claim of the ONE THEME ALL THE WAY THROUGH banner. The stages are
already tested on their own; what is tested here is that they still compose into that story.

The four stages are run once and shared, because the vocabulary pass reads the whole corpus.
"""
from __future__ import annotations

import importlib
from datetime import datetime

import polars as pl

import config

det = importlib.import_module("1_detectors.graph_anomaly.detect")
mon = importlib.import_module("2_monitors.monitor")
bas = importlib.import_module("3_baskets.uniform_method.basket")

ANCHOR, ASOF = "chatgpt", datetime(2023, 1, 2)
_run: dict = {}


def _genai() -> dict:
    """Detect's vocabulary for `chatgpt`, both monitors, and the basket — computed once."""
    if _run:
        return _run
    corpus = pl.read_parquet(config.CORPUS)
    p = config.params("detect")
    vocab, wide = det.vocabulary(corpus, ANCHOR, ASOF, p["llm_model"],
                                 p["theme_bag_of_words_size"], p["max_doc_freq"])

    def frame(words):
        return pl.DataFrame([{"theme": "genai", "week": ASOF, "vocab": words}],
                            schema={"theme": pl.String, "week": pl.Datetime,
                                    "vocab": pl.List(pl.String)})

    m = {k: v for k, v in config.params("monitor").items() if k != "method"}
    entry = {}
    for name, words in (("vocab", vocab), ("vocab_wide", wide)):
        signal, _ = mon.monitor(corpus, frame(words), ASOF, report=False, **m)
        held = signal.filter("position")
        entry[name] = None if held.is_empty() else held["week"].min()

    themes = frame(vocab).with_columns(vocab_wide=pl.Series([wide], dtype=pl.List(pl.String)))
    b = {k: v for k, v in config.params("basket").items() if k != "method"}
    weights, notes = bas.weights(themes, report=False, **b)

    _run.update(vocab=vocab, wide=wide, entry=entry, weights=weights, notes=notes)
    return _run


def the_vocabulary_splits_on_frequency_alone():
    """the two lists differ only by words too common to count a theme by — here google and iphone"""
    r = _genai()
    if not set(r["vocab"]) <= set(r["wide"]):
        return f"vocab is not inside vocab_wide: {sorted(set(r['vocab']) - set(r['wide']))}"
    extra = [w for w in r["wide"] if w not in r["vocab"]]
    missing = {"google", "iphone"} - set(extra)
    return f"the cap removed {extra}, expected google and iphone among them" if missing else None


def the_model_removed_the_bylines():
    """ENTITIES drops what names nothing: the columnist's name and the wire's own are gone"""
    r = _genai()
    debris = [w for w in ("olson", "parmy olson", "reuters", "way", "peril", "asia league")
              if w in r["wide"]]
    return f"still in the vocabulary: {debris}" if debris else None


def the_theme_is_entered_on_the_narrow_list_only():
    """in on `vocab` at 2023-01-02, never in on `vocab_wide` — the reason there are two lists"""
    r = _genai()
    if r["entry"]["vocab"] != ASOF:
        return f"entered on {r['entry']['vocab']}, expected 2023-01-02"
    if r["entry"]["vocab_wide"] is not None:
        return f"entered on vocab_wide too, on {r['entry']['vocab_wide']}"
    return None


def the_basket_is_the_three_listed_names():
    """aapl, googl and msft, a third each — one holding per company, Alphabet's classes merged"""
    r = _genai()
    held = r["weights"]["ticker"].to_list()
    if set(held) != {"aapl", "googl", "msft"}:
        return f"held {sorted(held)}"
    if len(held) != len(set(held)) or abs(r["weights"]["weight"].sum() - 1.0) > 1e-9:
        return f"weights {r['weights']['weight'].to_list()}"
    return None


def openai_is_in_the_theme_and_not_in_the_basket():
    """the centre of the theme cannot be bought, and the stage says so by leaving it out"""
    r = _genai()
    if "openai" not in r["wide"]:
        return "openai is not even in the vocabulary, so this no longer tests anything"
    stray = [t for t, c in zip(r["notes"]["ticker"], r["notes"]["company"]) if c == "openai"]
    return f"openai reached the basket as {stray}" if stray else None


def microsoft_arrives_through_the_bigram():
    """msft comes from `microsoft-backed chatgpt`, never from `microsoft`, which is too common"""
    r = _genai()
    if "microsoft" in r["wide"]:
        return "`microsoft` is in the vocabulary, so the bigram is no longer what carries msft"
    word = dict(zip(r["notes"]["ticker"], r["notes"]["word"])).get("msft")
    return None if word and "microsoft" in word and word != "microsoft" else f"msft came from {word!r}"


CHECKS = [the_vocabulary_splits_on_frequency_alone, the_model_removed_the_bylines,
          the_theme_is_entered_on_the_narrow_list_only, the_basket_is_the_three_listed_names,
          openai_is_in_the_theme_and_not_in_the_basket, microsoft_arrives_through_the_bigram]
