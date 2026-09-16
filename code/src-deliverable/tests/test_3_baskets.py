"""3_baskets — turning a theme's words into the companies to hold."""
from __future__ import annotations

import importlib
import os

import polars as pl

bas = importlib.import_module("3_baskets.uniform_method.basket")
uni = importlib.import_module("3_baskets.universe")

REGISTER = [("aapl", "apple inc.", 320193),
            ("aple", "apple hospitality reit, inc.", 1418121),
            ("aapi", "apple isports group, inc.", 1134982),
            ("googl", "alphabet inc.", 1652044),
            ("goog", "alphabet inc.", 1652044),
            ("googm", "alphabet inc.", 1652044),
            ("msft", "microsoft corp", 789019),
            ("tri", "thomson reuters corp /can/", 1075124),
            ("tmsof", "thomson reuters corp /can/", 1075124)]

NAMES = pl.DataFrame([{"ticker": t, "company": c, "cik": k} for t, c, k in REGISTER]) \
          .with_columns(stem=pl.col("company").map_elements(uni.stem, return_dtype=pl.String))


def _themes(vocab: list[str], theme: str = "genai") -> pl.DataFrame:
    return pl.DataFrame({"theme": [theme], "week": ["2023-01-02"], "vocab": [vocab]})


def stem_stops_at_a_real_word():
    """a register name loses its corporate tail and nothing else"""
    wrong = {n: uni.stem(n) for n in ("apple inc.", "apple hospitality reit, inc.",
                                      "thomson reuters corp /can/")
             if uni.stem(n) not in ("apple", "apple hospitality reit", "thomson reuters")}
    return f"stemmed wrong: {wrong}" if wrong else None


def a_company_is_not_a_substring():
    """the register name must BE the company, so `apple` is not Apple Hospitality REIT"""
    found = bas.matches(["apple"], NAMES)
    return f"matched {found}" if found != ["aapl"] else None


def share_classes_take_one_slot():
    """four tickers on one cik are one company, not four"""
    found = bas.matches(["alphabet"], NAMES)
    return f"matched {found}" if found != ["googl"] else None


def weights_are_equal_and_sum_to_one():
    """every name in a theme carries the same weight, and the theme sums to 1"""
    w, _ = bas.weights(_themes(["apple", "alphabet", "microsoft"]), NAMES,
                       llm_model=None, report=False)
    if len(w) != 3:
        return f"{len(w)} names, expected 3"
    if abs(w["weight"].sum() - 1.0) > 1e-9 or w["weight"].n_unique() != 1:
        return f"weights {w['weight'].to_list()}"
    return None


def top_n_cuts_the_tail():
    """a theme holds at most top_n names, taken in vocabulary order"""
    w, _ = bas.weights(_themes(["apple", "alphabet", "microsoft"]), NAMES,
                       llm_model=None, top_n=2, report=False)
    return f"held {w['ticker'].to_list()}" if w["ticker"].to_list() != ["aapl", "googl"] else None


def diagnostics_name_the_word_responsible():
    """the diagnostics say, per ticker, which of the theme's words put it there"""
    _, d = bas.weights(_themes(["microsoft"]), NAMES, llm_model=None, report=False)
    got = dict(zip(d["ticker"], d["word"]))
    return f"got {got}" if got != {"msft": "microsoft"} else None


def a_brand_is_not_a_registered_name():
    """untranslated, the words a theme is made of find nothing — the reason the translation exists"""
    w, _ = bas.weights(_themes(["google", "iphone", "chatgpt"]), NAMES,
                       llm_model=None, report=False)
    return f"matched {w['ticker'].to_list()}" if len(w) else None


def the_translation_reaches_the_companies():
    """the case this stage exists for: genAI's words become Alphabet, Apple and Microsoft"""
    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not set"
    w, _ = bas.weights(_themes(["google", "iphone", "microsoft-backed chatgpt", "openai"]),
                       NAMES, llm_model="gpt-4o", report=False)
    held = set(w["ticker"].to_list())
    missing = {"googl", "aapl", "msft"} - held
    return f"missing {sorted(missing)}; held {sorted(held)}" if missing else None


def the_wide_list_is_the_one_read():
    """given both lists the basket takes vocab_wide, the one the common names survive in"""
    themes = pl.DataFrame({"theme": ["genai"], "week": ["2023-01-02"],
                           "vocab": [["chatgpt"]], "vocab_wide": [["chatgpt", "microsoft"]]})
    w, _ = bas.weights(themes, NAMES, llm_model=None, report=False)
    return f"held {w['ticker'].to_list()}" if w["ticker"].to_list() != ["msft"] else None


CHECKS = [stem_stops_at_a_real_word, the_wide_list_is_the_one_read, a_company_is_not_a_substring, share_classes_take_one_slot,
          weights_are_equal_and_sum_to_one, top_n_cuts_the_tail,
          diagnostics_name_the_word_responsible, a_brand_is_not_a_registered_name,
          the_translation_reaches_the_companies]
