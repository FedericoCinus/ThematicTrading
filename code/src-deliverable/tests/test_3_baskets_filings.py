"""3_baskets/filings — counting a theme's words inside 10-K/10-Q text.

Offline: the network halves (`documents`, `edgar.hits`, `edgar.text`) are not exercised here.
What is tested is what decides the basket once the text is in hand.
"""
from __future__ import annotations

import importlib

fil = importlib.import_module("3_baskets.filings_method.basket")

COMPANY = {1: ("aaa", "alpha inc."), 2: ("bbb", "beta corp"), 3: ("ccc", "gamma plc")}


def _tally(rows: dict[int, tuple]) -> dict:
    """{cik: (hits, words, matched words)} -> what `weights` hands to `rank`."""
    return {cik: {"hits": h, "words": w, "terms": set(t)} for cik, (h, w, t) in rows.items()}


def a_phrase_survives_its_separators():
    """`chatgpt-like` is found however the filing spells the gap — EFTS phrase semantics"""
    counts, _ = fil.count("ChatGPT like, chatgpt-like and ChatGPT  like", ["chatgpt-like"])
    return None if counts["chatgpt-like"] == 3 else f"counted {counts['chatgpt-like']}, expected 3"


def a_word_is_not_found_inside_another():
    """`ai` must not match `said`, `chain` or `email`"""
    counts, _ = fil.count("said the chain in an email; ai is different", ["ai"])
    return None if counts["ai"] == 1 else f"counted {counts['ai']}, expected 1"


def the_document_length_is_counted():
    """density needs a denominator, so the word total comes back with the counts"""
    _, words = fil.count("one two three four five", ["one"])
    return None if words == 5 else f"counted {words} words, expected 5"


def density_leads_the_ranking():
    """a company that is ABOUT a theme repeats its words; one brushing past it does not"""
    ranked = fil.rank(_tally({1: (50, 100_000, {"a"}), 2: (40, 10_000, {"a"})}),
                      {1: COMPANY[1], 2: COMPANY[2]}, min_hits=10)
    order = [r["ticker"] for r in ranked]
    return None if order == ["bbb", "aaa"] else f"ranked {order}, expected the denser one first"


def breadth_breaks_a_tie():
    """equal density, and the company that matched more of the theme's words wins"""
    ranked = fil.rank(_tally({1: (20, 10_000, {"a"}), 2: (20, 10_000, {"a", "b"})}),
                      {1: COMPANY[1], 2: COMPANY[2]}, min_hits=10)
    order = [r["ticker"] for r in ranked]
    return None if order == ["bbb", "aaa"] else f"ranked {order}, expected the broader one first"


def a_short_document_cannot_win_on_two_mentions():
    """the floor is why density is usable: 2 hits in 500 words is not a signal"""
    ranked = fil.rank(_tally({1: (2, 500, {"a"}), 2: (269, 53_000, {"a", "b"})}),
                      {1: COMPANY[1], 2: COMPANY[2]}, min_hits=10)
    order = [r["ticker"] for r in ranked]
    return None if order == ["bbb"] else f"ranked {order}, expected the short document dropped"


def a_company_outside_the_universe_is_not_bought():
    """a filing's co-registrant that is not listed cannot enter the basket"""
    ranked = fil.rank(_tally({1: (50, 10_000, {"a"}), 9: (99, 1_000, {"a"})}),
                      {1: COMPANY[1]}, min_hits=10)
    order = [r["ticker"] for r in ranked]
    return None if order == ["aaa"] else f"ranked {order}, expected only the listed one"


def top_n_cuts_the_tail():
    """a theme holds at most top_n names"""
    ranked = fil.rank(_tally({1: (30, 1_000, {"a"}), 2: (20, 1_000, {"a"}),
                                3: (10, 1_000, {"a"})}), COMPANY, min_hits=10, top_n=2)
    return None if len(ranked) == 2 else f"kept {len(ranked)}, expected 2"


CHECKS = [a_phrase_survives_its_separators, a_word_is_not_found_inside_another,
          the_document_length_is_counted, density_leads_the_ranking, breadth_breaks_a_tie,
          a_short_document_cannot_win_on_two_mentions, a_company_outside_the_universe_is_not_bought,
          top_n_cuts_the_tail]
