"""Stage 4 — turn a theme into the companies to hold, using the theme's own words.

News names brands, a stock register names legal entities, and they are rarely the same word: no
listed company is called `google`, `iphone` or `chatgpt`. So the stage is two steps — translate each
of the theme's words into the company it belongs to, then look that company up in the register.

    weights(themes, universe) -> weights [theme, ticker, weight] · diagnostics
"""
from __future__ import annotations

import json
import re

import polars as pl

import config
import llm
from importlib import import_module

universe = import_module("3_baskets.universe")
progress = import_module("progress")

# ======================================================================================
# THE MAPPING — a theme's words in, a portfolio out
#
#   themes.vocab_wide  ──▶  issuers()  ──▶  matches()  ──▶  one per cik  ──▶  top_n, 1/n each
#                            a model         the register      share classes      weights
#                            word→company     name→ticker       collapse
#
#     word                      issuers()            matches()          held
#     ───────────────────────   ──────────────────   ────────────────   ────
#     chatgpt                →  openai            →  — not listed
#     iphone                 →  apple             →  aapl            →  aapl
#     google                 →  alphabet          →  googl goog        googl    same cik,
#                                                    googm googn                one holding
#     microsoft-backed …     →  openai, microsoft →  msft            →  msft
#     openai                 →  openai            →  — not listed
#                                                                       ────
#                                                          equal weight  3 names at 1/3
#
#   why a model       no listed company is called `google`, `iphone`, `chatgpt` or `openai`. It
#                     translates what a word BELONGS TO — never who would benefit from it, which
#                     would be a claim about the future                      (ISSUERS, below)
#   why exact match   the register name must BE the company
#                       apple  →  apple inc.  ✓   apple hospitality reit · apple isports  ✗
#   why vocab_wide    `google` and `iphone` are in the theme but not in the list the monitor
#                     counts. On `vocab` this same basket is msft alone — and reaches even that
#                     only because the model reads a company out of a bigram.
#   order             vocabulary order: how often the word shares a headline with the anchor
#
#   Names are normalized on both sides; only explicit aliases extend exact matching.
#   Distinct CIKs with the same normalized name are ambiguous and never bought automatically.
#   LEFT OVER   today's registrants only, and the theme's entry date is never enforced
#                                                              → FUTUREWORK_uniform.md
# ======================================================================================
# ======================================================================================
# TRANSLATE — the brand a headline uses, into the company a register lists
# ======================================================================================
# Lexical only, like gate 4 and for the same reason: what a word belongs to is a fact about the
# language, what a word will be worth is a fact about the future. The last line forbids the second,
# and the answers show it holds — asked about `chatgpt` the model says OpenAI, never Microsoft.
#
# Two things measuring forced into the wording:
#   more than one   `microsoft-backed chatgpt` names two companies. Asked for one the model returns
#                   OpenAI, and MSFT — the only listed name in the entire theme — is lost.
#   nothing         a word naming no company must come back empty, which makes this a filter too:
#                   of 29 raw partners 20 returned nothing, `parmy olson` (a columnist) among them.
#
# Residual leakage: the model knows today's owner, so a brand acquired after the theme's date is
# attributed to the company that holds it now. Recorded in FUTUREWORK_uniform.md, not fixed.
ISSUERS = """\
You are shown words taken from news headlines. Each may name a product, a brand, a service or an
organisation. Name the companies it belongs to.

  belongs to      the maker of a product, the owner of a brand, the operator of a service, or the
                  organisation itself. Name the company as a stock register lists it — the maker
                  of the iPhone is Apple, the operator of Google Search is Alphabet.

  more than one   when the word names more than one company: `microsoft-backed chatgpt` names both
                  the product's owner and the company the word itself mentions.

  nothing         for a word that names no product, brand, service or organisation.

Do NOT name a company because it would benefit from the subject, compete with it, supply it or be
exposed to it. Name only the companies the word itself names or belongs to.
"""

def issuers(vocab: list[str], model: str | None) -> list[tuple[str, str]]:
    """Which company each of the theme's words names.

    INPUT   vocab   the theme's words, most co-occurring first
            model   which model translates; None passes each word through as its own company
            (reads OPENAI_API_KEY)
    OUTPUT  [(word, company)] — one pair per company named, vocabulary order kept

    A word may name no company and is then absent, or several and is then repeated. Without a model
    the step is the identity, which still works for a word that is itself a registered name
    (`microsoft`) and never for one that is not (`google`).
    """
    if not model or not vocab:
        return [(w, w) for w in vocab]

    import os

    from pydantic import BaseModel

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("the basket needs OPENAI_API_KEY in code/.env — "
                           "set basket.llm_model to null in config.yaml to run without it")

    # No docstrings on these: pydantic sends them to the model as schema descriptions, so anything
    # written here is prompt text. `companies` is a list because the singular loses MSFT (see above).
    class Owner(BaseModel):
        word: str
        companies: list[str]

    class Owners(BaseModel):
        owners: list[Owner]

    def batch(words):
        answer = llm.parse(
            stage="issuers", model=model, response_format=Owners, max_completion_tokens=4096,
            messages=[{"role": "system", "content": ISSUERS},
                      {"role": "user", "content": "Words: " + ", ".join(words)}])
        return [o for o in answer.owners if o.word.lower() in words]

    owners = llm.batches(vocab, batch)

    rank = {w: i for i, w in enumerate(vocab)}
    pairs = [(o.word.lower(), c.lower())
             for o in owners for c in o.companies
             if c.strip().lower() not in ("", "nothing", "none", "n/a")]
    return sorted(pairs, key=lambda pair: rank.get(pair[0], len(vocab)))


ALIASES = {"jpmorgan": "jpmorgan chase"}


def resolve(companies: list[str], names: pl.DataFrame) -> list[dict]:
    """Exact normalized name first, then a one-hop explicit alias, never fuzzy matching.

    Multiple tickers sharing a CIK are one issuer (first register ticker retained).
    Multiple distinct CIKs are ambiguous: do not pick any ticker or try an alias.
    Rebuild stems from company when available, including for old cached universes.
    """
    index = {}
    for row in names.to_dicts():
        key = universe.stem(row.get("company") or row["stem"])
        if key:
            index.setdefault(key, []).append(row)
    resolved = []
    for company in companies:
        normalized = universe.stem(company)
        key, method = normalized, "exact"
        candidates = index.get(key, [])
        if not candidates and key in ALIASES:
            key, method = universe.stem(ALIASES[key]), "alias"
            candidates = index.get(key, [])
        ciks = {r["cik"] for r in candidates}
        status = "unmatched" if not candidates else "matched" if len(ciks) == 1 and None not in ciks else "ambiguous"
        chosen = candidates[0] if status == "matched" else None
        resolved.append({"input_company": company, "normalized_company": normalized,
                         "lookup_company": key, "match_method": method, "match_status": status,
                         "candidate_tickers": [r["ticker"] for r in candidates],
                         "ticker": chosen["ticker"] if chosen else None,
                         "company": key, "cik": chosen["cik"] if chosen else None})
    return resolved


def matches(companies: list[str], names: pl.DataFrame) -> list[str]:
    """Which of these companies are listed, one ticker each.

    INPUT   companies   company names, most relevant first — the output of `issuers`
            names       the universe, [ticker, stem, cik]
    OUTPUT  one ticker per company found, in the order asked

    Both sides are normalized; explicit aliases are tried only if there is no exact match.
    Distinct CIKs are rejected with a warning; share classes of one CIK take one slot.
    """
    found, seen = [], set()
    import warnings
    for row in resolve(companies, names):
        if row["match_status"] == "ambiguous":
            warnings.warn(f"ambiguous company {row['input_company']!r}: {row['candidate_tickers']}",
                          UserWarning, stacklevel=2)
        if row["match_status"] == "matched" and row["cik"] not in seen:
            seen.add(row["cik"])
            found.append(row["ticker"])
    return found


def weights(themes: pl.DataFrame, names: pl.DataFrame | None = None, *, top_n: int = 10,
            llm_model: str | None = None, report: bool = True, **_) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The portfolio: which tickers to hold for each theme, and in what proportion.

    INPUT   themes      [theme, week, vocab, vocab_wide] from the detector; `vocab_wide` is read
                        when present, because it is the list that still holds the buyable names
            names       the universe from 3_baskets/universe.py; loaded when not given
            top_n       how many names a theme holds, in vocabulary order
            llm_model   which model translates word into company; None skips that step
    OUTPUT  weights       [theme, ticker, weight] — equal weight, summing to 1 per theme
            diagnostics   held matches plus unmatched/ambiguous proposals (ticker=null),
                          with original/normalized names, match method and candidate tickers
    """
    names = universe.load() if names is None else names
    # `vocab_wide` keeps the names too common for the monitor to count by, which are exactly the
    # large listed ones this stage can buy. A themes frame without that column still works.
    column = "vocab_wide" if "vocab_wide" in themes.columns else "vocab"
    rows, notes = [], []
    for theme, vocab in progress.each(list(zip(themes["theme"], themes[column])),
                                      "word -> company", on=report, unit="theme"):
        pairs = issuers(list(vocab), llm_model)
        resolved = resolve([company for _, company in pairs], names)
        held, seen = [], set()
        for (word, _), match in zip(pairs, resolved):
            note = {k: v for k, v in dict(match, theme=theme, word=word).items() if k in _NOTES}
            if match["match_status"] != "matched":
                notes.append(note)
                if report and match["match_status"] == "ambiguous":
                    print(f"   AMBIGUOUS {match['input_company']}: {match['candidate_tickers']}")
            elif match["cik"] not in seen and len(held) < top_n:
                seen.add(match["cik"])
                held.append(match["ticker"])
                notes.append(note)
        for ticker in held:
            rows.append({"theme": theme, "ticker": ticker, "weight": 1 / len(held)})
        if report:
            print(f"   {theme[:34]:36} {len(held):>3} names" + (f"   {', '.join(held[:8])}" if held else "   EMPTY"))
    return pl.DataFrame(rows, schema=_WEIGHTS), pl.DataFrame(notes, schema=_NOTES)


_WEIGHTS = {"theme": pl.String, "ticker": pl.String, "weight": pl.Float64}
_NOTES = {"theme": pl.String, "ticker": pl.String, "company": pl.String, "word": pl.String,
          "input_company": pl.String, "normalized_company": pl.String, "lookup_company": pl.String,
          "match_method": pl.String, "match_status": pl.String, "candidate_tickers": pl.List(pl.String)}
