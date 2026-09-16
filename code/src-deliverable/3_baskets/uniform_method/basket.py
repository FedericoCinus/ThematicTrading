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
#   LEFT OVER   the model must spell a company as the register does: `meta` where the register
#               says `meta platforms` finds nothing, silently, like every miss here
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

    from openai import OpenAI
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

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None)
    answer = client.beta.chat.completions.parse(
        model=model, temperature=0, response_format=Owners,
        messages=[{"role": "system", "content": ISSUERS},
                  {"role": "user", "content": "Words: " + ", ".join(vocab)}])

    rank = {w: i for i, w in enumerate(vocab)}
    pairs = [(o.word.lower(), c.lower())
             for o in answer.choices[0].message.parsed.owners for c in o.companies
             if c.strip().lower() not in ("", "nothing", "none", "n/a")]
    return sorted(pairs, key=lambda pair: rank.get(pair[0], len(vocab)))


def matches(companies: list[str], names: pl.DataFrame) -> list[str]:
    """Which of these companies are listed, one ticker each.

    INPUT   companies   company names, most relevant first — the output of `issuers`
            names       the universe, [ticker, stem, cik]
    OUTPUT  one ticker per company found, in the order asked

    Matched against `stem`, so the name must BE the company: `apple` finds Apple Inc. and not Apple
    Hospitality REIT. Deduplicated by cik, so Alphabet's four share classes take one slot, not four.
    """
    found, seen = [], set()
    for company in companies:
        for ticker, cik in names.filter(pl.col("stem") == company).select("ticker", "cik").iter_rows():
            if cik not in seen:
                seen.add(cik)
                found.append(ticker)
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
            diagnostics   [theme, ticker, company, word] — which word put each ticker in the basket
    """
    names = universe.load() if names is None else names
    stem = dict(zip(names["ticker"], names["stem"]))
    # `vocab_wide` keeps the names too common for the monitor to count by, which are exactly the
    # large listed ones this stage can buy. A themes frame without that column still works.
    column = "vocab_wide" if "vocab_wide" in themes.columns else "vocab"
    rows, notes = [], []
    for theme, vocab in progress.each(list(zip(themes["theme"], themes[column])),
                                      "word -> company", on=report, unit="theme"):
        pairs = issuers(list(vocab), llm_model)
        held = matches([company for _, company in pairs], names)[:top_n]
        said = {}                                   # company -> the first word that named it
        for word, company in pairs:
            said.setdefault(company, word)
        for ticker in held:
            rows.append({"theme": theme, "ticker": ticker, "weight": 1 / len(held)})
            notes.append({"theme": theme, "ticker": ticker, "company": stem[ticker],
                          "word": said.get(stem[ticker], "")})
        if report:
            print(f"   {theme[:34]:36} {len(held):>3} names" + (f"   {', '.join(held[:8])}" if held else "   EMPTY"))
    return pl.DataFrame(rows, schema=_WEIGHTS), pl.DataFrame(notes, schema=_NOTES)


_WEIGHTS = {"theme": pl.String, "ticker": pl.String, "weight": pl.Float64}
_NOTES = {"theme": pl.String, "ticker": pl.String, "company": pl.String, "word": pl.String}
