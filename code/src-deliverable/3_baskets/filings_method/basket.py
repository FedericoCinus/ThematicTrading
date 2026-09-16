"""Stage 4, second method — the companies whose own filings talk about the theme.

    weights(themes) -> weights [theme, ticker, weight] · diagnostics

Same contract as 3_baskets/uniform_method/basket.py, different evidence: uniform asks whether a company is
NAMED by the theme's words, this asks whether a company WRITES them.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from importlib import import_module

import polars as pl

edgar = import_module("3_baskets.filings_method.edgar")
universe = import_module("3_baskets.universe")

# ======================================================================================
# THE MAPPING — a theme's words in, a portfolio out
#
#   IN    themes       [theme, week, vocab, vocab_wide]     1_detectors
#   IN    universe     [ticker, company, cik, stem]         3_baskets/universe.py
#   OUT   weights      [theme, ticker, weight]
#   OUT   diagnostics  [theme, ticker, company, hits, words, per_1k, n_terms]
#
#   HOW
#     vocab_wide  ──▶  edgar.hits()  ──▶  edgar.text()  ──▶  count()  ──▶  rank  ──▶  top_n, 1/n
#                      which filings      the document       occurrences   density
#                      said each word     itself             per company
#
#   window       [theme date − lookback_days, theme date]   on the FILING date, never today
#   shortlist    a word with more than max_hits_per_term filings is too common to shortlist on —
#                skipped there, still counted inside the documents the other words found
#   rank         per_1k = hits / words × 1000, and n_terms breaks ties
#                below min_hits total, a density is noise: a short document trivially scores high
#
#   WHY THIS AND NOT uniform.py
#     a name reaches a filing through its PEOPLE, a technical term through its BUSINESS
#     measured at 2023-01-02:  chatgpt 0 filings · openai 11, every one of them naming OpenAI in
#     a director's biography because Sam Altman sat on those boards · generative ai 9, of which
#     Innodata is a real link.  At 2022-09-30, quantum computing: 374.
#     → right and sometimes late, where names are early and sometimes wrong
#
#   COST      one EFTS request per word, then one download per shortlisted document.
#             max_docs caps a theme; the text cache makes the second run free.
#   NEEDS     SEC_USER_AGENT in code/.env
#
#   MEASURED, and it is the warning on this method — genAI at 2023-01-02, one-year window:
#       chatgpt                     0 filings
#       iphone                    146 filings,  89 in the universe   ← built the whole basket
#       google                  1,701 filings — too common to shortlist on
#       openai                      2 filings,   1 in the universe
#       microsoft-backed chatgpt    0 filings
#     → aapl · kdozf · zdge · bkng · ipm · ai · drio · trip · apps · logi
#       mobile-app companies that write about iPhones. An iPhone basket, not a genAI one.
#
#     The method did its job; the vocabulary carried a word that is not the theme. Note how much
#     worse that is here than in uniform.py: there `iphone` cost one name of three, because a
#     word buys one company. Here it bought ten, because ranking by how much a company WRITES a
#     word hands the basket to whoever writes it most.   → FUTUREWORK_filings.md
# ======================================================================================


def count(text: str, vocab: list[str]) -> tuple[dict[str, int], int]:
    """How often each of the theme's words appears in one filing.

    INPUT   text    the filing's text
            vocab   the theme's words
    OUTPUT  ({word: occurrences}, total words in the document)

    A word is matched the way EFTS matches a phrase: its tokens in order, separated by up to
    three non-word characters, so `chatgpt-like` also finds `ChatGPT like` and `ChatGPT‑Like`.
    """
    low = text.lower()
    return ({word: len(_pattern(word).findall(low)) for word in vocab},
            sum(1 for _ in re.finditer(r"\S+", low)))


def documents(vocab: list[str], until: str, names: pl.DataFrame, *, since: str | None = None,
              forms: str = "10-K,10-Q", max_hits_per_term: int = 1000,
              report: bool = True) -> tuple[dict, list[str]]:
    """Which filings in the universe said any of the theme's words, by a date.

    INPUT   vocab              the theme's words
            until              the last filing date read — the theme's date, never today
            names              the universe, for the cik join
            since              the first; defaults to until − a year
            forms              which form types to search
            max_hits_per_term  above this a word is too common to shortlist on
            report             print one line per word
    OUTPUT  ({(cik, adsh, doc): {matched words}}, the words skipped as too common)
    """
    since = since or (date.fromisoformat(until) - timedelta(days=365)).isoformat()
    listed = set(names["cik"].to_list())
    session, found, generic = edgar._session(), {}, []

    for word in vocab:
        total = edgar.how_many_filings(word, until, since, forms, session)
        if total > max_hits_per_term:
            generic.append(word)
            if report:
                print(f"      {word!r:28} {total:>6} filings — too common to shortlist on")
            continue
        inside = 0
        for hit in edgar.hits(word, until, since, forms, session):
            for cik in hit["ciks"]:
                if cik in listed:
                    found.setdefault((cik, hit["adsh"], hit["doc"]), set()).add(word)
                    inside += 1
        if report:
            print(f"      {word!r:28} {total:>6} filings, {inside} in the universe")
    return found, generic


def rank(tally: dict[int, dict], company: dict[int, tuple[str, str]], *,
         min_hits: int = 10, top_n: int = 10) -> list[dict]:
    """Which counted companies make the basket, and in what order.

    INPUT   tally      {cik: {hits, words, terms}} — one entry per company counted
            company    {cik: (ticker, name)} from the universe
            min_hits   below this many occurrences a density is noise, not a signal
            top_n      how many to keep
    OUTPUT  [{ticker, company, hits, words, per_1k, n_terms}] — best first

    Density leads: a company that is ABOUT a theme repeats a few of its words, while one that
    brushes past it touches many words once. The floor exists because density is a ratio — two
    mentions in a 1,000-word document would otherwise outrank 269 in 53,000.
    """
    return sorted(
        ({"ticker": company[cik][0], "company": company[cik][1],
          "hits": v["hits"], "words": v["words"], "n_terms": len(v["terms"]),
          "per_1k": v["hits"] / v["words"] * 1000 if v["words"] else 0.0}
         for cik, v in tally.items() if cik in company and v["hits"] >= min_hits),
        key=lambda r: (-r["per_1k"], -r["n_terms"]))[:top_n]


def weights(themes: pl.DataFrame, names: pl.DataFrame | None = None, *, top_n: int = 10,
            forms: str = "10-K,10-Q", lookback_days: int = 365, max_hits_per_term: int = 1000,
            max_docs: int = 200, min_hits: int = 10, asof: dict | None = None,
            report: bool = True, **_) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The portfolio: the companies whose filings use the theme's words most densely.

    INPUT   themes             [theme, week, vocab, vocab_wide]
            names              the universe; loaded when not given
            top_n              how many names a theme holds
            forms              which form types to search
            lookback_days      how far back from the theme's date filings are read
            max_hits_per_term  a word in more filings than this is skipped for shortlisting
            max_docs           ceiling on documents downloaded per theme
            min_hits           below this many occurrences a density is noise
            asof               {theme: date} — defaults to the theme's emergence week, which is
                               earlier than its entry date and therefore never reads more
            report             print progress
    OUTPUT  weights       [theme, ticker, weight] — equal weight, summing to 1 per theme
            diagnostics   [theme, ticker, company, hits, words, per_1k, n_terms]
    """
    names = universe.load() if names is None else names
    column = "vocab_wide" if "vocab_wide" in themes.columns else "vocab"
    company = {cik: (t, c) for t, c, cik in zip(names["ticker"], names["company"], names["cik"])}
    session = edgar._session()
    rows, notes = [], []

    for theme, week, vocab in zip(themes["theme"], themes["week"], themes[column]):
        until = str((asof or {}).get(theme, week))[:10]
        since = (date.fromisoformat(until) - timedelta(days=lookback_days)).isoformat()
        vocab = list(vocab)
        found, _ = documents(vocab, until, names, since=since, forms=forms,
                             max_hits_per_term=max_hits_per_term, report=report)

        tally: dict[int, dict] = {}
        for cik, adsh, doc in sorted(found)[:max_docs]:
            counts, words = count(edgar.text(cik, adsh, doc, session), vocab)
            seen = tally.setdefault(cik, {"hits": 0, "words": 0, "terms": set()})
            seen["hits"] += sum(counts.values())
            seen["words"] += words
            seen["terms"] |= {w for w, n in counts.items() if n}

        ranked = rank(tally, company, min_hits=min_hits, top_n=top_n)

        for r in ranked:
            rows.append({"theme": theme, "ticker": r["ticker"], "weight": 1 / len(ranked)})
            notes.append({"theme": theme} | {k: r[k] for k in
                                             ("ticker", "company", "hits", "words", "per_1k", "n_terms")})
        if report:
            held = ", ".join(r["ticker"] for r in ranked[:8])
            print(f"   {theme[:34]:36} {len(ranked):>3} names   {held or 'EMPTY'}")

    return pl.DataFrame(rows, schema=_WEIGHTS), pl.DataFrame(notes, schema=_NOTES)


_PATTERNS: dict[str, re.Pattern] = {}


def _pattern(word: str) -> re.Pattern:
    if word not in _PATTERNS:
        tokens = [t for t in re.split(r"[\s\-‐-―]+", word.lower()) if t]
        _PATTERNS[word] = re.compile(r"(?<!\w)" + r"\W{1,3}".join(map(re.escape, tokens)) + r"(?!\w)")
    return _PATTERNS[word]


_WEIGHTS = {"theme": pl.String, "ticker": pl.String, "weight": pl.Float64}
_NOTES = {"theme": pl.String, "ticker": pl.String, "company": pl.String, "hits": pl.Int64,
          "words": pl.Int64, "per_1k": pl.Float64, "n_terms": pl.Int64}
