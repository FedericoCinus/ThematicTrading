"""Who can be bought — the investable set both basket methods choose from.

    load() -> [ticker, company, cik, stem]
"""
from __future__ import annotations

import json
import re
from importlib import import_module

import polars as pl

import config

# ======================================================================================
# THE UNIVERSE — the list of companies a basket may hold
#
#   IN     nothing
#   OUT    [ticker, company, cik, stem]              ~10,400 rows
#
#   WHAT IT IS TODAY
#     https://www.sec.gov/files/company_tickers.json      every CURRENT SEC registrant
#         {"cik_str": 789019,  "ticker": "MSFT", "title": "MICROSOFT CORP"}
#     access   plain HTTP, fetched once     cache  data/raw/edgar/company_tickers.json
#     needs    SEC_USER_AGENT in code/.env  — the SEC answers 403 to anonymous requests
#
#   SEC     the US market regulator; every listed American company files its reports there
#   EDGAR   the SEC's public archive of those reports
#   CIK     the number the SEC gives each filer — names and tickers change, it does not
#   stem    the company name without its corporate tail: `apple inc.` -> `apple`
#
#   WHAT IT IS NOT
#     a registrant is not an investable company. The list holds shells, tiny filers and names
#     that barely trade, which is how `quantum` reaches Quantum Leap — a shell company.
#     today's registrants only: a company delisted before today is absent → survivorship bias
#
#   THE ALTERNATIVE, not implemented
#     a Bloomberg extract, data/raw/tickers/ticker_list_US.csv, one line per line of stock
#     ("AAPL US Equity"). It was the investable universe of the earlier work and is not in the
#     repo, with no script to regenerate it. Config keeps a slot for it; nothing reads it.
#
#     The reproducible version of the same idea, and the better one: screen this list by
#     liquidity using the prices 4_backtesting/prices.py already caches — keep what traded
#     more than X a day AS AT THE THEME'S DATE, never as at today, or the screen picks the
#     companies that became liquid afterwards.
# ======================================================================================

TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"

# A register name carries a corporate tail no headline ever uses. Stripping it is what lets a
# lookup demand an exact company rather than a substring: `apple inc.` becomes `apple`, while
# `apple hospitality reit, inc.` becomes `apple hospitality reit` and no longer answers to `apple`.
SUFFIX = {"inc", "incorporated", "corp", "corporation", "co", "company", "plc", "ltd", "limited",
          "holding", "holdings", "group", "sa", "nv", "ag", "se", "the", "class", "com"}


def load(force: bool = False) -> pl.DataFrame:
    """Every listed company the SEC knows.

    INPUT   force   re-download instead of using the cached copy
    OUTPUT  [ticker, company, cik, stem] — `stem` is the company name without its corporate
            tail, which is what a name lookup matches on
    """
    cached = config.EDGAR / "company_tickers.json"
    if force or not cached.exists():
        edgar = import_module("3_baskets.filings_method.edgar")      # the SEC session lives with the source
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(edgar._session().get(TICKER_MAP, timeout=30).text)
    rows = json.loads(cached.read_text()).values()
    return pl.DataFrame([{"ticker": r["ticker"].lower(), "company": r["title"].lower(),
                          "cik": r["cik_str"]} for r in rows]) \
             .with_columns(stem=pl.col("company").map_elements(stem, return_dtype=pl.String))


def stem(name: str) -> str:
    """The register name without its corporate tail.

    INPUT   name   a company name as the register spells it, lower case
    OUTPUT  the same without trailing corporate words: `apple inc.` -> `apple`
    """
    words = [w for w in re.split(r"[^a-z0-9&]+", re.sub(r"/[a-z]{2,4}/", " ", name)) if w]
    while words and words[-1] in SUFFIX:
        words.pop()
    return " ".join(words)
