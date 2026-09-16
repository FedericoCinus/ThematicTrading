"""SEC EDGAR: which companies had written about a theme's words, and by when.

EDGAR is the SEC's filing archive. Two things in it matter here:

  CIK    the identifier the SEC gives every filer — a number, Microsoft is 789019. Tickers and
         company names change; the CIK does not, so it is what a filing is joined on.
  EFTS   EDGAR Full-Text Search: ask it for a phrase and it returns the FILINGS containing it,
         filterable by form and by filing date. You never ask "which company is this" — you ask
         which documents say it, and the filer comes back attached.

The filing date filter is the whole point of using it here. A company writes about a theme in the
10-K it files months after the fact, so asking "who had said this by DATE" is a genuinely
point-in-time question — and its answer at a theme's birth is often nobody.
"""
from __future__ import annotations

import json
import lzma
import os
import re
import shutil
import time
from datetime import date, timedelta
from html.parser import HTMLParser

import requests

import config

EFTS = "https://efts.sec.gov/LATEST/search-index"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
THROTTLE = 0.12          # the SEC allows 10 requests a second; stay under it


def _session() -> requests.Session:
    """A session carrying the identifying header the SEC requires.

    INPUT   nothing — reads SEC_USER_AGENT from the environment
    OUTPUT  a requests session; raises if the header is missing, because without it every call
            comes back 403 and the failure would otherwise look like "no filings found"
    """
    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        raise RuntimeError("EDGAR needs SEC_USER_AGENT in code/.env, in the form "
                           "'Name Surname you@example.com' — it refuses anonymous requests")
    session = requests.Session()
    session.headers.update({"User-Agent": agent, "Accept-Encoding": "gzip, deflate"})
    return session


def how_many_filings(term: str, until: str, since: str = "2001-01-01",
                     forms: str = "10-K,10-Q", session=None) -> int:
    """How many filings had said this phrase by a given date.

    INPUT   term     the exact phrase, quoted for EFTS so it is not tokenised
            until    the last filing date counted — the theme's entry date, never today
            since    the first; EDGAR's full-text coverage starts in 2001
            forms    which form types to look in
    OUTPUT  the number of filings, capped at 10,000 by EFTS

    One cheap request per term: enough to answer whether the filings knew about a theme yet.
    """
    session = session or _session()
    time.sleep(THROTTLE)
    answer = session.get(EFTS, params={"q": f'"{term}"', "forms": forms,
                                       "startdt": since, "enddt": until}, timeout=30)
    answer.raise_for_status()
    return answer.json()["hits"]["total"]["value"]


# ======================================================================================
# WHICH FILINGS — the hit list, paged and date-bisected
#
#   IN     term, since, until, forms
#   OUT    [{adsh, doc, ciks, form, file_date}]   one entry per document, not per company
#   cache  data/raw/edgar/fts/{term}_{forms}_{since}_{until}.json
#
#   EFTS returns 100 hits a page and refuses to page past 10,000 in one window. Past that it
#   answers `"relation": "gte"` instead of a count, so:
#
#     ask ──▶ relation == "eq" and total < 9,900 ? ──▶ page through it
#                              otherwise          ──▶ split the date range in half, ask both
#
#   A window that cannot split any further (one day) keeps the first 10,000 and says so.
# ======================================================================================
WINDOW_CAP = 10_000
COVERAGE_START = "2001-01-01"      # EDGAR full-text search reaches no further back


def hits(term: str, until: str, since: str = COVERAGE_START, forms: str = "10-K,10-Q",
         session=None, force: bool = False) -> list[dict]:
    """Every filing document containing a phrase, by filing date.

    INPUT   term, until, since, forms   as in `how_many_filings`
            session                     reuse one across calls; made if not given
            force                       ignore the cache and re-ask EDGAR
    OUTPUT  [{adsh, doc, ciks, form, file_date}] — one entry per DOCUMENT; a document can carry
            several CIKs (co-registrants), so the company join happens downstream

    Raises ValueError for a window starting before EDGAR's full-text coverage, rather than
    returning the empty list that a genuinely silent period would also give.
    """
    if since < COVERAGE_START:
        raise ValueError(f"EDGAR full-text search starts {COVERAGE_START}; asked from {since}")
    cache = config.EDGAR / "fts" / f"{_slug(term)}_{_slug(forms)}_{since}_{until}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())["hits"]

    found = _collect(term, since, until, forms, session or _session())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"term": term, "forms": forms, "since": since, "until": until,
                                 "n": len(found), "hits": found}))
    return found


def _collect(term: str, since: str, until: str, forms: str, session) -> list[dict]:
    """All hits for one window, halving the dates when EDGAR will not page far enough."""
    answer = _ask(session, term, since, until, forms)
    total, relation = answer["hits"]["total"]["value"], answer["hits"]["total"]["relation"]
    if relation != "eq" or total > WINDOW_CAP - 100:
        a, b = date.fromisoformat(since), date.fromisoformat(until)
        if a < b:
            middle = a + (b - a) // 2
            return (_collect(term, since, middle.isoformat(), forms, session)
                    + _collect(term, (middle + timedelta(days=1)).isoformat(), until, forms, session))
        print(f"      {term!r}: over {WINDOW_CAP} hits on {since} alone — keeping the first page")

    found = [_row(h) for h in answer["hits"]["hits"]]
    while len(found) < min(total, WINDOW_CAP - 100):
        page = _ask(session, term, since, until, forms, start=len(found))["hits"]["hits"]
        if not page:
            break
        found += [_row(h) for h in page]
    return found


def _ask(session, term: str, since: str, until: str, forms: str, start: int = 0) -> dict:
    time.sleep(THROTTLE)
    params = {"q": f'"{term}"', "forms": forms, "startdt": since, "enddt": until}
    answer = session.get(EFTS, params=params | ({"from": start} if start else {}), timeout=30)
    answer.raise_for_status()
    return answer.json()


def _row(hit: dict) -> dict:
    source = hit["_source"]
    return {"adsh": source["adsh"], "doc": hit["_id"].split(":", 1)[1],
            "ciks": [int(c) for c in source["ciks"]],
            "form": source.get("form"), "file_date": source.get("file_date")}


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", text.lower()).strip("-")


# ======================================================================================
# THE TEXT — one document, HTML thrown away, cached compressed
#
#   IN     cik, adsh, doc                       from a hit
#   OUT    the filing's text, whitespace collapsed
#   cache  data/raw/edgar/filings/{cik}/{adsh}/{doc}.txt.xz
#   needs  SEC_USER_AGENT
#
#   EXTRACT_RULE names the extraction. Change the rule and the cache no longer certifies:
#   it is wiped rather than silently mixed, exactly as the corpus does with its own RULE.
#
#   Streamed and parsed in chunks: an inline-XBRL 10-K can be 50 MB of HTML and never needs
#   to be in memory whole.
# ======================================================================================
EXTRACT_RULE = "extract-v1"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


def text(cik: int, adsh: str, doc: str, session=None, force: bool = False) -> str:
    """The text of one filing document.

    INPUT   cik, adsh, doc   the document's identity, from a `hits` entry
            session, force   as in `hits`
    OUTPUT  the extracted text, whitespace collapsed to single spaces
    """
    _certify()
    path = config.EDGAR / "filings" / str(cik) / adsh / f"{doc}.txt.xz"
    if path.exists() and not force:
        with lzma.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()

    session = session or _session()
    time.sleep(THROTTLE)
    answer = session.get(f"{ARCHIVES}/{cik}/{adsh.replace('-', '')}/{doc}", stream=True, timeout=60)
    answer.raise_for_status()
    answer.encoding = answer.encoding or "utf-8"
    reader = _Text()
    for chunk in answer.iter_content(chunk_size=1 << 16, decode_unicode=True):
        reader.feed(chunk)
    reader.close()

    out = re.sub(r"\s+", " ", " ".join(reader.parts)).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wt", encoding="utf-8", preset=6) as handle:
        handle.write(out)
    return out


class _Text(HTMLParser):
    """Character data only, with the contents of <script> and <style> dropped."""

    SKIP = {"script", "style"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        self._depth += tag in self.SKIP

    def handle_endtag(self, tag):
        self._depth -= tag in self.SKIP and self._depth > 0

    def handle_data(self, data):
        if not self._depth:
            self.parts.append(data)


def _certify() -> None:
    """Wipe the text cache when it predates the current EXTRACT_RULE — it is regenerable."""
    folder = config.EDGAR / "filings"
    stamp = folder / "manifest.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("rule") == EXTRACT_RULE:
        return
    if folder.exists() and any(folder.iterdir()):
        print(f"   filing text cache predates {EXTRACT_RULE} — wiping {folder}")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"rule": EXTRACT_RULE}))
