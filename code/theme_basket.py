"""Map a theme vocabulary (bag of entity words) to a basket of US companies.

Baseline scoring (deliberately naive): count how often each theme term occurs in a
company's SEC filings (10-K/10-Q by default). Filings are selected by *filing date*
inside a configurable window [as_of - lookback, as_of + lookahead], so the same tool
supports strict point-in-time backtests (lookahead=0) and "how fast does the signal
appear" studies (lookahead>0).

Fetch strategy: downloading every filing for the ~9k-ticker universe would be tens of
GB, so we first ask EDGAR full-text search (EFTS) which filings mention each term and
download only those. Companies with no EFTS hit for any term score 0 by construction.
Terms with more than ``max_hits_per_term`` EFTS hits ("google", "microsoft", ...) are
too generic to shortlist on — they are skipped for shortlisting but still counted
inside the filings other terms selected, and reported as ``generic_terms``.

EDGAR facts this module relies on (verified 2026-07):
  - EFTS: GET https://efts.sec.gov/LATEST/search-index?q="term"&forms=10-K,10-Q
    &startdt=YYYY-MM-DD&enddt=YYYY-MM-DD&from=N. Quoted q = exact phrase; ``forms``
    filters on root_forms (amendments 10-K/A come for free); pages of <=100 hits;
    hard cap from+size <= 10000 -> bisect the date range when total.relation=="gte".
  - Hit ``_id`` = "{accession-with-dashes}:{filename}"; document URL is then
    https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/{filename}.
  - Ticker->CIK: https://www.sec.gov/files/company_tickers.json.
  - Max 10 requests/second and an identifying User-Agent is REQUIRED (else 403):
    put ``SEC_USER_AGENT=Your Name you@example.com`` in code/.env.
  - Full-text coverage starts 2001-01-01 (hard error for earlier windows).

Out of scope for the baseline: 20-F foreign private issuers, section-aware parsing
(risk factors vs. boilerplate), and any term weighting beyond raw counts.

File layout:
  data/raw/edgar/company_tickers.json          ticker->CIK map (refreshed after 30 days)
  data/raw/edgar/fts/{term}_{...}.json         cached EFTS hit lists per (term, forms, window)
  data/raw/edgar/filings/{cik}/{adsh}/{doc}.txt.xz  extracted filing text (raw HTML discarded)
  data/raw/edgar/filings/manifest.json         EXTRACT_RULE certification for the text cache
All caches are disposable (regenerable from the network).
"""
import json
import lzma
import os
import re
import shutil
import time
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))   # theme_detect lives in scripts/
from theme_detect import _load_env

_ROOT = Path(__file__).resolve().parent                      # code/
EDGAR_DIR = _ROOT / "data" / "raw" / "edgar"
TICKER_CSV = _ROOT / "data" / "raw" / "tickers" / "ticker_list_US.csv"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
FTS_COVERAGE_START = date(2001, 1, 1)
FTS_MAX_WINDOW = 10_000                                      # EFTS result-window hard cap
EXTRACT_RULE = "extract-v1"                                  # bump when text extraction changes
SCORE_RULE = "fts-count-v1"                                  # bump when scoring semantics change


# ======================================================================================
# HTTP layer: identified session + polite throttle (SEC: max 10 req/s, UA required)
# ======================================================================================
class _Throttle:
    """Keep consecutive requests >= min_interval apart; back off on 403/429/503."""

    def __init__(self, min_interval: float = 0.12):
        self.min_interval, self._last = min_interval, 0.0

    def wait(self):
        dt = self.min_interval - (time.monotonic() - self._last)
        if dt > 0:
            time.sleep(dt)
        self._last = time.monotonic()


_THROTTLE = _Throttle()
_session_obj = None


def _session() -> requests.Session:
    global _session_obj
    if _session_obj is None:
        _load_env()                                          # code/.env -> os.environ
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise RuntimeError(
                "SEC requires an identifying User-Agent. Add a line like\n"
                "  SEC_USER_AGENT=Your Name you@example.com\n"
                "to code/.env (see https://www.sec.gov/os/accessing-edgar-data).")
        _session_obj = requests.Session()
        _session_obj.headers["User-Agent"] = ua
    return _session_obj


_RETRY_STATUS = {403, 429, 500, 502, 503, 504}              # EFTS throws transient 500s


def _get(url: str, params: dict | None = None, stream: bool = False) -> requests.Response:
    for backoff in (2, 10, 30, 120, None):
        _THROTTLE.wait()
        r = _session().get(url, params=params, stream=stream, timeout=60)
        if r.status_code in _RETRY_STATUS and backoff is not None:
            print(f"  HTTP {r.status_code} from {url.split('/')[2]} — backing off {backoff}s")
            time.sleep(backoff)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("unreachable")


# ======================================================================================
# Universe: Bloomberg ticker list -> SEC CIKs
# ======================================================================================
def load_universe(path: Path = TICKER_CSV) -> list[str]:
    """Read the headerless Bloomberg ticker list ('AAPL US Equity') into SEC-style tickers."""
    out, seen = [], set()
    for line in Path(path).read_text().splitlines():
        t = line.strip().upper().removesuffix(" US EQUITY").strip().replace("/", "-")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_cik_map(force: bool = False, max_age_days: int = 30) -> dict[str, tuple[int, str]]:
    """ticker -> (cik, company title) from SEC's company_tickers.json (cached on disk)."""
    path = EDGAR_DIR / "company_tickers.json"
    stale = not path.exists() or (time.time() - path.stat().st_mtime) > max_age_days * 86400
    if force or stale:
        EDGAR_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(TICKER_MAP_URL).content)
    out: dict[str, tuple[int, str]] = {}
    for rec in json.loads(path.read_text()).values():
        out.setdefault(rec["ticker"].upper(), (int(rec["cik_str"]), rec["title"]))
    return out


def resolve_universe(tickers: list[str], cik_map: dict[str, tuple[int, str]]
                     ) -> tuple[dict[int, tuple[str, str]], list[str]]:
    """Map universe tickers to CIKs. Returns (cik -> (ticker, title), unresolved tickers).

    Unresolved names (ETFs, SPAC units, warrants, delisted) are expected and harmless:
    they simply cannot receive a score. Share classes of one issuer share a CIK; the
    first ticker encountered represents it.
    """
    by_cik: dict[int, tuple[str, str]] = {}
    unresolved = []
    for t in tickers:
        if t in cik_map:
            cik, title = cik_map[t]
            by_cik.setdefault(cik, (t, title))
        else:
            unresolved.append(t)
    return by_cik, unresolved


# ======================================================================================
# EDGAR quarterly master index (coverage diagnostics & retrieval-by-date, 1993+)
#
# Full-text SEARCH only reaches back to 2001, but RETRIEVAL of filings goes back to
# 1993 via the quarterly master indexes (one pipe-delimited file per quarter listing
# every filing). We keep only 10-K*/10-Q* rows (incl. legacy 10-K405/10-KSB variants).
# ======================================================================================
INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
INDEX_DIR = EDGAR_DIR / "full_index"
INDEX_START_YEAR = 1993
_INDEX_FORMS = re.compile(r"^10-[KQ]")


def load_quarter_index(year: int, qtr: int, force: bool = False) -> pd.DataFrame:
    """10-K/10-Q rows of one quarter's master index: [cik, company, form, date, filename]."""
    cache = INDEX_DIR / f"{year}Q{qtr}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    r = _get(INDEX_URL.format(year=year, q=qtr))
    rows = []
    for line in r.content.decode("latin-1").splitlines():
        parts = line.split("|")
        if len(parts) == 5 and parts[0].isdigit() and _INDEX_FORMS.match(parts[2]):
            rows.append(parts)
    df = pd.DataFrame(rows, columns=["cik", "company", "form", "date", "filename"])
    df["cik"] = df["cik"].astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def filing_index(start_year: int = INDEX_START_YEAR, end_year: int | None = None,
                 force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """All 10-K/10-Q index rows from start_year through today (quarter-cached).

    The still-open current quarter grows daily; pass force=True to refresh it.
    """
    today = date.today()
    end_year = end_year or today.year
    frames = []
    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            if (year, q) > (today.year, (today.month - 1) // 3 + 1):
                break
            try:
                frames.append(load_quarter_index(year, q, force=force))
            except requests.HTTPError as e:                  # not-yet-published quarter
                if verbose:
                    print(f"  {year}Q{q}: index unavailable ({e.response.status_code}) — skipped")
        if verbose and year % 5 == 0:
            print(f"  index loaded through {year}")
    out = pd.concat(frames, ignore_index=True)
    out["root_form"] = out["form"].str.extract(r"^(10-[KQ])")[0]
    return out


def filings_asof(index: pd.DataFrame, cik: int, as_of: str,
                 lookback_days: int = 365) -> pd.DataFrame:
    """Filings by one company with filing date in [as_of - lookback, as_of], with URLs."""
    d1 = pd.Timestamp(as_of)
    d0 = d1 - pd.Timedelta(days=lookback_days)
    hit = index[(index["cik"] == cik) & index["date"].between(d0, d1)].copy()
    hit["url"] = "https://www.sec.gov/Archives/" + hit["filename"]
    return hit.sort_values("date", ascending=False)


# ======================================================================================
# EDGAR full-text search (EFTS)
# ======================================================================================
def _slug(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", s.lower()).strip("-")


def _parse_hit(hit: dict, term: str) -> dict:
    src = hit["_source"]
    return {"term": term,
            "adsh": src["adsh"],
            "ciks": [int(c) for c in src["ciks"]],
            "doc": hit["_id"].split(":", 1)[1],
            "form": src.get("form"),
            "root_form": (src.get("root_forms") or [src.get("form")])[0],
            "file_date": src.get("file_date"),
            "period_ending": src.get("period_ending") or "",
            "file_type": src.get("file_type")}


def fts_total(term: str, startdt: str, enddt: str, forms: str = "10-K,10-Q") -> tuple[int, str]:
    """One cheap request: (hits.total.value, relation). relation=='gte' means capped at 10k."""
    j = _get(EFTS_URL, {"q": f'"{term}"', "forms": forms,
                        "startdt": startdt, "enddt": enddt}).json()
    tot = j["hits"]["total"]
    return tot["value"], tot["relation"]


def _fts_collect(term: str, startdt: str, enddt: str, forms: str) -> list[dict]:
    """All hits for one (term, window), recursively bisecting dates around the 10k cap."""
    j = _get(EFTS_URL, {"q": f'"{term}"', "forms": forms,
                        "startdt": startdt, "enddt": enddt}).json()
    total, rel = j["hits"]["total"]["value"], j["hits"]["total"]["relation"]
    if rel != "eq" or total > FTS_MAX_WINDOW - 100:
        d0, d1 = date.fromisoformat(startdt), date.fromisoformat(enddt)
        if d0 >= d1:                                         # cannot split further
            print(f"  WARNING: >{FTS_MAX_WINDOW} '{term}' hits on {startdt}; keeping first {FTS_MAX_WINDOW}")
        else:
            mid = d0 + (d1 - d0) // 2
            return (_fts_collect(term, startdt, mid.isoformat(), forms)
                    + _fts_collect(term, (mid + timedelta(days=1)).isoformat(), enddt, forms))
    hits = [_parse_hit(h, term) for h in j["hits"]["hits"]]
    while len(hits) < min(total, FTS_MAX_WINDOW - 100):
        page = _get(EFTS_URL, {"q": f'"{term}"', "forms": forms, "startdt": startdt,
                               "enddt": enddt, "from": len(hits)}).json()["hits"]["hits"]
        if not page:
            break
        hits += [_parse_hit(h, term) for h in page]
    return hits


def fts_hits(term: str, startdt: str, enddt: str, forms: str = "10-K,10-Q",
             force: bool = False) -> list[dict]:
    """Cached EFTS hit list for one (term, forms, window)."""
    cache = EDGAR_DIR / "fts" / f"{_slug(term)}_{_slug(forms)}_{startdt}_{enddt}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())["hits"]
    hits = _fts_collect(term, startdt, enddt, forms)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"term": term, "forms": forms, "startdt": startdt,
                                 "enddt": enddt, "n": len(hits), "hits": hits}))
    return hits


# ======================================================================================
# Filing text: streamed download -> incremental HTML->text -> xz cache
# ======================================================================================
class _TextExtractor(HTMLParser):
    """Incremental HTML->text: collects character data, drops <script>/<style> content.

    Feeding chunks keeps memory bounded even for 50+ MB inline-XBRL filings; plain-text
    documents (old .txt filings) pass straight through handle_data.
    """

    _SKIP = {"script", "style"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _check_extract_manifest():
    """Certify the text cache against EXTRACT_RULE; wipe it (regenerable) on mismatch."""
    filings_dir = EDGAR_DIR / "filings"
    manifest = filings_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("rule") == EXTRACT_RULE:
        return
    if filings_dir.exists() and any(filings_dir.iterdir()):
        print(f"filing-text cache predates {EXTRACT_RULE} — wiping {filings_dir}")
        shutil.rmtree(filings_dir)
    filings_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"rule": EXTRACT_RULE}))


def fetch_filing_text(cik: int, adsh: str, doc: str, force: bool = False) -> str:
    """Extracted text of one filing document, from cache or EDGAR (raw HTML is discarded)."""
    _check_extract_manifest()
    path = EDGAR_DIR / "filings" / str(cik) / adsh / f"{doc}.txt.xz"
    if path.exists() and not force:
        with lzma.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    url = f"{ARCHIVES_URL}/{cik}/{adsh.replace('-', '')}/{doc}"
    r = _get(url, stream=True)
    r.encoding = r.encoding or "utf-8"
    ex = _TextExtractor()
    for chunk in r.iter_content(chunk_size=1 << 16, decode_unicode=True):
        ex.feed(chunk)
    ex.close()
    text = re.sub(r"\s+", " ", " ".join(ex.parts)).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wt", encoding="utf-8", preset=6) as f:
        f.write(text)
    return text


# ======================================================================================
# Term counting (mirrors EFTS phrase semantics: hyphens/whitespace are token breaks)
# ======================================================================================
_pattern_cache: dict[str, re.Pattern] = {}


def _term_pattern(term: str) -> re.Pattern:
    """'chatgpt-like' -> tokens joined by 1-3 non-word chars, word-boundary guarded,
    so it matches 'ChatGPT like', 'chatgpt-like' and 'ChatGPT‑Like' alike."""
    if term not in _pattern_cache:
        toks = [t for t in re.split(r"[\s\-‐-―]+", term.lower()) if t]
        _pattern_cache[term] = re.compile(
            r"(?<!\w)" + r"\W{1,3}".join(map(re.escape, toks)) + r"(?!\w)")
    return _pattern_cache[term]


def count_terms(text: str, vocab: list[str]) -> tuple[dict[str, int], int]:
    """Case-insensitive occurrence count per vocab term, plus total word count."""
    low = text.lower()
    counts = {t: len(_term_pattern(t).findall(low)) for t in vocab}
    n_words = sum(1 for _ in re.finditer(r"\S+", low))
    return counts, n_words


# ======================================================================================
# Shortlist -> score -> basket
# ======================================================================================
def make_window(as_of: str, lookback_days: int, lookahead_days: int) -> tuple[str, str]:
    d = date.fromisoformat(str(as_of)[:10])
    start, end = d - timedelta(days=lookback_days), d + timedelta(days=lookahead_days)
    if start < FTS_COVERAGE_START:
        raise ValueError(f"window starts {start}, before EDGAR full-text coverage "
                         f"({FTS_COVERAGE_START}) — shorten lookback or move as_of")
    return start.isoformat(), end.isoformat()


def shortlist(vocab: list[str], startdt: str, enddt: str, by_cik: dict[int, tuple[str, str]],
              forms: str = "10-K,10-Q", max_hits_per_term: int = 1000,
              force: bool = False, verbose: bool = True
              ) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """EFTS pass: which universe filings mention which terms.

    Returns (one row per (cik, adsh, doc) with its matched terms; generic terms skipped
    for shortlisting; per-term EFTS totals).
    """
    totals: dict[str, int] = {}
    generic: list[str] = []
    rows: dict[tuple, dict] = {}
    for term in vocab:
        total, rel = fts_total(term, startdt, enddt, forms)
        totals[term] = total
        if rel != "eq" or total > max_hits_per_term:
            generic.append(term)
            if verbose:
                print(f"  {term!r}: {total:,}{'+' if rel != 'eq' else ''} filings — too generic, "
                      "skipped for shortlisting (still counted inside shortlisted filings)")
            continue
        hits = fts_hits(term, startdt, enddt, forms, force=force)
        kept = 0
        for h in hits:
            for cik in h["ciks"]:
                if cik in by_cik:
                    key = (cik, h["adsh"], h["doc"])
                    row = rows.setdefault(key, {**h, "cik": cik, "matched_terms": set()})
                    row["matched_terms"].add(term)
                    kept += 1
        if verbose:
            print(f"  {term!r}: {total:,} filings, {kept} in universe")
    if not rows:
        return (pd.DataFrame(columns=["cik", "ticker", "company", "adsh", "doc", "form",
                                      "root_form", "file_date", "period_ending", "matched_terms"]),
                generic, totals)
    df = pd.DataFrame(rows.values())
    df["ticker"] = df["cik"].map(lambda c: by_cik[c][0])
    df["company"] = df["cik"].map(lambda c: by_cik[c][1])
    df["matched_terms"] = df["matched_terms"].map(sorted)
    cols = ["cik", "ticker", "company", "adsh", "doc", "form", "root_form",
            "file_date", "period_ending", "file_type", "matched_terms"]
    return df[cols].sort_values(["cik", "adsh"]).reset_index(drop=True), generic, totals


def _dedupe_amendments(short: pd.DataFrame) -> pd.DataFrame:
    """One accession per (cik, root_form, period): prefer the original over /A amendments."""
    acc = (short.groupby(["cik", "adsh"])
                .agg(form=("form", "first"), root_form=("root_form", "first"),
                     file_date=("file_date", "first"), period_ending=("period_ending", "first"))
                .reset_index())
    acc["is_amend"] = acc["form"].str.contains("/A", regex=False)
    acc = acc.sort_values(["is_amend", "file_date"])         # originals first, then oldest first
    key = acc["period_ending"].where(acc["period_ending"] != "", acc["adsh"])
    keep = acc[~pd.DataFrame({"cik": acc["cik"], "root": acc["root_form"], "per": key})
               .duplicated()]["adsh"]
    return short[short["adsh"].isin(set(keep))]


def score_basket(short: pd.DataFrame, vocab: list[str], verbose: bool = True
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download shortlisted documents and count every vocab term (generic ones included).

    Returns (basket, filings): company-level scores sorted by (n_terms_hit, per_1k_words)
    descending, and the per-accession detail behind them.
    """
    short = _dedupe_amendments(short)
    frows = []
    groups = list(short.groupby(["cik", "adsh"], sort=False))
    for i, ((cik, adsh), g) in enumerate(groups, 1):
        counts = dict.fromkeys(vocab, 0)
        n_words = 0
        for doc in g["doc"]:
            text = fetch_filing_text(int(cik), adsh, doc)
            c, nw = count_terms(text, vocab)
            for t in vocab:
                counts[t] += c[t]
            n_words += nw
        r0 = g.iloc[0]
        frows.append({"cik": cik, "ticker": r0["ticker"], "company": r0["company"],
                      "adsh": adsh, "form": r0["form"], "file_date": r0["file_date"],
                      "period_ending": r0["period_ending"], "n_docs": len(g),
                      "n_words": n_words, **counts})
        if verbose and (i % 25 == 0 or i == len(groups)):
            print(f"  scored {i}/{len(groups)} filings")
    filings = pd.DataFrame(frows)
    if filings.empty:
        return pd.DataFrame(), filings
    basket = (filings.groupby(["cik", "ticker", "company"], as_index=False)
                     .agg({"adsh": "count", "n_words": "sum", **{t: "sum" for t in vocab}})
                     .rename(columns={"adsh": "n_filings"}))
    basket["total"] = basket[vocab].sum(axis=1)
    basket["n_terms_hit"] = (basket[vocab] > 0).sum(axis=1)
    basket["per_1k_words"] = (1000 * basket["total"] / basket["n_words"]).round(3)
    basket = (basket.sort_values(["n_terms_hit", "per_1k_words"], ascending=False)
                    .reset_index(drop=True))
    return basket[["ticker", "cik", "company", "n_filings", "n_words", *vocab,
                   "total", "n_terms_hit", "per_1k_words"]], filings


def build_basket(vocab: list[str], as_of: str, *, lookback_days: int = 365,
                 lookahead_days: int = 0, forms: str = "10-K,10-Q",
                 universe_csv: Path = TICKER_CSV, max_hits_per_term: int = 1000,
                 force: bool = False, verbose: bool = True) -> dict:
    """End-to-end: theme vocab -> scored company basket.

    Returns {"basket", "filings", "shortlist": DataFrames, "meta": run parameters,
    per-term EFTS totals, generic/unresolved diagnostics}.
    """
    startdt, enddt = make_window(as_of, lookback_days, lookahead_days)
    if verbose:
        print(f"window: {startdt} .. {enddt}  forms: {forms}")
    by_cik, unresolved = resolve_universe(load_universe(universe_csv), load_cik_map())
    if verbose:
        print(f"universe: {len(by_cik)} issuers resolved to CIKs, "
              f"{len(unresolved)} tickers unresolved (ETFs/units/warrants score 0)")
    short, generic, totals = shortlist(vocab, startdt, enddt, by_cik, forms=forms,
                                       max_hits_per_term=max_hits_per_term,
                                       force=force, verbose=verbose)
    if verbose:
        print(f"shortlist: {short['adsh'].nunique() if len(short) else 0} filings "
              f"across {short['cik'].nunique() if len(short) else 0} companies")
    basket, filings = score_basket(short, vocab, verbose=verbose)
    meta = {"score_rule": SCORE_RULE, "extract_rule": EXTRACT_RULE, "vocab": list(vocab),
            "as_of": str(as_of)[:10], "lookback_days": lookback_days,
            "lookahead_days": lookahead_days, "forms": forms, "startdt": startdt,
            "enddt": enddt, "max_hits_per_term": max_hits_per_term,
            "fts_totals": totals, "generic_terms": generic,
            "n_universe_resolved": len(by_cik), "n_unresolved": len(unresolved),
            "unresolved_sample": unresolved[:20],
            "n_filings_scored": len(filings), "n_companies": len(basket)}
    return {"basket": basket, "filings": filings, "shortlist": short, "meta": meta}
