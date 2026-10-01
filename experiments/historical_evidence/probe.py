"""Il test minimo: c'e' evidenza pre-cutoff che nomina gli emittenti giusti?

    python probe.py chatgpt 2023-01-16

Tre fonti, tutte gratuite e senza chiave, tutte datate alla fonte:

    gdelt()   news full-text, 2017+, 1 richiesta ogni 5s con penalita' appiccicosa
    hn()      Hacker News via Algolia, 2006+, nessun rate limit osservato
    wiki()    la revisione della pagina COM'ERA al cutoff — un solo GET, niente Wayback

Non e' il motore di ricerca della proposta. E' solo la domanda che decide se valga
la pena scriverlo: le fonti pre-cutoff nominano MSFT/GOOGL/AAPL meglio dei tag
Bloomberg che il preprocessing gia' scarta?
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "thematic-trading-probe/0.1"}


def get(url: str, timeout: int = 60, tries: int = 5) -> bytes:
    """Un GET con backoff esponenziale sul 429 — GDELT lo richiede davvero."""
    wait = 15
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            print(f"    429 · attendo {wait}s", file=sys.stderr)
            time.sleep(wait)
            wait *= 2
    return b""


# ======================================================================================
# LE FONTI — ognuna torna [(data, titolo, url)], tutto rigorosamente <= cutoff
# ======================================================================================
def gdelt(term: str, day: dt.date) -> list[tuple[str, str, str]]:
    """News di UN giorno. Un giorno per richiesta: la finestra intera costa N richieste."""
    q = urllib.parse.urlencode({
        "query": f"{term} sourcelang:english", "mode": "artlist", "maxrecords": "250",
        "startdatetime": day.strftime("%Y%m%d") + "000000",
        "enddatetime": day.strftime("%Y%m%d") + "235959",
        "format": "json", "sort": "datedesc"})
    raw = get(f"https://api.gdeltproject.org/api/v2/doc/doc?{q}")
    try:
        arts = json.loads(raw)["articles"]
    except Exception:
        return []                                   # rate-limited: il body e' testo, non JSON
    return [(a["seendate"][:8], a["title"], a["url"]) for a in arts]


def hn(term: str, cutoff: dt.date, days: int) -> list[tuple[str, str, str]]:
    """Storie HN della finestra. Una sola richiesta per l'intero intervallo."""
    end = int(dt.datetime.combine(cutoff + dt.timedelta(days=1), dt.time()).timestamp())
    start = end - days * 86400
    q = urllib.parse.urlencode({
        "query": term, "tags": "story", "hitsPerPage": "1000",
        "numericFilters": f"created_at_i>={start},created_at_i<{end}"})
    raw = get(f"https://hn.algolia.com/api/v1/search_by_date?{q}")
    hits = json.loads(raw)["hits"]
    return [(h["created_at"][:10], h.get("title") or "", h.get("url") or "") for h in hits]


def wiki(term: str, cutoff: dt.date) -> str:
    """Il testo della pagina COM'ERA al cutoff — point-in-time, verificabile, un GET."""
    q = urllib.parse.urlencode({
        "action": "query", "prop": "revisions", "titles": term, "rvlimit": "1",
        "rvstart": cutoff.strftime("%Y-%m-%dT23:59:59Z"), "rvdir": "older",
        "rvprop": "ids|timestamp", "format": "json"})
    pages = json.loads(get(f"https://en.wikipedia.org/w/api.php?{q}"))["query"]["pages"]
    revs = next(iter(pages.values())).get("revisions")
    if not revs:
        return ""
    rev = revs[0]
    print(f"    revisione {rev['revid']} del {rev['timestamp']}", file=sys.stderr)
    return get(f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}&action=raw").decode()


# ======================================================================================
# LA MISURA — quali aziende nomina il testo pre-cutoff
# ======================================================================================
#   Deliberatamente lessicale, come ISSUERS in 3_baskets: si conta chi e' NOMINATO, non
#   chi ne trarra' beneficio. Il secondo e' un fatto sul futuro e il modello lo sa.
BRANDS = {
    "openai": "—", "microsoft": "MSFT", "bing": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "deepmind": "GOOGL", "nvidia": "NVDA", "apple": "AAPL",
    "iphone": "AAPL", "meta": "META", "amazon": "AMZN", "baidu": "BIDU",
    "tesla": "TSLA", "twitter": "TWTR", "ibm": "IBM", "buzzfeed": "BZFD",
    "salesforce": "CRM", "shutterstock": "SSTK", "chegg": "CHGG", "stack overflow": "—",
}


def named(texts: list[str]) -> collections.Counter:
    joined = " ".join(texts).lower()
    c = collections.Counter()
    for brand in BRANDS:
        n = len(re.findall(r"\b" + re.escape(brand) + r"\b", joined))
        if n:
            c[brand] = n
    return c


def report(label: str, counts: collections.Counter, n_docs: int) -> None:
    print(f"\n{label}  ({n_docs} documenti)")
    if not counts:
        print("    nessuna azienda nominata")
        return
    for brand, n in counts.most_common(12):
        print(f"    {n:4d}  {brand:16s} -> {BRANDS[brand]}")


def main() -> None:
    term = sys.argv[1] if len(sys.argv) > 1 else "chatgpt"
    cutoff = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2023-01-16")
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    print(f"termine={term}  cutoff={cutoff}  finestra={days}g", file=sys.stderr)

    print("\nHN…", file=sys.stderr)
    h = hn(term, cutoff, days)
    report("HACKER NEWS · titoli", named([t for _, t, _ in h]), len(h))

    print("\nWikipedia…", file=sys.stderr)
    w = wiki(term, cutoff)
    report("WIKIPEDIA · revisione al cutoff", named([w]), 1 if w else 0)

    print("\nGDELT (lento: ~1 richiesta ogni 10s)…", file=sys.stderr)
    arts: list[tuple[str, str, str]] = []
    for i in range(days):
        day = cutoff - dt.timedelta(days=days - 1 - i)
        got = gdelt(term, day)
        arts += got
        print(f"    {day}  {len(got):4d} articoli", file=sys.stderr)
        time.sleep(10)
    report("GDELT · titoli news", named([t for _, t, _ in arts]), len(arts))


if __name__ == "__main__":
    main()
