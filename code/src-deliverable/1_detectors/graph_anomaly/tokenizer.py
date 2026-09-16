"""What a word is, for the whole pipeline.

A theme is a set of words, and the monitor decides a headline belongs to a theme by intersecting
that set with the headline's `terms`. Comparing two bags of words only means something if both were
built the same way — so this module is the single definition, applied once by `first_publication`
when it writes the corpus, and reused by the detector when it reads terms back out.

Which makes every constant here load-bearing: change a regex or a stopword and the `terms` column no
longer matches what this file would produce, so the corpus has to be rebuilt.

Copied verbatim from scripts/theme_detect.py (lines 32-118).
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ======================================================================================
# WHAT COUNTS AS A WORD — the patterns, and the three lists of words that never make it
#
#   EXTRACT_STOP   dropped before `terms` is written    -> never in the corpus at all
#   TERM_STOP      dropped when picking a theme's words -> in the corpus, useless as a theme term
#   EVENT_STOP     a happening, not a lasting thing     -> in the corpus, wrong kind of theme term
# ======================================================================================
TOKEN_RE = re.compile(r"(?u)\b[a-z][a-z0-9\-]{2,}\b")
BIGRAM_RE = re.compile(r"(?u)\b[a-z][a-z0-9\-]{2,}\s+[a-z][a-z0-9\-]{2,}\b")

# Never enters `terms`, so no headline in the corpus ever contributes these words. sklearn's English
# stopwords plus ~100 added by hand: corporate suffixes (inc, plc, ag), job titles (ceo, chairman),
# wire verbs (says, announces, reports), calendar and units (q1, mln, pct, march), analyst jargon
# (eps, overweight, downgrade). They turn up in headlines about anything, so keeping them would make
# every story co-occur with every other and drown the graph the detector builds.
EXTRACT_STOP = set(ENGLISH_STOP_WORDS) | { 
    "inc", "plc", "ltd", "llc", "corp", "co", "sa", "ag", "nv", "group", "holdings", "ceo", "cfo", "says", "said", "new", "year",
    "today", "week", "day", "update", "report", "reports", "results", "announces", "announced", "shares", "stock", "stocks",
    "stake", "dividend", "q1", "q2", "q3", "q4", "fy", "unit", "mln", "bln", "pct", "jan", "feb", "mar", "apr", "may", "jun", "jul",
    "aug", "sep", "oct", "nov", "dec", "sales", "deal", "chief", "cut", "raised", "raise", "buy", "sell", "net", "revenue", "beat",
    "miss", "forecast", "outlook", "business", "market", "markets", "global", "first", "second", "third", "fourth", "annual",
    "meeting", "plans", "plan", "bn", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "rated", "hold", "neutral", "perform", "outperform", "underperform", "overweight", "underweight", "equal-weight", "lowers",
    "upgrade", "downgrade", "maintains", "reiterates", "est", "eps", "adj", "sees", "expects", "names", "appoints", "hires",
    "officer", "director", "chairman", "executive", "promotes", "tender", "offering", "offer", "notes", "bond", "bonds", "debt", "bills", "yield"}

# Too generic to be one of a theme's words. Note it is entirely contained in EXTRACT_STOP — all 337
# words, none of them new — so for a term that came out of the corpus this check can never fire. It
# only bites on a vocabulary supplied from somewhere else, and is kept as that guard.
TERM_STOP = set(ENGLISH_STOP_WORDS) | {
    "says", "said", "new", "year", "week", "report", "shares", "stock",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}

# Words naming something that happens once — a bankruptcy, a lawsuit, a crash — rather than something
# that lasts. To anything that merely counts headlines, a week of coverage of one collapse looks
# exactly like a theme emerging, so the detector drops candidate words found here. Seven of these 83
# are also in EXTRACT_STOP and therefore never reach a term at all.
EVENT_STOP = {
    "collapse", "rout", "bankruptcy", "bankrupt", "fraud", "lawsuit", "sue", "sues", "sued", "probe", "hearing",
    "trial", "court", "arrest", "arrested", "resign", "resigns", "ban", "bans", "banned", "outage", "recall", "default", "slump",
    "slumps", "plunge", "plunges", "crash", "derailment", "strike", "quake", "earthquake", "protest", "protests", "unrest",
    "attack", "war", "sanctions", "fine", "fined", "scandal", "layoffs", "layoff", "pivot", "halt", "halts", "delay", "delays",
    "death", "dies", "killed", "guilty", "charges", "charged", "indicted", "crisis", "takeover", "merger", "deal", "acquires",
    "acquire", "buys", "stake", "ipo", "listing", "bond", "bonds", "notes", "debt", "offering", "case", "settlement", "shortage",
    "shutdown", "tumbles", "soars", "jumps", "rises", "falls", "drops", "gains", "cuts", "raises"}


# ======================================================================================
# TRANSFORM HEADLINES — HEADLINE IN, CORPUS COLUMN OUT (used by first_publication)
# ======================================================================================
def strip_prefix(headline: str) -> str:
    """Drop up to two leading `PREFIX:` segments.

    INPUT   headline   the text as the wire published it
    OUTPUT  the text without them; a prefix counts as one only if short (<= 30 chars, <= 4 words),
            so a headline that merely contains a colon is left alone
    """
    for _ in range(2):
        if ":" not in headline:
            return headline
        prefix, _, rest = headline.partition(":")
        if not prefix or not rest or len(prefix) > 30 or len(prefix.split()) > 4:
            return headline
        headline = rest.strip()
    return headline


def extract_terms(headline: str) -> list[str]:
    """Turn a headline into the words the pipeline matches on.

    INPUT   headline   any text
    OUTPUT  its sorted unigrams and bigrams, minus EXTRACT_STOP. This is what `terms` means.
    """
    lowered = headline.lower()
    terms = set()
    for bigram in BIGRAM_RE.findall(lowered):
        if all(token not in EXTRACT_STOP for token in bigram.split()):
            terms.add(bigram)
    for token in TOKEN_RE.findall(lowered):
        if token not in EXTRACT_STOP and len(token) >= 3:
            terms.add(token)
    return sorted(terms)


# ======================================================================================
# JUDGE TERMS — TERM IN, VERDICT OUT (used by the detector to filter a theme's words)
# ======================================================================================
def is_generic(term: str) -> bool:
    """INPUT a term · OUTPUT True when it is too short or too common to say anything."""
    return len(term) < 3 or any(token in TERM_STOP for token in term.split())


def is_event(term: str) -> bool:
    """INPUT a term · OUTPUT True when it names a happening rather than a lasting thing."""
    return any(token in EVENT_STOP for token in term.split())
