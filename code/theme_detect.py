"""theme_detect — one reusable function that runs the 0.24 theme-emergence funnel.

Point the unsupervised detector at an *important date* of any theme, feed it a raw
corpus of headlines, and get back the tidy per-(week, anchor) dataframe with every
funnel step as a column:

    candidates -> caught -> promoted -> llm_kept   (+ per-step rank columns)

The detection logic is lifted verbatim from notebook 0.24-quantum-multidate.ipynb so
results match. Every threshold/window is a flat keyword argument on ``detect_theme`` so
a single call can tune it inline. The LLM reject-gate is OFF by default.

    from theme_detect import detect_theme
    df = detect_theme(raw_headlines, "2025-01-15", theme_hint=r"\\bionq\\b|quantum comput")
"""
from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ======================================================================================
# Constants (verbatim from 0.24 cells c01 / c03 / c07) — not per-call knobs
# ======================================================================================
WIRES = ["BN", "BFW", "BBO"]
TOKEN_RE = re.compile(r"(?u)\b[a-z][a-z0-9\-]{2,}\b")
BIGRAM_RE = re.compile(r"(?u)\b[a-z][a-z0-9\-]{2,}\s+[a-z][a-z0-9\-]{2,}\b")

TERM_STOP = set(ENGLISH_STOP_WORDS) | {
    "says", "said", "new", "year", "week", "report", "shares", "stock",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}

EVENT_STOP = {
    "collapse", "rout", "bankruptcy", "bankrupt", "fraud", "lawsuit", "sue", "sues", "sued", "probe", "hearing",
    "trial", "court", "arrest", "arrested", "resign", "resigns", "ban", "bans", "banned", "outage", "recall", "default", "slump",
    "slumps", "plunge", "plunges", "crash", "derailment", "strike", "quake", "earthquake", "protest", "protests", "unrest",
    "attack", "war", "sanctions", "fine", "fined", "scandal", "layoffs", "layoff", "pivot", "halt", "halts", "delay", "delays",
    "death", "dies", "killed", "guilty", "charges", "charged", "indicted", "crisis", "takeover", "merger", "deal", "acquires",
    "acquire", "buys", "stake", "ipo", "listing", "bond", "bonds", "notes", "debt", "offering", "case", "settlement", "shortage",
    "shutdown", "tumbles", "soars", "jumps", "rises", "falls", "drops", "gains", "cuts", "raises"}

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

REJECT_SYS = ("You are a conservative FILTER that REMOVES clusters of news headlines that are NOT emerging themes. "
              "You never decide what IS a theme; only flag clusters that CLEARLY are: 1. single_entity_event, "
              "2. macro_market_aggregate, 3. boilerplate_wire. If unclear, KEEP. KEEP anything describing a SPECIFIC "
              "technological/industrial/product development across multiple actors.")


# ======================================================================================
# Verbatim helpers from 0.24 (thresholds threaded through as arguments)
# ======================================================================================
def _is_entity(t: str) -> bool:
    return not any(tok in EVENT_STOP for tok in t.split())


def _normalize_terms(v):
    return v.tolist() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, (list, tuple)) else [])


def _filter_terms(ts):
    return [t for t in ts if len(t) >= 3 and not any(tok in TERM_STOP for tok in t.split())]


def _week_ts(w, freq):
    return pd.Period(w, freq=freq).start_time


def _clustcoef(P, adj):
    if len(P) < 2:
        return 0.0
    tot = pairs = 0
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            tot += 1
            if P[j] in adj[P[i]]:
                pairs += 1
    return pairs / tot if tot else 0.0


def _strip_prefix(text):
    for _ in range(2):
        if ":" not in text:
            return text
        pre, _, rest = text.partition(":")
        if not pre or not rest or len(pre) > 30 or len(pre.split()) > 4:
            return text
        text = rest.strip()
    return text


def _extract(text):
    t = text.lower()
    terms = set()
    for bg in BIGRAM_RE.findall(t):
        if all(tok not in EXTRACT_STOP for tok in bg.split()):
            terms.add(bg)
    for tok in TOKEN_RE.findall(t):
        if tok not in EXTRACT_STOP and len(tok) >= 3:
            terms.add(tok)
    return sorted(terms)


def _detect(news, baseline_end, ds, de, *, freq, min_mentions, cluster_k, rep_hl):
    """Fixed-baseline novelty + bridging detector — one row per (novel entity, week)."""
    df = news.copy()
    df["terms_f"] = df["terms"].map(_normalize_terms).map(_filter_terms)
    df["week"] = df["date"].dt.to_period(freq).astype(str)
    baseline_terms = set()
    for tl in df.loc[df.date <= baseline_end, "terms_f"]:
        baseline_terms.update(tl)
    disc = [w for w in sorted(df["week"].unique(), key=lambda w: _week_ts(w, freq))
            if ds <= _week_ts(w, freq) <= de]
    rows = []
    for week in disc:
        grp = df.loc[df.week == week]
        twc, adj = Counter(), defaultdict(Counter)
        for tl in grp["terms_f"]:
            s = set(tl)
            for t in s:
                twc[t] += 1
            for a, b in combinations(sorted(s), 2):
                adj[a][b] += 1
                adj[b][a] += 1
        for t, cnt in twc.items():
            if t in baseline_terms or cnt < min_mentions or not _is_entity(t):
                continue
            epart = [p for p, _ in adj[t].most_common() if _is_entity(p)]
            P = epart[:cluster_k]
            reps = [hl for hl, tl in zip(grp["Headline"], grp["terms_f"]) if t in set(tl)][:rep_hl]
            rows.append({"week": week, "anchor": t, "partners": ", ".join(epart[:8]),
                         "mentions": int(cnt), "degree": len(epart), "clustering": round(_clustcoef(P, adj), 3),
                         "subgraph": [t] + P[:10], "reps": reps, "example": reps[0] if reps else ""})
    return pd.DataFrame(rows)


def _promotion_week(sub, *, freq, persist_weeks, cluster_max):
    """First week an anchor graduates: caught >=persist_weeks, degree at a new high, median clustering a bridge."""
    sub = sub.sort_values("week", key=lambda s: s.map(lambda w: _week_ts(w, freq)))
    wk, deg, clu = list(sub.week), list(sub.degree), list(sub.clustering)
    for i in range(len(wk)):
        if (i + 1) >= persist_weeks and deg[i] >= max(deg[:i] or [0]) and float(np.median(clu[:i + 1])) <= cluster_max:
            return wk[i]
    return None


# ======================================================================================
# LLM reject-gate (optional; OpenAI). Reads code/.env for credentials.
# ======================================================================================
try:
    from pydantic import BaseModel
    from typing import Literal

    class Reject(BaseModel):
        verdict: "Literal['keep','reject']"
        category: "Literal['single_entity_event','macro_market_aggregate','boilerplate_wire','none']"
        reason: str
except Exception:  # pydantic not importable — LLM path simply unavailable
    Reject = None

_client = None


def _llm_keep(subgraph, headlines, llm_model):
    """Return True (keep) / False (reject); None if no API key configured."""
    global _client
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from openai import OpenAI
    if _client is None:
        base = os.environ.get("OPENAI_BASE") or os.environ.get("OPENAI_BASE_URL") or None
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base)
    user = ("Cluster entities: " + ", ".join(subgraph[:14]) + "\nHeadlines:\n"
            + "\n".join(f"- {h}" for h in headlines[:8]) + "\n\nClassify this cluster.")
    model = os.environ.get("OPENAI_DEFAULT_MODEL", llm_model)
    r = _client.beta.chat.completions.parse(model=model, temperature=0, response_format=Reject,
                                            messages=[{"role": "system", "content": REJECT_SYS},
                                                      {"role": "user", "content": user}])
    return r.choices[0].message.parsed.verdict == "keep"


def _load_env(env_path=None):
    """Set OPENAI_* from a .env file if not already in the environment (default: code/.env)."""
    p = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ======================================================================================
# Corpus preparation (raw headlines -> Headline, date, terms)
# ======================================================================================
_HEAD_CANDS = ["Headline", "headline", "title", "text"]
_TIME_CANDS = ["date", "CaptureTime", "timestamp", "time"]


def _resolve_columns(corpus):
    hcol = next((c for c in _HEAD_CANDS if c in corpus.columns), None)
    tcol = next((c for c in _TIME_CANDS if c in corpus.columns), None)
    if hcol is None or tcol is None:
        raise ValueError(
            f"corpus needs a headline column (one of {_HEAD_CANDS}) and a timestamp column "
            f"(one of {_TIME_CANDS}); got columns {list(corpus.columns)}")
    return hcol, tcol


def _to_naive_date(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def _prepare_corpus(corpus):
    """Return a frame with canonical [Headline, date, terms]; extract terms if absent.

    If a ``terms`` column is already present the corpus is treated as PRE-PREPARED (the
    cached 0.24 format, already wire-filtered / deduped per-year) and the cleaning steps
    are skipped — re-deduping globally would drop cross-year recurrences that 0.24 keeps.
    """
    hcol, tcol = _resolve_columns(corpus)
    d = corpus.copy()
    has_terms = "terms" in d.columns
    if not has_terms:  # RAW path — full cleaning mirrors 0.24 build_year_terms
        if "WireName" in d.columns:
            d = d[d["WireName"].isin(WIRES)]
        d = d.dropna(subset=[hcol]).drop_duplicates(subset=[hcol])
    d["date"] = _to_naive_date(d[tcol])
    if hcol != "Headline":
        d = d.rename(columns={hcol: "Headline"})
    if not has_terms:
        d["Headline"] = d["Headline"].map(_strip_prefix)
        d = d[d["Headline"].str.split().map(len) >= 4]
        d["terms"] = [_extract(h) for h in d["Headline"]]
    d = d.dropna(subset=["date"]).reset_index(drop=True)
    return d[["Headline", "date", "terms"]]


# ======================================================================================
# Ranks + output schema
# ======================================================================================
def _output_cols(theme_hint, keep_debug_cols):
    cols = ["week", "anchor", "partners", "mentions", "degree", "clustering"]
    if theme_hint is not None:
        cols.append("is_theme")
    cols += ["caught", "promoted", "llm_kept", "example",
             "rank_candidate", "rank_caught", "rank_promoted", "llm_judged"]
    if keep_debug_cols:
        cols += ["subgraph", "reps"]
    return cols


def _add_rank_cols(df, llm_max):
    df = df.copy()
    df["rank_candidate"] = df.groupby("week")["mentions"].rank(ascending=False, method="min").astype("Int64")
    df["rank_caught"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    cr = df[df.caught].groupby("week")["degree"].rank(ascending=False, method="min")
    df.loc[cr.index, "rank_caught"] = cr.astype(int)
    df["rank_promoted"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    pm = df[df.promoted].sort_values(["degree", "anchor"], ascending=[False, True])  # anchor = deterministic tie-break
    df.loc[pm.index, "rank_promoted"] = list(range(1, len(pm) + 1))
    df["llm_judged"] = df["rank_promoted"].notna() & (df["rank_promoted"] <= llm_max)
    return df


def _empty_result(theme_hint, keep_debug_cols):
    cols = _output_cols(theme_hint, keep_debug_cols)
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    df["mentions"] = df["mentions"].astype("int64")
    df["degree"] = df["degree"].astype("int64")
    df["clustering"] = df["clustering"].astype("float64")
    for c in ("caught", "promoted", "llm_judged"):
        df[c] = df[c].astype("bool")
    for c in ("rank_candidate", "rank_caught", "rank_promoted"):
        df[c] = df[c].astype("Int64")
    if theme_hint is not None:
        df["is_theme"] = df["is_theme"].astype("bool")
    return df


# ======================================================================================
# The public function
# ======================================================================================
def detect_theme(
    corpus,
    date,
    theme_hint=None,
    *,
    # window
    detect_months=5,
    baseline_months=24,
    # candidate gate
    min_mentions=3,
    # caught gate
    degree_min=8,
    # promote gate
    persist_weeks=2,
    cluster_k=15,
    cluster_max=0.60,
    # llm gate (OFF by default)
    run_llm=False,
    llm_max=25,
    llm_model="gpt-4o-mini",
    # misc
    freq="W-MON",
    rep_hl=8,
    keep_debug_cols=False,
    env_path=None,
):
    """Run the 0.24 theme-emergence funnel on a raw headline corpus, centered on ``date``.

    Detection window = ``[date - detect_months, date]``; baseline = corpus history with
    ``date <= date - detect_months``. Returns one row per ``(week, anchor)`` with every
    funnel step as a column, matching 0.24's ``DFS[<date>]`` plus rank columns.

    Parameters
    ----------
    corpus : pandas.DataFrame
        Raw headlines. Needs a headline-text column (one of Headline/headline/title/text)
        and a timestamp column (one of date/CaptureTime/timestamp/time). If it already has
        a ``terms`` column (the cached 0.24 format) extraction is skipped; otherwise terms
        (unigrams+bigrams, finance stopwords stripped) are extracted internally. A
        ``WireName`` column, if present, is filtered to {BN, BFW, BBO}.
    date : str | pandas.Timestamp
        The important date D (e.g. an ETF-investability date).
    theme_hint : str | re.Pattern | None
        Optional regex. Anchors matching it get ``is_theme=True`` (inspection only; does not
        affect the funnel). When None, the ``is_theme`` column is omitted.
    detect_months, baseline_months, min_mentions, degree_min, persist_weeks, cluster_k,
    cluster_max, freq, rep_hl : the detector thresholds/windows (0.24 defaults). Tune inline.
    run_llm : bool
        Default False -> ``llm_kept`` stays <NA>. When True and an OpenAI key is found,
        run the conservative reject-gate on the top-``llm_max`` promoted rows.
    llm_max, llm_model, env_path : LLM options.
    keep_debug_cols : bool
        If True, also return the ``subgraph`` and ``reps`` list-columns.

    Returns
    -------
    pandas.DataFrame
        Sorted by (week asc, degree desc), columns:
        week, anchor, partners, mentions, degree, clustering, [is_theme,]
        caught, promoted, llm_kept, example,
        rank_candidate, rank_caught, rank_promoted, llm_judged [, subgraph, reps].
        Empty detection returns an empty DataFrame with exactly these columns.
    """
    D = pd.Timestamp(date)
    det_start = (D - pd.DateOffset(months=detect_months)).normalize()
    base_start = (det_start - pd.DateOffset(months=baseline_months)).normalize()

    _load_env(env_path)
    df = _prepare_corpus(corpus)
    win = df[(df.date >= base_start) & (df.date <= D)]

    # --- history / coverage guards ---
    if int((win.date <= det_start).sum()) == 0:
        warnings.warn("detect_theme: baseline slice is empty (no history before the window); "
                      "novelty is unreliable — every term will look new.")
    if win[win.date >= det_start].empty:
        warnings.warn(f"detect_theme: detection window [{det_start.date()}, {D.date()}] has no "
                      "headlines (date beyond corpus?); returning empty result.")
        return _empty_result(theme_hint, keep_debug_cols)

    R = _detect(win, det_start, det_start, D,
                freq=freq, min_mentions=min_mentions, cluster_k=cluster_k, rep_hl=rep_hl)
    if R.empty:
        return _empty_result(theme_hint, keep_debug_cols)

    # --- per-row measures ---
    if theme_hint is not None:
        pat = re.compile(theme_hint, re.I) if isinstance(theme_hint, str) else theme_hint
        R["is_theme"] = R["anchor"].str.contains(pat, na=False)
    R["caught"] = (R.mentions >= min_mentions) & (R.degree >= degree_min)
    R["promoted"] = False
    for a, sub in R[R.caught].groupby("anchor"):
        pw = _promotion_week(sub, freq=freq, persist_weeks=persist_weeks, cluster_max=cluster_max)
        if pw is not None:
            R.loc[(R.anchor == a) & (R.week == pw), "promoted"] = True

    # --- llm_kept: only on promoted rows, only if run_llm (the pipeline's last gate) ---
    R["llm_kept"] = pd.NA
    if run_llm:
        for i in list(R.index[R.promoted])[:llm_max]:
            try:
                R.at[i, "llm_kept"] = _llm_keep(R.at[i, "subgraph"], R.at[i, "reps"], llm_model)
            except Exception as e:  # one bad API call shouldn't sink the dataframe
                warnings.warn(f"detect_theme: LLM call failed for row {i}: {e}")
                R.at[i, "llm_kept"] = pd.NA

    # --- sort, ranks, select (matches 0.24 Step 4 -> 4b order) ---
    # anchor is a deterministic tie-break so output is reproducible run-to-run
    # (0.24 sorted by [week, degree] only, so its within-tie order is arbitrary).
    R = R.sort_values(["week", "degree", "anchor"], ascending=[True, False, True]).reset_index(drop=True)
    R = _add_rank_cols(R, llm_max=llm_max)
    return R[_output_cols(theme_hint, keep_debug_cols)]
