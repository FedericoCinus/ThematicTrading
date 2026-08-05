"""theme_monitor — acceleration scoring for the topics that theme_detect discovers.

Two-step v0, the natural companion to ``theme_detect.detect_theme``:

    1. discover_topics(corpus, date, theme_hint)  -> a set of topic terms as of `date`
    2. weekly_topic_counts(corpus, topics, d0, d1) -> weekly headline count per topic
       acceleration_score(weekly)                 -> rolling z-score of those counts

A headline "mentions a topic" iff the topic term is in that headline's ``terms`` list
(the corpus already carries ``terms``, extracted by the same ``_extract`` the discovery
pipeline uses — so containment is exact and consistent, bigrams like "quantum comput"
included). The acceleration formula is the rolling z-score used verbatim on the sibling
Theme-Detection project for Bloomberg theme codes; here it runs on discovered topics.

    from theme_monitor import discover_topics, weekly_topic_counts, acceleration_score
    topics = discover_topics(corpus, "2023-05-18", theme_hint=r"chatgpt", detect_months=9)
    weekly = weekly_topic_counts(corpus, topics, "2022-06-01", "2023-05-18")
    scores = acceleration_score(weekly)          # z-score at the last week = accel as of d1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))   # theme_detect lives in scripts/
from theme_detect import detect_theme, EXTRACT_STOP, TERM_STOP, TOKEN_RE

# reuse theme_detect's finance/boilerplate stopword sets so the BOW vocabulary drops the
# obvious junk words for free; the data-driven high-frequency cut (below) handles the rest.
_BOW_STOP = set(EXTRACT_STOP) | set(TERM_STOP)


# ======================================================================================
# 1 · Discovery — thin wrapper over detect_theme returning the promoted topic terms
# ======================================================================================
def discover_topics(corpus, date, theme_hint=None, **detect_kwargs) -> list[str]:
    """Run the emergence funnel at one date; return the sorted promoted anchor terms.

    Parameters
    ----------
    corpus : pandas.DataFrame
        Headlines, passed straight through to ``detect_theme`` (raw or terms-prepared).
    date : str | pandas.Timestamp
        The important date D to center discovery on (e.g. an ETF-investability date).
    theme_hint : str | re.Pattern | None
        Optional regex forwarded to ``detect_theme`` (inspection only; does not gate).
    **detect_kwargs
        Any ``detect_theme`` threshold/window (e.g. ``detect_months=9``).

    Returns
    -------
    list[str]
        Sorted, unique anchor terms with ``promoted == True`` — the "set of themes".
    """
    df = detect_theme(corpus, date, theme_hint=theme_hint, **detect_kwargs)
    return sorted(df.loc[df["promoted"], "anchor"].unique())


# ======================================================================================
# 2 · Monitoring — weekly headline counts per topic over a [date_from, date_to] window
# ======================================================================================
def weekly_topic_counts(corpus, topics, date_from, date_to, *, freq="W-MON") -> pd.DataFrame:
    """Count, per week, how many headlines in [date_from, date_to] mention each topic.

    Containment is exact term-list membership: a headline counts toward a topic iff the
    topic string is one of its ``terms``. The result is reindexed onto a *contiguous*
    weekly grid (zero-filled) so the rolling window in ``acceleration_score`` sees every
    week, and onto the full ``topics`` set (a topic never mentioned becomes an all-zero
    column) so the output shape is stable regardless of what actually appeared.

    Parameters
    ----------
    corpus : pandas.DataFrame
        Must have a ``date`` (datetime) and ``terms`` (list[str]) column — the canonical
        corpus format produced by ``preprocess_news`` and consumed by ``detect_theme``.
    topics : sequence[str]
        Topic terms to monitor (typically the output of ``discover_topics``).
    date_from, date_to : str | pandas.Timestamp
        Inclusive window bounds.
    freq : str
        Weekly period frequency (default ``"W-MON"``, matching ``detect_theme``).

    Returns
    -------
    pandas.DataFrame
        Index = week start (datetime), columns = ``topics`` (in given order),
        values = weekly headline count (int).
    """
    topics = list(dict.fromkeys(topics))                     # dedupe, preserve order
    d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
    if "date" not in corpus.columns or "terms" not in corpus.columns:
        raise ValueError(f"corpus needs 'date' and 'terms' columns; got {list(corpus.columns)}")

    d = corpus.loc[(corpus["date"] >= d0) & (corpus["date"] <= d1), ["date", "terms"]]
    topic_set = set(topics)
    exploded = d.explode("terms")
    exploded = exploded[exploded["terms"].isin(topic_set)]
    exploded["week"] = exploded["date"].dt.to_period(freq).dt.start_time

    counts = (exploded.groupby(["week", "terms"]).size()
              .unstack(fill_value=0) if len(exploded) else pd.DataFrame())

    # contiguous weekly grid over the whole window, all topics as columns (zero-filled)
    weeks = pd.period_range(d0, d1, freq=freq).start_time
    counts = (counts.reindex(index=weeks, columns=topics, fill_value=0)
              .fillna(0).astype(int))
    counts.index.name = "week"
    return counts


def weekly_topic_counts_from_parquet(corpus_path, topics, date_from, date_to, *,
                                     freq="W-MON") -> pd.DataFrame:
    """Memory-safe ``weekly_topic_counts`` that reads straight from the corpus parquet.

    Uses a polars *streaming* scan that filters to the date window and to rows whose
    ``terms`` include a monitored topic **before** anything is materialized — so only the
    (tiny) set of topic-mentioning headlines is ever pulled into memory, even against the
    full multi-year, 22M-row corpus. This is the notebook-safe entry point; the in-memory
    ``weekly_topic_counts`` above is for when you already hold a corpus slice.
    """
    topics = list(dict.fromkeys(topics))
    d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
    small = (pl.scan_parquet(str(corpus_path))
               .select(["date", "terms"])
               .filter(pl.col("date").is_between(d0.to_pydatetime(), d1.to_pydatetime()))
               .filter(pl.col("terms").list.eval(pl.element().is_in(topics)).list.any())
               .collect(engine="streaming").to_pandas())
    return weekly_topic_counts(small, topics, date_from, date_to, freq=freq)


# ======================================================================================
# 2b · BOW theme monitoring — a theme = anchor+partner words; headline hits on ANY word
# ======================================================================================
def bow_words(strings, *, extra_stop=()) -> set[str]:
    """Tokenize anchor/partner strings into a unigram word set (theme_detect tokenizer),
    dropping finance/boilerplate stopwords. Bigrams like 'chatgpt maker' contribute both
    'chatgpt' and 'maker'; the corpus always carries the unigram term, so unigram matching
    is sufficient."""
    stop = _BOW_STOP | set(extra_stop)
    words: set[str] = set()
    for s in strings:
        if not s:
            continue
        for w in TOKEN_RE.findall(str(s).lower()):
            if len(w) >= 3 and w not in stop:
                words.add(w)
    return words


def word_doc_freq(corpus_path, words, *, date_from=None, date_to=None) -> pd.Series:
    """Document frequency (how many headlines contain each word) via a streaming scan.
    Memory-safe: only the candidate words are ever grouped."""
    words = list(dict.fromkeys(words))
    lf = pl.scan_parquet(str(corpus_path)).select(["terms"] if date_from is None and date_to is None
                                                  else ["date", "terms"])
    if date_from is not None:
        lf = lf.filter(pl.col("date") >= pd.Timestamp(date_from).to_pydatetime())
    if date_to is not None:
        lf = lf.filter(pl.col("date") <= pd.Timestamp(date_to).to_pydatetime())
    # keep only rows that contain a candidate word BEFORE exploding — avoids blowing the
    # whole 22M-row corpus up into a term stream (bounded memory).
    got = (lf.filter(pl.col("terms").list.eval(pl.element().is_in(words)).list.any())
             .explode("terms").filter(pl.col("terms").is_in(words))
             .group_by("terms").len().collect(engine="streaming").to_pandas())
    return (pd.Series(got["len"].values, index=got["terms"].values)
              .reindex(words, fill_value=0).sort_values(ascending=False))


def build_theme_vocab(anchors, partners, corpus_path, *, max_df=None, max_df_frac=None,
                      min_df=1, corpus_rows=None, extra_stop=(), return_freq=False):
    """BOW vocabulary for one theme = words of (anchor ∪ partners), minus generics.

    "Generic" = high document frequency across the whole corpus (bank, million, companies,
    investment…), which is what makes an OR-match drown in off-theme news. Words whose
    corpus DF exceeds ``max_df`` (or ``max_df_frac`` * ``corpus_rows``) are dropped;
    theme-specific words (bitcoin, ionq, chatgpt, xrp…) sit well below the cut and survive.
    ``min_df`` drops words that never occur (e.g. truncation fragments) — they can't match
    anything anyway.
    """
    words = bow_words(list(anchors) + list(partners), extra_stop=extra_stop)
    freq = word_doc_freq(corpus_path, words)
    if max_df is None:
        if max_df_frac is None:
            max_df_frac = 0.005
        if corpus_rows is None:
            corpus_rows = pl.scan_parquet(str(corpus_path)).select(pl.len()).collect().item()
        max_df = max_df_frac * corpus_rows
    kept = set(freq[(freq >= min_df) & (freq <= max_df)].index)
    return (kept, freq) if return_freq else kept


def weekly_theme_hits_from_parquet(corpus_path, vocab, date_from, date_to, *,
                                   freq="W-MON") -> pd.Series:
    """Weekly count of headlines that hit a theme: a headline is one hit if ANY of its
    ``terms`` is in ``vocab`` (bag-of-words OR-match, each headline counted at most once).
    Streaming + reduced-to-dates, so it stays memory-safe over the full corpus."""
    vocab = list(vocab)
    d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
    hits = (pl.scan_parquet(str(corpus_path)).select(["date", "terms"])
              .filter(pl.col("date").is_between(d0.to_pydatetime(), d1.to_pydatetime()))
              .filter(pl.col("terms").list.eval(pl.element().is_in(vocab)).list.any())
              .select("date").collect(engine="streaming").to_pandas())
    weeks = pd.period_range(d0, d1, freq=freq).start_time
    if len(hits):
        wk = hits["date"].dt.to_period(freq).dt.start_time
        s = wk.value_counts().reindex(weeks, fill_value=0).sort_index()
    else:
        s = pd.Series(0, index=weeks)
    s.index.name = "week"
    return s.astype(int)


# ======================================================================================
# 3 · Acceleration — rolling z-score of the weekly counts (Theme-Detection's rolling_z)
# ======================================================================================
def rolling_z(s: pd.Series, window: int = 12, min_periods: int = 4) -> pd.Series:
    """(s - rolling mean) / rolling std — the acceleration primitive ported verbatim
    from Theme-Detection (identical across its pipeline_lookup / icln / arty notebooks).
    A zero rolling std becomes NaN (undefined acceleration) rather than +/-inf."""
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (s - mu) / sd


def acceleration_score(weekly: pd.DataFrame, *, roll_weeks: int = 12,
                       min_periods: int = 4) -> pd.DataFrame:
    """Per-topic rolling z-score of the weekly counts.

    Same shape as ``weekly``; the value at the last week is each topic's acceleration
    "as of ``date_to``". Early weeks are NaN during the ``min_periods`` warm-up.
    """
    return weekly.apply(rolling_z, window=roll_weeks, min_periods=min_periods)


# ======================================================================================
# 4 · Thresholding / detection rules — the calibrated detector from Theme-Detection
#     (icln_clean_energy / arty_genai / cannabis_survivor_bias notebooks, verbatim rules)
# ======================================================================================
def acceleration_delta(z: pd.Series, lag: int = 4) -> pd.Series:
    """Second derivative of the count series: the ``lag``-week change in the z-score
    (Theme-Detection's ``news_z_accel = news_z - news_z.shift(ACCEL_LAG)``)."""
    return z - z.shift(lag)


def level_rule(z: pd.Series, threshold: float = 1.0, n_consecutive: int = 4) -> pd.Series:
    """LEVEL rule (slow/persistent): fires once z has been > ``threshold`` for
    ``n_consecutive`` consecutive weeks. Verbatim run-length logic from the emergence
    notebooks. Returns a boolean Series (``detection_fires``)."""
    above = z > threshold
    consec = above.astype(int).groupby((above != above.shift()).cumsum()).cumsum()
    return consec >= n_consecutive


def accel_rule(z_accel: pd.Series, threshold: float = 3.0) -> pd.Series:
    """ACCELERATION rule (fast/early): fires on a single week whose z-acceleration
    exceeds ``threshold`` (Theme-Detection used +3.0). Leads the level rule on sharp
    emergences but is noisier."""
    return z_accel > threshold


def first_fire(fires: pd.Series, after=None):
    """First week a boolean fire-Series is True (optionally on/after ``after``)."""
    s = fires if after is None else fires.loc[pd.Timestamp(after):]
    hits = s.index[s.fillna(False)]
    return hits[0] if len(hits) else None


def calibration_stats(z: pd.Series, lo, hi, *, threshold: float = 1.0,
                      n_consecutive: int = 4, accel_lag: int = 4,
                      accel_threshold: float = 3.0) -> dict:
    """Behaviour of the z-signal on a *quiet* calibration window [lo, hi] — the
    false-positive check that sets the thresholds. By construction the rolling z should
    have mean≈0, std≈1; a good threshold fires ~0 times here. Mirrors the calibration
    blocks in the ICLN/ARTY/cannabis notebooks."""
    c = z.loc[pd.Timestamp(lo):pd.Timestamp(hi)].dropna()
    lvl = level_rule(z, threshold, n_consecutive).loc[pd.Timestamp(lo):pd.Timestamp(hi)]
    acc = accel_rule(acceleration_delta(z, accel_lag), accel_threshold
                     ).loc[pd.Timestamp(lo):pd.Timestamp(hi)]
    return {"weeks": len(c), "mean_z": round(float(c.mean()), 2),
            "std_z": round(float(c.std()), 2),
            "frac_z>1": round(float((c > 1).mean()), 2),
            "frac_z>1.5": round(float((c > 1.5).mean()), 2),
            "max_z": round(float(c.max()), 2) if len(c) else float("nan"),
            "level_fires": int(lvl.sum()), "accel_fires": int(acc.sum())}
