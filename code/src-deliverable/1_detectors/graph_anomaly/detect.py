"""Stage 2 — find themes emerging in the news, without being told what to look for.

A theme is a set of words. This detector finds them by asking which words are *new*, and which of
those start keeping company with many things that were not talking to each other before.

    detect(corpus, asof) -> themes [theme, week, vocab, vocab_wide] · diagnostics · rejects
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations

import polars as pl

import progress

from . import tokenizer

EXAMPLES = 10         # headlines kept per candidate: gate 4 judges a cluster by reading them, and
                      # one is not enough — a lone headline reads as one company's news

# ======================================================================================
# FLOW — one run of the stage
#
#   You pick one date, `asof` — the day you are pretending it is (e.g., TODAY). 
#   Nothing published after it may touch the result, which is what makes a backtest honest. 
#   Everything else follows from it:
#
#     2020-12-17 ────── BASELINE 24 months ────── 2022-12-17 ── DETECTION 5 months ── 2023-05-18
#                                                 detect_start                        asof
#            "what was normally said"                       "where I look for something new"
#
#   Two spans because one date cannot do two jobs: you need a history to establish what is normal,
#   and a recent stretch to search. `asof` says where everything ends; `detect_start`, which is just
#   `asof - detect_months`, says where the past stops and the present begins.
#
#   news_corpus.parquet
#        |   SLICE [baseline_start, asof]   and TOKENIZE, since the corpus stores words, not terms
#        v
#   BASELINE  [.. , detect_start]     what a term must be absent from to count as new
#   DETECTION [detect_start, asof]    the weeks a theme may emerge in
#        |
#        |   GATE 1  CANDIDATE   new, said min_mentions times in a week, naming a thing
#        |   GATE 2  CAUGHT      co-occurs with degree_min distinct entities
#        |   GATE 3  PROMOTED    lasted persist_weeks, degree at a new high, and its
#        |                       partners do NOT know each other
#        |   GATE 4  SHAPE       a model removes clusters with a plainly non-thematic shape
#        v
#   themes · diagnostics · rejects
#
#   Gate 4 is there because gates 1-3 are statistics. They measure that something is new, widely
#   mentioned and bridging — all true of an IPO, a court case and a recurring wire table. Telling
#   those from a theme is a question about meaning, so a model answers it, and only ever by
#   rejecting: it never decides what IS a theme, so it cannot invent one.
#
#   Gate 3 is the idea. Clustering says how interconnected an anchor's partners already are.
#   LOW means it bridges worlds that were separate — quantum computing joining hardware,
#   defence and pharma. HIGH means one clique that was already talking to itself, which is a
#   single company's news cycle wearing a theme's clothes.
# ======================================================================================


def tokenize(headlines: pl.DataFrame) -> dict[datetime, list[tuple[str, set[str]]]]:
    """Group headlines by their week, each with the terms it carries.

    INPUT   headlines   a slice of the corpus: [Headline, date]
    OUTPUT  {week start: [(headline, its terms), ...]}, weeks in order

    Leaves polars here and works in Python from now on: the graph below is pure Python, and
    crossing back into a dataframe for every row costs more than the whole computation.
    No prefix stripping — a speaker is an entity worth having.
    """
    weeks = defaultdict(list)
    for headline, date in zip(headlines["Headline"].to_list(), headlines["date"].to_list()):
        monday = datetime(date.year, date.month, date.day) - timedelta(days=date.weekday())
        weeks[monday].append((headline, set(tokenizer.extract_terms(headline))))
    return dict(sorted(weeks.items()))


# ======================================================================================
# GATE 1 — CANDIDATES: a term nobody used before, and the company it keeps
# ======================================================================================
def candidates(weeks: dict, baseline_end: datetime, *, report: bool = False,
               min_mentions: int = 3, cluster_k: int = 15) -> list[dict]:
    """Find terms nobody used before, and the company they keep.

    INPUT   weeks          the output of `tokenize`
            baseline_end   everything up to here is what "before" means
            min_mentions   headlines a term needs in one week to be worth a row
            cluster_k      how many partners the clustering is measured over
    OUTPUT  a list of rows, one per (new term, week): anchor, week, mentions, degree, clustering,
            partners, examples

    Novelty is decided against everything up to `baseline_end`, so a term ever used before is never
    a candidate again. Partners are the entity terms sharing a headline with it, most frequent
    first; `degree` counts them, `clustering` says how many of the top `cluster_k` know each other.

    A term containing the anchor is not a partner but the anchor again — `chatgpt requires` and
    `rapidus partner` are bigrams the tokenizer made from the anchor and its neighbour — so they are
    dropped. What remains is still imperfect: with few mentions, a term's closest company is the rest
    of its own sentence, so `policy response` outranks `microsoft`. See FUTUREWORK, section 3.
    """
    seen = {term for week, items in weeks.items() if week <= baseline_end
            for _, terms in items for term in terms}

    rows = []
    later = [(week, items) for week, items in weeks.items() if week > baseline_end]
    for week, items in progress.each(later, "gates 1-3", on=report, unit="wk"):
        mentions, adjacent = Counter(), defaultdict(Counter)
        for _, terms in items:
            mentions.update(terms)
            for a, b in combinations(sorted(terms), 2):
                adjacent[a][b] += 1
                adjacent[b][a] += 1

        for term, said in mentions.items():
            if term in seen or said < min_mentions or tokenizer.is_event(term):
                continue
            words = set(term.split())
            partners = [p for p, _ in adjacent[term].most_common()
                        if not tokenizer.is_event(p) and not words & set(p.split())]
            rows.append({"week": week, "anchor": term, "mentions": said, "degree": len(partners),
                         "clustering": _clustering(partners[:cluster_k], adjacent),
                         "partners": partners[:cluster_k],
                         "examples": [h for h, terms in items if term in terms][:EXAMPLES]})
    return rows


def _clustering(partners: list[str], adjacent: dict) -> float:
    """Share of partner pairs that also co-occur. Low means the anchor bridges rather than belongs."""
    pairs = list(combinations(partners, 2))
    return round(sum(b in adjacent[a] for a, b in pairs) / len(pairs), 3) if pairs else 0.0


# ======================================================================================
# GATES 2 AND 3 — CAUGHT, then PROMOTED: it lasts, it peaks, and it bridges
# ======================================================================================
def promote(rows: list[dict], *, min_mentions: int = 3, degree_min: int = 8,
            persist_weeks: int = 2, cluster_max: float = 0.60) -> list[dict]:
    """Decide which candidates graduate, and in which week.

    INPUT   rows   the output of `candidates`
            degree_min, persist_weeks, cluster_max, min_mentions   the two gates below
    OUTPUT  the same rows, each with `caught` and `promoted` set; an anchor promotes at most once

    That week is the first where it has been caught `persist_weeks` times, its degree is at a high
    it never reached before, and the clustering it has shown so far is still low enough to bridge.
    """
    for row in rows:
        row["caught"] = row["mentions"] >= min_mentions and row["degree"] >= degree_min
        row["promoted"] = False

    caught = defaultdict(list)
    for row in sorted((r for r in rows if r["caught"]), key=lambda r: r["week"]):
        caught[row["anchor"]].append(row)

    for history in caught.values():
        degrees, clusterings = [], []
        for i, row in enumerate(history):
            degrees.append(row["degree"])
            clusterings.append(row["clustering"])
            if i + 1 >= persist_weeks and row["degree"] >= max(degrees[:-1], default=0) \
               and _median(clusterings) <= cluster_max:
                row["promoted"] = True
                break
    return rows


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    half = len(ordered) // 2
    return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2


# ======================================================================================
# GATE 4 — SHAPE: remove what the headlines themselves show is not a theme
# ======================================================================================
# Remove only, and only for reasons visible in the headlines. A model trained past 2023 knows how
# ChatGPT turned out, so asking it whether something is an emerging theme imports the answer from the
# future — and a filter that keeps only what it knows succeeded leaks as much as one that picks
# winners. Every question below is structural and timeless: it would read the same in 1990.
#
# The five names are reasons to REMOVE. `none` means no reason applies, which is how a cluster is
# kept. So `one_actor` is a defect — one subject doing everything — and a cluster where many
# organisations act is not one_actor and survives.
#
# The model sees the anchor and up to eight headlines, never the theme's word list: whether the same
# subject is acting or many are reacting exists only in sentences, not in a bag of words.
#
# No line tells the model to ignore how promising the subject is. One used to, and measuring it over
# eight calls per cluster showed it did the opposite of its purpose: an instruction not to judge is
# the last thing the model reads, it generalises past the future tense to the structural question
# too, and falls back on `none` — which here means keep.
#
#            cluster    expected    with the line   without
#            chatgpt    keep        8/8             8/8       no cost to the theme
#            eqs-dd     remove      8/8             1/8       insider-dealing bulletin
#            ukraine    remove      4/8             1/8       a war
#
# Nothing replaced it, because nothing needs to: all five patterns ask what the headlines show —
# who acts, what shape the line has — and none asks whether the subject will grow.
#
# The wording is brittle. Measure any edit over a dozen calls, never one — and note that a model
# too small for the question answers a near-tie, which floating-point noise then decides.
PATTERNS = """\
You are shown news headlines that share a term. Say which pattern they match, judging ONLY from the
headlines in front of you.

  one_actor     the SAME organisation or person is the ACTOR in every headline — it announces,
                reports, appoints, acquires, files, is charged. If DIFFERENT organisations each
                act on or react to the shared term it is NOT one_actor, even when that term is
                one company's product.

  news_event    a specific event in the world — a war, a crime, a court case, an election.

  aggregate     a market-wide summary, survey, index level or rate forecast.

  wire_format   a routine table or bulletin the wire repeats on a schedule.

  none          none of the above.
"""

def is_real(anchor: str, headlines: list[str], model: str) -> bool:
    """Ask whether the headlines have the shape of something that is plainly not a theme.

    INPUT   anchor      the term the cluster is built around
            headlines   up to eight of its headlines — the evidence judged
            model       which model answers
    OUTPUT  True to keep the cluster, False to remove it

    The gate may only remove. It is never asked whether something is a theme, only whether it has the
    shape of something that plainly is not one.

    Judged on several headlines, not one: a theme looks like a theme because different actors keep
    appearing around it, and that is invisible in a single line.

    Requires OPENAI_API_KEY. A missing key is an error, not a silent pass: a run that quietly skipped
    this gate would look like a run that passed it, and would hand eighty stories to the next stage.
    """
    import os
    from typing import Literal

    from openai import OpenAI
    from pydantic import BaseModel

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("gate 4 needs OPENAI_API_KEY in code/.env — "
                           "set detect.llm_model to null in config.yaml to run without it")

    # No docstring on this class: pydantic docstrings travel to the model as the schema's
    # description, so anything written here is prompt text. `actors` earns its place by making the
    # model name who appears before it decides — a theme and one company's news differ by how many
    # different names are acting. `Literal` earns its place because asked for a free string the model
    # answers whatever it likes, and a comparison against one spelling silently rejects everything.
    class Verdict(BaseModel):
        pattern: Literal["one_actor", "news_event", "aggregate", "wire_format", "none"]
        actors: list[str]
        reason: str

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None)
    answer = client.beta.chat.completions.parse(
        model=model, temperature=0, response_format=Verdict,
        messages=[{"role": "system", "content": PATTERNS},
                  {"role": "user", "content": f"Cluster term: {anchor}\nHeadlines:\n"
                                              + "\n".join(f"- {h}" for h in headlines[:EXAMPLES])}])
    return answer.choices[0].message.parsed.pattern == "none"


# ======================================================================================
# VOCABULARY — what the theme is made of, gathered from everything before the as-of date
# ======================================================================================
# Detection sees one week. A theme's company is spread over every week it has existed, so the words
# that define it are gathered here instead: one pass over all history up to `asof`, which stays
# point-in-time because there is no history after `asof` to leak.
#
# The same model then keeps only the words that name something. That filter is what the
# notebooks did by hand — "every company/product/person kept, only verbs, generic nouns and
# truncation fragments removed" — and without it the vocabulary is mostly sentence debris, which
# matters because the monitor counts headlines against these words and exposure searches filings for
# them. Measured on ChatGPT at 2023-01-16: 95 raw partners become
# chatgpt, microsoft, openai, google, alphabet, bing and five more.
#
# TWO LISTS COME OUT, because the two consumers want opposite words:
#
#   vocab        without the common names          MONITOR   counts headlines week by week, and
#                chatgpt · openai projects ·                 `google` in the list would count
#                microsoft-backed chatgpt · openai           Google — flat through the arrival
#
#   vocab_wide   with them                         BASKET    only the large names can be bought;
#                + google · iphone                           capped it holds msft alone,
#                                                            uncapped aapl · googl · msft
#
# One model call, on the wider list. `max_doc_freq` then separates them, so the difference between
# the two is exactly the words too common to measure a theme by — and nothing else.
def vocabulary(corpus, anchor: str, asof: datetime, model: str | None, size: int = -1,
               max_doc_freq: float = 0.0001, since: datetime | None = None) -> list[str]:
    """What the theme is made of.

    INPUT   corpus         the canonical corpus, [Headline, date]
            anchor         the theme's term
            asof           nothing published after this is read
            model          which model separates names from debris; None skips that filter
            size           how many words to keep, most co-occurring first; -1 keeps all
            max_doc_freq   share of headlines above which a word says nothing
            since          start of the period the frequency is measured over
    OUTPUT  (vocab, vocab_wide) — each the anchor followed by its words, most co-occurring first.
            `vocab` is the theme the monitor counts, `vocab_wide` the same plus the names too
            common to count by, which are the ones the basket can buy.

    Two filters, doing different jobs, and both are needed. The model removes what does not name a
    thing — `policy response`, `work shift`, the debris co-occurrence counts are full of. Then
    `max_doc_freq` removes what names a thing too common to signal anything: measured on this corpus,
    `chatgpt` and `openai` appear 0 and 6 times in two million baseline headlines while `microsoft`
    and `google` appear 2,380 and 3,350. Counting a theme by its large partners measures how much
    Microsoft is in the news — the weekly series comes out flat through ChatGPT's entire arrival.

    `size` then caps how many words remain, most co-occurring first; -1 keeps all.
    """
    seen = (corpus.lazy()
            .filter((pl.col("date") <= asof) & pl.col("Headline").str.to_lowercase().str.contains(anchor, literal=True))
            .select("Headline").collect())

    company = Counter()
    for headline in seen["Headline"].to_list():
        terms = set(tokenizer.extract_terms(headline))
        if anchor in terms:
            company.update(terms - {anchor})

    since = since or asof - timedelta(days=730)      # two years is as separating as all history
    reference = corpus.lazy().filter(pl.col("date").is_between(since, asof))
    everywhere = _too_common(reference, set(company), max_doc_freq)
    partners = [term for term, _ in company.most_common()]
    if not model or not partners:
        return _two(anchor, partners, everywhere, size)

    import os

    from openai import OpenAI
    from pydantic import BaseModel

    class Entities(BaseModel):
        entities: list[str]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None)
    answer = client.beta.chat.completions.parse(
        model=model, temperature=0, response_format=Entities,
        messages=[{"role": "system", "content": ENTITIES},
                  {"role": "user", "content": "Terms: " + ", ".join(partners)}])
    # lowercased because the corpus stores terms lowercase and the model hands back `Microsoft`
    # intersected with `partners`, not with every term the anchor ever met: the model normalises,
    # answering `microsoft` when shown `microsoft-backed chatgpt`, and that would walk a word the
    # frequency cap had just removed straight back in
    kept = sorted({e.lower() for e in answer.choices[0].message.parsed.entities} & set(partners),
                  key=lambda t: -company[t])
    return _two(anchor, kept, everywhere, size)


def _two(anchor: str, words: list[str], everywhere: set[str], size: int) -> tuple[list[str], list[str]]:
    """Split one filtered word list into the monitor's and the basket's.

    INPUT   anchor       the theme's term, which heads both lists
            words        what survived the model, most co-occurring first
            everywhere   the words too common for their presence to mean anything
            size         how many to keep per list; -1 keeps all
    OUTPUT  (vocab, vocab_wide)
    """
    narrow = [w for w in words if w not in everywhere]
    cut = (lambda ws: ws if size < 0 else ws[:size])
    return [anchor] + cut(narrow), [anchor] + cut(words)


# Purely descriptive, like gate 4 and for the same reason: "could one invest in this" is a question
# about the future, and naming the anchor as a theme would tell the model the answer it is being run
# to find out. Asking only what a word denotes keeps the judgement inside the text.
def _too_common(reference, candidates: set[str], max_doc_freq: float) -> set[str]:
    """Which of these words are so common in the reference period that their presence says nothing."""
    headlines = reference.select("Headline").collect()["Headline"].to_list()
    counts = Counter()
    for headline in headlines:
        counts.update(set(tokenizer.extract_terms(headline)) & candidates)
    return {term for term, n in counts.items() if n > max_doc_freq * len(headlines)}


ENTITIES = (
    "You are given terms extracted from news headlines. Keep only those that NAME a company, an "
    "institution, a product or a technology. Drop people, news outlets, places, generic nouns, verbs "
    "and sentence fragments. Return them lowercase, exactly as given.")


# ======================================================================================
# THE STAGE — corpus in, themes out
# ======================================================================================
def detect(corpus, asof: str, *, detect_months: int = 5, baseline_months: int = 24,
           min_mentions: int = 3, degree_min: int = 8, persist_weeks: int = 2,
           cluster_max: float = 0.60, cluster_k: int = 15,
           llm_model: str | None = "gpt-4o", llm_max: int = 100,
           theme_bag_of_words_size: int = -1, max_doc_freq: float = 0.0001,
           report: bool = True, **_) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Find the themes that emerged in the months before `asof`.

    INPUT   corpus   the canonical corpus, [Headline, date]
            asof     the day we are pretending it is; both windows are derived from it
            gates    the thresholds of the four gates, from config.yaml
    OUTPUT  themes        [theme, week, vocab, vocab_wide] — what any detector must produce
                          `vocab` is what the monitor counts, `vocab_wide` what the basket buys
            diagnostics   [theme, anchor, mentions, degree, clustering, example] — why this one
            rejects       the candidates that never promoted, which is what you read when tuning
    """
    asof = datetime.fromisoformat(asof) if isinstance(asof, str) else asof
    detect_start = asof - timedelta(days=round(detect_months * 30.44))
    baseline_start = detect_start - timedelta(days=round(baseline_months * 30.44))

    headlines = corpus.lazy().filter(pl.col("date").is_between(baseline_start, asof)).collect()
    if headlines.is_empty():
        raise ValueError(f"no headlines between {baseline_start.date()} and {asof.date()}")
    weeks = tokenize(headlines)

    rows = promote(candidates(weeks, detect_start, report=report,
                              min_mentions=min_mentions, cluster_k=cluster_k),
                   min_mentions=min_mentions, degree_min=degree_min,
                   persist_weeks=persist_weeks, cluster_max=cluster_max)
    themes = sorted((r for r in rows if r["promoted"]), key=lambda r: -r["degree"])
    promoted = len(themes)                          # before gate 4, so the funnel can be read
    if llm_model:
        # Judged most connected first, and anything past the cap is dropped rather than waved
        # through: unjudged is not vetted, and a gate that checks only part of the traffic is not a
        # gate. The cap is a cost ceiling, not a sampling rule — raise it if it ever bites.
        if len(themes) > llm_max:
            print(f"   gate 4: {len(themes) - llm_max} themes past llm_max={llm_max} dropped unjudged")
        for row in progress.each(themes[:llm_max], "gate 4", on=report, unit="cluster"):
            row["promoted"] = is_real(row["anchor"], row["examples"], llm_model)
        themes = [r for r in themes[:llm_max] if r["promoted"]]

    if report:
        _report(weeks, detect_start, asof, rows, promoted, themes, llm_model)

    words = {_id(r): vocabulary(corpus, r["anchor"], asof, llm_model, theme_bag_of_words_size,
                                max_doc_freq, baseline_start)
             for r in progress.each(themes, "vocabulary", on=report, unit="theme")}

    return (pl.DataFrame([{"theme": _id(r), "week": r["week"],
                           "vocab": words[_id(r)][0], "vocab_wide": words[_id(r)][1]}
                          for r in themes], schema=_THEMES),
            pl.DataFrame([{"theme": _id(r)} | {k: _with_example(r)[k] for k in _DIAGNOSTIC}
                          for r in themes], schema=_DIAG),
            pl.DataFrame([{"theme": _id(r)} | {k: _with_example(r)[k] for k in _REJECT}
                          for r in rows if not r["promoted"]], schema=_REJECTS))


def _id(row: dict) -> str:
    """`anchor@week`: readable, and the same on every run."""
    return f"{row['anchor']}@{row['week'].date()}"


_DIAGNOSTIC = ["anchor", "mentions", "degree", "clustering", "example"]
_REJECT = ["anchor", "week", "mentions", "degree", "clustering", "caught", "example"]
_THEMES = {"theme": pl.String, "week": pl.Datetime,
           "vocab": pl.List(pl.String), "vocab_wide": pl.List(pl.String)}
_DIAG = {"theme": pl.String, "anchor": pl.String, "mentions": pl.Int64, "degree": pl.Int64,
         "clustering": pl.Float64, "example": pl.String}
_REJECTS = {"theme": pl.String, "anchor": pl.String, "week": pl.Datetime, "mentions": pl.Int64,
            "degree": pl.Int64, "clustering": pl.Float64, "caught": pl.Boolean, "example": pl.String}


def _with_example(row: dict) -> dict:
    """Diagnostics show one headline; the gate reads them all."""
    return row | {"example": row["examples"][0] if row["examples"] else ""}
def _report(weeks: dict, baseline_end: datetime, asof: datetime, rows: list,
            promoted: int, themes: list, llm_model: str | None) -> None:
    """The funnel, one line per gate, so it reads while it runs."""
    caught = sum(r["caught"] for r in rows)
    base = sum(len(i) for w, i in weeks.items() if w <= baseline_end)
    total = sum(len(i) for i in weeks.values())
    print(f"   window      {min(weeks).date()} .. {asof.date()}   {total:,} headlines"
          f"  ({base:,} baseline, {total - base:,} detection)")
    print(f"   candidates  new, said enough, naming a thing        {len(rows):>8,}")
    print(f"   caught      co-occurring widely enough              {caught:>8,}")
    print(f"   promoted    lasting, peaking, bridging              {promoted:>8,}")
    if llm_model:
        print(f"   gate 4      a model removes non-thematic shapes      {len(themes):>8,}")
    if themes:
        print("      " + ", ".join(_id(r) for r in themes[:6]))
