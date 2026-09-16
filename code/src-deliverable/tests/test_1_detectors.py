"""1_detectors — what a word is, and the method that finds themes in words."""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import polars as pl

tok = importlib.import_module("1_detectors.graph_anomaly.tokenizer")
det = importlib.import_module("1_detectors.graph_anomaly.detect")


def _corpus(new_term_weeks: int = 3, partners: int = 10) -> pl.DataFrame:
    """A corpus with one genuinely new term, so the novelty gate has something true to find.

    The baseline talks about `acme` for months. Then `widget` appears, each time beside a different
    company that never appears beside any other — high degree, zero clustering, the shape of a
    bridging theme.
    """
    rows = []
    for week in range(26):                                   # baseline: nothing new ever happens
        day = datetime(2023, 10, 2) + timedelta(weeks=week)
        rows += [{"Headline": f"acme reports numbers for period {week}", "date": day}] * 4
    for week in range(new_term_weeks):                       # detection: widget arrives
        day = datetime(2024, 4, 8) + timedelta(weeks=week)
        rows += [{"Headline": f"widget chosen by company{i}", "date": day} for i in range(partners)]
    return pl.DataFrame(rows)


# Gate 4 off for the synthetic checks below: they exercise the statistical gates, and a model asked
# whether `widget chosen by company0` is a real theme is right to say no. The one check that runs it
# is the genAI case at the end, on the real corpus.
ASOF = "2024-05-06"
GATES = dict(detect_months=2, baseline_months=6, llm_model=None, report=False)


def prefix_stripping():
    """strip_prefix drops a short wire prefix and leaves anything longer alone"""
    cases = {
        "BN: Apple Delays iPhone Launch": "Apple Delays iPhone Launch",
        "No colon here at all": "No colon here at all",
        "A prefix running well past thirty characters: kept whole":
            "A prefix running well past thirty characters: kept whole",
    }
    wrong = [h for h, want in cases.items() if tok.strip_prefix(h) != want]
    return f"{len(wrong)} of {len(cases)} cases wrong: {wrong}" if wrong else None


def stripping_can_cost_a_speaker():
    """strip_prefix also removes a real speaker, which is why it is not applied to the corpus"""
    got = tok.strip_prefix("Daniel Yergin: Surveillance With Prewitt and Keene")
    return None if got == "Surveillance With Prewitt and Keene" else f"got {got!r} — the known behaviour changed"


def extracted_terms():
    """extract_terms returns sorted unigrams and bigrams, minus the stopwords"""
    got = tok.extract_terms("Quantum Computing Stocks Rally on IonQ Results")
    if got != sorted(got):
        return "terms are not sorted"
    if "stocks" in got or "results" in got:
        return f"a stopword survived: {got}"
    return None if "quantum computing" in got and "ionq" in got else f"missing an expected term: {got}"


def term_verdicts():
    """is_generic and is_event answer as their stopword lists say"""
    wrong = [c for c, got, want in [
        ("stock", tok.is_generic("stock"), True),
        ("quantum comput", tok.is_generic("quantum comput"), False),
        ("bankruptcy filing", tok.is_event("bankruptcy filing"), True),
        ("quantum comput", tok.is_event("quantum comput"), False),
    ] if got is not want]
    return f"wrong verdict for {wrong}" if wrong else None


# ------------------------------------------------------------------- the detector
def a_new_word_is_found_an_old_one_is_not():
    """the novelty gate lets `widget` through and never `acme`, which was always there"""
    themes, _, rejects = det.detect(_corpus(), ASOF, **GATES)
    anchors = set(themes["theme"].str.split("@").list.first()) | set(rejects["anchor"])
    if "widget" not in anchors:
        return "widget was never even a candidate"
    return "acme became a candidate although it is in the baseline" if "acme" in anchors else None


def a_theme_emerges_once():
    """an anchor is promoted in at most one week — the week it emerged"""
    themes, _, _ = det.detect(_corpus(new_term_weeks=6), ASOF, **GATES)
    anchors = themes["theme"].str.split("@").list.first().to_list()
    repeated = [a for a in set(anchors) if anchors.count(a) > 1]
    return f"promoted more than once: {repeated}" if repeated else None


def the_three_frames_line_up():
    """themes, diagnostics and rejects are all keyed on `theme`, and the first two agree"""
    themes, diagnostics, rejects = det.detect(_corpus(), ASOF, **GATES)
    for name, frame, columns in [("themes", themes, ["theme", "week", "vocab"]),
                                 ("diagnostics", diagnostics, ["theme", "anchor", "mentions", "degree"]),
                                 ("rejects", rejects, ["theme", "anchor", "week", "caught"])]:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            return f"{name} is missing {missing}"
    return None if set(themes["theme"]) == set(diagnostics["theme"]) else "themes and diagnostics disagree"


def bridging_is_measured_not_assumed():
    """a promoted anchor bridges: its partners are less connected than cluster_max allows"""
    themes, diagnostics, _ = det.detect(_corpus(), ASOF, **GATES)
    widget = themes.join(diagnostics, on="theme").filter(pl.col("anchor") == "widget")
    if widget.is_empty():
        return "widget was not promoted, so there is nothing to measure"
    clustering, vocab = widget["clustering"][0], widget["vocab"][0].to_list()
    if not 0.0 <= clustering <= 0.60:
        return f"clustering is {clustering}, which is not the bridging shape promotion requires"
    return None if vocab and vocab[0] == "widget" else f"vocabulary does not start with the anchor: {vocab[:3]}"


def genai_is_found_again():
    """the case this detector solved in reality: `chatgpt` emerges in the week of 2023-01-02

    The only end-to-end check there is — real corpus, default gates, no hint of what to look for.
    Slow (it reads 29 months of headlines) and worth it: everything else here is synthetic.
    """
    import config
    themes, diagnostics, _ = det.detect(pl.scan_parquet(config.CORPUS), "2023-01-16",
                                        report=False, **config.params("detect"))
    found = themes.join(diagnostics, on="theme").filter(pl.col("anchor") == "chatgpt")
    if found.is_empty():
        return f"chatgpt was not promoted; {len(themes)} other themes were"
    week = str(found["week"][0])[:10]
    return None if week == "2023-01-02" else f"promoted in the week of {week}, expected 2023-01-02"


def the_contract_carries_a_list_per_consumer():
    """themes is [theme, week, vocab, vocab_wide] — one word list for each stage downstream"""
    want = ["theme", "week", "vocab", "vocab_wide"]
    got = list(det._THEMES)
    return f"themes is {got}" if got != want else None


def the_two_lists_differ_only_by_the_common_words():
    """vocab is vocab_wide minus the words too common for their presence to mean anything"""
    narrow, wide = det._two("chatgpt", ["google", "openai", "iphone"], {"google", "iphone"}, -1)
    if wide != ["chatgpt", "google", "openai", "iphone"]:
        return f"vocab_wide is {wide}"
    if narrow != ["chatgpt", "openai"]:
        return f"vocab is {narrow}"
    short, _ = det._two("chatgpt", ["google", "openai", "iphone"], {"google", "iphone"}, 1)
    return f"size ignored: {short}" if short != ["chatgpt", "openai"] else None


CHECKS = [prefix_stripping, stripping_can_cost_a_speaker, extracted_terms, term_verdicts,
          a_new_word_is_found_an_old_one_is_not, a_theme_emerges_once,
          the_three_frames_line_up, bridging_is_measured_not_assumed,
          the_contract_carries_a_list_per_consumer, the_two_lists_differ_only_by_the_common_words,
          genai_is_found_again]
