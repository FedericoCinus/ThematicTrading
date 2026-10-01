"""Scoring for notebook 0.42 — entities, relations, and GPT leakage probes.

Matching is lenient on surface form and strict on meaning: "Broadcom" matches "Broadcom Inc."
(containment after normalisation), but a predicate must be exactly the gold one and "no edge"
matches only "no edge".

RECALL, DEFINED PROPERLY
    recall = correct edges / relations in the gold set.
    0.41 divided by (correct + missed), which silently drops the true relations on which the
    pipeline drew an edge with the wrong subject, object or predicate. That flatters recall —
    on 0.41's held-out set it reads 56 % one way and 45 % the other.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata

PRED_CANON = {"acquires", "sells", "invests in", "supplies", "partners with", "sues"}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"'s\b", "", s)
    return re.sub(r"[^a-z0-9&]+", " ", s).strip()


def match(a, b) -> bool:
    a, b = norm(a), norm(b)
    return len(a) > 1 and len(b) > 1 and (a == b or a in b or b in a)


def any_match(x, xs) -> bool:
    return any(match(x, y) for y in (xs or []))


# ────────────────────────────── entities ──────────────────────────────

def entity_counts(pred, gold):
    """(tp, fp, fn) for one headline. pred: list[str]; gold: list[str]."""
    tp = sum(1 for g in gold if any_match(g, pred))
    fp = sum(1 for p in pred if not any_match(p, gold))
    return tp, fp, len(gold) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)


# ────────────────────────────── relations ──────────────────────────────

def edge_outcome(pred, gold, alternatives=()):
    """Best outcome over the gold relation and any alternative the guideline also accepts."""
    outs = [_edge_outcome(pred, g) for g in [gold, *alternatives]]
    for o in ("correct", "clean", "wrong", "missed", "false edge"):
        if o in outs:
            return o


def _edge_outcome(pred, gold):
    """One of 'correct', 'wrong', 'missed', 'clean', 'false edge'.

    pred: dict with subject/predicate/object (subject None = no edge).
    gold: dict with predicate ('none' = no edge), subjects[], objects[].
    """
    is_gold = gold["predicate"] != "none"
    drew = pred.get("subject") is not None
    if not is_gold:
        return "false edge" if drew else "clean"
    if not drew:
        return "missed"
    if pred.get("predicate") != gold["predicate"]:
        return "wrong"
    s, o = pred["subject"], pred["object"]
    direct = any_match(s, gold["subjects"]) and any_match(o, gold["objects"])
    swapped = any_match(s, gold["objects"]) and any_match(o, gold["subjects"])
    ok = direct or (gold["predicate"] == "partners with" and swapped)
    return "correct" if ok else "wrong"


def relation_metrics(outcomes):
    c = {k: outcomes.count(k) for k in ("correct", "wrong", "missed", "clean", "false edge")}
    drawn = c["correct"] + c["wrong"] + c["false edge"]
    gold_edges = c["correct"] + c["wrong"] + c["missed"]
    gold_none = c["clean"] + c["false edge"]
    p = c["correct"] / drawn if drawn else 0.0
    r = c["correct"] / gold_edges if gold_edges else 0.0
    return dict(**c, precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
                specificity=c["clean"] / gold_none if gold_none else 0.0)


def bootstrap(outcomes, stat, n=2000, seed=0):
    """95 % percentile interval of `stat(sampled outcomes)` — resampling headlines."""
    rng = random.Random(seed)
    vals = sorted(stat([outcomes[rng.randrange(len(outcomes))] for _ in outcomes]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def mcnemar(a_ok, b_ok):
    """Exact two-sided McNemar p-value on paired correctness (lists of bools)."""
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), b, c


# ────────────────────────────── leakage probes ──────────────────────────────

def _strings(out):
    """Every string a model produced for one headline, for signature search."""
    xs = []
    for e in out.get("entities", []) or []:
        if isinstance(e, dict):
            xs += [e.get("text"), e.get("canonical_name"), e.get("ticker")]
        else:
            xs.append(e)
    rel = out.get("relation") or {}
    xs += [rel.get("subject"), rel.get("object")]
    return [x for x in xs if x]


def leak_flags(out, probe):
    """Which kinds of future knowledge `out` contains, given the probe's signatures.

    A future name or counterparty only counts if it is NOT in the headline itself: a model that
    copies the text cannot be leaking.
    """
    head = norm(probe["headline"])
    produced = _strings(out)
    rel = out.get("relation") or {}

    def contains(big, small):           # whole words, one direction: the signature inside the answer
        b, t = norm(big).split(), norm(small).split()
        return bool(t) and any(b[i:i + len(t)] == t for i in range(len(b) - len(t) + 1))

    def novel(sig):
        return bool(norm(sig)) and not contains(probe["headline"], sig) and any(contains(x, sig) for x in produced)

    name = any(novel(s) for s in probe.get("future_names") or [])
    party = any(novel(s) for s in probe.get("future_counterparties") or [])

    def tick(t):                            # "VMW US" and "VMW" are the same ticker
        return str(t or "").upper().split()[0] if str(t or "").strip() else ""
    fut_t, pit_t = tick(probe.get("future_ticker")), tick(probe.get("main_entity_ticker_pit"))
    tickers = [tick(e.get("ticker")) for e in out.get("entities", []) or [] if isinstance(e, dict)]
    ticker = bool(fut_t) and fut_t != pit_t and fut_t in tickers

    # a later predicate replacing the one the headline states — a co-mention promoted to a
    # takeover, or a live deal dropped because it later died
    fp, pp = (probe.get("future_predicate") or "").strip(), probe["relation_pit"]["predicate"]
    predicate = bool(fp) and fp != pp and (rel.get("predicate") or "none") == fp

    fs, ds = (probe.get("future_status") or "").lower(), (rel.get("deal_status") or "none")
    ended = ("terminat" in fs or "abandon" in fs or "block" in fs or "collaps" in fs or "withdr" in fs)
    status = bool(fs) and ds != probe.get("deal_status_pit") and (
        (ended and ds == "terminated") or ("complet" in fs and ds == "completed"))
    return dict(name=name, counterparty=party, ticker=ticker, predicate=predicate, status=status,
                any=name or party or ticker or predicate or status)


def pit_correct(out, probe):
    """Is the extraction right as of the headline date? relation (+ status when there is a deal)."""
    rel = out.get("relation") or {}
    gold = probe["relation_pit"]
    pred = dict(subject=rel.get("subject") if rel.get("predicate", "none") != "none" else None,
                predicate=rel.get("predicate"), object=rel.get("object"))
    ok = edge_outcome(pred, gold) in ("correct", "clean")
    if gold["predicate"] != "none" and probe.get("deal_status_pit") not in (None, "", "none"):
        ok = ok and rel.get("deal_status") in ("announced", "pending") and \
            probe["deal_status_pit"] in ("announced", "pending")
    return ok
