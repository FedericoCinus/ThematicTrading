"""Scoring for notebook 0.44: extraction F1 on real headlines (E1) and future leaks on probes (E2).

E1 — four scores per system, all micro-averaged over the 50 headlines:
    entities      mention + type. Mentions match after normalisation if equal or one contains the other.
    claims        subject + predicate + object (PARTNERS_WITH in either order); "strict" also needs an
                  accepted modality and, for IMPACTS, polarity.
    participants  entities linked to an event as ACTOR or TARGET (role label not scored): the
                  company-to-product/concept links the theme graph is built from.
    events        aligned by trigger (any shared content word); driver and modality are scored on
                  aligned events only.
A gold item marked optional is neither required nor wrong: matching it earns nothing and costs nothing.
Alignment is one-to-one; a type error counts once as FP and once as FN.

E2 — a leak is a future signature (from the probe) appearing in anything the model wrote but absent
from the headline, a deal predicate between the two co-mentioned firms, or 'negated' on a deal that
later failed. Every flag is shown verbatim for a human to confirm.
"""
from __future__ import annotations

import random
import re
import unicodedata

STOP = {"to", "the", "a", "an", "of", "in", "on", "for", "with", "and", "at", "by", "from", "as", "its", "s"}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’]s\b", "", s)
    return re.sub(r"[^a-z0-9&]+", " ", s).strip()


def same(a, b) -> bool:
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a == b or (min(len(a), len(b)) > 1 and (a in b or b in a)))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f)


# ─────────────────────────────── alignment ───────────────────────────────

def align_entities(pred, gold, typed):
    """{pred_id: gold_id}, one-to-one: exact normalised mention first, then containment."""
    out, used = {}, set()
    for exact in (True, False):
        for p in pred:
            if p["id"] in out:
                continue
            for g in gold:
                if g["id"] in used or (typed and p["type"] not in g["types"]):
                    continue
                ok = norm(p["mention"]) == norm(g["mention"]) if exact else same(p["mention"], g["mention"])
                if ok:
                    out[p["id"]] = g["id"]; used.add(g["id"]); break
    return out


def words(s):
    return {w for w in norm(s).split() if w not in STOP}


def align_events(pred, gold, ent_map=None):
    """Shared trigger word first; then, for events still unaligned, a shared participant.

    The fallback was added after reading the outputs: in verbless headlines ("TOPBUILD 2Q NET SALES
    $1.37B") the model's trigger ("$1.37B") and the gold's ("NET SALES") share no word although they
    are the same fact. It applies identically to every system.
    """
    pairs = sorted(((len(words(p["trigger"]) & words(g["trigger"])), p["id"], g["id"])
                    for p in pred for g in gold), reverse=True)
    out, used = {}, set()
    for k, pid, gid in pairs:
        if k and pid not in out and gid not in used:
            out[pid] = gid; used.add(gid)
    if ent_map is not None:
        for p in pred:
            if p["id"] in out:
                continue
            mine = {ent_map.get(x) for x in p.get("actor_ids", []) + p.get("target_ids", [])} - {None}
            for g in gold:
                if g["id"] not in used and mine & set(g["parts"]):
                    out[p["id"]] = g["id"]; used.add(g["id"]); break
    return out


# ─────────────────────────────── E1 ───────────────────────────────

def _count(pred_keys, gold_req, gold_opt):
    tp = len(pred_keys & gold_req)
    fp = len(pred_keys - gold_req - gold_opt)
    return tp, fp, len(gold_req - pred_keys)


def score_record(pred, gold, has_events=True):
    """Counts for one headline. pred: model record; gold: one line of real_50_gold.jsonl."""
    P = {k: pred.get(k) or [] for k in ("entities", "events", "claims")}
    G = gold
    opt_e = {e["id"] for e in G["entities"] if e["opt"]}
    req_e = {e["id"] for e in G["entities"] if not e["opt"]}
    out = {}

    typed = align_entities(P["entities"], G["entities"], typed=True)
    ent_map = align_entities(P["entities"], G["entities"], typed=False)
    tp = sum(1 for g in typed.values() if g in req_e)
    fp = sum(1 for p in P["entities"] if p["id"] not in typed and ent_map.get(p["id"]) not in opt_e)
    out["entities"] = (tp, fp, len(req_e) - tp)

    pmention = {p["id"]: p["mention"] for p in P["entities"]}
    ev_map = align_events(P["events"], G["events"], ent_map) if has_events else {}

    def end(x):
        if x in ent_map:
            return ent_map[x]
        if x in ev_map:
            return ev_map[x]
        return "?" + norm(pmention.get(x, x))

    # claims: dedupe identical predictions, then match one-to-one against gold claims
    preds, seen = [], set()
    for c in P["claims"]:
        k = (end(c["subject_id"]), c["predicate"], end(c["object_id"]))
        if k not in seen:
            seen.add(k); preds.append((k, c))
    ctp = cfp = strict = 0
    used = set()
    for (s, pr, o), c in preds:
        hit = None
        for j, g in enumerate(G["claims"]):
            if j in used or g["predicate"] != pr:
                continue
            ok = s in g["subjects"] and o in g["objects"]
            if pr == "PARTNERS_WITH":
                ok = ok or (o in g["subjects"] and s in g["objects"])
            if ok:
                hit = j; break
        if hit is None:
            cfp += 1
            continue
        used.add(hit)
        g = G["claims"][hit]
        if g["opt"]:
            continue
        ctp += 1
        pol_ok = pr != "IMPACTS" or c.get("impact_polarity") in g["polarities"]
        strict += int(c.get("modality") in g["modalities"] and pol_ok)
    creq = sum(1 for g in G["claims"] if not g["opt"])
    out["claims"] = (ctp, cfp, creq - ctp)
    if not has_events:                      # no events, no modality: strict and event scores are NA
        return out
    out["claims_strict"] = (strict, cfp + ctp - strict, creq - strict)
    req_ev = [v for v in G["events"] if not v["opt"]]
    gold_parts = {x for v in req_ev for x in v["parts"]}
    opt_parts = {x for v in G["events"] for x in v["opt_parts"]} | {x for v in G["events"] if v["opt"] for x in v["parts"]} | opt_e
    pred_parts = {end(x) for v in P["events"] for x in v.get("actor_ids", []) + v.get("target_ids", [])}
    out["participants"] = _count(pred_parts, gold_parts, opt_parts)

    req_ids = {v["id"] for v in req_ev}
    etp = sum(1 for g in ev_map.values() if g in req_ids)
    efp = sum(1 for v in P["events"] if v["id"] not in ev_map)
    out["events"] = (etp, efp, len(req_ids) - etp)
    gv = {v["id"]: v for v in G["events"]}
    drv = [(v.get("driver") in gv[ev_map[v["id"]]]["drivers"]) for v in P["events"]
           if v["id"] in ev_map and ev_map[v["id"]] in req_ids]
    mod = [(v.get("modality") in gv[ev_map[v["id"]]]["modalities"]) for v in P["events"]
           if v["id"] in ev_map and ev_map[v["id"]] in req_ids]
    out["driver_ok"] = (sum(drv), len(drv))
    out["modality_ok"] = (sum(mod), len(mod))
    return out


FAMILIES = ["entities", "claims", "claims_strict", "participants", "events"]


def aggregate(per_headline):
    """per_headline: list of score_record outputs. Returns {family: prf} + driver/modality accuracy."""
    res = {}
    for fam in FAMILIES:
        rows = [r[fam] for r in per_headline if fam in r]
        if rows:
            res[fam] = prf(*map(sum, zip(*rows)))
    for k in ("driver_ok", "modality_ok"):
        rows = [r[k] for r in per_headline if k in r]
        if rows:
            ok, n = map(sum, zip(*rows))
            res[k] = ok / n if n else float("nan")
    return res


def bootstrap_f1(per_headline, fam, n=2000, seed=0):
    rng = random.Random(seed)
    rows = [r[fam] for r in per_headline if fam in r]
    vals = []
    for _ in range(n):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        tp, fp, fn = map(sum, zip(*s))
        vals.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def paired_diff(a, b, fam, n=2000, seed=0):
    """95% interval of F1(a) - F1(b), resampling the same headlines for both systems."""
    rng = random.Random(seed)
    ra, rb = [r[fam] for r in a], [r[fam] for r in b]
    vals = []
    for _ in range(n):
        idx = [rng.randrange(len(ra)) for _ in ra]
        fa = [sum(ra[i][k] for i in idx) for k in range(3)]
        fb = [sum(rb[i][k] for i in idx) for k in range(3)]
        f = lambda t: 2 * t[0] / (2 * t[0] + t[1] + t[2]) if sum(t) else 0.0
        vals.append(f(fa) - f(fb))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


# ─────────────────────────────── E2 ───────────────────────────────

def _strings(out):
    xs = []
    for k in ("entities", "events", "claims"):
        for item in out.get(k) or []:
            xs += [v for f, v in item.items() if isinstance(v, str) and not f.endswith("id") and f not in (
                "type", "predicate", "driver", "modality", "impact_polarity")]
    return xs


def _sig_re(sig):
    body = re.escape(sig.lower()).replace(r"\*", r"\w*")
    return re.compile(rf"(?<![\w]){body}(?![\w])" if not sig.endswith("*") else rf"(?<![\w]){body}")


def leak_flags(out, probe):
    sigs = _sig_hits(" \n ".join(_strings(out)), probe)
    names = {e["id"]: e["mention"] for e in out.get("entities") or []}

    def firm(mention, name):             # "PDC" is PDC Energy, "Energy" is not
        m, n = norm(mention).split(), norm(name).split()
        return bool(m) and bool(n) and m[0] == n[0]

    pair = False
    if probe.get("pair"):
        a, b = probe["pair"]
        for c in out.get("claims") or []:
            s, o = names.get(c["subject_id"], ""), names.get(c["object_id"], "")
            if c["predicate"] in ("ACQUIRES", "PARTNERS_WITH", "INVESTS_IN") and (
                    (firm(s, a) and firm(o, b)) or (firm(s, b) and firm(o, a))):
                pair = True
    status = bool(probe.get("status")) and any(
        x.get("modality") == "negated" for x in (out.get("events") or []) + (out.get("claims") or []))
    return dict(signatures=sigs, pair=pair, status=status, any=bool(sigs) or pair or status)


def entity_recall(out, probe):
    got = [e["mention"] for e in out.get("entities") or []]
    return sum(1 for g in probe["entities"] if any(same(g, m) for m in got)), len(probe["entities"])


def unsupported(out, headline):
    """Entity mentions that are not in the headline: the channel through which a future name would enter."""
    return [e["mention"] for e in out.get("entities") or [] if norm(e["mention"]) not in norm(headline)]


# ─────────────────────────────── E1 by node and edge type ───────────────────────────────

SIGNATURE = {   # allowed endpoint types of schema 0.3
    "ACQUIRES": (["COMPANY"], ["COMPANY"]),
    "INVESTS_IN": (["COMPANY", "ORGANIZATION"], ["COMPANY", "PRODUCT", "CONCEPT", "SECTOR", "LOCATION"]),
    "PARTNERS_WITH": (["COMPANY"], ["COMPANY"]),
    "SUPPLIES_TO": (["COMPANY"], ["COMPANY", "ORGANIZATION"]),
    "OFFERS": (["COMPANY", "ORGANIZATION"], ["PRODUCT", "CONCEPT"]),
    "ACTIVE_IN": (["COMPANY"], ["SECTOR", "CONCEPT", "PRODUCT", "LOCATION"]),
    "IMPACTS": (["CONCEPT", "PRODUCT", "EVENT", "ORGANIZATION", "INDICATOR"], ["COMPANY", "SECTOR", "PRODUCT", "INDICATOR"]),
}
NODE_KINDS = ["COMPANY", "ORGANIZATION", "PRODUCT", "CONCEPT", "SECTOR", "LOCATION", "INDICATOR", "EVENT"]
EDGE_KINDS = list(SIGNATURE) + ["ACTOR/TARGET"]


def _entity_fp_category(p, gold):
    m = p["mention"]
    if any(norm(m) and norm(m) in norm(g["mention"]) and norm(m) != norm(g["mention"]) for g in gold["entities"]):
        return "pezzo di un nome"
    if re.search(r"[\d$€¥£%]", m):
        return "importo o valore come entità"
    if re.search(r"\b(shares?|bonds?|futures|loans?|stocks?|report|10-k|notes?|bills?)\b", m, re.I):
        return "strumento finanziario o documento"
    if p["type"] == "LOCATION":
        return "luogo da nazionalità o possessivo"
    if p["type"] == "INDICATOR":
        return "parola generica come INDICATOR"
    return "entità generica o senza nome"


def _event_fp_category(v, headline):
    if re.search(r"\b(rated|raised to|cut to|upgraded?|downgraded?|affirms?|initiated|reinstated)\b", headline, re.I):
        return "rating come evento"
    if re.search(r"buy ?back", headline, re.I):
        return "buyback come evento"
    if re.search(r"gain|bounce|climb|jump|soar|fall|slump|hit|rall|advance|decline|plunge|record", v["trigger"], re.I):
        return "movimento di mercato come evento"
    return "altro (commento, stima, doppione)"


def breakdown(pred, gold, has_events=True):
    """Counts by node and edge type, plus one labelled line per error.

    counts[(level, kind)] = [tp, fp, fn], level 'nodo' or 'arco'. A missed gold entity with several
    accepted types is counted under its first type; a false positive under the predicted type.
    errors: dict(level, kind, esito 'FP'/'FN', categoria, dettaglio). Categories come from the
    explicit rules above, not from reading each case.
    """
    P = {k: pred.get(k) or [] for k in ("entities", "events", "claims")}
    G, head = gold, gold["headline"]
    counts = {("nodo", k): [0, 0, 0] for k in NODE_KINDS} | {("arco", k): [0, 0, 0] for k in EDGE_KINDS}
    errors = []

    def err(level, kind, esito, cat, det):
        errors.append(dict(livello=level, tipo=kind, esito=esito, categoria=cat, dettaglio=det))

    opt_e = {e["id"] for e in G["entities"] if e["opt"]}
    gent = {e["id"]: e for e in G["entities"]}
    typed = align_entities(P["entities"], G["entities"], typed=True)
    ent_map = align_entities(P["entities"], G["entities"], typed=False)
    for p in P["entities"]:
        if p["id"] in typed:
            if typed[p["id"]] not in opt_e:
                counts[("nodo", p["type"])][0] += 1
        elif ent_map.get(p["id"]) not in opt_e:
            counts[("nodo", p["type"])][1] += 1
            cat = "tipo sbagliato" if ent_map.get(p["id"]) in gent else _entity_fp_category(p, G)
            err("nodo", p["type"], "FP", cat, f"{p['mention']} [{p['type']}]")
    got = set(typed.values())
    untyped_hit = set(ent_map.values())
    for g in G["entities"]:
        if not g["opt"] and g["id"] not in got:
            counts[("nodo", g["types"][0])][2] += 1
            err("nodo", g["types"][0], "FN", "tipo sbagliato" if g["id"] in untyped_hit else "entità non trovata",
                f"{g['mention']} [{'|'.join(g['types'])}]")

    ev_map = align_events(P["events"], G["events"], ent_map) if has_events else {}
    ptype = {e["id"]: e["type"] for e in P["entities"]} | {v["id"]: "EVENT" for v in P["events"]}
    pname = {e["id"]: e["mention"] for e in P["entities"]} | {v["id"]: "«" + v["trigger"] + "»" for v in P["events"]}
    gname = {e["id"]: e["mention"] for e in G["entities"]} | {v["id"]: "«" + v["trigger"] + "»" for v in G["events"]}

    def end(x):
        return ent_map.get(x) or ev_map.get(x) or "?" + norm(pname.get(x, x))

    if has_events:
        req_ids = {v["id"] for v in G["events"] if not v["opt"]}
        for v in P["events"]:
            if v["id"] in ev_map:
                counts[("nodo", "EVENT")][0] += ev_map[v["id"]] in req_ids
            else:
                counts[("nodo", "EVENT")][1] += 1
                err("nodo", "EVENT", "FP", _event_fp_category(v, head), f"«{v['trigger']}»")
        for v in G["events"]:
            if v["id"] in req_ids and v["id"] not in ev_map.values():
                counts[("nodo", "EVENT")][2] += 1
                err("nodo", "EVENT", "FN", "evento non trovato", f"«{v['trigger']}»")

    seen, used = set(), set()
    for c in P["claims"]:
        s, pr, o = end(c["subject_id"]), c["predicate"], end(c["object_id"])
        if (s, pr, o) in seen:
            continue
        seen.add((s, pr, o))
        hit = None
        for j, g in enumerate(G["claims"]):
            if j in used or g["predicate"] != pr:
                continue
            if (s in g["subjects"] and o in g["objects"]) or (
                    pr == "PARTNERS_WITH" and o in g["subjects"] and s in g["objects"]):
                hit = j; break
        text = f"{pname.get(c['subject_id'], '?')} —{pr}→ {pname.get(c['object_id'], '?')}"
        if hit is not None:
            used.add(hit)
            counts[("arco", pr)][0] += not G["claims"][hit]["opt"]
            continue
        counts[("arco", pr)][1] += 1
        heads, tails = SIGNATURE[pr]
        if s.startswith("?") or o.startswith("?"):
            cat = "estremo che non è un'entità del gold"
        elif ptype.get(c["subject_id"]) not in heads or ptype.get(c["object_id"]) not in tails:
            cat = "tipo di estremo non ammesso"
        elif pr == "IMPACTS":
            cat = "IMPACTS senza causa esplicita"
        else:
            cat = f"{pr} non affermato"
        err("arco", pr, "FP", cat, text)
    for j, g in enumerate(G["claims"]):
        if not g["opt"] and j not in used:
            counts[("arco", g["predicate"])][2] += 1
            err("arco", g["predicate"], "FN", "relazione non trovata",
                f"{'|'.join(gname[x] for x in g['subjects'])} —{g['predicate']}→ {'|'.join(gname[x] for x in g['objects'])}")

    if has_events:
        req_ev = [v for v in G["events"] if not v["opt"]]
        gold_parts = {x for v in req_ev for x in v["parts"]}
        opt_parts = ({x for v in G["events"] for x in v["opt_parts"]} |
                     {x for v in G["events"] if v["opt"] for x in v["parts"]} | opt_e)
        pred_parts = {end(x) for v in P["events"] for x in v.get("actor_ids", []) + v.get("target_ids", [])}
        k = counts[("arco", "ACTOR/TARGET")]
        k[0] = len(pred_parts & gold_parts)
        for x in sorted(pred_parts - gold_parts - opt_parts):
            k[1] += 1
            err("arco", "ACTOR/TARGET", "FP", "partecipante che non è un'entità del gold" if x.startswith("?")
                else "partecipante in più", x.lstrip("?") if x.startswith("?") else gname.get(x, x))
        for x in sorted(gold_parts - pred_parts):
            k[2] += 1
            err("arco", "ACTOR/TARGET", "FN", "partecipante non trovato", gname.get(x, x))
    else:
        for kind in ("EVENT",):
            counts.pop(("nodo", kind))
        counts.pop(("arco", "ACTOR/TARGET"))
    return counts, errors


# ─────────────────────────────── E2 controls ───────────────────────────────

def _all_strings(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for k, v in x.items() if not k.startswith("_") for s in _all_strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _all_strings(v)]
    return []


def _sig_hits(text, probe):
    head = probe["headline"].lower()
    for ok in probe.get("allowed") or []:            # names correct on as_of that contain a signature
        text = re.sub(re.escape(ok), " ", text, flags=re.I)
    return [s for s in probe["sig"] if not _sig_re(s).search(head) and _sig_re(s).search(text.lower())]


def leak_flags_naive(out, probe):
    """Leaks in the naive record: signatures anywhere (names, tickers, industries, relation), a deal
    between the co-mentioned pair (F2), or the later deal status (F4)."""
    sigs = _sig_hits(" \n ".join(_all_strings({k: out.get(k) for k in ("entities", "relation")})), probe)
    rel = out.get("relation") or {}
    pair = False
    if probe.get("pair") and rel.get("predicate") in ("acquires", "partners with", "invests in"):
        a, b = probe["pair"]
        s, o = rel.get("subject") or "", rel.get("object") or ""
        first = lambda m, n: bool(norm(m)) and norm(m).split()[0] == norm(n).split()[0]
        pair = (first(s, a) and first(o, b)) or (first(s, b) and first(o, a))
    status = bool(probe.get("future_status")) and rel.get("deal_status") == probe["future_status"]
    return dict(signatures=sigs, pair=pair, status=status, any=bool(sigs) or pair or status)


def knows_future(answer, probe):
    """Positive control: the direct answer names the later fact (signature, or the later status word)."""
    hits = _sig_hits(answer, probe)
    fs = probe.get("future_status")
    if fs == "terminated" and re.search(r"terminat|abandon|scrap|collaps|called off|walked away|blocked|withdr", answer, re.I):
        hits.append("(esito: " + fs + ")")
    if fs == "completed" and re.search(r"complet|closed", answer, re.I):
        hits.append("(esito: " + fs + ")")
    return hits


# ─────────────────────────────── schema 0.4: nodes and edges only ───────────────────────────────

SIGNATURE_V04 = {
    "OFFERS": (["COMPANY", "ORGANIZATION"], ["PRODUCT", "CONCEPT"]),
    "ACTIVE_IN": (["COMPANY"], ["PRODUCT", "CONCEPT", "SECTOR", "LOCATION"]),
    "EXPOSED_TO": (["COMPANY", "SECTOR"], ["PRODUCT", "CONCEPT"]),
    "INVESTS_IN": (["COMPANY", "ORGANIZATION"], ["COMPANY", "PRODUCT", "CONCEPT", "SECTOR", "LOCATION"]),
    "ACQUIRES": (["COMPANY"], ["COMPANY"]),
    "PARTNERS_WITH": (["COMPANY"], ["COMPANY"]),
    "SUPPLIES_TO": (["COMPANY"], ["COMPANY", "ORGANIZATION"]),
}
NODE_KINDS_V04 = ["COMPANY", "ORGANIZATION", "PRODUCT", "CONCEPT", "SECTOR", "LOCATION"]
EDGE_KINDS_V04 = list(SIGNATURE_V04)
THEME_EDGES = ["OFFERS", "ACTIVE_IN", "EXPOSED_TO", "INVESTS_IN"]     # company -> product/concept/sector (INVESTS_IN: when the object is not a company)


def breakdown_v04(pred, gold):
    """Schema 0.4 counts by node and edge type, one labelled line per error, and modality agreement.

    pred: {nodes, edges}; gold: one line of real_50_gold_v04.jsonl. Same matching rules as 0.3:
    normalised mentions (equal or contained), one-to-one, optional gold items neither required nor wrong.
    Returns (counts, errors, modality) with counts[(level, kind)] = [tp, fp, fn] and
    modality = (edges matched with an accepted modality, edges matched).
    """
    nodes, edges = pred.get("nodes") or [], pred.get("edges") or []
    G = gold
    counts = {("nodo", k): [0, 0, 0] for k in NODE_KINDS_V04} | {("arco", k): [0, 0, 0] for k in EDGE_KINDS_V04}
    errors = []

    def err(level, kind, esito, cat, det):
        errors.append(dict(livello=level, tipo=kind, esito=esito, categoria=cat, dettaglio=det))

    opt_e = {e["id"] for e in G["entities"] if e["opt"]}
    gent = {e["id"]: e for e in G["entities"]}
    typed = align_entities(nodes, G["entities"], typed=True)
    ent_map = align_entities(nodes, G["entities"], typed=False)
    for p in nodes:
        kind = p["type"] if p["type"] in NODE_KINDS_V04 else "COMPANY"
        if p["id"] in typed:
            if typed[p["id"]] not in opt_e:
                counts[("nodo", kind)][0] += 1
        elif ent_map.get(p["id"]) not in opt_e:
            counts[("nodo", kind)][1] += 1
            cat = "tipo sbagliato" if ent_map.get(p["id"]) in gent else _entity_fp_category(p, G)
            if cat == "parola generica come INDICATOR":
                cat = "entità generica o senza nome"
            err("nodo", kind, "FP", cat, f"{p['mention']} [{p['type']}]")
    got, untyped_hit = set(typed.values()), set(ent_map.values())
    for g in G["entities"]:
        if not g["opt"] and g["id"] not in got:
            counts[("nodo", g["types"][0])][2] += 1
            err("nodo", g["types"][0], "FN", "tipo sbagliato" if g["id"] in untyped_hit else "entità non trovata",
                f"{g['mention']} [{'|'.join(g['types'])}]")

    ptype = {n["id"]: n["type"] for n in nodes}
    pname = {n["id"]: n["mention"] for n in nodes}
    gname = {e["id"]: e["mention"] for e in G["entities"]}
    end = lambda x: ent_map.get(x) or "?" + norm(pname.get(x, x))
    seen, used, mod_ok, mod_n = set(), set(), 0, 0
    for c in edges:
        s, pr, o = end(c["subject_id"]), c["predicate"], end(c["object_id"])
        if (s, pr, o) in seen:
            continue
        seen.add((s, pr, o))
        hit = None
        for j, g in enumerate(G["edges"]):
            if j in used or g["predicate"] != pr:
                continue
            if (s in g["subjects"] and o in g["objects"]) or (
                    pr == "PARTNERS_WITH" and o in g["subjects"] and s in g["objects"]):
                hit = j; break
        text = f"{pname.get(c['subject_id'], '?')} —{pr}→ {pname.get(c['object_id'], '?')}"
        if hit is not None:
            used.add(hit)
            if not G["edges"][hit]["opt"]:
                counts[("arco", pr)][0] += 1
                mod_n += 1
                mod_ok += c.get("modality") in G["edges"][hit]["modalities"]
            continue
        counts[("arco", pr)][1] += 1
        heads, tails = SIGNATURE_V04[pr]
        if s.startswith("?") or o.startswith("?"):
            cat = "estremo che non è un nodo del gold"
        elif ptype.get(c["subject_id"]) not in heads or ptype.get(c["object_id"]) not in tails:
            cat = "tipo di estremo non ammesso"
        else:
            cat = f"{pr} non affermato"
        err("arco", pr, "FP", cat, text)
    for j, g in enumerate(G["edges"]):
        if not g["opt"] and j not in used:
            counts[("arco", g["predicate"])][2] += 1
            err("arco", g["predicate"], "FN", "relazione non trovata",
                f"{'|'.join(gname[x] for x in g['subjects'])} —{g['predicate']}→ {'|'.join(gname[x] for x in g['objects'])}")
    return counts, errors, (mod_ok, mod_n)


def as_v03(out):
    """A 0.4 record seen through the 0.3 field names, to reuse leak_flags on E2."""
    return {"entities": [dict(n) for n in out.get("nodes") or []], "events": [],
            "claims": [dict(e) for e in out.get("edges") or []]}
