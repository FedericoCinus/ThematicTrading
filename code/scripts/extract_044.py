"""GPT extraction for notebook 0.44 — schema 0.3, guideline 044-v1, cached and repeatable.

    from extract_044 import extract, run_many
    extract("Visa to Buy Fintech Startup Plaid for $5.3 Billion", "2020-01-13", model="gpt-4o", dated=True)

The model gets the same guideline the gold annotator used (0.42 showed that judging GPT against
rules it never saw measures the rules, not GPT) plus three short invented examples. The only
difference between the two arms of the leakage test is the line `Current date: ...`.

Every call is keyed by (model, messages, full JSON schema, params, rep) and appended to
data/processed/gpt_cache_0.44.jsonl, so re-running the notebook costs nothing and returns what was
measured. Set RUN_API = False to forbid new calls (a missing answer then raises).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
GOLD = CODE / "notebooks" / "gold" / "0.44"
CACHE = CODE / "data" / "processed" / "gpt_cache_0.44.jsonl"
RUN_API = True
API_MODELS = ["gpt-4o", "gpt-5.5"]
# open-weight models served locally by llama.cpp (llama-server, OpenAI-compatible, JSON-schema grammar):
#   llama-server -m <gguf> -c 32768 -np 4 --port 8080
LOCAL = {"qwen3-30b-a3b": dict(url="http://127.0.0.1:8080/v1", repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                               file="Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")}
MODELS = API_MODELS + list(LOCAL)
PARAMS = {"gpt-4o": {"temperature": 0}, "gpt-5.5": {"reasoning_effort": "none"}, "qwen3-30b-a3b": {"temperature": 0}}

ENTITY_TYPES = ["COMPANY", "ORGANIZATION", "PRODUCT", "CONCEPT", "SECTOR", "LOCATION", "INDICATOR"]
DRIVERS = ["SUPPLY", "DEMAND", "REVENUE", "EFFICIENCY_COST", "STRATEGIC_ACTION", "TECHNOLOGY_INNOVATION",
           "POLICY_REGULATION", "MACRO"]
PREDICATES = ["ACQUIRES", "INVESTS_IN", "PARTNERS_WITH", "SUPPLIES_TO", "OFFERS", "ACTIVE_IN", "IMPACTS"]
MODALITIES = ["reported", "planned", "forecast", "uncertain", "negated"]


def _nullable(values):
    return {"anyOf": [{"type": "string", "enum": values}, {"type": "null"}]}


def _obj(props):
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


STR, NSTR, IDS = {"type": "string"}, {"type": ["string", "null"]}, {"type": "array", "items": {"type": "string"}}
SCHEMA = {
    "name": "headline_graph_044",
    "strict": True,
    "schema": _obj({
        "entities": {"type": "array", "items": _obj({
            "id": STR, "type": {"type": "string", "enum": ENTITY_TYPES}, "mention": STR, "owner_id": NSTR})},
        "events": {"type": "array", "items": _obj({
            "id": STR, "trigger": STR, "description": STR, "driver": _nullable(DRIVERS),
            "actor_ids": IDS, "target_ids": IDS, "modality": _nullable(MODALITIES), "event_time": NSTR,
            "evidence": STR})},
        "claims": {"type": "array", "items": _obj({
            "subject_id": STR, "predicate": {"type": "string", "enum": PREDICATES}, "object_id": STR,
            "event_id": NSTR, "product_id": NSTR, "modality": _nullable(MODALITIES),
            "impact_polarity": _nullable(["positive", "negative"]), "evidence": STR})},
    }),
}

INSTRUCTION = ("Extract entities, events and relations explicitly stated in the headline.\n"
               "Preserve mentions and modality; quote evidence.\n"
               "Leave unsupported fields empty or null.")


def _e(i, t, m, o=None):
    return {"id": i, "type": t, "mention": m, "owner_id": o}


def _v(i, trig, desc, drv, act, tgt, mod, ev):
    return {"id": i, "trigger": trig, "description": desc, "driver": drv, "actor_ids": act, "target_ids": tgt,
            "modality": mod, "event_time": None, "evidence": ev}


def _c(s, p, o, ev_id, mod, pol, ev):
    return {"subject_id": s, "predicate": p, "object_id": o, "event_id": ev_id, "product_id": None,
            "modality": mod, "impact_polarity": pol, "evidence": ev}


EXAMPLES = [
    ("Chip Maker Corvo Agrees to Buy Rival Dune for $2 Billion", {
        "entities": [_e("e1", "COMPANY", "Corvo"), _e("e2", "PRODUCT", "Chip"), _e("e3", "COMPANY", "Dune")],
        "events": [_v("v1", "Agrees to Buy", "Corvo agreed to buy Dune for $2 billion.", "STRATEGIC_ACTION",
                      ["e1"], ["e3"], "planned", "Corvo Agrees to Buy Rival Dune for $2 Billion")],
        "claims": [_c("e1", "ACTIVE_IN", "e2", None, "reported", None, "Chip Maker Corvo"),
                   _c("e1", "ACQUIRES", "e3", "v1", "planned", None, "Corvo Agrees to Buy Rival Dune")]}),
    ("Borea Raised to Buy at Vela Securities on AI Chip Demand", {
        "entities": [_e("e1", "COMPANY", "Borea"), _e("e2", "COMPANY", "Vela Securities"),
                     _e("e3", "PRODUCT", "AI Chip"), _e("e4", "INDICATOR", "Demand", "e3")],
        "events": [],
        "claims": [_c("e4", "IMPACTS", "e1", None, "forecast", "positive",
                      "Borea Raised to Buy at Vela Securities on AI Chip Demand")]}),
    ("Alba Plans to Expand Battery Capacity After Record Orders", {
        "entities": [_e("e1", "COMPANY", "Alba"), _e("e2", "PRODUCT", "Battery"),
                     _e("e3", "INDICATOR", "Capacity", "e1"), _e("e4", "INDICATOR", "Orders")],
        "events": [_v("v1", "Plans to Expand", "Alba plans to expand battery capacity.", "SUPPLY", ["e1"],
                      ["e3", "e2"], "planned", "Alba Plans to Expand Battery Capacity"),
                   _v("v2", "Record Orders", "Orders reached a record.", "DEMAND", [], ["e4"], "reported",
                      "Record Orders")],
        "claims": []}),
]


def system_prompt():
    guideline = (GOLD / "guideline.md").read_text().strip()
    shots = "\n\n".join(f"Headline: {h}\nOutput: {json.dumps(o, ensure_ascii=False)}" for h, o in EXAMPLES)
    return f"{INSTRUCTION}\n\n{guideline}\n\nExamples (invented):\n\n{shots}"


def messages(headline, as_of=None, dated=True):
    user = (f"Current date: {as_of}.\n" if dated else "") + f"Headline: {headline}"
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


_cache: dict | None = None
_client = None
_local_clients = {}
_lock = threading.Lock()


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    _cache[r["key"]] = r
    return _cache


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI
        env = (CODE / ".env").read_text()
        key = re.search(r"^OPENAI_API_KEY=(.+)$", env, re.M).group(1).strip().strip('"').strip("'")
        _client = OpenAI(api_key=key)
    return _client


def _client_for(model):
    if model in LOCAL:
        if model not in _local_clients:
            from openai import OpenAI
            _local_clients[model] = OpenAI(base_url=LOCAL[model]["url"], api_key="local", timeout=600)
        return _local_clients[model]
    return _openai()


def call(model, msgs, rep=0, schema="default"):
    schema = SCHEMA if schema == "default" else schema
    params = PARAMS[model]
    key = hashlib.sha256(json.dumps([model, msgs, schema, params, rep], sort_keys=True).encode()).hexdigest()
    with _lock:
        if key in _load():
            return _cache[key]
    if not RUN_API:
        raise RuntimeError(f"not in cache and RUN_API is False: {model} {msgs[-1]['content'][:60]}")
    t0 = time.time()
    kw = dict(model=model, messages=msgs, **params)
    if schema is not None:
        kw["response_format"] = {"type": "json_schema", "json_schema": schema}
    for attempt in range(4):
        try:
            r = _client_for(model).chat.completions.create(**kw)
            break
        except Exception as exc:                       # rate limits and transient errors: retry, then record
            if attempt == 3:
                rec = dict(key=key, model=model, served_by=None, rep=rep, content=None, error=repr(exc)[:300],
                           seconds=round(time.time() - t0, 2))
                break
            time.sleep(2 ** attempt * 3)
    else:
        rec = None
    if "rec" not in locals() or rec is None:
        u = r.usage
        details = getattr(u, "completion_tokens_details", None)
        rec = dict(key=key, model=model, served_by=r.model, rep=rep, content=r.choices[0].message.content,
                   refusal=getattr(r.choices[0].message, "refusal", None), finish=r.choices[0].finish_reason,
                   prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                   reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
                   seconds=round(time.time() - t0, 2))
    with _lock:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _cache[key] = rec
    return rec


def extract(headline, as_of=None, model="gpt-4o", dated=True, rep=0):
    """Parsed record (entities, events, claims) plus `_meta`; a failed call returns empty lists + error."""
    rec = call(model, messages(headline, as_of, dated), rep)
    try:
        out = json.loads(rec["content"])
    except (TypeError, json.JSONDecodeError):
        out = {"entities": [], "events": [], "claims": [], "_failed": rec.get("error") or rec.get("refusal") or "unparsable"}
    out["_meta"] = {k: rec.get(k) for k in ("model", "served_by", "prompt_tokens", "completion_tokens",
                                            "reasoning_tokens", "seconds", "finish")}
    out["_raw"] = rec.get("content")
    return out


def run_many(fn, jobs, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: fn(*j), jobs))


def load_jsonl(name):
    return [json.loads(l) for l in (GOLD / name).read_text().splitlines() if l.strip()]


# ─────────── E2 controls: does the model know the future, and does it leak when not told not to? ───────────

NAIVE_SCHEMA = {
    "name": "headline_extraction_naive",
    "strict": True,
    "schema": _obj({
        "entities": {"type": "array", "items": _obj({
            "text": STR, "type": {"type": "string", "enum": ["ORG", "PERSON"]}, "canonical_name": STR,
            "ticker": NSTR, "industry": NSTR})},
        "relation": _obj({
            "predicate": {"type": "string", "enum": ["acquires", "sells", "invests in", "supplies", "partners with",
                                                    "sues", "none"]},
            "subject": NSTR, "object": NSTR,
            "deal_status": {"type": "string", "enum": ["announced", "pending", "completed", "terminated", "rumored", "none"]}}),
    }),
}
NAIVE_TASK = (
    "Extract from this financial news headline:\n"
    "1. entities: every organisation or person named in the headline. For each give the text as written, its type "
    "(ORG or PERSON), the company's canonical name, the stock ticker of its primary listing if it is a listed "
    "company, otherwise null, and its industry.\n"
    "2. relation: at most one economic relation between two actors named in the headline - predicate one of "
    "acquires, sells, invests in, supplies, partners with, sues, or none; the subject and the object; and the "
    "deal_status (announced, pending, completed, terminated, rumored, or none).")


def naive_messages(headline, as_of=None, dated=True):
    """The prompt of 0.42 ('naive'): no guideline, no rule against memory, fields that invite completion."""
    user = (f"Current date: {as_of}.\n" if dated else "") + f"Headline: {headline}\n\n{NAIVE_TASK}"
    return [{"role": "system", "content": "You extract structured information from financial news headlines."},
            {"role": "user", "content": user}]


def extract_naive(headline, as_of=None, model="gpt-4o", dated=True, rep=0):
    rec = call(model, naive_messages(headline, as_of, dated), rep, schema=NAIVE_SCHEMA)
    try:
        out = json.loads(rec["content"])
    except (TypeError, json.JSONDecodeError):
        out = {"entities": [], "relation": {}, "_failed": rec.get("error") or "unparsable"}
    out["_raw"] = rec.get("content")
    return out


def knowledge_messages(headline, as_of):
    """Positive control: ask directly what happened next. If the model cannot say, the trap is void."""
    return [{"role": "user", "content": f"A news headline from {as_of}: \"{headline}\"\n"
             "What happened afterwards to the companies or the deal in this headline? Answer in at most three sentences."}]


def knowledge(headline, as_of, model="gpt-4o", rep=0):
    rec = call(model, knowledge_messages(headline, as_of), rep, schema=None)
    return rec.get("content") or ""


# ─────────────────────────── schema 0.4: nodes and edges only (no events, no indicators) ───────────────────────────

NODE_TYPES_V04 = ["COMPANY", "ORGANIZATION", "PRODUCT", "CONCEPT", "SECTOR", "LOCATION"]
PREDICATES_V04 = ["OFFERS", "ACTIVE_IN", "EXPOSED_TO", "INVESTS_IN", "ACQUIRES", "PARTNERS_WITH", "SUPPLIES_TO"]
MODALITIES_V04 = ["reported", "planned", "uncertain", "negated"]
SCHEMA_V04 = {
    "name": "headline_graph_044_v04",
    "strict": True,
    "schema": _obj({
        "nodes": {"type": "array", "items": _obj({
            "id": STR, "type": {"type": "string", "enum": NODE_TYPES_V04}, "mention": STR})},
        "edges": {"type": "array", "items": _obj({
            "subject_id": STR, "predicate": {"type": "string", "enum": PREDICATES_V04}, "object_id": STR,
            "modality": {"type": "string", "enum": MODALITIES_V04}, "evidence": STR})},
    }),
}
INSTRUCTION_V04 = ("Extract the named nodes and the edges explicitly stated in the headline.\n"
                   "Copy mentions verbatim; quote evidence.\n"
                   "A headline with no relation gives only its nodes.")


def _n(i, t, m):
    return {"id": i, "type": t, "mention": m}


def _ed(s, p, o, mod, ev):
    return {"subject_id": s, "predicate": p, "object_id": o, "modality": mod, "evidence": ev}


EXAMPLES_V04 = [
    ("Chip Maker Corvo Agrees to Buy Rival Dune for $2 Billion", {
        "nodes": [_n("n1", "COMPANY", "Corvo"), _n("n2", "PRODUCT", "Chip"), _n("n3", "COMPANY", "Dune")],
        "edges": [_ed("n1", "ACTIVE_IN", "n2", "reported", "Chip Maker Corvo"),
                  _ed("n1", "ACQUIRES", "n3", "planned", "Corvo Agrees to Buy Rival Dune")]}),
    ("Borea Raised to Buy at Vela Securities on AI Chip Demand", {
        "nodes": [_n("n1", "COMPANY", "Borea"), _n("n2", "COMPANY", "Vela Securities"), _n("n3", "PRODUCT", "AI Chip")],
        "edges": [_ed("n1", "EXPOSED_TO", "n3", "uncertain", "Borea Raised to Buy at Vela Securities on AI Chip Demand")]}),
    ("Alba Wins Order to Supply Batteries to Orsa After Record Quarter", {
        "nodes": [_n("n1", "COMPANY", "Alba"), _n("n2", "PRODUCT", "Batteries"), _n("n3", "COMPANY", "Orsa")],
        "edges": [_ed("n1", "SUPPLIES_TO", "n3", "reported", "Alba Wins Order to Supply Batteries to Orsa"),
                  _ed("n1", "OFFERS", "n2", "reported", "Alba Wins Order to Supply Batteries")]}),
    ("Italy's Nerina Bank Shares Fall 5%, Pippo Research Says", {
        "nodes": [_n("n1", "COMPANY", "Nerina Bank"), _n("n2", "COMPANY", "Pippo Research")],
        "edges": []}),
]


def system_prompt_v04():
    guideline = (GOLD / "guideline_v04.md").read_text().strip()
    shots = "\n\n".join(f"Headline: {h}\nOutput: {json.dumps(o, ensure_ascii=False)}" for h, o in EXAMPLES_V04)
    return f"{INSTRUCTION_V04}\n\n{guideline}\n\nExamples (invented):\n\n{shots}"


def messages_v04(headline, as_of=None, dated=True):
    user = (f"Current date: {as_of}.\n" if dated else "") + f"Headline: {headline}"
    return [{"role": "system", "content": system_prompt_v04()}, {"role": "user", "content": user}]


def extract_v04(headline, as_of=None, model="gpt-4o", dated=True, rep=0):
    """Schema 0.4 record: {nodes, edges} plus `_meta`; a failed call returns empty lists + `_failed`."""
    rec = call(model, messages_v04(headline, as_of, dated), rep, schema=SCHEMA_V04)
    try:
        out = json.loads(rec["content"])
    except (TypeError, json.JSONDecodeError):
        out = {"nodes": [], "edges": [], "_failed": rec.get("error") or rec.get("refusal") or "unparsable"}
    out["_meta"] = {k: rec.get(k) for k in ("model", "served_by", "prompt_tokens", "completion_tokens",
                                            "reasoning_tokens", "seconds", "finish")}
    return out
