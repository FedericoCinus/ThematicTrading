"""GPT as a schema extractor, for comparison with the point-in-time models — cached and repeatable.

    from gpt_extract import extract, chooser
    extract("Credit Suisse Shares Slump to Record Low", "2023-03-15", model="gpt-4o", mode="dated")

Every call is keyed by (model, messages, params, rep) and stored in a JSON-lines cache under
data/processed/, so re-running a notebook costs nothing and returns exactly what was measured.
`rep` exists because the same request is not guaranteed to return the same answer: the
project's own FUTUREWORK_reproducibility.md records gpt-4o changing its output at temperature 0.
Asking three times is how that variance is measured instead of assumed away.

Two prompt modes, because "is GPT reliable in the past?" has two readings:
    naive   the headline alone — how an extractor is usually called
    dated   the headline plus its date and an explicit instruction to ignore anything later —
            the most a careful user can do. If GPT still leaks here, telling it not to is no fix.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
CACHE = CODE / "data" / "processed" / "gpt_cache_0.42.jsonl"
PREDICATES = ["acquires", "sells", "invests in", "supplies", "partners with", "sues", "none"]
STATUSES = ["announced", "pending", "completed", "terminated", "rumored", "none"]

# Knowledge cutoffs per OpenAI's model pages, checked by the 0.42 audit (developers.openai.com,
# fetched 2026-09-28). The alias "gpt-4o" is served by gpt-4o-2024-08-06. OpenAI prints the date
# as a month, and ChatGPT's release notes call gpt-4o's knowledge "November 2023", so outcomes
# between the two bounds are treated as undecidable for gpt-4o rather than as known or unknown.
CUTOFF = {"gpt-4o": "2023-10-01", "gpt-5.5": "2025-12-01"}
CUTOFF_LATE = {"gpt-4o": "2023-12-01", "gpt-5.5": "2026-01-01"}

_cache: dict | None = None
_client = None
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


def _reasoning(model):
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def call(model, messages, schema=None, rep=0):
    """One cached chat completion. Returns the stored record: content, usage, model id."""
    params = {"schema": schema["name"] if schema else None}
    if not _reasoning(model):
        params["temperature"] = 0
    key = hashlib.sha256(json.dumps([model, messages, params, rep], sort_keys=True).encode()).hexdigest()
    with _lock:
        cache = _load()
        if key in cache:
            return cache[key]
    kw = dict(model=model, messages=messages)
    if not _reasoning(model):
        kw["temperature"] = 0
    if schema:
        kw["response_format"] = {"type": "json_schema", "json_schema": schema}
    r = _openai().chat.completions.create(**kw)
    rec = dict(key=key, model=model, served_by=r.model, rep=rep, content=r.choices[0].message.content,
               prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens)
    with _lock:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _cache[key] = rec
    return rec


def run_many(fn, jobs, workers=8):
    """Run fn(*job) for every job concurrently; results in input order. Cached calls return at once."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: fn(*j), jobs))


SCHEMA = {
    "name": "headline_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entities": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ["ORG", "PERSON"]},
                    "canonical_name": {"type": "string"},
                    "ticker": {"type": ["string", "null"]},
                },
                "required": ["text", "type", "canonical_name", "ticker"]}},
            "relation": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "predicate": {"type": "string", "enum": PREDICATES},
                    "subject": {"type": ["string", "null"]},
                    "object": {"type": ["string", "null"]},
                    "deal_status": {"type": "string", "enum": STATUSES},
                },
                "required": ["predicate", "subject", "object", "deal_status"]},
        },
        "required": ["entities", "relation"],
    },
}

TASK = (
    "Extract from this financial news headline:\n"
    "1. entities: every organisation or person named in the headline. For each give the text as "
    "written, its type (ORG or PERSON), the company's canonical name, and the stock ticker of its "
    "primary listing if it is a listed company, otherwise null.\n"
    "2. relation: at most one economic relation between two economic actors named in the headline — "
    "predicate one of acquires, sells, invests in, supplies, partners with, sues, or none; the subject "
    "(the actor performing it) and the object; and the deal_status (announced, pending, completed, "
    "terminated, rumored, or none).\n"
    "Analyst ratings, bond issues, buybacks, proxy votes and market moves are not relations; "
    "governments, regulators and courts are not economic actors."
)


def messages(headline, on_date, mode):
    if mode == "naive":
        user = f"Headline: {headline}\n\n{TASK}"
    elif mode == "dated":
        user = (f"Today is {on_date}. You are reading this headline on the day it was published. "
                f"Use only what the headline says and what was known on or before {on_date}; do not "
                f"use any knowledge of events after {on_date}.\n\nHeadline: {headline}\n\n{TASK}")
    elif mode == "guideline":
        # the same written guideline the gold annotators followed — the fair comparison
        rules = (CODE / "notebooks" / "gold" / "guideline.txt").read_text()
        user = (f"Today is {on_date}. Use only what the headline says and what was known on or before {on_date}.\n\n"
                f"Follow this annotation guideline exactly:\n\n{rules}\n\nHeadline: {headline}\n\n{TASK}")
    else:
        raise ValueError(mode)
    return [{"role": "system", "content": "You extract structured information from financial news headlines."},
            {"role": "user", "content": user}]


def extract(headline, on_date, model="gpt-4o", mode="dated", rep=0):
    """Free-form schema extraction. Returns the parsed dict plus call metadata."""
    rec = call(model, messages(headline, on_date, mode), SCHEMA, rep)
    out = json.loads(rec["content"])
    out["_meta"] = {k: rec[k] for k in ("model", "served_by", "rep", "prompt_tokens", "completion_tokens")}
    return out


def chooser(model="gpt-4o", on_date=None, rep=0):
    """A drop-in for pit_extract.relation(chooser=...): GPT fills only the predicate slot."""
    def pick(prompt, options):
        lead = (f"Today is {on_date}. Do not use knowledge of events after {on_date}.\n\n" if on_date else "")
        msg = [{"role": "user", "content":
                f"{lead}{prompt}\n\nComplete the last 'Predicate:' with exactly one of: "
                f"{', '.join(options)}. Reply with that phrase only."}]
        text = call(model, msg, None, rep)["content"].strip().lower().strip(" .\"'")
        hit = [o for o in options if o == text] or [o for o in options if o in text]
        return hit[0] if hit else options[0]
    return pick
