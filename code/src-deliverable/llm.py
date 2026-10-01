"""Bounded structured calls; optional durable per-run cache and spending guard.

Prices: https://developers.openai.com/api/docs/models/gpt-4o (2026-09-21).
The guard covers this runner only, not other jobs/account charges or taxes.
Call under the runner's exclusive run lock when configuring a persistent session.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from openai import OpenAI, LengthFinishReasonError


class OutputTruncated(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f)
    os.replace(f.name, path)


def validate_budget(budget, models):
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("LLM budget must be a finite positive number of USD")
    supported = {"gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"}
    if any(model and model not in supported for model in models):
        raise ValueError("Spending guard currently supports standard GPT-4o pricing only")
    endpoint = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if endpoint and endpoint != "https://api.openai.com/v1":
        raise ValueError("Spending guard requires standard OpenAI endpoint/pricing")


class Session:
    def __init__(self, folder, budget):
        validate_budget(budget, [])
        self.folder = Path(folder)
        self.budget = budget
        self.path = self.folder / "llm_usage.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"calls": []}

    @property
    def spent(self):
        return sum(c["charged_or_reserved_usd"] for c in self.data["calls"])

    def save(self):
        self.data.update(budget_usd=self.budget, charged_or_reserved_usd=self.spent)
        atomic_json(self.path, self.data)

    def reserve(self, model, stage, key, max_tokens):
        validate_budget(self.budget, [model])
        # Full context at the uncached input price is a conservative upper bound;
        # avoids guessing prompt/schema token counts. Output is explicitly capped.
        reserve = (128_000 * 2.50 + max_tokens * 10.00) / 1_000_000
        if self.spent + reserve > self.budget:
            raise BudgetExceeded(f"LLM budget ${self.budget:.2f}: ${self.spent:.4f} spent/reserved; "
                                 f"next call requires ${reserve:.4f} headroom. "
                                 "Increase --budget-usd to resume the same run.")
        call = dict(model=model, stage=stage, key=key, status="pending_or_unknown",
                    charged_or_reserved_usd=reserve)
        self.data["calls"].append(call)
        self.save()  # A crash/timeout never erases a potentially billed request.
        return call

    def finish(self, call, usage, status):
        call["status"] = status
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            call.update(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                        cached_tokens=cached,
                        charged_or_reserved_usd=((usage.prompt_tokens - cached) * 2.50 +
                                                 cached * 1.25 + usage.completion_tokens * 10) / 1e6)
        self.save()


_session = None


def configure(folder, budget):
    global _session
    _session = Session(folder, budget)
    return _session


def parse(*, stage, model, response_format, messages, max_completion_tokens):
    """Cache parsed responses and truncations; meter successful and failed calls."""
    request = dict(model=model, temperature=0, messages=messages,
                   max_completion_tokens=max_completion_tokens, service_tier="default")
    key = hashlib.sha256(json.dumps(dict(request, schema=response_format.model_json_schema()),
                                    sort_keys=True).encode()).hexdigest()
    session = _session
    path = session.folder / "responses" / (key + ".json") if session else None
    if path and path.exists():
        saved = json.loads(path.read_text())
        if saved["status"] == "length":
            raise OutputTruncated(f"{stage}: cached truncated response")
        return response_format.model_validate(saved["value"])
    call = session.reserve(model, stage, key, max_completion_tokens) if session else None
    try:
        # No implicit retries: an uncertain request retains its full reservation.
        with OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                    max_retries=0, timeout=60) as client:
            answer = client.beta.chat.completions.parse(response_format=response_format, **request)
    except LengthFinishReasonError as exc:
        if session:
            session.finish(call, exc.completion.usage, "length")
            atomic_json(path, {"status": "length"})
        raise OutputTruncated(f"{stage}: response exceeded output limit") from exc
    except Exception:
        if session:
            session.finish(call, None, "error_usage_unknown")
        raise
    if session:
        session.finish(call, answer.usage, answer.choices[0].finish_reason)
    value = answer.choices[0].message.parsed
    if value is None:
        raise RuntimeError(f"{stage}: no parsed response (possibly a refusal); no result cached")
    if path:
        atomic_json(path, {"status": "ok", "value": value.model_dump()})
    return value


def batches(items, fn, size=50):
    """Process every input; recursively split truncated batches, never silently drop."""
    def process(chunk):
        try:
            return fn(chunk)
        except OutputTruncated as exc:
            if len(chunk) == 1:
                raise RuntimeError(f"LLM output truncated even for one term: {chunk[0]!r}") from exc
            middle = len(chunk) // 2
            return process(chunk[:middle]) + process(chunk[middle:])
    return [value for start in range(0, len(items), size)
            for value in process(items[start:start + size])]
