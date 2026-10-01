"""Offline regression tests: no network calls and no real API key required."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI as RealOpenAI

import run

sys.path.insert(0, str(run.SOURCE))
import llm
from pydantic import BaseModel

detector = importlib.import_module("1_detectors.graph_anomaly.detect")
basket = importlib.import_module("3_baskets.uniform_method.basket")


class Result(BaseModel):
    entities: list[str]


def usage(prompt=1000, output=200, cached=0):
    return NS(prompt_tokens=prompt, completion_tokens=output,
              prompt_tokens_details=NS(cached_tokens=cached))


def response(value, tokens=None):
    return NS(usage=tokens or usage(),
              choices=[NS(finish_reason="stop", message=NS(parsed=value))])


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.env = patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test", "OPENAI_BASE_URL": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        old = llm._session
        self.addCleanup(setattr, llm, "_session", old)
        self.session = llm.configure(self.folder, 5)
        self.client_patch = patch.object(llm, "OpenAI")
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.api = self.client.return_value.__enter__.return_value.beta.chat.completions.parse
        self.api.return_value = response(Result(entities=["apple"]))

    def call(self, **overrides):
        return llm.parse(**(dict(stage="vocabulary", model="gpt-4o", response_format=Result,
                                messages=[{"role": "user", "content": "apple"}],
                                max_completion_tokens=2048) | overrides))

    def test_cache_and_cost_survive_restart(self):
        self.assertEqual(self.call().entities, ["apple"])
        self.assertAlmostEqual(self.session.spent, .0045)
        restarted = llm.configure(self.folder, 10)
        self.assertEqual(self.call().entities, ["apple"])
        self.assertAlmostEqual(restarted.spent, .0045)
        self.api.assert_called_once()
        self.assertEqual(self.api.call_args.kwargs["max_completion_tokens"], 2048)
        self.assertEqual(self.client.call_args.kwargs["max_retries"], 0)

    def test_cached_input_discount(self):
        self.api.return_value = response(Result(entities=[]), usage(cached=500))
        self.call()
        self.assertAlmostEqual(self.session.spent, .003875)

    def test_real_sdk_parses_mock_http_and_records_usage(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["max_completion_tokens"], 2048)
            self.assertEqual(body["response_format"]["type"], "json_schema")
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0, "model": "gpt-4o",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"entities":["apple"]}'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}})
        self.client.side_effect = lambda **kwargs: RealOpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertEqual(self.call().entities, ["apple"])
        self.assertAlmostEqual(self.session.spent, .0045)

    def test_budget_blocks_before_network(self):
        self.session.budget = .01
        with self.assertRaises(llm.BudgetExceeded):
            self.call()
        self.client.assert_not_called()
        self.assertEqual(self.session.spent, 0)

    def test_budget_includes_previous_and_unknown_calls(self):
        self.api.side_effect = TimeoutError("offline timeout")
        with self.assertRaises(TimeoutError):
            self.call()
        self.assertAlmostEqual(self.session.spent, .34048)
        llm.configure(self.folder, .5)
        with self.assertRaises(llm.BudgetExceeded):
            self.call()
        self.api.assert_called_once()

    def test_length_error_is_charged_and_cached(self):
        self.api.side_effect = llm.LengthFinishReasonError(completion=NS(usage=usage(4422, 16384)))
        for _ in range(2):
            with self.assertRaises(llm.OutputTruncated):
                self.call()
        self.api.assert_called_once()
        self.assertAlmostEqual(self.session.spent, .174895)
        self.assertEqual(json.loads(self.session.path.read_text())["calls"][0]["status"], "length")

    def test_refusal_billed_but_not_cached(self):
        self.api.return_value = response(None)
        with self.assertRaisesRegex(RuntimeError, "no parsed response"):
            self.call()
        self.assertAlmostEqual(self.session.spent, .0045)
        self.assertFalse((self.folder / "responses").exists())

    def test_prompt_changes_invalidate_cache(self):
        self.call()
        self.call(messages=[{"role": "user", "content": "microsoft"}])
        self.assertEqual(self.api.call_count, 2)

    def test_vocabulary_chunks_keep_order_and_intersect_per_chunk(self):
        terms = [f"term{i}" for i in range(121)]
        sizes = []
        def answer(**kwargs):
            words = kwargs["messages"][1]["content"].removeprefix("Terms: ").split(", ")
            sizes.append(len(words))
            return response(kwargs["response_format"](entities=[w.upper() for w in reversed(words)] +
                                                              ["invented", "term120"]))
        self.api.side_effect = answer
        self.assertEqual(detector.filter_entities(terms, "gpt-4o"), terms)
        self.assertEqual(sizes, [50, 50, 21])
        self.assertEqual(detector.filter_entities(terms, "gpt-4o"), terms)
        self.assertEqual(self.api.call_count, 3)

    def test_length_recursively_splits_and_restart_uses_successes(self):
        terms = ["apple", "microsoft", "openai", "google"]
        def answer(**kwargs):
            words = kwargs["messages"][1]["content"].removeprefix("Terms: ").split(", ")
            if len(words) > 1:
                raise llm.LengthFinishReasonError(completion=NS(usage=usage()))
            return response(kwargs["response_format"](entities=words))
        self.api.side_effect = answer
        self.assertEqual(detector.filter_entities(terms, "gpt-4o"), terms)
        self.assertEqual(self.api.call_count, 7)
        self.assertEqual(len(self.session.data["calls"]), 7)
        llm.configure(self.folder, 5)
        self.assertEqual(detector.filter_entities(terms, "gpt-4o"), terms)
        self.assertEqual(self.api.call_count, 7)

    def test_singleton_failure_does_not_silently_drop(self):
        self.api.side_effect = llm.LengthFinishReasonError(completion=NS(usage=usage()))
        with self.assertRaisesRegex(RuntimeError, "even for one term"):
            detector.filter_entities(["apple"], "gpt-4o")
        self.api.assert_called_once()

    def test_empty_vocab_makes_no_calls(self):
        self.assertEqual(detector.filter_entities([], "gpt-4o"), [])
        self.client.assert_not_called()

    def test_gate_and_issuer_costs_use_same_session(self):
        def answer(**kwargs):
            schema = kwargs["response_format"]
            if schema.__name__ == "Verdict":
                value = schema(pattern="none", actors=["apple"], reason="several actors")
            else:
                value = schema(owners=[{"word": "iphone", "companies": ["Apple"]}])
            return response(value)
        self.api.side_effect = answer
        audit = {}
        self.assertTrue(detector.is_real("iphone", ["headline"], "gpt-4o", audit=audit))
        self.assertEqual(audit["llm_reason"], "several actors")
        self.assertEqual(basket.issuers(["iphone"], "gpt-4o"), [("iphone", "apple")])
        self.assertEqual([c["stage"] for c in self.session.data["calls"]], ["gate", "issuers"])
        self.assertAlmostEqual(self.session.spent, .009)

    def test_invalid_budgets_or_unpriced_models_fail(self):
        for budget in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                llm.validate_budget(budget, ["gpt-4o"])
        with self.assertRaisesRegex(ValueError, "GPT-4o"):
            llm.validate_budget(5, ["unpriced-model"])
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://example.invalid"}):
            with self.assertRaisesRegex(ValueError, "endpoint"):
                llm.validate_budget(5, ["gpt-4o"])


if __name__ == "__main__":
    unittest.main()
