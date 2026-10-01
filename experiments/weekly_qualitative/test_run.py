"""Offline checks: python -m unittest discover -s experiments/weekly_qualitative."""
import csv
import importlib
import os
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import polars as pl

import run

sys.path.insert(0, str(run.SOURCE))
sys.path.insert(0, str(run.SOURCE / "tests"))
detector = importlib.import_module("1_detectors.graph_anomaly.detect")
basket = importlib.import_module("3_baskets.uniform_method.basket")
fixtures = importlib.import_module("test_1_detectors")


class StudyTests(unittest.TestCase):
    def test_corpus_built_once_then_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cfg = SimpleNamespace(CORPUS=folder / "news_corpus_press.parquet",
                                  CORPUS_MANIFEST=folder / "manifest.json", FEED=folder, WIRES="press")
            (folder / "raw_news_2022.csv.xz").touch()
            def build():
                fixtures._corpus().write_parquet(cfg.CORPUS)
                run.atomic_json(cfg.CORPUS_MANIFEST, {"corpus": {"rows": 1}})
            with patch.object(run.importlib, "import_module") as loader:
                loader.return_value.build_corpus.side_effect = build
                run.ensure_corpus(cfg)
                run.ensure_corpus(cfg)
                loader.return_value.build_corpus.assert_called_once_with()
            self.assertTrue(cfg.CORPUS.exists())

    def test_incomplete_corpus_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cfg = SimpleNamespace(CORPUS=folder / "news_corpus_press.parquet",
                                  CORPUS_MANIFEST=folder / "manifest.json", FEED=folder, WIRES="press")
            cfg.CORPUS.touch()
            (folder / "raw_news_2022.csv.xz").touch()
            run.atomic_json(cfg.CORPUS_MANIFEST, {"corpus": None})
            with patch.object(run.importlib, "import_module") as loader:
                loader.return_value.build_corpus.side_effect = lambda: run.atomic_json(
                    cfg.CORPUS_MANIFEST, {"corpus": {"rows": 1}})
                run.ensure_corpus(cfg)
                loader.return_value.build_corpus.assert_called_once_with()

    def test_missing_raw_does_not_start_build(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cfg = SimpleNamespace(CORPUS=folder / "news_corpus_press.parquet",
                                  CORPUS_MANIFEST=folder / "manifest.json", FEED=folder, WIRES="press")
            with patch.object(run.importlib, "import_module") as loader:
                with self.assertRaisesRegex(RuntimeError, "no raw captures"):
                    run.ensure_corpus(cfg)
                loader.assert_not_called()
            self.assertEqual(list(folder.iterdir()), [])

    def test_wires_cli_override_preserves_other_settings(self):
        for prefix in ("", "+", "++", "~"):
            current = f"detect.llm_max=5 {prefix}preprocessing.wires=all basket.top_n=3"
            self.assertEqual(run.wire_overrides(current, "press"),
                             "detect.llm_max=5 basket.top_n=3 preprocessing.wires=press")

    def test_wires_selects_corpus_and_fails_clearly_if_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, TT_DATA=directory, TT_OVERRIDES="preprocessing.wires=bloomberg")
            result = subprocess.run([sys.executable, str(Path(run.__file__)), "--wires", "press", "--check"],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("news_corpus_press.parquet", result.stderr)
            self.assertIn("Wires: press", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_wire_selection(self):
        result = subprocess.run([sys.executable, str(Path(run.__file__)), "--wires", "invalid"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def setUp(self):
        self.companies = [{"ticker": "aaa", "weight": .5}, {"ticker": "bbb", "weight": .5}]
        self.quotes = pd.DataFrame({"SPY": [100., 100., 110.], "AAA": [50., 100., 120.],
                                   "BBB": [50., 100., 90.]},
                                  index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-09"]))

    def test_fixed_horizon(self):
        got = run.evaluate(self.companies, "2024-01-01", self.quotes, [1], "SPY")
        self.assertAlmostEqual(got["return_1w"], .05)
        self.assertAlmostEqual(got["excess_1w"], -.05)
        self.assertEqual(got["evaluation"]["1"]["entry"], "2024-01-02")

    def test_missing_prices_not_renormalized(self):
        got = run.evaluate(self.companies, "2024-01-01", self.quotes.drop(columns="BBB"), [1], "SPY")
        self.assertIsNone(got["return_1w"])
        self.assertEqual(got["basket_status"], "missing_prices")
        self.assertAlmostEqual(got["spy_1w"], .1)

    def test_incomplete_and_empty(self):
        self.assertEqual(run.evaluate([], "2024-01-01", self.quotes, [1], "SPY")["basket_status"], "empty")
        got = run.evaluate(self.companies, "2024-01-01", self.quotes, [1, 2], "SPY")
        self.assertEqual(got["basket_status"], "incomplete_horizon")
        self.assertIsNone(got["return_2w"])
        self.assertIsNotNone(got["return_1w"])

    def test_audit_and_strict_cutoff(self):
        corpus = fixtures._corpus()
        at_cutoff = pl.DataFrame({"Headline": ["widget secretfuturebrand"] * 10,
                                  "date": [datetime(2024, 5, 6)] * 10})
        names = pl.DataFrame({"ticker": ["wdgt"], "stem": ["widget"], "cik": [1]})
        got = run.collect_week(datetime(2024, 5, 6), pl.concat([corpus, at_cutoff]).lazy(),
                               detector, basket, names,
                               {"detect": fixtures.GATES, "basket": {"llm_model": None, "report": False}},
                               "hash", "commit")
        widget = next(r for r in got["positive"] if r["anchor"] == "widget")
        self.assertIn("widget", widget["words_before_llm"])
        self.assertEqual(widget["companies"][0]["ticker"], "wdgt")
        self.assertNotIn("secretfuturebrand", widget["words_before_llm"])
        self.assertTrue(widget["headlines"][0]["dates"])
        for category in ("positive", "negative"):
            ids = [r["theme_id"] for r in got[category]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_llm_cap_is_a_rejection(self):
        audit = []
        args = fixtures.GATES | {"llm_model": "fake", "llm_max": 0}
        themes, _, rejects = detector.detect(fixtures._corpus(), fixtures.ASOF, audit=audit, **args)
        self.assertTrue(themes.is_empty())
        dropped = [r for r in audit if r.get("rejection_stage") == "llm_max"]
        self.assertTrue(dropped)
        self.assertTrue({r["detector_theme_id"] for r in dropped} <= set(rejects["theme"]))

    def test_gate_explanation_is_exported(self):
        def reject(anchor, headlines, model, *, audit):
            audit.update(llm_pattern="one_actor", llm_reason="one company")
            return False
        audit = []
        with patch.object(detector, "is_real", reject):
            detector.detect(fixtures._corpus(), fixtures.ASOF, audit=audit,
                            **(fixtures.GATES | {"llm_model": "fake"}))
        rejected = [r for r in audit if r.get("rejection_stage") == "llm"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[0]["llm_reason"], "one company")
        names = pl.DataFrame({"ticker": ["wdgt"], "stem": ["widget"], "cik": [1]})
        with patch.object(detector, "is_real", reject):
            week = run.collect_week(datetime(2024, 5, 6), fixtures._corpus().lazy(),
                                    detector, basket, names,
                                    {"detect": fixtures.GATES | {"llm_model": "fake"},
                                     "basket": {"llm_model": None, "report": False}}, "hash", "commit")
        widget = next(r for r in week["negative"] if r["anchor"] == "widget")
        self.assertEqual(widget["rejection_stage"], "llm")

    def test_price_history_gap_is_not_a_late_purchase(self):
        got = run.evaluate(self.companies, "2023-01-01", self.quotes, [1], "SPY")
        self.assertEqual(got["basket_status"], "missing_prices")
        self.assertIsNone(got["return_1w"])

    def test_cache_restores_audit(self):
        calls = []
        def verdict(anchor, *, audit):
            calls.append(anchor)
            audit["llm_reason"] = "cached reason"
            return True
        with tempfile.TemporaryDirectory() as directory:
            cached = run.cached_call(verdict, Path(directory))
            cached("widget", audit={})
            details = {}
            self.assertTrue(cached("widget", audit=details))
            self.assertEqual(details["llm_reason"], "cached reason")
            self.assertEqual(len(calls), 1)

    def test_resume_export_preserves_reviews_and_first_basket(self):
        class Prices:
            def load(inner, tickers):
                return self.quotes
        row = {"asof": "2024-01-01", "theme_id": "widget", "companies": self.companies}
        checkpoints = [{"positive": [row], "negative": []},
                       {"positive": [row | {"asof": "2024-01-08", "companies": []}], "negative": []}]
        study = {"horizons_weeks": [1], "benchmark": "SPY",
                 "positive_file": "positive.csv", "negative_file": "negative.csv"}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / ".cache").mkdir()
            run.export(folder, checkpoints, study, Prices())
            with (folder / "positive.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[1]["return_1w"], "")
            self.assertEqual(rows[1]["first_detected_asof"], "2024-01-01")
            rows[0]["human_notes"] = "keep, checked"
            run.write_csv(folder / "positive.csv", [], list(rows[0]))
            run.write_csv(folder / "positive.csv", rows, list(rows[0]))
            run.export(folder, checkpoints, study, Prices())
            with (folder / "positive.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["human_notes"], "keep, checked")
            self.assertAlmostEqual(float(rows[0]["return_1w"]), .05)


if __name__ == "__main__":
    unittest.main()
