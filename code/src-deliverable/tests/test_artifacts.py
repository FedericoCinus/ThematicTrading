"""artifacts — one run, one name, three files per stage."""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import polars as pl

import artifacts
import config

FRAME = pl.DataFrame({"theme": ["genai", "quantum"], "weight": [0.5, 0.5]})
DIAG = pl.DataFrame({"theme": ["genai", "quantum"], "why": ["a word", "another"]})


@contextmanager
def _sandbox():
    """config.PROC pointed at an empty directory, and put back afterwards.

    artifacts reads config.PROC at call time, so swapping the attribute is enough — and restoring
    it matters, because a later block reading a leftover temp directory would find it empty and
    pass for the wrong reason.
    """
    real = config.PROC
    config.PROC = Path(tempfile.mkdtemp()) / "processed"
    try:
        yield config.PROC
    finally:
        config.PROC = real


def what_goes_in_comes_out():
    """a saved frame reads back identical, diagnostics included"""
    with _sandbox():
        artifacts.save("weights", FRAME, DIAG)
        back, diag = artifacts.load("weights")
        if not back.equals(FRAME):
            return "the canonical frame changed on the way through parquet"
        return None if diag is not None and diag.equals(DIAG) else "the diagnostics did not survive"


def diagnostics_are_optional():
    """a stage with nothing to explain writes no .diag file, and load says None"""
    with _sandbox() as root:
        artifacts.save("weights", FRAME)
        if (root / "weights").glob("*.diag.parquet") and list((root / "weights").glob("*.diag.parquet")):
            return "a .diag file was written for a stage that passed none"
        return None if artifacts.load("weights")[1] is None else "load invented a diagnostics frame"


def the_manifest_records_what_produced_it():
    """the parameters travel in the .json, never in the filename"""
    with _sandbox():
        artifacts.save("themes", FRAME, asof="2023-01-02", llm_model="gpt-4o", max_doc_freq=0.0001)
        m = artifacts.manifest("themes")
        for key in ("run", "stage", "rows", "written", "asof", "llm_model", "max_doc_freq"):
            if key not in m:
                return f"the manifest has no {key}: {sorted(m)}"
        return None if m["rows"] == len(FRAME) else f"rows says {m['rows']}, frame has {len(FRAME)}"


def a_run_leaves_no_half_written_file():
    """the temporary files are renamed away, so a directory listing is only finished artifacts"""
    with _sandbox() as root:
        artifacts.save("signal", FRAME, DIAG, asof="2023-01-02")
        leftover = [p.name for p in (root / "signal").iterdir() if p.name.endswith(".tmp")]
        return f"left behind {leftover}" if leftover else None


def two_runs_sit_side_by_side():
    """a parameter variant is a second run under the same stage, not an overwrite"""
    with _sandbox():
        artifacts.save("themes", FRAME, run="graph_anomaly-rolling_z-uniform")
        artifacts.save("themes", FRAME.head(1), run="graph_anomaly-rolling_z-uniform@z2")
        found = artifacts.runs("themes")
        if found != ["graph_anomaly-rolling_z-uniform", "graph_anomaly-rolling_z-uniform@z2"]:
            return f"runs() returned {found}"
        return None if len(artifacts.load("themes", run=found[1])[0]) == 1 else "the two runs collided"


def rerunning_replaces_its_own_files():
    """the same configuration overwrites itself instead of accumulating copies"""
    with _sandbox():
        artifacts.save("themes", FRAME, run="same")
        artifacts.save("themes", FRAME.head(1), run="same")
        if artifacts.runs("themes") != ["same"]:
            return f"a rerun created {artifacts.runs('themes')}"
        return None if len(artifacts.load("themes", run="same")[0]) == 1 else "the old frame survived"


def a_missing_run_is_an_error_not_an_empty_frame():
    """a stage that was never written must not look like a stage that found nothing"""
    with _sandbox():
        try:
            artifacts.load("weights", run="never-ran")
        except FileNotFoundError:
            return None
        return "load returned something for a run that was never written"


CHECKS = [what_goes_in_comes_out, diagnostics_are_optional, the_manifest_records_what_produced_it,
          a_run_leaves_no_half_written_file, two_runs_sit_side_by_side,
          rerunning_replaces_its_own_files, a_missing_run_is_an_error_not_an_empty_frame]
