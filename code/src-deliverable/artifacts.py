"""Every stage writes the same three files, under the name of the run that produced them.

    save(stage, canonical, diagnostics, **manifest) -> Path
    load(stage) -> canonical, diagnostics
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import polars as pl

import config

# ======================================================================================
# THE LAYOUT — one run, one name, three files per stage
#
#   IN    stage         a directory name: themes · signal · weights · backtest
#   IN    canonical     the frame the next stage reads
#   IN    diagnostics   the method's own columns, same index            optional
#   IN    **manifest    what produced it: method, parameters, asof
#   OUT   data/processed/{stage}/{run}.parquet          canonical
#                                 {run}.diag.parquet    diagnostics
#                                 {run}.json            manifest
#
#   run     config.run_id()      graph_anomaly-rolling_z+news_level-uniform-89ea2b
#           the method names say WHAT ran, the digest says WITH WHICH PARAMETERS. Change a value
#           and the id changes with it, so a run can never quietly overwrite another; change
#           nothing and it does, which is what rerunning a configuration should mean.
#           `index(stage)` turns the digests back into the parameters they stand for.
#
#   HOW     frame ──▶ {file}.tmp ──▶ os.replace ──▶ {file}
#           os.replace is atomic on one filesystem: a reader sees the old file or the new one,
#           never half of either, and an interrupted run leaves only a .tmp to ignore
#
#   data/processed/
#     themes/    graph_anomaly-rolling_z-uniform.parquet  .diag.parquet  .json
#     signal/    graph_anomaly-rolling_z-uniform.parquet  .diag.parquet  .json
#     weights/   graph_anomaly-rolling_z-uniform.parquet  .diag.parquet  .json
#     backtest/  graph_anomaly-rolling_z-uniform.parquet  .diag.parquet  .json
# ======================================================================================


def save(stage: str, canonical: pl.DataFrame, diagnostics: pl.DataFrame | None = None,
         *, run: str | None = None, **manifest) -> Path:
    """Write one stage's output under the run id.

    INPUT   stage         the directory to write into
            canonical     the frame the next stage reads
            diagnostics   the method's own columns; None writes no .diag file
            run           overrides config.run_id()
            **manifest    anything describing the run — method, parameters, asof
    OUTPUT  the path of the canonical parquet

    Overwrites the same run in place: re-running a configuration replaces its own files rather
    than accumulating copies.
    """
    run = run or config.run_id()
    folder = config.PROC / stage
    folder.mkdir(parents=True, exist_ok=True)

    _write(canonical, folder / f"{run}.parquet")
    if diagnostics is not None:
        _write(diagnostics, folder / f"{run}.diag.parquet")
    _replace(folder / f"{run}.json", json.dumps(
        {"run": run, "stage": stage, "rows": len(canonical),
         "written": datetime.now().isoformat(timespec="seconds"), **manifest},
        indent=2, default=str))
    return folder / f"{run}.parquet"


def load(stage: str, *, run: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Read back one stage's output.

    INPUT   stage   the directory to read from
            run     overrides config.run_id()
    OUTPUT  (canonical, diagnostics) — diagnostics is None when the stage wrote none

    Raises FileNotFoundError when that run never wrote this stage, rather than returning an empty
    frame that would look like a run which found nothing.
    """
    run = run or config.run_id()
    path = config.PROC / stage / f"{run}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no {stage} for run {run} — looked in {path}")
    diag = path.with_name(f"{run}.diag.parquet")
    return pl.read_parquet(path), pl.read_parquet(diag) if diag.exists() else None


def manifest(stage: str, *, run: str | None = None) -> dict:
    """What produced a saved frame.

    INPUT   stage, run   as in `load`
    OUTPUT  the manifest dict: run, stage, rows, written, and the parameters the stage was given
    """
    run = run or config.run_id()
    path = config.PROC / stage / f"{run}.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest for run {run} — looked in {path}")
    return json.loads(path.read_text())


def runs(stage: str) -> list[str]:
    """Which runs are on disk for one stage.

    INPUT   stage   the directory to list
    OUTPUT  run ids, sorted — what can be compared against what
    """
    folder = config.PROC / stage
    if not folder.exists():
        return []
    # `{run}.diag.parquet` has stem `{run}.diag`, which is not a run
    return sorted(p.stem for p in folder.glob("*.parquet") if not p.stem.endswith(".diag"))


def index(stage: str) -> pl.DataFrame:
    """Every run of one stage, with the parameters that produced it.

    INPUT   stage   the directory to read
    OUTPUT  one row per run: its manifest, flattened — which is how a digest becomes readable

    The run id alone says what ran, not with what. This is the lookup table.
    """
    rows = [manifest(stage, run=run) for run in runs(stage)]
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _write(frame: pl.DataFrame, path: Path) -> None:
    tmp = path.parent / f"{path.name}.tmp"
    frame.write_parquet(tmp)
    os.replace(tmp, path)


def _replace(path: Path, text: str) -> None:
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(text)
    os.replace(tmp, path)
