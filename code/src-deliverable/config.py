"""Where the data lives and how the run is tuned — the only module that knows either.

Paths are resolved once, here. Every module in the old scripts computed its own
`_ROOT = Path(__file__).parent`, so moving a file silently repointed its data directory and two of
them broke that way; nothing else in this package touches `__file__` or names a directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"

# ======================================================================================
# THE RUN — one file per experiment, each defining the WHOLE pipeline
#
#   experiments/default.yaml         the pipeline as it stands
#   experiments/<name>.yaml          a different one, in the same five blocks
#
#   HOW      experiments/<name>.yaml ──▶ compose ──▶ CONFIG, a plain dict ──▶ params(stage)
#
#   TT_EXPERIMENT=filings_basket        which file to run          default: default
#   TT_OVERRIDES="basket.top_n=20 detect.min_mentions=5"      ad hoc, space separated
#   TT_DATA=/somewhere                  a different data directory
#
#   A file is whole and self-contained — no defaults list, no config groups — so reading one
#   tells you everything that ran. Hydra is here for its override parser and for the sweeps a
#   command-line entrypoint could later run, not for composition.
#
#   Composed here rather than through @hydra.main, because the front door of this package is a
#   notebook, not a command line: `import config` has to work the same in both.
#
#   A SWEEP is a loop over overrides, one run id each:
#       for z in (0.5, 1.0, 2.0):
#           os.environ["TT_OVERRIDES"] = f"monitor.z_threshold={z}"
#           importlib.reload(config); pipeline.run("2023-01-02", until="2025-12-31")
# ======================================================================================


def _compose() -> dict:
    """One experiment file, composed into a plain dict, with the environment's overrides applied."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    name = os.environ.get("TT_EXPERIMENT", "default")
    if not (EXPERIMENTS / f"{name}.yaml").exists():
        known = sorted(p.stem for p in EXPERIMENTS.glob("*.yaml"))
        raise FileNotFoundError(f"no experiment {name!r} in {EXPERIMENTS} — there are {known}")

    GlobalHydra.instance().clear()          # so a re-import in a notebook does not raise
    with initialize_config_dir(config_dir=str(EXPERIMENTS), version_base=None):
        config = compose(config_name=name,
                         overrides=(os.environ.get("TT_OVERRIDES") or "").split())
    return OmegaConf.to_container(config, resolve=True)


CONFIG = _compose()

DATA = Path(os.environ.get("TT_DATA") or HERE / CONFIG["data_root"]).resolve()

# external sources
FEED = DATA / "raw"                                  # raw_news_{year}.csv.xz
EDGAR = DATA / "raw" / "edgar"                       # SEC register, filings, full-text-search cache
PRICES = DATA / "raw" / "prices"                     # Yahoo adjusted-close cache

# The investable universe is the SEC register, cached under EDGAR — see 3_baskets/universe.py for
# what that includes and what it does not. `universe_csv` in the basket's config is a slot for a
# Bloomberg extract instead; nothing reads it today.
UNIVERSE = Path(CONFIG["basket"]["universe_csv"]) if CONFIG["basket"].get("universe_csv") else None

# stage 1 artifact, the one canonical corpus
PROC = DATA / "processed"
CORPUS = PROC / "news_corpus.parquet"
CORPUS_MANIFEST = PROC / "news_corpus_manifest.json"


def params(stage: str) -> dict:
    """The tunables of one stage.

    INPUT   stage   a top-level key of the composed config: preprocessing, detect, monitor, ...
    OUTPUT  a plain dict, ready to pass on as keyword arguments

    `method` is included: it names the variant module to run, so a stage's whole configuration —
    which method and how it is set — travels together.
    """
    return dict(CONFIG.get(stage, {}))


def run_id() -> str:
    """The name every file a run writes is saved under.

    INPUT   nothing — read from the composed config
    OUTPUT  `detector-entry+exit-basket-xxxxxx`

    The three method names say WHAT ran; the six hex characters say WITH WHICH PARAMETERS. They are
    a digest of the whole composed config, so:

      same configuration   same id   a rerun replaces its own files instead of piling up copies
      one value changed    new id    the earlier run is still there to compare against

    There is nothing to remember to set. A suffix you had to pass by hand was a rule that two runs
    must not collide, enforced by whoever remembered it — and the run that silently overwrote
    yesterday's looked exactly like the run that did not.

    What the digest stands for is in the manifest beside every frame, and `artifacts.index(stage)`
    lays those out side by side.
    """
    monitor = CONFIG["monitor"]
    triple = "-".join((CONFIG["detect"]["method"],
                       f'{monitor["entry"]["method"]}+{monitor["exit"]["method"]}',
                       CONFIG["basket"]["method"]))
    return f"{triple}-{fingerprint()}"


def fingerprint() -> str:
    """Six hex characters standing for the whole configuration.

    INPUT   nothing
    OUTPUT  a short digest, stable across processes and machines

    blake2b and not `hash()`, which is randomised per process — the same config has to give the
    same id tomorrow, or a rerun writes a second copy instead of replacing the first.
    """
    return hashlib.blake2b(json.dumps(CONFIG, sort_keys=True, default=str).encode(),
                           digest_size=3).hexdigest()


def load_env(path: str | Path | None = None) -> None:
    """Load secrets into the environment.

    INPUT   path   a .env file; defaults to code/.env
    OUTPUT  nothing — os.environ is filled in, never overwriting what is already set

    Provides SEC_USER_AGENT, which EDGAR refuses requests without, and OPENAI_API_KEY for the
    optional LLM gate in detection.
    """
    p = Path(path) if path else HERE.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
