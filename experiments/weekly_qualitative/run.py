"""Weekly qualitative study. Run from any directory; all paths are repository-relative."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "code/src-deliverable"
REVIEW = ["human_label", "human_notes", "llm_label", "llm_notes", "description",
          "review_model", "review_prompt_version"]
COMMON = ["asof", "theme_id", "detector_theme_id", "anchor", "emergence_week",
          "mentions", "degree", "clustering", "headlines", "words_before_llm",
          "words_after_llm", "monitor_words", "llm_pattern", "llm_reason",
          "config_hash", "git_commit"] + REVIEW


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, default=str)
        temp = f.name
    os.replace(temp, path)


def write_csv(path, rows, columns):
    # Preserve manual/LLM annotations when rebuilding tables from checkpoints.
    reviews = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reviews = {(r["asof"], r["theme_id"]): {k: r.get(k, "") for k in REVIEW}
                       for r in csv.DictReader(f)}
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row, **reviews.get((row["asof"], row["theme_id"]), {}))
            writer.writerow({k: json.dumps(v, ensure_ascii=False, default=str)
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})
        temp = f.name
    os.replace(temp, path)


def evaluate(companies, asof, quotes, horizons, benchmark):
    """Freeze equal weights; never drop a missing asset or fill a stale close."""
    import pandas as pd
    result = {f"{kind}_{h}w": None for h in horizons for kind in ("return", "spy", "excess")}
    result["evaluation"] = {}
    if not companies:
        return dict(result, basket_status="empty")
    if benchmark not in quotes or quotes[benchmark].dropna().empty:
        return dict(result, basket_status="missing_prices", evaluation={"reason": "benchmark missing"})
    sessions = quotes[benchmark].dropna().sort_index().index
    entries = sessions[sessions > pd.Timestamp(asof)]
    if not len(entries):
        return dict(result, basket_status="incomplete_horizon", evaluation={"reason": "no entry date"})
    entry = entries[0]
    if entry > pd.Timestamp(asof) + pd.Timedelta(days=7):
        return dict(result, basket_status="missing_prices", evaluation={"reason": "entry price history gap"})
    statuses = []
    for h in horizons:
        exits = sessions[sessions >= entry + pd.Timedelta(weeks=h)]
        detail = {"entry": entry.date().isoformat(), "status": "incomplete_horizon"}
        result["evaluation"][str(h)] = detail
        if not len(exits):
            statuses.append("incomplete_horizon")
            continue
        end = exits[0]
        if end > entry + pd.Timedelta(weeks=h, days=7):
            detail.update(status="missing_prices", reason="exit price history gap")
            statuses.append("missing_prices")
            continue
        detail["exit"] = end.date().isoformat()
        spy = float(quotes.loc[end, benchmark] / quotes.loc[entry, benchmark] - 1)
        result[f"spy_{h}w"] = spy
        holdings, missing = [], []
        for company in companies:
            ticker = company["ticker"].upper()
            start_price = quotes.loc[entry, ticker] if ticker in quotes else float("nan")
            end_price = quotes.loc[end, ticker] if ticker in quotes else float("nan")
            if not all(math.isfinite(float(p)) and p > 0 for p in (start_price, end_price)):
                missing.append(ticker)
            else:
                holdings.append({"ticker": ticker, "weight": company["weight"],
                                 "return": float(end_price / start_price - 1)})
        detail.update(holdings=holdings, missing_tickers=missing)
        status = "missing_prices" if missing else "ok"
        detail["status"] = status
        statuses.append(status)
        if not missing:
            ret = sum(r["weight"] * r["return"] for r in holdings)
            result[f"return_{h}w"] = ret
            result[f"excess_{h}w"] = ret - spy
    result["basket_status"] = ("missing_prices" if "missing_prices" in statuses else
                                "incomplete_horizon" if "incomplete_horizon" in statuses else "ok")
    return result


def cached_call(fn, folder):
    """Cache identical gate/issuer requests, including gate explanations, across weeks."""
    def call(*args, **kwargs):
        audit = kwargs.pop("audit", None)
        path = folder / (digest([fn.__module__, fn.__name__, args, kwargs]) + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
        else:
            details = {}
            value = fn(*args, **kwargs, **({"audit": details} if audit is not None else {}))
            saved = {"value": value, "audit": details}
            atomic_json(path, saved)
        if audit is not None:
            audit.update(saved["audit"])
        return saved["value"]
    return call


def calendar(study):
    start = datetime.fromisoformat(study["start"])
    if start.weekday() != 0 or start.time() != datetime.min.time():
        raise ValueError("study.start must be a Monday date at midnight")
    if not isinstance(study["weeks"], int) or study["weeks"] < 1:
        raise ValueError("study.weeks must be a positive integer")
    required = {"step_days": 7, "cutoff": "00:00:00 UTC", "weighting": "equal",
                "rebalance": False, "exit": "fixed_horizon", "format": "csv",
                "entry": "first_price_date_after_first_detection", "row_key": ["asof", "theme_id"]}
    for key, value in required.items():
        if study[key] != value:
            raise ValueError(f"unsupported study.{key}: expected {value!r}")
    horizons = study["horizons_weeks"]
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons_weeks must contain positive integers")
    return [start + timedelta(weeks=i) for i in range(study["weeks"])]


def collect_week(asof, corpus, detector, basket, names, settings, fingerprint, commit):
    import polars as pl
    audit = []
    # Enforce a strict cutoff also inside vocabulary(), whose default boundary is inclusive.
    themes, _, _ = detector.detect(corpus.filter(pl.col("date") < asof), asof.isoformat(),
                                    audit=audit, **settings["detect"])
    weights, notes = basket.weights(themes, names=names, **settings["basket"])
    mappings = {(r["theme"], r["ticker"]): r for r in notes.to_dicts()}
    holdings = {}
    for row in weights.to_dicts():
        detail = mappings.get((row["theme"], row["ticker"]), {})
        holdings.setdefault(row["theme"], []).append(
            {k: (row | detail).get(k) for k in ("ticker", "weight", "company", "word",
                                               "input_company", "normalized_company", "match_method")})
    # One candidate per anchor: promotion, then LLM/cap rejection, then statistical rejection.
    # Among equal stages keep the latest week, without hiding a gate-4 decision behind gate 3.
    def rank(row):
        return 2 if row["promoted"] else int(row.get("rejection_stage") in ("llm", "llm_max"))

    chosen = {}
    for row in sorted(audit, key=lambda r: (r["week"], r["anchor"])):
        previous = chosen.get(row["anchor"])
        if previous is None or rank(row) >= rank(previous):
            chosen[row["anchor"]] = row
    positive, negative = [], []
    for anchor, row in sorted(chosen.items()):
        common = {k: row.get(k) for k in COMMON}
        common.update(asof=asof.date().isoformat(), theme_id=anchor,
                      emergence_week=row["week"].date().isoformat(),
                      config_hash=fingerprint, git_commit=commit)
        if row["promoted"]:
            issues = [n for n in notes.to_dicts() if n["theme"] == row["detector_theme_id"]
                      and n.get("match_status") in ("unmatched", "ambiguous")]
            positive.append(dict(common, companies=holdings.get(row["detector_theme_id"], []),
                                 matching_issues=issues))
        else:
            negative.append(dict(common, rejection_stage=row["rejection_stage"],
                                 rejection_reason=row["rejection_reason"]))
    return {"positive": positive, "negative": negative}


def export(folder, checkpoints, study, prices):
    positive, negative, first = [], [], {}
    horizons, benchmark = study["horizons_weeks"], study["benchmark"].upper()
    for checkpoint in checkpoints:
        for original in checkpoint["positive"]:
            row = dict(original)
            if row["theme_id"] not in first:
                first[row["theme_id"]] = row["asof"]
                tickers = sorted({c["ticker"].upper() for c in row["companies"]} | {benchmark})
                snapshot = folder / ".cache" / ("prices-" + digest(tickers) + ".parquet")
                if snapshot.exists():
                    import pandas as pd
                    quotes = pd.read_parquet(snapshot)
                else:
                    quotes = prices.load(tickers)
                    temp = snapshot.with_suffix(".tmp")
                    quotes.to_parquet(temp)
                    os.replace(temp, snapshot)
                row.update(evaluate(row["companies"], row["asof"], quotes, horizons, benchmark))
            row["first_detected_asof"] = first[row["theme_id"]]
            positive.append(row)
        negative.extend(checkpoint["negative"])
    positive_columns = COMMON + ["first_detected_asof", "companies", "matching_issues", "basket_status", "evaluation"]
    positive_columns += [f"{kind}_{h}w" for h in horizons for kind in ("return", "spy", "excess")]
    write_csv(folder / study["positive_file"], positive, positive_columns)
    write_csv(folder / study["negative_file"], negative, COMMON + ["rejection_stage", "rejection_reason"])
    print(f"Tables: {len(positive)} positive, {len(negative)} negative -> {folder}", flush=True)


def ensure_corpus(config, *, check_only=False):
    """Build the selected corpus on demand; --check never writes or starts preprocessing."""
    def ready():
        if not config.CORPUS.is_file():
            return False
        # An interrupted merge can leave a partial parquet with corpus=null in the manifest.
        return (not config.CORPUS_MANIFEST.exists() or
                json.loads(config.CORPUS_MANIFEST.read_text()).get("corpus") is not None)

    if ready():
        return
    if check_only:
        raise RuntimeError(f"corpus missing or incomplete: {config.CORPUS}\n"
                           "A normal run will build it automatically; --check does not build anything.")
    if not any(config.FEED.glob("raw_news_*.csv.xz")):
        raise RuntimeError(f"cannot build {config.CORPUS}: no raw captures in {config.FEED}")
    config.CORPUS.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with config.CORPUS.with_suffix(".build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"corpus build already in progress: {config.CORPUS}") from None
        if ready():
            return
        print(f"Building corpus automatically: wires={config.WIRES} -> {config.CORPUS}\n"
              "Processing all available years; completed yearly caches are reused.", flush=True)
        importlib.import_module("0_preprocessing.first_publication").build_corpus()
        if not ready():
            raise RuntimeError(f"preprocessing did not produce a complete corpus: {config.CORPUS}")


def wire_overrides(existing, selection):
    """CLI selection wins over this one environment key; keep all other Hydra overrides."""
    overrides = [part for part in existing.split()
                 if part.split("=", 1)[0].lstrip("+~") != "preprocessing.wires"]
    return " ".join(overrides + [f"preprocessing.wires={selection}"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--check", action="store_true", help="validate local prerequisites, no API calls")
    parser.add_argument("--weeks", type=int, help="override study.weeks (e.g. 1 for a smoke test)")
    parser.add_argument("--attempt", help="override study.attempt")
    parser.add_argument("--budget-usd", type=float, help="total LLM budget for this run, including restarts")
    parser.add_argument("--wires", choices=("bloomberg", "press", "all"),
                        help="select wires; build the corpus automatically if missing (except --check)")
    args = parser.parse_args(argv)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    if args.weeks is not None:
        cfg.study.weeks = args.weeks
    if args.attempt:
        cfg.study.attempt = args.attempt
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Operational limit, not scientific identity: raising it must resume the same folder.
    budget = cfg.pop("llm_budget_usd", 5.0)
    if args.budget_usd is not None:
        budget = args.budget_usd
    study = cfg["study"]
    dates = calendar(study)
    os.environ["TT_EXPERIMENT"] = cfg["pipeline_experiment"]
    if args.wires is not None:
        os.environ["TT_OVERRIDES"] = wire_overrides(os.environ.get("TT_OVERRIDES", ""), args.wires)
    sys.path.insert(0, str(SOURCE))
    import config
    import pipeline
    import polars as pl
    import llm
    config.load_env()
    llm.validate_budget(budget, [config.CONFIG[s].get("llm_model") for s in ("detect", "basket")])
    if config.CONFIG["basket"]["method"] != "uniform":
        raise ValueError("this study supports only basket.method=uniform")
    for key in ("positive_file", "negative_file"):
        if Path(study[key]).name != study[key]:
            raise ValueError(f"{key} must be a filename")
    if study["positive_file"] == study["negative_file"]:
        raise ValueError("output filenames must differ")
    valid = pipeline.check_environment()
    print(f"Wires: {config.WIRES}; corpus: {config.CORPUS}")
    if args.check:
        ensure_corpus(config, check_only=True)
    if not valid:
        raise RuntimeError("missing environment prerequisites listed above")
    ensure_corpus(config, check_only=args.check)
    corpus = pl.scan_parquet(config.CORPUS)
    bounds = corpus.select(pl.col("date").min().alias("start"), pl.col("date").max().alias("end")).collect().row(0)
    if bounds[0] is None or bounds[1] < dates[-1] - timedelta(days=7):
        raise ValueError(f"corpus does not cover the requested weeks: {bounds}")
    baseline = dates[0] - timedelta(days=round((config.params("detect")["detect_months"] +
                                               config.params("detect")["baseline_months"]) * 30.44))
    if bounds[0] > baseline:
        print(f"WARNING: truncated baseline; corpus begins {bounds[0]}, requested {baseline}")
    stat = config.CORPUS.stat()
    code_hash = digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(SOURCE.rglob("*.py"))} | {"runner": Path(__file__).read_text()})
    identity = {"study": cfg, "pipeline": config.CONFIG, "code_hash": code_hash,
                "corpus": {"path": str(config.CORPUS), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}}
    fingerprint = digest(identity)[:12]
    base = (ROOT / study["output_dir"]).resolve()
    allowed = (ROOT / "experiments/weekly_qualitative/results").resolve()
    if not base.is_relative_to(allowed) or base == allowed:
        raise ValueError("output_dir must be a run directory under experiments/weekly_qualitative/results")
    folder = base.with_name(base.name + "-" + fingerprint)
    print(f"Calendar: {len(dates)} weeks, {dates[0].date()} -> {dates[-1].date()}")
    print(f"Results: {folder}")
    print(f"LLM budget: ${budget:.2f} for this run (including restarts)")
    if args.check:
        print("Local checks passed. API connectivity/credit not checked; no API calls made.")
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (folder / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = folder / ".cache"
        cache.mkdir(exist_ok=True)
        atomic_json(cache / "manifest.json", identity)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=False).stdout.strip()
        detector, basket = pipeline.stage("detect"), pipeline.stage("basket")
        spending = llm.configure(cache, budget)
        spending.save()
        universe_path = cache / "universe.parquet"
        if universe_path.exists():
            names = pl.read_parquet(universe_path)
        else:
            names = basket.universe.load()
            names.write_parquet(universe_path.with_suffix(".tmp"))
            os.replace(universe_path.with_suffix(".tmp"), universe_path)
        prices = importlib.import_module("4_backtesting.prices")
        checkpoints = []
        for i, asof in enumerate(dates, 1):
            path = cache / f"week-{asof.date()}.json"
            resumed = path.exists()
            print(f"\nWeek {i}/{len(dates)}: {asof.date()} {'(cached)' if resumed else ''}", flush=True)
            if resumed:
                checkpoint = json.loads(path.read_text())
            else:
                try:
                    checkpoint = collect_week(asof, corpus, detector, basket, names,
                                               config.CONFIG, fingerprint, commit)
                finally:
                    print(f"LLM spent/reserved: ${spending.spent:.4f} / ${budget:.2f}; "
                          f"details: {spending.path}", flush=True)
                atomic_json(path, checkpoint)
            checkpoints.append(checkpoint)
            # Do not truncate existing tables while replaying cached weeks: that would erase
            # review annotations on the later rows before we reach their checkpoints.
            if not resumed or i == len(dates):
                export(folder, checkpoints, study, prices)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
