"""Build a company basket for a theme vocabulary from SEC filings (naive count baseline).

Given a theme = bag of entity words (output of the detection pipeline, e.g. notebook
0.29's vocab dicts), find US companies whose 10-K/10-Q filings mention the theme terms,
using EDGAR full-text search to shortlist and local occurrence counts to score.
See theme_basket.py for method, caveats and the EDGAR cache layout.

File layout produced:

  data/processed/theme_basket_{name}.parquet           company-level basket (scores)
  data/processed/theme_basket_{name}_filings.parquet   per-filing count detail
  data/processed/theme_basket_{name}_manifest.json     build record (rule, params, diagnostics)

Re-running with unchanged parameters is a no-op (manifest-certified); EDGAR responses
and filing texts are cached under data/raw/edgar/, so parameter changes only fetch
what is new.

Usage:
    python scripts/build_theme_basket.py --name genai --as-of 2023-01-17 \
        --vocab chatgpt openai altman microsoft nvidia
    python scripts/build_theme_basket.py --name genai --as-of 2023-01-17 \
        --vocab-file themes/genai.txt --lookback 365 --lookahead 120
    python scripts/build_theme_basket.py --name genai ... --force      # rebuild outputs
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]                  # code/
sys.path.insert(0, str(_ROOT))
from theme_basket import SCORE_RULE, EXTRACT_RULE, TICKER_CSV, build_basket  # noqa: E402

OUT_DIR = _ROOT / "data" / "processed"


def _paths(name: str) -> dict[str, Path]:
    return {"basket": OUT_DIR / f"theme_basket_{name}.parquet",
            "filings": OUT_DIR / f"theme_basket_{name}_filings.parquet",
            "manifest": OUT_DIR / f"theme_basket_{name}_manifest.json"}


def _params(a: argparse.Namespace, vocab: list[str]) -> dict:
    return {"score_rule": SCORE_RULE, "extract_rule": EXTRACT_RULE,
            "vocab": sorted(vocab), "as_of": a.as_of, "lookback_days": a.lookback,
            "lookahead_days": a.lookahead, "forms": a.forms,
            "max_hits_per_term": a.max_hits_per_term, "universe": str(a.universe)}


def _print_basket(basket: pd.DataFrame, top_n: int):
    if basket.empty:
        print("basket: EMPTY — no universe filing mentions any theme term in the window")
        return
    cols = ["ticker", "company", "n_filings", "total", "n_terms_hit", "per_1k_words"]
    with pd.option_context("display.width", 160, "display.max_colwidth", 40):
        print(basket[cols].head(top_n).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="basket slug used in output filenames")
    ap.add_argument("--as-of", required=True, help="reference date YYYY-MM-DD (e.g. theme promotion)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vocab", nargs="+", help="theme terms (entities)")
    g.add_argument("--vocab-file", type=Path, help="file with one theme term per line")
    ap.add_argument("--lookback", type=int, default=365, help="days before as-of (filing date)")
    ap.add_argument("--lookahead", type=int, default=0, help="days after as-of (0 = point-in-time)")
    ap.add_argument("--forms", default="10-K,10-Q", help="EDGAR root form filter")
    ap.add_argument("--max-hits-per-term", type=int, default=1000,
                    help="terms with more EDGAR hits are treated as generic (no shortlisting)")
    ap.add_argument("--universe", type=Path, default=TICKER_CSV, help="ticker list CSV")
    ap.add_argument("--top-n", type=int, default=25, help="rows of the basket to print")
    ap.add_argument("--force", action="store_true", help="rebuild outputs even if certified")
    a = ap.parse_args()

    vocab = a.vocab or [w.strip() for w in a.vocab_file.read_text().splitlines()
                        if w.strip() and not w.startswith("#")]
    paths, params = _paths(a.name), _params(a, vocab)

    if not a.force and paths["manifest"].exists() and paths["basket"].exists():
        m = json.loads(paths["manifest"].read_text())
        if {k: m.get(k) for k in params} == params:
            print(f"certified build for '{a.name}' exists — skipping (use --force to rebuild)")
            _print_basket(pd.read_parquet(paths["basket"]), a.top_n)
            return

    t0 = time.time()
    res = build_basket(vocab, a.as_of, lookback_days=a.lookback, lookahead_days=a.lookahead,
                       forms=a.forms, universe_csv=a.universe,
                       max_hits_per_term=a.max_hits_per_term)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res["basket"].to_parquet(paths["basket"], index=False)
    res["filings"].to_parquet(paths["filings"], index=False)
    manifest = {**params, **{k: v for k, v in res["meta"].items() if k not in params},
                "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0)}
    paths["manifest"].write_text(json.dumps(manifest, indent=2))

    print(f"\nbasket '{a.name}': {len(res['basket'])} companies from "
          f"{res['meta']['n_filings_scored']} filings  [{time.time() - t0:.0f}s]")
    if res["meta"]["generic_terms"]:
        print(f"generic terms (counted but not shortlisted): {res['meta']['generic_terms']}")
    print(f"-> {paths['basket'].relative_to(_ROOT)}\n")
    _print_basket(res["basket"], a.top_n)


if __name__ == "__main__":
    main()
