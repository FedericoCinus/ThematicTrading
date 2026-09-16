"""Green ticks and red dots — no framework.

Every `test_<stage>.py` exposes CHECKS: a list of functions. A check takes nothing, returns None
when it passes, or a short string saying what is wrong. Its docstring is the line you read.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):                      # the package, then the tests themselves
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def imports() -> None:
    """Load every module before anything else runs.

    A missing name takes a tenth of a second to find this way and two and a half minutes to find
    through a test that reads the corpus first.
    """
    for name in ("config", "0_preprocessing.first_publication",
                 "1_detectors.graph_anomaly.tokenizer", "1_detectors.graph_anomaly.detect",
                 "2_monitors.monitor", "2_monitors.entry.rolling_z",
                 "2_monitors.exit.news_level", "3_baskets.universe", "3_baskets.uniform_method.basket",
                 "3_baskets.filings_method.edgar", "3_baskets.filings_method.basket",
                 "4_backtesting.prices", "4_backtesting.rebalance",
                 "artifacts", "progress", "pipeline"):
        try:
            importlib.import_module(name)
        except Exception as e:
            print(f"   🔴  import {name}   →  {type(e).__name__}: {e}")
            return
    print("   ✅  every module imports")


def block(stage: str) -> None:
    """Run every check of one stage, printing one line each."""
    try:
        module = importlib.import_module(f"test_{stage}")
    except ModuleNotFoundError as missing:
        if missing.name and missing.name.startswith("test_"):
            print(f"{stage}\n   ·  not written yet")
            return
        print(f"{stage}\n   🔴  the block does not import   →  {type(missing).__name__}: {missing}")
        return
    except Exception as broken:
        # anything else raised while importing: report it as this block's failure rather than
        # letting it stop every block after this one
        print(f"{stage}\n   🔴  the block does not import   →  {type(broken).__name__}: {broken}")
        return

    print(stage)
    for check in module.CHECKS:
        t0 = time.time()
        try:
            failure = check()
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        mark = "🔴" if failure else "✅"
        label = (check.__doc__ or check.__name__).strip().splitlines()[0]
        print(f"   {mark}  {label}"
              + (f"   →  {failure}" if failure else "")
              + (f"   [{elapsed:.0f}s]" if elapsed > 1 else ""))
