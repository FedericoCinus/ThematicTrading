"""What a long run prints while it is running.

    with step("🔎", "detect") as done: ...        a stage header, and its elapsed time on exit
    for x in each(items, "gate 4"): ...           a bar with a count and an ETA
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

# ======================================================================================
# PROGRESS — the only place that decides how a run looks while it works
#
#   IN    on      every call takes it; False and nothing is printed at all
#   OUT   stderr  bars and headers, so stdout stays the run's actual output
#
#   step("🔎", "detect")        ▶️  detect
#                               ✅ detect  3m 41s
#   each(rows, "gate 4")        🔎 gate 4            68/240  28%  [01:12<02:47,  2.0it/s]
#
#   Bars go to STDERR on purpose: `python RUN_FULL_PIPELINE.py > run.txt` then keeps the numbers
#   and leaves the animation on the terminal, instead of writing a file full of carriage returns.
#
#   Two rules for what gets a bar:
#     a loop is worth one when it is slow AND its length is known in advance — an LLM call per
#     theme, a corpus pass per headline. A fast loop with a bar is noise.
#     the ETA comes free from tqdm and is worth more than the percentage: it is the number that
#     answers "do I wait or do I go and do something else".
#
#   tqdm missing is not an error: the loop still runs, the bar simply is not there. Progress is
#   not a result, and a run must not fail for want of decoration.
# ======================================================================================

ICONS = {"preprocess": "📰", "detect": "🔎", "monitor": "📈",
         "basket": "🧺", "backtest": "💰", "save": "💾"}


def each(items, label: str, *, total: int | None = None, on: bool = True, unit: str = ""):
    """Iterate, showing a bar with a count and an estimated time left.

    INPUT   items   anything iterable; a list so its length is known, or pass `total`
            label   what is being worked through, shown to the left of the bar
            total   how many, when `items` cannot say
            on      False returns `items` untouched — what tests and library calls want
            unit    what is being counted, for the rate: "wk", "headline", …
    OUTPUT  the same items, one at a time
    """
    if not on:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, total=total, file=sys.stderr, leave=False, dynamic_ncols=True,
                desc=f"   {label}", unit=unit or "it",
                unit_scale=True,      # 4687138 -> 4.69M, because nobody reads seven digits
                mininterval=0.3,      # redraws a few times a second, not thousands
                smoothing=0.1,        # the ETA follows the average rate, not the last instant
                bar_format="{desc:<26} {n_fmt}/{total_fmt} {percentage:3.0f}% |{bar}| {remaining} left")


@contextmanager
def step(name: str, *, on: bool = True, detail: str = ""):
    """Announce a stage, then say how long it took.

    INPUT   name    a key of ICONS, or any label
            on      False prints nothing
            detail  appended to the opening line
    OUTPUT  nothing — the block runs inside
    """
    if not on:
        yield
        return
    icon = ICONS.get(name, "▶️")
    print(f"\n{icon}  {name}{'   ' + detail if detail else ''}", file=sys.stderr, flush=True)
    started = time.monotonic()
    try:
        yield
    finally:
        print(f"✅  {name}   {clock(time.monotonic() - started)}", file=sys.stderr, flush=True)


def clock(seconds: float) -> str:
    """Seconds as something a person reads.

    INPUT   seconds   elapsed
    OUTPUT  `4.2s`, `3m 41s`, `1h 02m`
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m"
