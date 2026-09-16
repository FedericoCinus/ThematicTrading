"""Exit method — a theme is left when its news flow falls back to what it was before.

    week(counts, entry) -> (the week to leave on, the levels it was judged on)
"""
from __future__ import annotations

# ======================================================================================
# NEWS LEVEL — loudness as a multiple of the theme's own pre-entry normal
#
#   IN    counts   headlines per week for one theme, the same grid the entry read
#   IN    entry    index of the week the position opened
#   OUT   (index of the week to leave or None if it never fires, the level series)
#
#   base      mean of counts over the base_weeks BEFORE entry           "normal", fixed
#   level(i)  mean of the trailing roll_weeks / base                    1.0 = back to normal
#   leave     the first week after arming with level below threshold
#             for n_consecutive weeks running
#
#   ARMING is not a knob, it is what makes the rule mean anything. At the entry week the trailing
#   mean is still mostly pre-theme, so the level sits near 1.0 and an unarmed rule closes the
#   position a month after opening it, every time. "The story is over" presupposes it began, so
#   the rule waits for level >= arm before it may fire at all.
#
#   WHY NOT THE ENTRY'S OWN SCORE — a rolling z measures ACCELERATION: its window catches up with
#   any sustained level within roll_weeks and returns to zero while the theme is still loud. It
#   can find a beginning and cannot find an end. Measured on genAI, exiting on that score held the
#   position two weeks on a theme that ran for years.
#
#     week      entry  +4   +8   +12  ...  +40  +44  +48
#     hits        11   24   28   31        14   9    7
#     z          7.6  1.1   0.6  0.2       0.1  0.0  0.0    ← back to zero while it is still loud
#     level      1.0  6.2   9.4 10.8       2.1  1.2  0.8    ← still says the story is running
#                                                      └─ below 1.0: back to normal, leave
#
#   THE BLIND SPOT — it never fires on a theme born from a brand-new word. Gate 1 only promotes
#   terms ABSENT from the baseline, so the base is near zero almost by construction and the level
#   sits at 30x, 80x, 140x forever. Measured: it closes 5 of 7 themes 100-141 weeks early, and on
#   `chatgpt` and `rapidus` it is exactly buy-and-hold.          → FUTUREWORK_exit.md
# ======================================================================================


def week(counts: list[int], entry: int, *, roll_weeks: int = 12, base_weeks: int = 52,
         threshold: float = 1.0, n_consecutive: int = 4, arm: float | None = None,
         **_) -> tuple[int | None, list[float | None]]:
    """When to leave this theme.

    INPUT   counts         headlines per week, one entry per week of the grid
            entry          index of the week the position opened
            roll_weeks     the trailing window the level is measured over
            base_weeks     the weeks before entry that define "normal"
            threshold      the level below which the story is over; 1.0 is pre-entry normal
            n_consecutive  weeks it must stay below before the rule fires
            arm            the level the theme must first reach; defaults to `threshold`
    OUTPUT  (index of the week to leave, or None to hold to the end of the grid,
             the level of every week — None before there is a trailing window to take)
    """
    level = _level(counts, entry, roll_weeks, base_weeks)
    if level is None:
        return None, [None] * len(counts)

    arm = threshold if arm is None else arm
    lit = next((i for i in range(entry, len(level))
                if level[i] is not None and level[i] >= arm), None)
    if lit is None:
        return None, level                      # never got loud, so nothing to declare over

    quiet = 0
    for i in range(lit + 1, len(level)):
        quiet = quiet + 1 if level[i] is not None and level[i] < threshold else 0
        if quiet >= n_consecutive:
            return i, level
    return None, level


def _level(counts: list[int], entry: int, roll_weeks: int, base_weeks: int) -> list[float | None] | None:
    """Trailing mean over the fixed pre-entry mean. None when the theme had no history to be
    normal against — a theme born at the very start of the grid cannot be measured this way."""
    before = counts[max(0, entry - base_weeks):entry]
    base = sum(before) / len(before) if before else 0.0
    if base <= 0:
        return None
    out = []
    for i in range(len(counts)):
        window = counts[max(0, i - roll_weeks + 1):i + 1]
        out.append(sum(window) / len(window) / base if len(window) == roll_weeks else None)
    return out
