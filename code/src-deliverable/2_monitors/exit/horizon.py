"""Exit method — leave after a fixed number of weeks, whatever the theme is doing.

    week(counts, entry) -> (the week to leave on, no series: this rule reads nothing)
"""
from __future__ import annotations

# ======================================================================================
# HORIZON — the baseline every other exit has to beat
#
#   IN    counts   ignored; it is here so every exit method has the same shape
#   IN    entry    index of the week the position opened
#   OUT   (entry + weeks, or None when that falls past the end of the grid)
#
#   THIS IS NOT A RULE ABOUT THE THEME. It is the control: holding for a fixed time needs no
#   signal, no model and no data, so any exit that reads the news has to do better than it or it
#   is measuring nothing. Reporting `news_level` without this number beside it says how the
#   pipeline did, not whether its exit contributed.
#
#   Counted in WEEKS, not months, because an exit method is handed the weekly grid and no dates.
#   52 ≈ the twelve months the earlier work used.
# ======================================================================================


def week(counts: list[int], entry: int, *, weeks: int = 52,
         **_) -> tuple[int | None, list[float | None]]:
    """When to leave this theme: a fixed number of weeks after entering it.

    INPUT   counts   the weekly grid, read only for its length
            entry    index of the week the position opened
            weeks    how many weeks to hold
    OUTPUT  (index of the week to leave, or None if the grid ends first,
             a series of None — this rule judges nothing, and says so)
    """
    leave = entry + weeks
    return (leave if leave < len(counts) else None), [None] * len(counts)
