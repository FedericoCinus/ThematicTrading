"""Entry method — a theme is entered when its weekly volume ACCELERATES.

    week(counts, born) -> (the week to enter on, the scores it was judged on)
"""
from __future__ import annotations

# ======================================================================================
# ROLLING Z — how unusual this week is against the weeks just before it
#
#   IN    counts   headlines per week for one theme, on a contiguous zero-filled grid
#   IN    born     index of the theme's emergence week; nothing before it can be entered
#   OUT   (index of the entry week or None, the score series)
#
#   score(i) = (counts[i] − mean(previous roll_weeks)) / std(previous roll_weeks)
#              the week itself is excluded, or a spike would flatten inside its own window
#              None until min_periods weeks of history exist, and None against a flat window
#
#   TWO WAYS IN, both read off the same score
#     level   loud for n_consecutive weeks running          a theme that lasts
#     accel   the score jumps by accel_threshold over
#             accel_lag weeks                               a theme arriving too fast to wait for
#
#   WHAT THIS CANNOT DO: say when to leave. The window catches up with any sustained level within
#   roll_weeks, so the score returns to zero while the theme is still loud. Acceleration detects
#   a beginning and is blind to an end — which is why the exit is a different measure entirely,
#   in ../exit/.
# ======================================================================================


def week(counts: list[int], born: int, *, roll_weeks: int = 12, min_periods: int = 4,
         z_threshold: float = 1.0, n_consecutive: int = 4, accel_lag: int = 4,
         accel_threshold: float = 3.0, **_) -> tuple[int | None, list[float | None]]:
    """When to enter this theme.

    INPUT   counts           headlines per week, one entry per week of the grid
            born             index of the emergence week
            roll_weeks       how many previous weeks a week is compared against
            min_periods      weeks of history needed before a score exists at all
            z_threshold      standard deviations above the recent mean that count as loud
            n_consecutive    weeks in a row it must be loud to enter on the level rule
            accel_lag        weeks back the score is compared with itself
            accel_threshold  how much it must have risen over that span to enter on its own
    OUTPUT  (index of the week to enter, or None if the theme never convinces us,
             the score of every week — None where there was not enough history)
    """
    scores = _z(counts, roll_weeks, min_periods)
    loud = [s is not None and s > z_threshold for s in scores]
    for i in range(born, len(scores)):
        level = i + 1 >= n_consecutive and all(loud[i - n_consecutive + 1:i + 1])
        accel = (scores[i] is not None and i >= accel_lag and scores[i - accel_lag] is not None
                 and scores[i] - scores[i - accel_lag] > accel_threshold)
        if level or accel:
            return i, scores
    return None, scores


def _z(counts: list[int], roll_weeks: int, min_periods: int) -> list[float | None]:
    """The rolling z-score of each week against the weeks before it.

    A theme going from exactly zero to loud scores nothing in its FIRST loud week: a window of
    zeros has no deviation to divide by. From the second week the spike is inside the window, the
    deviation exists, and the score appears. The sharpest arrivals are entered one week late,
    never missed.
    """
    out = []
    for i, count in enumerate(counts):
        past = counts[max(0, i - roll_weeks):i]
        if len(past) < min_periods:
            out.append(None)
            continue
        mean = sum(past) / len(past)
        variance = sum((p - mean) ** 2 for p in past) / (len(past) - 1)
        out.append(None if variance == 0 else (count - mean) / variance ** 0.5)
    return out
