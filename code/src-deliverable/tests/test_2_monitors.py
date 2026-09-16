"""2_monitors — counting a theme's volume and turning it into a position."""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import polars as pl

mon = importlib.import_module("2_monitors.monitor")
entry = importlib.import_module("2_monitors.entry.rolling_z")
leave = importlib.import_module("2_monitors.exit.news_level")

THEMES = pl.DataFrame([{"theme": "widget", "week": datetime(2024, 3, 4),
                        "vocab": ["widget", "acmeone"]}],
                      schema={"theme": pl.String, "week": pl.Datetime, "vocab": pl.List(pl.String)})


def _corpus(quiet_weeks: int = 20, loud_weeks: int = 8) -> pl.DataFrame:
    """Months of silence, then a theme arrives — the shape the score is built to find."""
    rows = []
    for week in range(quiet_weeks):
        day = datetime(2023, 10, 16) + timedelta(weeks=week)
        rows += [{"Headline": f"acme reports numbers for period {week}", "date": day}] * 3
    for week in range(loud_weeks):
        day = datetime(2024, 3, 4) + timedelta(weeks=week)
        rows += [{"Headline": f"widget chosen by company{i} in week {week}", "date": day}
                 for i in range(12)]
    return pl.DataFrame(rows)


def grid_has_no_gaps():
    """weeks_between returns consecutive Mondays, so a silent week is a zero and not a hole"""
    weeks = mon.weeks_between(datetime(2024, 1, 3), datetime(2024, 2, 20))
    if any(w.weekday() != 0 for w in weeks):
        return "not every week starts on a Monday"
    gaps = [(b - a).days for a, b in zip(weeks, weeks[1:]) if (b - a).days != 7]
    return f"{len(gaps)} gaps in the grid" if gaps else None


def a_headline_counts_once():
    """a headline carrying several of a theme's words is still one story"""
    corpus = pl.DataFrame([{"Headline": "widget and acmeone together", "date": datetime(2024, 3, 6)}])
    weeks = mon.weeks_between(datetime(2024, 3, 4), datetime(2024, 3, 10))
    hits = mon.weekly_hits(corpus, THEMES, weeks)
    return None if hits["widget"] == [1] else f"counted {hits['widget']}, expected [1]"


def score_waits_for_history():
    """no score before min_periods weeks, none against a flat stretch, and one once the past varies

    The flat case is worth knowing: a theme going from exactly zero to loud scores nothing in its
    first loud week, because a window of zeros has no deviation to divide by. From the second week
    on the spike is inside the window, the deviation exists, and the score appears.
    """
    scores = entry._z([0] * 10 + [50, 50], roll_weeks=12, min_periods=4)
    if any(s is not None for s in scores[:4]):
        return "a score appeared before there was history to compare against"
    if any(s is not None for s in scores[4:11]):
        return "a flat stretch produced a score, but it has no deviation to divide by"
    return None if scores[-1] is not None and scores[-1] > 2 else f"the spike scored {scores[-1]}"


def nothing_before_the_theme_existed():
    """a position cannot open before the emergence week, however loud the score"""
    counts = [1, 2] * 6 + [40] * 12          # a base that varies, or there is no deviation to divide by
    if entry.week(counts, born=0, min_periods=4, n_consecutive=1)[0] != 5:
        return "the fixture does not score loud early, so the born guard is not being tested"
    opened, _ = entry.week(counts, born=14, min_periods=4, n_consecutive=1)
    if opened is None:
        return "never entered a theme that was loud for twelve weeks"
    return None if opened == 14 else f"entered at week {opened}, expected the emergence week 14"


def the_exit_measures_level_not_acceleration():
    """a theme that stays loud is NOT left, however flat its acceleration has gone

    This is the whole reason the two halves are separate modules. The z-score of a sustained
    level returns to zero within `roll_weeks`; the level does not.
    """
    counts = [1] * 60 + [40] * 60                       # quiet, then loud and staying loud
    scores = entry._z(counts, roll_weeks=12, min_periods=4)
    tail = [s for s in scores[-12:] if s is not None]
    if tail and max(tail) > 1.0:
        return "the entry score never went quiet, so this does not test what it claims"
    left, level = leave.week(counts, entry=60, roll_weeks=12, base_weeks=52)
    if left is not None:
        return f"left at week {left} a theme still running at {level[-1]:.0f}x its normal"
    return None if level[-1] > 10 else f"the level fell to {level[-1]:.1f}x while still loud"


def a_theme_that_fades_is_left():
    """back to the pre-entry normal for n_consecutive weeks, and the position closes"""
    counts = [5] * 60 + [50] * 20 + [4] * 40            # quiet, loud, then back to normal
    left, level = leave.week(counts, entry=60, roll_weeks=12, base_weeks=52, n_consecutive=4)
    if left is None:
        return f"never left a theme that came back to {level[-1]:.2f}x normal"
    return None if left > 80 else f"left at week {left}, before the theme had faded"


def the_exit_waits_to_be_armed():
    """it may not fire before the theme has been loud: at entry the window is still pre-theme

    Without arming the trailing mean at entry is mostly the quiet past, the level sits near 1.0,
    and the rule closes the position a month after opening it — on every theme, every time.
    """
    counts = [5] * 60 + [50] * 40
    left, _ = leave.week(counts, entry=60, roll_weeks=12, base_weeks=52, n_consecutive=4)
    return None if left is None else f"closed at week {left} a theme that only got louder"


def a_theme_arriving_is_entered():
    """silence, then volume: the stage enters, and not before the emergence week"""
    signal, _ = mon.monitor(_corpus(), THEMES, "2024-04-29", lookback_months=7,
                            roll_weeks=12, min_periods=4, report=False)
    held = signal.filter("position")
    if held.is_empty():
        return "never entered a theme that went from 0 to 12 headlines a week"
    first = held["week"].min()
    return None if first >= datetime(2024, 3, 4) else f"entered {first}, before the theme emerged"


def the_entry_date_is_recoverable():
    """entry_dates gives the boundary everything downstream must stop reading at"""
    signal, _ = mon.monitor(_corpus(), THEMES, "2024-04-29", lookback_months=7,
                            roll_weeks=12, min_periods=4, report=False)
    dates = mon.entry_dates(signal)
    if "asof" not in dates.columns or len(dates) != 1:
        return f"expected one row of [theme, asof], got {dates.columns} with {len(dates)} rows"
    held = signal.filter("position")["week"].min()
    return None if dates["asof"][0] == held else f"asof {dates['asof'][0]} is not the first week held"



LOUD = "gigacorp"                                   # a word that is in the news every week anyway
BORN = datetime(2023, 12, 11)                       # the week the theme's own word arrives


def _corpus_beside_a_loud_word(weeks: int = 44, arrival: int = 36) -> pl.DataFrame:
    """A word that is everywhere all along, and beside it a theme word that arrives late.

    The loud word swings between 60 and 99 headlines a week on its own, which is the scale the
    z-score ends up dividing by — and the theme brings eleven, which fits inside that swing.

    Measured on this fixture, weeks above the score threshold once the theme exists:
    on `widget` alone five in a row, on `widget` + the loud word never more than one.
    """
    rows = []
    for week in range(weeks):
        day = datetime(2023, 4, 3) + timedelta(weeks=week)
        rows += [{"Headline": f"{LOUD} ships units number {i} of week {week}", "date": day}
                 for i in range(60 + (week * 23) % 40)]
        if week >= arrival:
            rows += [{"Headline": f"widget chosen by company{i} in week {week}", "date": day}
                     for i in range(11)]
    return pl.DataFrame(rows)


def _entered(corpus: pl.DataFrame, vocab: list[str]) -> bool:
    themes = pl.DataFrame([{"theme": "widget", "week": BORN, "vocab": vocab}],
                          schema={"theme": pl.String, "week": pl.Datetime,
                                  "vocab": pl.List(pl.String)})
    signal, _ = mon.monitor(corpus, themes, "2024-01-29", lookback_months=12,
                            roll_weeks=12, min_periods=4, report=False)
    return not signal.filter("position").is_empty()


def a_common_word_hides_the_arrival():
    """why the detector returns two lists: one common word and the same arrival stops being seen"""
    corpus = _corpus_beside_a_loud_word()
    if not _entered(corpus, ["widget"]):
        return "the arrival was not entered on the rare word alone"
    if _entered(corpus, ["widget", LOUD]):
        return f"still entered with `{LOUD}` in the list, so the common word did not hide it"
    return None


def a_common_word_sets_the_scale():
    """and how it hides it: its own weekly swing becomes the deviation the score divides by"""
    corpus = _corpus_beside_a_loud_word()
    weeks = mon.weeks_between(datetime(2023, 4, 3), datetime(2024, 1, 29))
    themes = pl.DataFrame([{"theme": "rare", "week": BORN, "vocab": ["widget"]},
                           {"theme": "wide", "week": BORN, "vocab": ["widget", LOUD]}],
                          schema={"theme": pl.String, "week": pl.Datetime,
                                  "vocab": pl.List(pl.String)})
    hits = mon.weekly_hits(corpus, themes, weeks)
    before = weeks.index(BORN)
    if any(hits["rare"][:before]):
        return "the rare word was already in the news before the theme existed"
    quiet = hits["wide"][:before]
    if min(quiet) < 11 or max(quiet) - min(quiet) < 11:
        return f"the loud word does not swing wider than the arrival: {min(quiet)}..{max(quiet)}"
    return None



def genai_is_entered_on_one_list_and_not_the_other():
    """the real case, on the real corpus: genAI is entered on `vocab` and invisible on `vocab_wide`

    The same arrival, the same week, the same thresholds. `google` and `iphone` carry tens of
    headlines a week of their own, so the deviation the score divides by goes from about one to
    about thirteen, and ChatGPT's eleven headlines stop being an anomaly. In the arrival week the
    wider count even falls, because Google and iPhone news stops over Christmas.
    """
    import config
    corpus = pl.read_parquet(config.CORPUS)
    asof = datetime(2023, 1, 2)
    params = {k: v for k, v in config.params("monitor").items() if k != "method"}
    entered = {}
    for name, vocab in [("vocab", ["chatgpt", "microsoft-backed chatgpt", "openai"]),
                        ("vocab_wide", ["chatgpt", "iphone", "google",
                                        "microsoft-backed chatgpt", "openai"])]:
        themes = pl.DataFrame([{"theme": "genai", "week": asof, "vocab": vocab}],
                              schema={"theme": pl.String, "week": pl.Datetime,
                                      "vocab": pl.List(pl.String)})
        signal, _ = mon.monitor(corpus, themes, asof, report=False, **params)
        entered[name] = not signal.filter("position").is_empty()
    if not entered["vocab"]:
        return "genAI was not entered on the list the monitor is given"
    if entered["vocab_wide"]:
        return "genAI was entered on the wider list too, so the common words did not drown it"
    return None


def the_control_holds_for_exactly_as_long_as_it_is_told():
    """`horizon` reads nothing: it leaves a fixed number of weeks after entering, and says so"""
    horizon = importlib.import_module("2_monitors.exit.horizon")
    counts = [5] * 200
    left, series = horizon.week(counts, entry=60, weeks=52)
    if left != 112:
        return f"left at week {left}, expected 60 + 52"
    if any(s is not None for s in series):
        return "the control returned a series, but it judges nothing and must say so"
    return None if horizon.week(counts, entry=60, weeks=500)[0] is None else \
        "ran past the end of the grid instead of holding to it"


CHECKS = [grid_has_no_gaps, a_headline_counts_once, score_waits_for_history,
          nothing_before_the_theme_existed, the_exit_measures_level_not_acceleration,
          a_theme_that_fades_is_left, the_exit_waits_to_be_armed,
          the_control_holds_for_exactly_as_long_as_it_is_told,
          a_theme_arriving_is_entered,
          the_entry_date_is_recoverable,
          a_common_word_hides_the_arrival, a_common_word_sets_the_scale,
          genai_is_entered_on_one_list_and_not_the_other]
