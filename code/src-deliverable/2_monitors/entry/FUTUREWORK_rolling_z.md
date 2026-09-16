# What the rolling z-score does and does not answer

## 1 · It detects arrival, not presence — solved, by not using it for both ends

The score asks how unusual a week is against its own recent past, so it measures *acceleration*.
Once a theme's new volume has been in the window for `roll_weeks`, that volume is the new normal
and the score falls back to zero — while the theme is still running.

Measured on genAI: March 2023 carries 28 headlines a week, ten times December, and scores 0.6.

This was the pipeline's binding constraint while the same score decided both ends. With one entry
and one exit, exiting on the score held the position two weeks on a theme that ran for years:

| roll_weeks | weeks held | exit | theme | SPY |
|---|---|---|---|---|
| 12 | 2 | 2023-01-16 | +3.8% | +87.7% |
| 52 | 8 | 2023-02-27 | +7.8% | +87.7% |

**The fix was not a longer window but a different measure**, and it lives in `../exit/`: loudness as
a multiple of the theme's own pre-entry normal. A level, not a derivative. The same theme is now
held 154 weeks for +145.0% against SPY's +78.6%.

What remains here is the entry only, and for entering, acceleration is the right question.

## 2 · `lookback_months` only has to clear the theme's birth

Beyond that it changes nothing, because `roll_weeks` already cuts the memory. Measured: 24 and 160
months give the identical entry date and the identical five weeks held, for 6 seconds against 62.
Set it to reach past the birth and stop paying.

An earlier note here claimed it moved the entry dates. It does not — that run had also changed the
theme's vocabulary, and the effect belonged to the vocabulary.

## 3 · A theme arriving from exactly zero is entered one week late

A window of zeros has no deviation to divide by, so the first loud week scores nothing. From the
second week the spike is inside the window and the score appears. The sharpest arrivals are the ones
delayed, which is the opposite of what one would want, and it is one week.
