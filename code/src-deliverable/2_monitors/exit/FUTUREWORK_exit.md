# What the exit rules do, measured against the control

## 0 · The study this was ported from concluded that none of them work

`news_level` is `theme_exit.news_exit` ported verbatim — the level series and the firing week match
to 0.00e+00 on the same input. What did not come across with the code is notebook 0.39's own verdict
on it, and it is not a footnote:

> **Nothing here beats holding by enough to be worth believing.** Of the 64 basket-and-stop
> combinations, none improved the Sharpe ratio. Across the news-rule sweep, exactly two cells beat
> holding [...] and the same setting *worsens* both filtered genAI baskets and all eight quantum
> ones. Two wins out of sixteen, with no sign that survives the $200m floor, is what noise looks
> like.

Rule by rule, from the same notebook:

| rule | verdict |
|---|---|
| news, **no re-entry** | *"Correct in kind, ruinous in practice."* On quantum it fires in September 2023 into a six-week lull and turns **+668% into +32%**; the news was back to 3.5× by December. |
| news, with re-entry | at threshold 1.0 a no-op — a monthly sampling of a four-week weekly rule cannot see a six-week event. Raising the threshold makes it fire, and makes it worse. |
| stop-loss | incoherent at any distance: a 20% stop against 30–54% monthly volatility fires on an ordinary month. Mean cost to Sharpe −0.59 to −1.63. |
| fixed horizon | *"loses to holding almost everywhere"* — 12 months costs more than 24, which costs more than not closing. |

**This pipeline's default is the row that study calls ruinous.** `exit: news_level` with one entry
and one exit is exactly *news, no re-entry*: the "close and stop" simplification the notebook names
as what converts a temporary quiet spell into a permanent loss. Our own genAI measurement is
consistent with the benign half of that — it never fires, so it costs nothing — but the case where
it does harm, quantum, is one this detector never reaches, so we cannot reproduce it here.

The caveat is the notebook's own: n = 2 themes, one entry each, one window. It does not establish
that theme exits cannot work. It does establish that on the only evidence anyone has gathered,
holding won.


## 1 · The control is not optional

`horizon` holds for a fixed number of weeks and reads nothing at all. It exists so that any exit
which *does* read the news has a number to beat. Measured on genAI, entry 2023-01-02, window to
2025-12-31, basket `aapl googl msft`:

| exit | weeks held | theme | SPY | sharpe |
|---|---|---|---|---|
| `news_level` | 157 | +163.7% | +87.7% | 1.79 |
| `horizon` 26 | 26 | +44.0% | +87.7% | 1.33 |
| `horizon` 52 | 52 | +54.6% | +87.7% | 1.29 |
| `horizon` 104 | 104 | +101.0% | +87.7% | 1.61 |
| never exit | 157 | +163.7% | +87.7% | 1.79 |

**On genAI `news_level` is exactly buy-and-hold.** Same weeks, same return, to the decimal. The
+163.7% is what the basket made; the exit contributed nothing to it. Reporting that number as the
pipeline's result without this table beside it would be claiming skill the rule did not show.

## 2 · It does fire, on themes that fade

The same comparison across the themes one run detected:

| theme | `news_level` | never exit | |
|---|---|---|---|
| alvotech exit | 41 | 177 | closes 136 weeks earlier |
| images exit | 28 | 164 | 136 |
| ftx latest | 23 | 164 | 141 |
| amoxicillin | 46 | 146 | 100 |
| semafor | 13 | 117 | 104 |
| rapidus | 153 | 153 | never fires |
| chatgpt | 157 | 157 | never fires |

Five of seven. So the rule works and the blind spot is specific.

## 3 · The blind spot: a theme born from a brand-new word has no normal to return to

`level = trailing mean / mean of the base_weeks before entry`. The detector only promotes terms that
were **absent** from the baseline — that is gate 1 — so the base is near zero almost by construction,
and the level sits at 30×, 80×, 140× forever. It can never come back to 1.0, so the rule never fires.

`chatgpt` and `rapidus` are exactly that case. `amoxicillin` and `ftx` are not: those words existed
before their story, so they had a normal to return to.

Two directions, neither tried:

- **a floor under the base**, so a theme that went from 0.1 to 40 headlines a week is measured
  against a plausible minimum rather than against nothing;
- **relative to the theme's own peak** instead of its past: "down to a tenth of the loudest it ever
  got" needs no pre-entry history and works for a word that did not exist.

## 4 · The stop-loss is not written, and the reason is the pipeline's shape

`theme_exit.py` had three rules; two are here. The third closes when the basket falls more than 20%
below its running peak, and it cannot be an exit method as this stage is built:

```
detect ──▶ monitor ──▶ basket ──▶ backtest
           no prices   the basket   the prices
                       exists here  exist here
```

The monitor runs before the basket and never sees a price, so there is no curve to draw down.

Two ways out, and they are not equivalent:

- **reorder to `detect → basket → monitor → backtest`** and hand the monitor prices and weights.
  The basket does not read the signal (`weights(themes)`), so nothing blocks this. The rule stays
  in one place and `signal` remains the whole trading rule.
- **put the stop in the backtest**, which already holds both. Cheaper, and it splits the rule across
  two stages: the signal would then say "in" on weeks the backtest was actually flat.
