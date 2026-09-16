# Pipeline — minimal I/O contracts

## How to run it

From `code/`, the directory this file sits in:

```
python src-deliverable/RUN_FULL_PIPELINE.py --asof 2023-01-16 --until 2025-12-31
```

`--asof` is the day the pipeline pretends it is: detection and the basket never read past it.
`--until` is how far the monitor is then run forward and the result backtested — leave it out and
the run stops at the portfolio, which is what a live Monday morning wants.

The as-of is **not** the day the theme was born. Gate 3 requires two weeks of persistence, so
ChatGPT is dated 2023-01-02 and becomes detectable on 2023-01-16. Asking on the 2nd returns nothing,
correctly: on the 2nd you could not yet know.

### The basket: names, or filings

```
python src-deliverable/RUN_FULL_PIPELINE.py --asof 2023-01-16 --until 2025-12-31                              # names  (default)
python src-deliverable/RUN_FULL_PIPELINE.py --asof 2023-01-16 --until 2025-12-31 --experiment filings_basket  # filings
```

| | what it reads | needs | genAI at 2023-01-16 |
|---|---|---|---|
| `uniform` | the theme's words against SEC **company names**, translated brand → company by a model | `OPENAI_API_KEY`, `SEC_USER_AGENT` | `msft, aapl, googl` |
| `filings` | the theme's words counted inside **10-K/10-Q text**, ranked by density | `SEC_USER_AGENT`, downloads filings | `aapl, kdozf, zdge, bkng, …` — an iPhone basket, see its FUTUREWORK |

### The monitor: the default is fine, the control is not optional

`entry: rolling_z` and `exit: news_level` are the default in every experiment file, and entry has
no second option to pick. The exit has two, and the second one exists to be beaten:

| exit | what it reads | genAI: weeks held, return |
|---|---|---|
| `news_level` | is the flow still above its pre-entry normal | 157, +163.7% |
| `horizon` | nothing — a fixed number of weeks | 52, +54.6% · 104, +101.0% |
| *(never exit)* | | 157, +163.7% |

```
python src-deliverable/RUN_FULL_PIPELINE.py --asof 2023-01-16 --until 2025-12-31 --experiment horizon_exit
```

Read that table before quoting a return. On genAI `news_level` is **exactly** buy-and-hold — the
theme's words did not exist before it, so the flow can never fall back to a normal it never had, and
the rule never fires. It does fire on five of the seven themes one run detected, closing them 100 to
141 weeks early. `2_monitors/exit/FUTUREWORK_exit.md` has both tables and the blind spot.

Change how a half behaves rather than which it is:

```
python src-deliverable/RUN_FULL_PIPELINE.py --asof 2023-01-16 --until 2025-12-31 --set monitor.entry.z_threshold=2.0
```

### Runs never overwrite each other

Nothing to name and nothing to remember. Every frame is written as
`data/processed/{stage}/{run id}.parquet`, and the run id is the three method names plus a digest of
the whole configuration:

```
  default                              graph_anomaly-rolling_z+news_level-uniform-89ea2b
  --set monitor.entry.z_threshold=2.0  graph_anomaly-rolling_z+news_level-uniform-6b09cf
  --experiment filings_basket          graph_anomaly-rolling_z+news_level-filings-7b8662
```

One value different, different id: the earlier run is still there to compare against. Nothing
different, same id: a rerun replaces its own files instead of piling up copies — including when you
`--set` a value to what it already was.

The digest says *that* the parameters differ, not *how*. That is in the manifest beside every frame:

```python
artifacts.index("backtest")     # one row per run: its id, and every parameter that produced it
```

### A sweep

```
for z in 0.5 1.0 2.0; do
    python src-deliverable/RUN_FULL_PIPELINE.py --asof 2023-01-16 --until 2025-12-31 --set monitor.entry.z_threshold=$z
done
```

### From a notebook

`RUN_FULL_PIPELINE.ipynb` — one cell per stage, each returning its frame so it can be looked at
before the next one runs, plus the weekly-volume and equity-curve plots. `pipeline.run(asof, until=)`
does the whole thing in one call.

## What a theme is

**A theme is a set of words.** That is the whole definition, and it is what makes the pipeline hold
together: `detect` produces the set, `monitor` and `basket` consume it — each asking the same
question at a different granularity, *how much does this document use these words?*
(headlines per week · filings per company). `basket` and `backtest` never see it.

If a theme later becomes something richer (a centroid, an entity set, a graph), only that one question
changes. Coupled to today's definition: `monitor` and the exposure half of `basket`.
Free of it: the weighting half and `backtest`, which are keyed on `theme × ticker` and know nothing else.

## Two rules that shape everything

**1. Two frames per stage.** Every stage with pluggable methods returns:
- the **canonical output** — identical columns for every method, the only thing downstream reads;
- a **diagnostics** frame on the **same index** — the method's own explanation columns, for inspection
  and plots only. Never a pipeline input.

**2. Three kinds of things.** External *sources* (read-only, cached, never produced here),
*artifacts* (the spine, one per stage), *stages* (the functions between them).

---

## External sources

| source | where | used by |
|---|---|---|
| news feed | `data/raw/raw_news_{year}.csv.xz` — Bloomberg capture, one row per feed *message* | preprocessing |
| ticker universe | SEC register → cache `data/raw/edgar/company_tickers.json` — every current registrant, which is not the same as every investable name (`3_baskets/universe.py`) | basket |
| SEC filings | SEC EDGAR API → cache `data/raw/edgar/` — company 10-K/10-Q text, the evidence of what a company does | basket |
| prices | Yahoo adjusted closes → cache `data/raw/prices/` | backtest **only** |

Sources are self-populating caches: first run needs network, reruns are offline, deleting a cache is safe.

---

## Schema

```
  source                 stage                    artifact          index
  ──────                 ─────                    ────────          ─────

  news feed ──────────▶  1 preprocessing  ──────▶ corpus            headline
                                                    │
                    ┌───────────────────────────────┤
                    │                               ▼
                    │      2 detect        ──────▶ themes           theme
                    │                               │   week · vocab · vocab_wide
                    │                               ▼
  corpus ───────────┴──▶  3 monitor        ──────▶ signal           theme × week
                                                    │   score · position ──────┐
                    entry_dates() = theme's asof    │                          │
                                                    ▼                          │
  universe ─┬──────────▶  4 basket         ──────▶ weights          theme × ticker
  filings ──┘                                       │                          │
                                                    ▼                          │
  prices ─────────────▶  5 backtest        ◀────────┴──────────────────────────┘
                                           ──────▶ performance      theme
                                                   curves           theme × date
```

Every stage is `f(previous artifact [, source]) → (canonical, diagnostics)`.
Stages 2–5 map over all themes, so every frame downstream of `detect` is keyed by `theme` first.

The **`signal` reaches the backtest twice over**: its first `position == True` per theme dates that
theme (`asof`, which keeps `exposure` point-in-time), and the full column is the entry/exit rule the
backtest follows. Nothing downstream may use data past a theme's `asof`.

---

## 1 · preprocessing — fixed, no variants

- **reads** news feed `data/raw/raw_news_{year}.csv.xz` (columns used: `Headline, CaptureTime, WireName, Event`)
- **returns** `corpus` — one row per publication: `Headline` str, **exactly as the wire published it**
  · `date` datetime, **the earliest publication message, never CaptureTime**. Sorted by date.
- **saves** `data/processed/news_corpus.parquet` (+ `news_corpus_manifest.json` for idempotent rebuilds)

The dating rule is this stage's whole reason to exist: most feed rows are re-transmissions, and dating
by CaptureTime injects fake bursts.

**It repairs, it does not prepare.** No stripping, no shortening, no tokenizing, no thinning — a
method that reads words, one that reads entities and one that embeds sentences each need something
different from a headline, and only they know what. Two consequences: a prefix is kept, because
`Daniel Yergin: Surveillance With Prewitt and Keene` loses its speaker when stripped; and a title
published again on a later day is kept, because a repeat is a fact about the news. Which repeated
titles are worth discarding is a judgement a human makes, reading `repeated_titles()` and filling in
`preprocessing.drop_titles`.

## 2 · detect — variants: `graph_anomaly`, …

- **reads** `corpus` (sliced to the method's window) + `asof` date
- **returns**
  - `themes` — one row per detected theme: `theme` id · `week` (emergence week) ·
    `vocab` list[str] · `vocab_wide` list[str]
  - `diagnostics` — same `theme` index, method's own columns. For `graph_anomaly`:
    `anchor, partners, mentions, degree, clustering, caught, promoted, rank_*, example`
- **saves** `data/processed/themes/{run}.parquet` + `{run}.diag.parquet`

`vocab` **is** the theme: it's the column stages 3 and 4 consume, nothing else. A method that finds
themes by embeddings or topic models fills the same columns and slots in unchanged.

**Two lists, because the two consumers want opposite words.** `vocab` is the theme without the words
too common for their presence to mean anything; `vocab_wide` is the same list with them. The monitor
counts headlines and so must not hold `google` — it would count Google, and the weekly series comes
out flat through ChatGPT's whole arrival. The basket buys companies and so needs exactly those large
names. Measured at 2023-01-02, the two differ by `iphone` and `google`, and the basket is `msft`
alone on the first and `aapl, googl, msft` on the second. One model call produces both.

**No human step in the flow.** Curation, when wanted, is filtering the `themes` frame
(`themes[themes.theme.isin(picked)]`) — a notebook display helper, not a stage and not a file format.

## 3 · monitor — the stage is fixed; its two halves are the variants

- **reads** `corpus` **+** `themes` — the series is built here, it is not an input: for each theme, count
  weekly the headlines whose `terms` intersect the theme's `vocab`, on a contiguous zero-filled weekly
  grid (contractual: rolling statistics depend on it)
- **returns**
  - `signal` — index `theme × week`: `score` float (method-defined intensity) ·
    `position` bool — **the trading rule**: in the theme, or out of it, that week
  - `diagnostics` — same index: `hits, score, level` — the series each half judged on
- **saves** `data/processed/signal/{run}.parquet` + `{run}.diag.parquet`

The monitor is not a date-finder: it decides **when to be invested**, week by week, and the backtest
follows it in and out. **One entry and one exit per theme**: a theme is a thing that happens once,
and a rule that lets the position reopen is no longer trading the theme — it is trading the score's
oscillation around its own threshold.

**The two halves do not share a measure, and that is the point.** `entry/` opens on ACCELERATION —
is this week unusual against the weeks just before it. `exit/` closes on LEVEL — is the flow still
above what it was before we bought. An acceleration score cannot find an end: its window catches up
with any sustained level within `roll_weeks` and returns to zero while the theme is still running.
Measured on genAI, using it for both held the position two weeks on a theme that ran for years; the
level rule holds it 154 weeks, for +145.0% against SPY's +78.6%.

Two constraints, both about not seeing the future:
`position` may not be True before the theme's own emergence `week`, and the first True per theme is
the theme's `asof` — the date that keeps `exposure` point-in-time. `entry_dates(signal)` is what
produces it: nothing downstream may read anything published after it.

## 4 · basket — variants: `uniform`, …

- **reads** `themes` (the `vocab_wide`, and the entry date `entry_dates` derives from the signal) +
  the ticker universe
- **returns**
  - `weights` — index `theme × ticker`: `weight` float > 0, sums to 1.0 per theme. **This is the
    portfolio.**
  - `diagnostics` — same index: which of the theme's words matched, and how
- **saves** `data/processed/weights/{run}.parquet` + `{run}.diag.parquet`

**The mapping uses the theme's own words and nothing else** — no filings, no third source to keep in
sync. But the words and the register speak differently: news names brands, the register names legal
entities. Measured on the genAI vocabulary at 2023-01-02, listed companies called `google`, `iphone`,
`chatgpt`, `openai`: none, none, none, none. The raw join returns an empty basket and no error.

So the stage is two steps. A model translates each word into the company it belongs to — the maker,
the owner, the operator, never who would benefit, which would be a claim about the future — and only
then is the register searched, demanding the name BE that company rather than contain it, and
collapsing share classes by cik. `google → alphabet → googl`; `iphone → apple → aapl`;
`microsoft-backed chatgpt → openai, microsoft → msft`.

External evidence (SEC filings, revenue segments) is a later variant of this stage, not a
prerequisite — as are the ticker tags Bloomberg already writes on every headline. Both are in
`3_baskets/uniform_method/FUTUREWORK_uniform.md`, with what was measured for each.

Whatever it reads must stop at the theme's entry date: a filing or a listing dated after it is
information we would not have had.

## 5 · backtest — fixed, pure math

- **reads** `weights` + `signal` (the entry/exit rule) + prices
- **returns**
  - `performance` — index `theme`: `total_return, ann_return, ann_vol, sharpe, max_dd` and the same
    against the benchmark
  - `curves` — index `theme × date`: the equity curve (summary/detail pair, same shape as canonical/diag)
- **saves** `data/processed/backtest/{run}.parquet` + `{run}.curves.parquet`

Hold the theme's basket while `position` is True, flat otherwise, trading on the first price date
**after** the week boundary (never same-bar). What is being measured is not a basket but the whole
chain that produced it — see below.

Stated, not modelled, in this version: no transaction costs, risk-free rate 0, weights renormalized
over the names that actually have prices, delisted names carried at their last close (optimistic),
and the ticker universe is current registrants, so results carry survivorship bias.

---

## A run is a whole pipeline

What the backtest measures is never a single stage: it is a **combination** — one detector, one
monitor, one basket construction, with their parameters. That combination is a *run*, and comparing
pipelines means comparing the `performance` of several runs.

So the run id names every file a run writes, which is also how `data/processed/` stays legible:

```
data/processed/{stage}/{run}.parquet         the canonical frame
data/processed/{stage}/{run}.diag.parquet    the diagnostics
data/processed/{stage}/{run}.json            manifest: methods, parameters, window, timestamp
```

Default id: `{detector}-{monitor}-{basket}`, e.g. `graph_anomaly-rolling_z-uniform`. Pass an explicit
label when comparing parameter variants of the same triple (`graph_anomaly-rolling_z-uniform@z2`).
The parameters themselves live in the manifest, not in the filename — the id stays readable and the
manifest stays complete. Artifacts are written to a temporary file and renamed, so an interrupted run
never leaves a half-written parquet behind.

## Layout

Every stage is a numbered directory, so the order is visible in the folder itself. Only what is not a
stage sits loose at the root.

```
src-deliverable/
  README.md                        these contracts
  PLAN.md                          the build order and what is done
  experiments/default.yaml         the whole pipeline in five blocks — one file per experiment
              filings_basket.yaml    the same, with the other basket method
              horizon_exit.yaml      the same, with the control exit
              no_llm.yaml            the same, with every model call off
  config.py                        where the data lives; loads one experiment
  artifacts.py                     every stage writes its frames under the run id
  progress.py                      what a long run prints while it is running

  pipeline.py                            run(asof, until=None)       -> every frame
  RUN_FULL_PIPELINE.py                   the same, from a terminal
  RUN_FULL_PIPELINE.ipynb                the same, a cell at a time, with plots

  0_preprocessing/first_publication.py   build_corpus()              -> corpus
  1_detectors/graph_anomaly/detect.py    detect(corpus, asof)        -> themes, diag, rejects
                            tokenizer.py  what a word is — the detector defines it, the monitor reuses it
  2_monitors/monitor.py                  monitor(corpus, themes, asof) -> signal, diag
             entry/rolling_z.py           when to open: ACCELERATION
             exit/news_level.py           when to close: LEVEL
  3_baskets/universe.py                  load()                      -> who can be bought
             uniform_method/basket.py     weights(themes)             -> weights, diag
             filings_method/basket.py     weights(themes)             -> the same, from filing text
             filings_method/edgar.py      the SEC source: hit lists and filing text
  4_backtesting/rebalance.py             backtest(weights, signal, prices) -> performance, curves
                prices.py                 the Yahoo cache — the only IO on this path
```

**Every stage is a directory**, uniformly, even where only one method exists today. The number is the
stage's place in the order, the directory names the role, and the file names the method — named for
what it does: `rolling_z` is a 12-week rolling z-score, `first_publication` dates a headline by its earliest
ADD message. A `detector-` prefix inside `1_detectors/` would only repeat the directory.

A stage directory also holds what that stage owns. `tokenizer.py` lives in `0_preprocessing/` because
preprocessing is what *writes* the `terms` column, so it is what defines a term; the detector and the
monitor import that same tokenizer instead of each keeping a copy, which is exactly the tangle the old
scripts fell into.

`src-deliverable/` goes on `sys.path` (`sys.path.insert(0, "../src-deliverable")` in a notebook).
Root modules are imported normally. **Anything inside a numbered directory is loaded with
`importlib.import_module("0_preprocessing.text")`, never with a bare `import` statement** — a name
starting with a digit is not a Python identifier, so `import 0_preprocessing` is a syntax error while
the importlib call works. This costs nothing here, because the experiment file chooses each stage's method
and `pipeline.py` therefore has to resolve the module dynamically in any case; `pipeline.stage("detect")`
returns it.

Only three modules sit at the root, and each is there because it belongs to no single stage:

| module | why it is not inside a stage |
|---|---|
| `config.py` + `experiments/` | every stage needs the paths, and the run is defined once for all of them |
| `artifacts.py` | every stage writes `(canonical, diag, manifest)` under the run id, atomically |
| `pipeline.py` | it is what resolves the stages from the experiment file and chains them |
| `progress.py` | every stage shows progress the same way, and bars go to stderr so stdout stays the result |

Everything else lives with the stage that owns it: the SEC session in `3_baskets/filings_method/edgar.py`, the
Yahoo price cache in `4_backtesting/prices.py`, the tokenizer in `1_detectors/graph_anomaly/`.

When a second stage needs one of those, it imports it from where it lives rather than having it
promoted to the root. `2_monitors/monitor.py` counting a theme's words must use the very tokenizer
that chose those words, and writing `1_detectors.tokenizer` at the top of the file says so. Moving it
to the root would not remove that dependency, only hide it.
