# Build plan

Contracts live in [README.md](README.md); this file tracks the work.

`[ ]` todo · `[~]` written, under review · `[x]` done (approved)

One item at a time: I write it, you read it, we discuss, it becomes `[x]`. Nothing moves to the next
item before that. Every item is a few dozen lines — if one turns out bigger, we split it.

Each item says where its logic comes from and how we check it. Automated tests are deliberately out of
scope for now; checks are parity against the notebook that already produces the number.

**Baseline before starting.** The working tree already has uncommitted moves (`theme_*.py` deleted at
`code/` and re-added under `code/scripts/`, `build_theme_basket.py` edited). Those are yours and stay
as they are — nothing in this plan touches `code/scripts/` until item 13.

---

## 0 · `[x]` Stubs — numbered stages, better names

Numbers stay: they make the order visible in the folder, and they cost nothing, because
`importlib.import_module` loads `0_preprocessing` and even hyphenated names without complaint — only
a bare `import 0_preprocessing` is a syntax error, and `pipeline.py` has to resolve stage modules
dynamically anyway, since the experiment file picks the method.

Names improved within that scheme, and slot 3 reserved for `exposure`, which shifts basket and
backtesting up by one:

```
1_detectors/detector-graph_anomaly.py   →  1_detectors/graph_anomaly.py
2_monitors/monitor-variance_outlier.py  →  2_monitors/rolling_z.py
3_baskets/constructor-uniform.py        →  4_baskets/uniform.py
4_backtesting.py                        →  5_backtesting.py
                                        →  3_exposures/edgar_counts.py (slot reserved, no file yet)
```

No empty placeholders: a file appears when it is written. Which non-stage modules are justified is
argued in the README table — `prices.py` starts inside the backtest, `edgar.py` splits only if the
exposure stage gets unwieldy.

Also verified here, not at the end: `code/.env` is ignored by `.gitignore:15` and has never been
committed. Nothing to clean up.

## 1 · `[x]` `config.py` — paths and secrets

Data root (`TT_DATA` env override, default `code/data`), the source paths, the corpus path,
`load_env()` for `.env`. The only module that knows where anything lives.

From: the `_ROOT` constants scattered across the four scripts, and `theme_detect._load_env`.
Check: every path constant resolves correctly from a notebook and from a script. This is what makes
the "moved file → wrong data directory" bug — which broke two modules already — impossible to repeat.

## 2 · `[~]` `0_preprocessing/tokenizer.py` — the tokenizer

`WIRES`, the regexes, the three stopword sets, `strip_prefix`, `extract_terms`.

From: `theme_detect.py` lines 32–118, verbatim — three scripts import these today through `sys.path`
hacks and private names.
Check: `extract_terms` on a handful of headlines returns exactly what the corpus already has in
`terms`. The 22.6M-row corpus must not need rebuilding.

Note: `is_entity` gets renamed. It does not recognise entities — it only checks that no word of the
term is in `EVENT_STOP` — and a name that oversells is worse than a blunt one.

## 3 · `[ ]` `artifacts.py` — stage I/O and run ids

`save(stage, run, canonical, diagnostics=None, **manifest)` and `load(stage, run)` over
`data/processed/{stage}/{run}.parquet`, `.diag.parquet`, `.json`.

Decided: the run id is `{detector}-{entry}+{exit}-{basket}-{digest}`; parameters go
in the manifest, not the filename. Writes go to a temp file and get renamed, so an interrupted run
leaves no half-written parquet.

## 4 · `[~]` `0_preprocessing/first_publication.py` — stage 1

`build_corpus(years=None, force=False) -> Path`. Feed messages → the one corpus
`[Headline, date, terms]`, ADD-event dating, per-year cache, manifest, streamed merge.

From: `scripts/preprocess_news.py` (173 lines), a fairly direct lift once it imports `text` and `config`.
Check: rebuild one year and diff against the existing corpus — must be identical.
Not migrated: `build_sample.py` stays a dev toy (it deliberately skips the dating rule).

## 5 · `[~]` `1_detectors/graph_anomaly/detect.py` — stage 2

`detect(corpus, asof, **gates) -> (themes, diagnostics, rejects)`. The novelty + co-occurrence funnel.

From: `scripts/theme_detect.py` (414 lines) minus what moves elsewhere (tokenizer → `tokenizer.py`, env →
`config.py`) and minus the raw-corpus path that dates by CaptureTime, which is deleted: it
reintroduces the replay artifact preprocessing exists to remove.

The one item that is a redesign rather than a lift — today it returns a single flat funnel frame:

- One row of `themes` per promoted anchor: `theme` id (`anchor@week`), `week` = emergence week,
  `vocab` from `vocabulary()`. Several anchors of one real theme still produce several rows with
  overlapping vocabularies; merging them is a later version.
- `vocab` replaces the human curation step, and `vocabulary()` is how: every term the anchor has
  shared a headline with in all history up to `asof`, filtered by the model to the words that name a
  company, institution, product or technology. `theme_bag_of_words_size` caps how many are kept.
- Rejected candidates get their own `rejects` frame. Diagnostics share the canonical index, so they
  can only explain promoted themes, and the near-misses are what you read when tuning gates.
- Gate 4 removes what the headlines show is not a theme, asking only structural questions so nothing
  the model knows about the future can enter. It needs `gpt-4o`: the mini is undecided on borderline
  clusters and its answer is then settled by floating-point noise.

Check: **done** — with the default gates, `chatgpt` is promoted in the week of 2023-01-02 from every
as-of date between January and April 2023, which is the case history says this detector solved.
Quantum is not found and cannot be: see `FUTUREWORK_graph_anomaly.md`, section 0.

What that same run also shows: 1 real theme among 24-81, the rest being single-entity events
(`gardaworld`, `apotea`), people in the news (`mangione`, and `luigi mangione` beside it), and
recurring wire tables (`communications peers` and five siblings). Those are exactly the three classes
the discarded LLM gate was written to reject, so dropping it was a mistake — the statistical gates do
not tell a theme from a story.

## 6 · `[x]` `2_monitors/` — stage 3, split in two

`monitor.py` counts and does not decide: it builds the contiguous weekly grid, then hands the counts
to two swappable halves that live in its own folder.

```
entry/rolling_z.py    week(counts, born)  -> (the week to open on, the scores it judged on)
exit/news_level.py    week(counts, entry) -> (the week to close on, the levels it judged on)
```

**They must not share a measure.** A rolling z is a derivative: its window catches up with any
sustained level within `roll_weeks` and returns to zero while the theme is still running, so it can
find a beginning and is blind to an end. The exit is a level instead — the trailing mean over the
mean of the `base_weeks` BEFORE entry, 1.0 being pre-entry normal — with an arming step, without
which the position closes a month after opening on every theme.

Measured on genAI, same detection, same basket, same window 2023 → 2025:

| exit | weeks held | theme | SPY |
|---|---|---|---|
| the entry's own z-score | 2 | +3.8% | +87.7% |
| news level | 154 | +145.0% | +78.6% |

One entry and one exit per theme, enforced by construction: `position` is one unbroken run of True.
Both method names reach the run id, so `rolling_z+news_level` says which pair traded.

From: `theme_exit.py`, brought over from the notebooks. Its stop-loss and fixed-horizon rules are
not ported — they are price rules, and this stage only ever reads news.

## 7 · `[x]` `3_baskets/filings_method/` — the SEC source and the second basket method

`edgar.py` is plumbing only: `how_many_filings` (one cheap count), `hits` (the document list, paged
and date-bisected around EFTS's 10,000-per-window cap), `text` (one document, HTML discarded, xz
cached under `EXTRACT_RULE`). `basket.py` is the method: shortlist by EFTS, count the theme's words
inside those documents, rank by occurrences per 1k words with a floor, take top_n at equal weight.

Folder rather than a flat file, like `1_detectors/graph_anomaly/`: the method owns its source.
`3_baskets/universe.py` came out of the uniform method at the same time, because who can be bought is the
stage's question and not one method's.

**Measured, and it is the finding**: genAI at 2023-01-02, one-year window — `chatgpt` 0 filings,
`iphone` 146 (89 in the universe), `google` 1,701 (too common to shortlist on), `openai` 2,
`microsoft-backed chatgpt` 0. The basket is `aapl, kdozf, zdge, bkng, ipm, ai, drio, trip, apps,
logi`: mobile-app companies that write about iPhones. The method worked; the vocabulary carried a
word that is not the theme, and this method amplifies that where `uniform` dilutes it — one bad word
buys one name there and the whole basket here. See `3_baskets/filings_method/FUTUREWORK_filings.md`.

Not on the critical path: `experiments/default.yaml` keeps `basket.method: uniform`; `TT_EXPERIMENT=filings_basket` swaps it.

## 8 · `[x]` `3_baskets/uniform_method/` — stage 4

`weights(themes, names=None, **params) -> (weights, diagnostics)`. Per theme: translate each word
into the company it belongs to, look that company up in the SEC register, take the first `top_n` in
vocabulary order at equal weight.

Decided: **not** the filing scorer. The mapping uses the theme's own words and one model call, so the
stage has no multi-GB cache and no network stage on its critical path. What that costs is in
`FUTUREWORK_uniform.md`, measured.

Three things the measurements forced, each with its own check in `tests/test_3_baskets.py`:

- **a model between word and register.** News names brands, the register names legal entities: listed
  companies called `google`, `iphone`, `chatgpt`, `openai` — none, none, none, none. Without the
  translation the genAI basket is empty either way, capped or not; with it, `msft` on the vocabulary
  the pipeline actually produces and `aapl, googl, msft` on the uncapped one.
- **the register name must BE the company, not contain it.** `apple` otherwise takes Apple
  Hospitality REIT and Apple iSports Group.
- **one holding per cik.** Alphabet's four share classes are one company; unmerged they took 44% of
  a nine-name basket that believed itself equal-weight.

Stage 5 as originally planned (a separate `weights()` over an `exposure` frame) is folded in here:
with no scoring step there is nothing to rank, only vocabulary order.

Blocked on nothing. `data/raw/tickers/ticker_list_US.csv` is still absent; the SEC register
substitutes for it, which is also what makes the universe today's registrants only.

## 9 · `[x]` `baskets/` weighting — stage 5

Folded into 8: with no scoring step there is no ranking to apply, only vocabulary order.

## 10 · `[x]` `4_backtesting/prices.py` — Yahoo source

`download(tickers)` fills a per-ticker parquet cache, `load(tickers)` returns the wide close frame.
Two functions, the only IO on the backtest path, no key needed.

Decided: a ticker Yahoo has nothing for is cached EMPTY rather than left missing, so delisted and
renamed symbols are not re-queried every run — and `load` simply omits it from the columns, which
the backtest then handles by renormalising. pandas here and nowhere else, because what follows is
matrix arithmetic on a date index.

## 11 · `[x]` `4_backtesting/rebalance.py` — stage 5

`backtest(weights, signal, prices, benchmark=, rebalance_months=) -> (performance, curves)`. Four
public functions: `segments` reads the holding stretches off the position column, `curve` marks one
stretch to market, `summarise` reads five numbers off a curve, `backtest` chains them per theme and
against the benchmark.

The five near-duplicates of `theme_backtest.py` (`equal_weight_curve`, `weighted_curve`,
`rebalance_curve`, `rebalance_backtest`, `rebalance_backtest_weighted`) collapse into one weighted
path: equal weights are weights like any others.

Decided, and each has its own check in `tests/test_4_backtesting.py`:
- **only the compounded curve.** The notebooks reported Σ of period returns beside it; two numbers
  from the same data differing by the compounding is a way to be wrong twice.
- **`sharpe = ann_return / ann_vol`**, risk-free 0 — consistent with the two numbers beside it
  rather than a third convention.
- **the window stops where the SIGNAL stops**, not where the price history does. Running a flat
  curve out to the last available price annualises a return over a life the strategy never had.
- **trade the first price date AFTER the week boundary**, never same-bar.
- Renamed `months` → `rebalance_months`: in the old scripts it reads like test duration.

Measured end to end at asof 2023-01-02, monitored to 2025-12-31: genAI holds `aapl, googl, msft`
for 21 weeks out of 210 and returns +28.2% against SPY's +87.7% — the monitor's early exit, already
recorded in `2_monitors/FUTUREWORK_rolling_z.md`, is what costs it.

Left: `run(asof, until=)` widens `lookback_months` so the monitor's window still contains the
theme's birth. That belongs in the monitor, which should take the window from the theme rather than
from its own end date.

## 12 · `[ ]` `pipeline.py`

One run = `(detector, monitor, basket)` end to end, ~60 lines. The executable version of the README
schema, the front door of the deliverable, and the thing the backtest actually evaluates.

## 13 · `[ ]` Close out

Point notebooks 0.28–0.37 at the package, verify parity, then delete the migrated scripts.
`build_sample.py` stays.


---

# Open — what is not settled

## 0 · Preprocessing

The question that was open — *when the same headline comes back, is it the same story?* — is
answered: **the same, and only the first one is kept**, matched character for character on the raw
title. That settles both faces of it at once, the same title on another day and in another year, and
it is the rule the earlier analysis had recommended all along.

- `[x]` **Rebuilt**: 22,761,199 rows, the count of distinct raw titles, down from 24,442,332.
  Re-run it with `REBUILD = True` in the notebook, or `pp.build_corpus(force=True)`.
- `[x]` **Recurring programmes: nothing more to drop.** A report that groups titles by what follows
  the last colon finds `Preview`, `WSJ`, `Reuters`, `CNBC` — source attributions, not programmes —
  so it was removed rather than left to mislead. A recurring programme and a sourced headline have
  the same shape, and no cheap rule separates them. `preprocessing.drop_titles` stays as a manual
  escape hatch, normally empty.
- `[x]` **Kept**: `data/processed/corpus_old_scripts/` (3.6 GB, renamed from `legacy_v1`, with a
  README saying what it is) until notebooks 0.28-0.37 are migrated.

## Later stages — nothing to do yet, but do not forget

- `[x]` ~~`artifacts.py` does not exist.~~ `save`/`load`/`manifest`/`runs` over
  `data/processed/{stage}/{run}.{parquet,diag.parquet,json}`, written through a `.tmp` and renamed.
  `pipeline.run(..., save=True)` writes all four stages.
- `[~]` **`data/raw/tickers/ticker_list_US.csv` does not exist** — a Bloomberg extract with no
  download script in the repo. The dead reference is gone from `config.py` and the README; the
  universe is now defined in one place, `3_baskets/universe.py`, and `config.yaml` keeps a
  `basket.universe_csv` slot that nothing reads. What remains is the difference that matters: the
  SEC register is every REGISTRANT, not every INVESTABLE name, which is how `quantum` reaches
  Quantum Leap, a shell. The reproducible fix is a liquidity screen off the prices
  `4_backtesting/prices.py` already caches, applied AS AT THE THEME'S DATE — not written.
- `[x]` ~~The vocabulary is capped for the monitor and the basket wants it uncapped.~~ Half of this
  was a wrong diagnosis: uncapped, the basket was empty too, because brands are not legal entities —
  fixed by translating in the basket. The other half was real, and `detect` now returns two lists off
  one model call: `vocab` without the too-common words for the monitor, `vocab_wide` with them for
  the basket. Measured at 2023-01-02 they differ by `iphone` and `google`, and the genAI basket goes
  from `msft` alone to `aapl, googl, msft`.
- `[ ]` **`TERM_STOP` is dead logic** in the detector's tokenizer: entirely inside `EXTRACT_STOP`, so
  it can never fire on a corpus term. Kept as a guard, documented. Remove it, or leave it.

## Debts

- `[ ]` **Notebooks 0.28–0.37 read a `terms` column the corpus no longer has.** Item 13, brought
  forward because the rebuild broke them today.
- `[ ]` **`pyyaml` is not declared** in `code/pyproject.toml`; `config.py` works only because another
  dependency happens to pull it in.
- `[ ]` Items **2** and **4** are written and green, still marked `[~]` awaiting your approval.
