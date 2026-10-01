# The same command does not give the same answer twice

## 1 · What was observed

Two runs, same command, same corpus, nothing changed between them:

```
python src-deliverable/RUN_FULL_PIPELINE.py --asof 2023-01-16 --until 2025-12-31
```

| | basket for `chatgpt@2023-01-02` | return |
|---|---|---|
| run 1 | `msft, googl, aapl` | **+145.0%** |
| run 2 | `msft, googl, aapl, nwl` | **+56.9%** |

`nwl` is Newell Brands. It arrived because that run's vocabulary contained `magic bullet` — from the
single headline *"ChatGPT Is No Magic Bullet for Microsoft's Bing"* — and the translation step
answered it as a blender brand, which it also is. The other run's vocabulary did not contain the
word at all.

The vocabulary itself has come back differently on different runs of the identical call:

```
['chatgpt', 'microsoft-backed chatgpt', 'openai']
['chatgpt', 'openai projects', 'microsoft-backed chatgpt', 'openai']
['chatgpt', 'microsoft', 'openai', 'fanatics', 'microsoft-backed chatgpt', 'iphone', 'google',
 'celsius', 'genesis', 'bing', 'nyc schools', 'alphabet', 'tata', 'oxbotica', 'magic bullet']
```

## 2 · Where it enters

Three model calls, all with `temperature: 0` already set:

| call | where | what varies |
|---|---|---|
| gate 4, `PATTERNS` | `1_detectors/graph_anomaly/detect.py` | whether a cluster is kept |
| vocabulary, `ENTITIES` | same file | which words the theme is made of |
| translation, `ISSUERS` | `3_baskets/uniform_method/basket.py` | which companies those words name |

The first two decide *what the pipeline even looks at*, so their variation propagates through every
stage after them.

## 3 · What does not fix it

- **`temperature: 0`** is already set everywhere. It fixes which token wins a comparison, not the
  floating-point arithmetic that ranks them — near-ties flip between requests. Measured earlier at
  gate 4: `gpt-4o-mini` keeps the ChatGPT cluster 6 times in 20; `gpt-4o` keeps it 8 in 8. Clear-cut
  clusters never wobbled on either model, so the instability lives entirely in the close calls.
- **A `seed`** is best-effort in the API and tied to a `system_fingerprint` that changes when the
  provider changes anything behind it. It narrows the window; it does not close it.

## 4 · Why it blocks parameter search, specifically

The run id carries a digest of the configuration, so two runs of one configuration write to the
same files and a rerun replaces its own. That is correct bookkeeping and it now hides a lie: the
id asserts *same configuration*, and the result underneath it can still differ.

```
change   --set monitor.entry.z_threshold=2.0
observe  +90% instead of +145%

was that the parameter, or was it the model answering differently?
```

There is no way to tell from one run each. Every comparison between two configurations is
contaminated, and the size of the contamination — +145% against +56.9% on the same command — is
larger than most parameter effects worth looking for.

## 5 · The fix, and what it does not buy

**Cache the model's answers on disk, keyed by `(model, the exact prompt sent)`.** The first run pays
and records; every run after it reads. The same question then always gets the same answer, two runs
of one configuration become identical, and a difference between two configurations is the parameter.

It belongs beside `artifacts.py` — the same idea as the Yahoo price cache and the EDGAR filing
cache, applied to the model — and it makes sweeps cheap as a side effect, since the model is asked
once per distinct question rather than once per run.

What it does **not** buy: a *new* configuration is still measured once, with whatever the model
answered that first time. The cache makes results reproducible, not correct. For a configuration
that has never been run, the honest procedure is still to repeat it and look at the spread — which
is what was done for the gate 4 prompt (a dozen calls per variant, never one) and is documented in
the comment above `PATTERNS`.

## 6 · The smaller sibling: n is tiny

A run detects 3 to 6 themes, of which usually one is interesting. Comparing two configurations on
one theme establishes nothing, cache or no cache. Accumulating themes means sweeping `--asof` as
well as the parameters, and that multiplies the model calls — which is the other reason the cache
comes first.
