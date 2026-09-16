# What the name lookup cannot do, and what is waiting behind it

## 1 · News speaks in brands, the register lists legal entities

The stage joins the theme's words against SEC company names. Measured on the genAI vocabulary at
2023-01-02:

| the theme's word | listed companies named that |
|---|---|
| `google` | 0 — the company is Alphabet Inc. |
| `iphone` | 0 — Apple Inc. |
| `chatgpt` | 0 — OpenAI, and not listed |
| `openai` | 0 |
| `microsoft` | 1 — microsoft corp |

So the raw join returns an empty basket for the one theme the pipeline detects. `issuers()` now
translates the brand into the company before the join — `iphone` → Apple → AAPL — and the genAI
vocabulary reaches `msft, googl, aapl` instead of nothing.

Two weaknesses it leaves:

- **the model must spell the company as the register does.** The lookup demands the register name BE
  the company, which is what stops `apple` from taking Apple Hospitality REIT — but it also means
  that a model answering `Meta` where the register says `Meta Platforms, Inc.` finds nothing. Silent,
  like every other miss in this stage.
- **the basket reads `vocab_wide`, not `vocab`.** The detector's `max_doc_freq` exists for the
  monitor: with `google` in the word list the weekly series counts Google, not the theme, and comes
  out flat through ChatGPT's entire arrival. But `google` and `iphone` are exactly what this stage
  translates into GOOGL and AAPL. The detector therefore returns both lists off one model call.
  Measured at 2023-01-02, end to end:

  | list | words | basket |
  |---|---|---|
  | `vocab` | chatgpt · microsoft-backed chatgpt · openai | msft |
  | `vocab_wide` | + iphone · google | aapl · googl · msft |

  Note what MSFT depends on: not the word `microsoft`, which is too common for either list, but the
  model reading a company out of the bigram `microsoft-backed chatgpt`.

## 2 · Bloomberg already tagged every headline with its companies

The raw feed carries `DerivedTickersId` and `AssignedTickersId`, and preprocessing drops both. On the
594 headlines of 2022 naming chatgpt or openai:

```
GOOGL 193 · MSFT 188 · 1554630D 60 · 321042Z 50 · AAPL 39 · TWTR 24 · META 21 · SSTK 17 · TSLA 12
                       └── Bloomberg ids of unlisted companies — OpenAI among them
```

Point-in-time by construction: the tag was written the day the story was published, not today. It
reaches names no vocabulary would produce (Shutterstock, Twitter), and it needs no model and no
network — only a column kept in `0_preprocessing/first_publication.py`.

Not tested in a backtest. Notebook 0.4 reached the same conclusion from the other side, recording that
the ICB-cosine map was unreliable (max cosine ≈ 0.5, AI → Leisure Goods) and that the ticker tags were
"more trustworthy … grounded in the data".

## 3 · The filing text is what the earlier work actually traded

`scripts/theme_basket.py` (`SCORE_RULE = "fts-density-v2"`) counts the theme's words inside 10-K/10-Q
text: EFTS shortlists the filings, the text is downloaded and cached, companies rank by occurrences
per 1k words. The genAI basket in deck 8 — NVDA, TRIP, GOOG, EA, MYPS, APPS, EXPE, EBAY, CRNC, FFAI —
came from there. It finds `google` inside Alphabet's filing, which is exactly what a name lookup
cannot do.

Its cost is the reason it is not here yet: a network stage with a multi-GB cache, and a date problem —
at 2023-01-02 `chatgpt` had 0 filings and `openai` 11, all of them naming OpenAI in a director's
biography because Sam Altman sat on those boards. It is right and late, where names are early and
sometimes wrong.

## 4 · Known holes in the current mapping

- **survivorship** — `company_tickers.json` lists today's registrants, so a company delisted before the
  basket's date cannot enter it. Every backtest on this is flattered.
- **the entry date is not enforced** — `weights()` never receives it. Harmless while the only source is
  a snapshot of today, wrong the moment a dated source (filings, tags) is added.
- **product owners change** — the translation asks a model what a brand belongs to, and the model knows
  today's owner. A brand acquired after the theme's date would be attributed to the wrong company.
- **`ticker_list_US.csv`** — named by the README and `config.py`, absent from the repo. The SEC register
  substitutes for it.
