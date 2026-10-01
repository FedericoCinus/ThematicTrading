# The entity recogniser is not point-in-time

## 1 · The decision

Notebook `0.41-ner-relations.ipynb` extracts entities with spaCy `en_core_web_trf`, a model
trained on modern data. Everything else in that pipeline is point-in-time: the predicate comes
from a trigger lexicon over the headline's own words, and the only model call goes to a DatedGPT
vintage. The recogniser is the one modern component, and it is knowingly left that way for now.

## 2 · Why it is a real leak

A NER model recognises a name partly from having seen it. One trained in 2026 knows names that
became prominent after the date being analysed; a 2022-vintage model would not. If recall is
higher for companies that later mattered, the graph's node coverage tilts toward themes that
worked out — the exact direction that flatters a backtest of emerging themes.

The known instance: *"Amazon Invests $4 Billion in Anthropic"* is currently the opposite case —
`en_core_web_trf` **misses** Anthropic, so the edge is never drawn. That miss is accepted for now.

## 3 · Why DatedGPT is not the substitute

Measured on the same eighteen hand-labelled headlines:

| | Entity |
|---|---|
| spaCy `en_core_web_trf`, sentence-cased input | **F1 96 %** (P 100 %, R 93 %) |
| DatedGPT-2024, few-shot | 73 % |

Swapping in DatedGPT trades 23 points of accuracy for an unquantified bias. Worse, its errors are
not *dated ignorance* but ordinary mistakes — it answers `Bing` for a Microsoft headline and
`Hyundai` for an IonQ one. A weaker model does not confer the temporal property; only a dated one
does.

## 4 · What to build instead

`manelalab/chrono-bert-v1-<yyyymmdd>` is a ModernBERT encoder with annual cutoffs 1999–2024. It
loads for token classification directly (150 M parameters, context 8192, classifier head
uninitialised — confirmed) and is small enough to fine-tune on a laptop.

Fine-tune it on **CoNLL-2003**, which is Reuters newswire from 1996–97: the label data predates
any period this project analyses, so the supervision adds no future information. Encoder cutoff at
year *Y* plus 1997 labels gives a recogniser that is point-in-time for any analysis date after
1997.

Cost: `datasets` is not installed; one fine-tune per vintage you intend to use, or a single fixed
cutoff if one suffices.

## 5 · An attempt that did not work, recorded so it is not repeated

Quantifying the leak by measuring DatedGPT surprisal of company names across vintages **fails**.
Bare names in a generic carrier sentence are split into subwords, and the per-token NLL is
dominated by subword statistics rather than entity familiarity:

- *Alphabet* (2.06 nats) and *Microsoft* (2.03) top the 2022→2024 table, above *OpenAI* (1.66) and
  *Anthropic* (1.24) — the opposite of what the hypothesis predicts;
- *Anthropic* was founded in 2021, yet 2020 and 2024 score it identically (−2.13 vs −2.24).

Normalising against the rest of the sentence does not fix it. The 2022 checkpoint is not weaker in
general either — on neutral undated text it is the *best* of the three (perplexity 18.5, against
20.6 for 2024 and 20.9 for 2020) — so the anomaly is specific to proper nouns and remains
unexplained.

The measurement that would work is a direct recall comparison: run a dated recogniser and a modern
one over the same headlines and count the entities only the modern one finds. That needs §4 built
first.
