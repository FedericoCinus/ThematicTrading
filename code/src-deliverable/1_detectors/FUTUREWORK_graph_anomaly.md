# What this detector can and cannot see

## 0 · It finds themes that arrive with a new word

This is the shape of the method, not a defect, and it decides which themes are reachable at all.

A term is a candidate only if it appears **nowhere** in the baseline. So the detector sees a theme
when the theme brings a name nobody had used before. That is why genAI was found — `chatgpt` was
genuinely new — and why quantum computing never was: `quantum`, `ionq` and `rigetti` were all in the
corpus years earlier, so no as-of date can make them novel. Measured on this corpus:

| term | first appears | reachable? |
|---|---|---|
| `chatgpt` | 2022-12-07 | yes |
| `generative ai` | 2023-02-01 | yes |
| `rigetti` | 2020-08-04 | only in 2020-22 |
| `openai` | 2013-02-20 | **never** |
| `ionq` | 2010-02-26 | **never** |

`openai` is the sharpest case: the token has existed since 2013, so the company's rise is invisible to
a novelty test however the window is placed.

**The window is geometry, not tuning.** A term born on date B is novel while `asof - detect_months < B`,
and inside the detection window while `asof >= B`. So it can be found for exactly `detect_months`
after its birth, and never again:

```
detect_months = 5, ChatGPT born 2022-11-30

  asof 2022-12-19   baseline ends 2022-07-20   candidate, not yet persistent
  asof 2023-01-16   baseline ends 2022-08-17   PROMOTED, week of 2023-01-02
  asof 2023-02-13   baseline ends 2022-09-14   PROMOTED, week of 2023-01-02
  asof 2023-03-13   baseline ends 2022-10-12   PROMOTED, week of 2023-01-02
  asof 2023-04-17   baseline ends 2022-11-16   PROMOTED, week of 2023-01-02
  asof 2023-05-15   baseline ends 2022-12-14   gone: the baseline swallowed its birth
```

Two things follow. **Running weekly removes the need to guess the window** — you sweep it, and the
emergence week comes out the same from every run that catches it, so the answer does not depend on
when you looked. And **the same theme is therefore found many times**, once per run in that band, so
anything downstream must deduplicate on `(anchor, week)`.

**Accepted for now, to be worked on.** Novelty is a heuristic for "something is starting", and it
fails for any theme whose words have been around too long — which is a large and interesting class,
not an edge case. Quantum computing is the example we care about: the words existed for years and
what changed was how often they were said. This detector will not find those, and we are running it
anyway, because the themes it *does* find are found early and cheaply.

Two directions when we come back to it, in increasing order of ambition:

1. **Rarity instead of absence.** Not "never said" but "said at most N times". `chatgpt` was said 4
   times before the cut and 312 in the months after — a 78:1 signal thrown away by a binary gate. One
   parameter, `max_baseline_mentions`, with 0 reproducing today's behaviour exactly. Cheap, and it
   rescues terms killed by a single coincidental old mention. It does **not** rescue `quantum`, which
   was said constantly.
2. **Acceleration instead of novelty.** For a theme made of old words, the signal is not *whether* the
   words appear but *how much more often*. That is what the monitor already computes with a rolling
   z-score — so the honest fix may be a second detector that finds themes by acceleration of known
   vocabulary, sitting beside this one rather than replacing it. The contract already allows it: both
   would fill the same `themes [theme, week, vocab]` frame.

---

# Known biases

Two things this detector gets wrong by construction. Neither blocks using it, both change how its
output should be read, and both were measured rather than suspected. Written down here so a result
is never explained by a bias we already knew about.

---

## 1 · The gates are absolute, but the corpus is not

`min_mentions` and `degree_min` are counts. The corpus supplies very different numbers of headlines
depending on the year:

| year | stories in the corpus |
|---|---|
| 2010 | ~2.28 M |
| 2017 | 1.23 M |
| 2019 | 1.00 M |

A thick year has more headlines per week, so more co-occurrences, so a higher `degree` for the *same
phenomenon*. A theme emerging in 2010 clears `degree_min = 8` far more easily than one emerging in
2025, and a term said three times in a week is a rarer event in a thin year than in a thick one.

**What it does to results.** The detector finds more themes in early years than in late ones, and the
difference is an artifact of how much news the capture carries, not of how much was happening. Any
comparison of theme counts across eras is invalid as it stands. A single theme's own trajectory is
unaffected, since it lives inside one window.

**What would fix it.** Make the gates relative instead of absolute: express `min_mentions` as a
quantile of that week's term-frequency distribution, and `degree_min` as a quantile of that week's
degree distribution. Then a gate means "unusually well connected *this week*", which is what it was
always trying to mean. The cost is one extra pass per week and a parameter that is harder to reason
about; the benefit is that runs on different periods become comparable.

**Until then.** Read theme counts within a period, never across. When sweeping the gates, sweep them
on one period at a time.

---

## 2 · Novelty favours bigrams, and we made that worse

`extract_terms` produces unigrams and bigrams. Measured on a 24-month baseline against the following
5-month window:

```
baseline vocabulary        932,464 terms
terms in the window        265,043
        of which novel     120,929   (45.6%)
        of those, bigrams       87%
```

A new *pair* of words is combinatorially far easier than a new word, so the candidate pool is
dominated by bigrams. That is not wrong in itself — themes are usually named by bigrams ("quantum
computing", "generative ai") — but it means most candidates are accidental pairings, and the gates
downstream are doing all the work of separating them.

**The part we caused.** The corpus deliberately keeps the headline as the wire published it, prefix
included, because stripping it deleted real information: `Daniel Yergin: Surveillance With Prewitt
and Keene` loses its speaker when stripped. The cost is that every person's name is a bigram, and the
first time anyone appears they are a novel candidate.

Worse, a recurring guest has exactly the profile this detector rewards. They appear alongside many
different entities that do not appear alongside each other — **high degree, low clustering** — which
is precisely the signature of a bridging theme. We should expect a class of false positives the
earlier analyses never saw, because they ran on stripped headlines.

**What to do about it.** Look at `rejects` first, and look for proper names. If the problem shows up,
fix it where it lives: a gate on the anchor being a person, or a list of programme hosts. Do **not**
reinstate prefix stripping — that trades a small, targeted problem for the larger one of throwing
away every speaker in the corpus.

**Also relevant.** The corpus carries source attributions inside the headline text (`wsj` appears
23,353 times, plus `reuters`, `cnbc`, `ft`, `nikkei`). These are never novel, so they cannot become
anchors, but they co-occur with everything and would enter every theme's vocabulary as partners.
`max_doc_freq` exists to remove them — if it is ever raised, check the vocabularies.

---

## 3 · A theme's vocabulary is thin at birth, and its ordering is noise

`vocabulary()` gathers every term the anchor has shared a headline with in all history up to the
as-of date, then has the model keep only the words that name a company, institution, product or
technology. That replaces what the notebooks did by hand, and it works: ChatGPT at 2023-01-16 yields
twelve words, stable across six runs bar one flicker (`nyc schools`, 2 of 6).

**But the theme is found early, so there is almost nothing to gather.** `chatgpt` appears in
**eighteen headlines** in the entire corpus at that date. The co-occurrence counts that order the
vocabulary are therefore 1, 2, at most 5 — and at those numbers the ranking past the first two words
is decided by ties:

```
chatgpt · microsoft · openai   the real ones, always first
google · alphabet · bing       correct, and ranked below
tata · oxbotica · fanatics · celsius · genesis · iphone · nyc schools
```

Those last ones are real companies that shared a single headline about something else. They are not a
filter failure — the model is asked what a word denotes, not whether it relates to the theme, and
that question is deliberately not asked because answering it needs to know how the story ended.

**Consequences.** `theme_bag_of_words_size` is a real knob only once there is enough co-occurrence to
rank; at emergence it cuts almost arbitrarily. And the vocabulary a theme is born with is *not* the
vocabulary it deserves: the notebooks' rich genAI list (alibaba, baidu, ernie, nvidia) was compiled in
May from months of hindsight, and none of those words exist beside `chatgpt` in January.

**What would fix it.** Let the monitor re-open the vocabulary each week using only data up to that
week. The theme then grows while it is watched — Baidu arrives in February, Alibaba in March — and
never sees the future. That is a change to stage 3's contract, not a bug fix, and it is the reason
the monitor takes `corpus` as well as `themes`.

## 4 · The gate needs a model big enough not to be undecided

`temperature: 0` fixes which token wins, not the arithmetic that ranks them. Logits are floating-point
sums whose reduction order varies between requests with how the batch is composed, and in
mixture-of-experts models routing varies with concurrent traffic. Those last-bit differences are
invisible — until two tokens are nearly tied, when they decide the answer.

That is what a model too small for the question looks like from outside: on ChatGPT's cluster
`gpt-4o-mini` keeps it 6 times in 20, while every clear-cut cluster never wobbles. `gpt-4o` keeps it
8 in 8. The instability was not the system being random; it was the model having no opinion, and the
noise casting the deciding vote.

**Consequences for anyone changing this.** Judging a prompt edit on one call, or even on six, measures
nothing — a 30% process looks perfect often enough to fool you, and it fooled us twice. A dozen calls
minimum. And if the small model is ever restored to save money, expect the acceptance test to fail
intermittently for reasons that have nothing to do with the code being edited at the time.
