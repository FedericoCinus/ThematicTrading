# What the filing method buys, and what it needs to be worth using

## 1 · It amplifies a bad vocabulary word instead of diluting it

Measured on genAI at 2023-01-02, one-year window, forms 10-K/10-Q:

| word | filings | in the universe |
|---|---|---|
| `chatgpt` | 0 | 0 |
| `iphone` | 146 | 89 |
| `google` | 1,701 | too common to shortlist on |
| `openai` | 2 | 1 |
| `microsoft-backed chatgpt` | 0 | 0 |

Basket: `aapl · kdozf · zdge · bkng · ipm · ai · drio · trip · apps · logi` — Apple, Kidoz, Zedge,
Booking, Digital Turbine, Logitech. Mobile-app companies that write about iPhones.

The same word costs the two methods very differently:

```
uniform.py    iphone → apple → AAPL          one name out of three
filings/      iphone → 89 filings → 10 names the whole basket
```

Because ranking by how much a company *writes* a word hands the basket to whoever writes it most,
and app companies write `iPhone` constantly. A name lookup buys one company per word; a text search
buys everyone who talks.

**So this method needs a cleaner vocabulary than uniform does, not the same one.** `iphone` is in
genAI's vocabulary because headlines compared ChatGPT's growth to the iPhone's — a real
co-occurrence, and not the theme.

## 2 · The concepts the earlier work used are not the ones the detector produces

The pinned vocabulary of notebook 0.36 (`code/themes/genai.txt`) held `generative ai`,
`large language model`, `chatbot`, `foundation model`, `conversational ai` — terms that a filing
uses when it describes its business. The detector produces brands and bigrams: `chatgpt`,
`microsoft-backed chatgpt`, `openai`, `google`, `iphone`.

That gap is the real reason the notebook baskets look sane and this one does not. Two ways to close
it, neither tried:

- ask the model for the theme's **technical terms** as well as its entities — a second, small
  vocabulary aimed at this method rather than at the monitor;
- search the filings for the entities and read the *sentences around them*, instead of ranking on
  raw occurrence density.

## 3 · Smaller things

- **`since` defaults to one year before the theme's date.** Longer windows reach more filings but
  cross into a period where the theme did not exist; shorter ones miss the annual 10-K of any
  company that filed early in the year. Not tuned.
- **`max_docs = 200`** silently truncates a popular theme, and `sorted(found)` makes that
  truncation a cik ordering rather than a relevance one — the low CIKs are the old companies.
- **`max_hits_per_term = 1000`** decides which words shortlist. At 1,701 filings `google` is
  dropped; the threshold has never been varied.
- **Co-registrants**: one document can carry several CIKs, and every one of them is credited with
  the whole document's occurrences.
- **Form scope**: 10-K and 10-Q only. 20-F foreign private issuers are invisible, which removes
  most non-US names from a theme that may be mostly non-US.
