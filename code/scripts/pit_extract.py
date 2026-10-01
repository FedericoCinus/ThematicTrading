"""Point-in-time entity and relation extraction — the frozen 0.41 pipeline, with NER and LLM switchable.

    from pit_extract import relation, entities, chat
    relation("Broadcom to Acquire VMware in $61 Billion Deal", "2024-03-01", ner="chronobert", llm="datedgpt")

THE LEAKAGE RULE, ENFORCED BY THE API
    A headline dated D may only be read by models whose training data ends before D. Every
    public vintage used here ends on 31 December, so the vintage is always year(D) - 1. Callers
    pass the headline DATE, never a vintage: there is no argument through which a later model
    can be requested by mistake. (0.40 and 0.41 ran vintage 2024 on 2023 and 2024 headlines —
    that was leakage, and it is why the date is now the only input.)

WHAT IS FROZEN
    The gates, trigger lexicon, colon rule, negation list, generic-name list, word-order rule and
    the six predicate shots are copied verbatim from 0.41's final state. Nothing here was tuned on
    the 2024 test set of 0.42; the only choice left open for 0.42 is the NER casing, and that is
    made on its dev split.

NER BACKENDS
    chronobert   ChronoBERT-<vintage> fine-tuned on CoNLL-2003 (1996-97) by train_chronobert_ner.py.
                 Point-in-time for any year: encoder and labels both predate the headline.
    spacy-trf    en_core_web_trf 3.8 = OntoNotes 5 + roberta-base. Its newest training text is
                 roberta's CC-News, which ends in February 2019 — so it is clean only for headlines
                 dated after that, and refused for anything earlier.

LLM BACKENDS (local, dated) — with the verdict of the leakage audit in 0.42 §1
    datedgpt-base  datedgpt/datedgpt-<v>-base       1.3 B  CLEAN with caveats: FineWeb-Edu pages
                   crawled by 31 Dec <v>; the Llama tokenizer and the 2024 quality classifier that
                   picked the pages are later, but select text rather than add it
    pit-4b         Diamegs/PIT-4B-<snapshot>        4.2 B  CLEAN with caveats: FineWeb pages fetched
                   by the end of the snapshot month; trained continually, so one gradient step may
                   carry text from the next shard — see vintage_for
    datedgpt       datedgpt/datedgpt-<v>-instruct   1.3 B  CAVEAT: instruction data written by
                   DeepSeek-V4-Pro (2026) under a lexical constraint; recipe of the released
                   weights undocumented
    pit-4b-ft      Diamegs/PIT-4B-FT-<snapshot>     4.2 B  LEAKY: the same 2023-24 instruction mix
                   (GPT-4, GPT-4o, Llama-3.1-405B, Qwen2.5 outputs) is merged into every vintage
    The two base models are the leakage-free pair. The instruct models answer questions and stay
    available for chat, but relation() and llm_entities() refuse a leaky one unless told otherwise.
"""
from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

import torch

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# ────────────────────────────── the leakage rule ──────────────────────────────

def _as_date(d) -> _date:
    return d if isinstance(d, _date) else _date.fromisoformat(str(d)[:10])


PIT_MONTH = {2013: "201312", 2014: "201412", 2015: "201511", 2016: "201612", 2017: "201712",
             2018: "201812", 2019: "201912", 2020: "202012", 2021: "202112", 2022: "202212",
             2023: "202312", 2024: "202412"}

LLMS = {
    "datedgpt-base": dict(repo=lambda y: f"datedgpt/datedgpt-{y}-base", trust=False, years=range(2013, 2025),
                          chat=False, continual=False, status="clean"),
    "pit-4b":        dict(repo=lambda y: f"Diamegs/PIT-4B-{PIT_MONTH[y]}", trust=True, years=PIT_MONTH,
                          chat=False, continual=True, status="clean"),
    "datedgpt":      dict(repo=lambda y: f"datedgpt/datedgpt-{y}-instruct", trust=False, years=range(2013, 2025),
                          chat=True, continual=False, status="caveat"),
    "pit-4b-ft":     dict(repo=lambda y: f"Diamegs/PIT-4B-FT-{PIT_MONTH[y]}", trust=True, years=PIT_MONTH,
                          chat=True, continual=True, status="leaky"),
}
SPACY_CLEAN_FROM = _date(2019, 3, 1)      # roberta-base's CC-News ends Feb 2019


def vintage_for(d, family=None) -> int:
    """The newest vintage that cannot have seen the headline's date.

    Every vintage ends on 31 December, so the default is year(D) - 1. Families trained
    CONTINUALLY (PIT, the NER encoder is not) save a checkpoint when the loader moves to the next
    monthly shard, and the audit found that one gradient step can straddle that switch — up to
    ~17 M tokens of the next non-empty shard, which for a December checkpoint is late January or
    February. Those vintages are therefore not provably clean for January and February headlines
    of the following year, which fall back one more year.
    """
    d = _as_date(d)
    if family in LLMS and LLMS[family]["continual"] and d.month <= 2:
        return d.year - 2
    return d.year - 1

if torch.cuda.is_available():
    DEVICE, DTYPE = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE, DTYPE = "mps", torch.float16
else:
    DEVICE, DTYPE = "cpu", torch.float32

_MODELS, _TOKS, _NERS = {}, {}, {}
LOCAL_ONLY = False      # True: a vintage that is not on disk raises OSError instead of downloading


def _check_llm(llm, year):
    if llm not in LLMS:
        raise ValueError(f"unknown llm {llm!r}; choose from {list(LLMS)}")
    if year not in LLMS[llm]["years"]:
        raise ValueError(f"{llm} has no vintage {year}")


def tokenizer(llm, year):
    from transformers import AutoTokenizer
    _check_llm(llm, year)
    key = (llm, year)
    if key not in _TOKS:
        # DatedGPT's 24 checkpoints ship a byte-identical tokenizer (same SentencePiece hash), but
        # not every base repo includes tokenizer.json — so the family always loads one that does.
        repo = "datedgpt/datedgpt-2024-instruct" if llm.startswith("datedgpt") else LLMS[llm]["repo"](year)
        _TOKS[key] = AutoTokenizer.from_pretrained(repo, trust_remote_code=LLMS[llm]["trust"],
                                                   local_files_only=LOCAL_ONLY)
    return _TOKS[key]


def model(llm, year):
    from transformers import AutoModelForCausalLM
    _check_llm(llm, year)
    key = (llm, year)
    if key not in _MODELS:
        m = AutoModelForCausalLM.from_pretrained(LLMS[llm]["repo"](year), dtype=DTYPE,
                                                 trust_remote_code=LLMS[llm]["trust"], local_files_only=LOCAL_ONLY)
        _MODELS[key] = m.to(DEVICE).eval()
    return _MODELS[key]


def release():
    """Drop every resident model (LLMs and NERs). Weights stay on disk."""
    _MODELS.clear(); _NERS.clear(); gc.collect()
    if DEVICE != "cpu":
        getattr(torch, DEVICE).empty_cache()


@torch.no_grad()
def choose(prefix, options, llm, year):
    """Rank a closed list by mean token log-probability. Never generates."""
    tok, m = tokenizer(llm, year), model(llm, year)
    base, out = tok(prefix, add_special_tokens=False)["input_ids"], {}
    for o in options:
        ids = base + tok(" " + o, add_special_tokens=False)["input_ids"]
        lp = torch.log_softmax(m(input_ids=torch.tensor([ids]).to(DEVICE)).logits.float(), dim=-1)
        tail = ids[len(base):]
        out[o] = sum(lp[0, len(base) + j - 1, k].item() for j, k in enumerate(tail)) / len(tail)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _refuse_leaky(llm, allow_leaky):
    if LLMS[llm]["status"] == "leaky" and not allow_leaky:
        raise ValueError(f"{llm} is not point-in-time (see the leakage audit); pass allow_leaky=True "
                         f"only to study the leak itself")


@torch.no_grad()
def chat(question, on_date, llm="datedgpt", max_new_tokens=140):
    """Ask the newest vintage that could not know `on_date` a free question.

    Returns the answer; prints a warning when the model is not point-in-time for that date.
    """
    year = vintage_for(on_date, llm)
    if LLMS[llm]["status"] == "leaky":
        print(f"⚠ {llm} is LEAKY: its instruction data comes from 2023-24 models, so it may know "
              f"events after {on_date}. Use it to explore, not to measure.\n")
    elif LLMS[llm]["status"] == "caveat":
        print(f"note: {llm}'s instruction data was generated by a 2026 model under a lexical "
              f"constraint — point-in-time with a caveat.\n")
    tok = tokenizer(llm, year)
    if llm == "datedgpt":
        text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False)
        special = False                     # the template already emits <s>
    elif llm == "pit-4b-ft":
        text, special = f"<|user|>\n{question}\n<|assistant|>\n", True
    else:
        text, special = question, True      # pretrained only: it continues, it does not answer
    enc = tok(text, return_tensors="pt", add_special_tokens=special).to(DEVICE)
    out = model(llm, year).generate(**enc, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False,
                                    repetition_penalty=1.1, eos_token_id=tok.eos_token_id,
                                    pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True)
    return answer.split("<|end|>")[0].strip()


# ────────────────────────────── NER backends ──────────────────────────────

@dataclass
class Span:
    text: str
    label_: str
    start_char: int
    end_char: int


_WORD = re.compile(r"[’']s\b|\w+(?:[.&\-]\w+)*|[^\w\s]")


class ChronoNER:
    """ChronoBERT-<year> + CoNLL-2003 head, exposing spaCy's `.ents` interface."""

    def __init__(self, year, aug=True):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        path = MODELS_DIR / (f"chronobert-ner-aug-{year}" if aug else f"chronobert-ner-{year}")
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run scripts/train_chronobert_ner.py {year}"
                                    f"{' --aug' if aug else ''}")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(DEVICE).eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def __call__(self, text):
        words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]
        if not words:
            return type("Doc", (), {"ents": []})()
        enc = self.tok([w for w, _, _ in words], is_split_into_words=True, truncation=True,
                       max_length=128, return_tensors="pt").to(DEVICE)
        pred = self.model(**enc).logits.argmax(-1)[0].tolist()
        tags, prev = ["O"] * len(words), None
        for j, w in enumerate(enc.word_ids(0)):
            if w is not None and w != prev:
                tags[w] = self.id2label[pred[j]]
            prev = w
        ents, cur = [], None
        for k, t in enumerate(tags + ["O"]):
            opens = t.startswith("B-") or (t.startswith("I-") and (cur is None or t[2:] != cur[0]))
            if opens or t == "O":
                if cur:
                    s, e = words[cur[1]][1], words[k - 1][2]
                    lab = {"PER": "PERSON"}.get(cur[0], cur[0])
                    ents.append(Span(text[s:e], lab, s, e))
                cur = (t[2:], k) if opens else None
        return type("Doc", (), {"ents": ents})()


def ner(name, on_date):
    """The recogniser for `name`, at the vintage `on_date` allows — refused if it could leak."""
    d = _as_date(on_date)
    if name == "spacy-trf":
        if d < SPACY_CLEAN_FROM:
            raise ValueError(f"spacy-trf has seen text up to Feb 2019 and cannot read a headline of {d}")
        key = ("spacy-trf",)
        if key not in _NERS:
            import spacy
            _NERS[key] = spacy.load("en_core_web_trf")
        return _NERS[key]
    if name in ("chronobert", "chronobert-plain"):
        aug = name == "chronobert"          # casing-augmented is the default; -plain is the ablation
        key = (name, vintage_for(d))
        if key not in _NERS:
            _NERS[key] = ChronoNER(vintage_for(d), aug=aug)
        return _NERS[key]
    raise ValueError(f"unknown ner {name!r}")


KEEP = {"ORG", "PERSON"}


def _clean(t):
    return t.replace("’s", "").replace("'s", "").strip(" .,")


def entities(headline, on_date, ner_name="chronobert", casing="sentence"):
    """Company and person names in reading order. `casing`: 'sentence' (0.41's choice) or 'asis'."""
    rec = ner(ner_name, on_date)
    text = headline[0] + headline[1:].lower() if casing == "sentence" else headline
    seen, out = set(), []
    for e in rec(text).ents:
        if e.label_ not in KEEP:
            continue
        t = _clean(headline[e.start_char:e.end_char])      # same length: offsets carry over
        if t.lower() not in seen and len(t) > 1:
            seen.add(t.lower()); out.append(t)
    return out


def entities_for_edge(headline, on_date, ner_name="chronobert", casing="sentence"):
    """`entities`, topped up from the as-written reading when fewer than two were found (0.41)."""
    found = entities(headline, on_date, ner_name, casing)
    if len(found) >= 2 or casing == "asis":
        return found
    seen = {x.lower() for x in found}
    extra = [_clean(e.text) for e in ner(ner_name, on_date)(headline).ents if e.label_ in KEEP]
    return found + [x for x in extra if x.lower() not in seen and len(x) > 1 and not seen.add(x.lower())]


# ────────────────────────────── the frozen 0.41 relation pipeline ──────────────────────────────

PREDICATES = {
    "acquires":      ["acquire", "acquires", "acquisition", "buy", "buys", "bought",
                      "purchase", "purchases", "takeover", "to buy", "takeover of",
                      "purchase of", "approach for", "bought from"],
    "sells":         ["sell", "sells", "sold", "offload", "offloads", "divest", "divests",
                      "stake sale", "sheds", "stake falls", "exit", "disposes"],
    "invests in":    ["invest", "invests", "investment", "stake in", "funding", "injects",
                      "stake rises", "takes stake", "raises stake", "converts"],
    "supplies":      ["supply", "supplies", "supplier", "ship", "ships", "deliver", "delivers"],
    "partners with": ["deal", "deals", "sign", "signs", "signed", "partner", "partners",
                      "partnership", "agreement", "teams", "tie-up", "joint venture"],
    "sues":          ["sue", "sues", "sued", "lawsuit", "accuses", "litigation"],
}
MARKET_NOISE = re.compile(
    r"rated new |raised to |cut to |ups to |upgrade|downgrade|reiterat|resumed |initiated |"
    r"maintain|\bpt \$|price target|buy recommendation|\bbuy back\b|buyback|"
    r"tender:|seeks to buy|on \d+ of \d+ proposals|\bproposals\b|\bagm\b|\begm\b|"
    r"\brating\b|outperform|underperform|overweight|underweight|\bneutral\b", re.I)
SOURCE_TAG = re.compile(
    r"^(rtrs|reuters|faz|wsj|ft|bbg|bloomberg|times|toi|nikkei|handelsblatt|cnbc|dj|press|"
    r"update|m&a snapshot|tech watch|supply chain|new deal)\b", re.I)
NEGATION = re.compile(
    r"\b(withdraw\w*|abandon\w*|scrap\w*|call\w* off|rule\w* out|end\w* talks|terminat\w*|"
    r"drop\w* bid|walk\w* away|denie\w*|no longer|fail\w* to)\b", re.I)
GENERIC = {"congress", "treasuries", "treasury", "fed", "federal reserve", "supply chain",
           "senate", "parliament", "government", "eu", "imf", "un", "white house", "sec",
           "ftc", "doj", "court", "reuters", "bloomberg", "wsj", "new deal", "holders",
           "shareholders", "board", "forex market", "street", "class a"}
PRED_SHOTS = ("Headline: Broadcom to Acquire VMware in $61 Billion Deal\n"
              "Subject: Broadcom\nObject: VMware\nPredicate: acquires\n\n"
              "Headline: TSMC to Supply Chips to Tesla Under New Agreement\n"
              "Subject: TSMC\nObject: Tesla\nPredicate: supplies\n\n"
              "Headline: Oracle Sues Google Over Use of Java in Android\n"
              "Subject: Oracle\nObject: Google\nPredicate: sues\n\n"
              "Headline: Ford and SK On Sign Battery Plant Joint Venture\n"
              "Subject: Ford\nObject: SK On\nPredicate: partners with\n\n"
              "Headline: Temasek Takes 5% Stake in Adyen\n"
              "Subject: Temasek\nObject: Adyen\nPredicate: invests in\n\n"
              "Headline: Vanguard Cuts Holding in Rivian to 3%\n"
              "Subject: Vanguard\nObject: Rivian\nPredicate: sells\n\n")


def split_tail(headline):
    if ":" not in headline:
        return headline, None
    body, tail = headline.rsplit(":", 1)
    tail = tail.strip()
    if not tail or len(tail.split()) > 4:
        return headline, None
    return body.strip(), tail


def triggered(headline):
    low = f" {re.sub(r'[^a-z0-9 ]', ' ', headline.lower())} "
    return [p for p, words in PREDICATES.items() if any(f" {w} " in low for w in words)]


def _edge(headline, why, ents=(), trig=(), s=None, p=None, o=None):
    return dict(headline=headline, entities=list(ents), triggers=list(trig),
                subject=s, predicate=p, object=o, why=why)


def relation(headline, on_date, ner_name="chronobert", llm="datedgpt-base", casing="sentence", chooser=None,
             allow_leaky=False):
    """(subject, predicate, object) or no edge, reading `headline` as of `on_date`.

    `chooser`, if given, replaces the local LLM for the one decision a model makes — the
    predicate among the triggered candidates. It receives (prompt, options) and returns one
    option; it is how a remote model (GPT) is dropped into the same slot for comparison.
    """
    if chooser is None:
        _refuse_leaky(llm, allow_leaky)
    if MARKET_NOISE.search(headline):
        return _edge(headline, "market noise")
    if NEGATION.search(headline):
        return _edge(headline, "negated")
    body, tail = split_tail(headline)
    trig = triggered(body)
    if not trig:
        return _edge(headline, "no trigger")
    ents = entities_for_edge(body, on_date, ner_name, casing)
    if tail and not SOURCE_TAG.match(tail):
        actor = entities_for_edge(tail + " acted", on_date, ner_name, casing)
        if actor:
            ents = [actor[0]] + [e for e in ents if e.lower() != actor[0].lower()]
    ents = [e for e in ents if e.lower().strip(" .,") not in GENERIC]
    if len(ents) < 2:
        return _edge(headline, "no second entity", ents, trig)
    subj, obj = ents[0], ents[1]
    if subj.lower() in obj.lower() or obj.lower() in subj.lower():
        return _edge(headline, "endpoints not distinct", ents, trig)
    prompt = f"{PRED_SHOTS}Headline: {body}\nSubject: {subj}\nObject: {obj}\nPredicate:"
    if len(trig) == 1:
        pred = trig[0]                      # a single candidate: nothing for any model to decide
    elif chooser is not None:
        pred = chooser(prompt, trig)
    else:
        pred = next(iter(choose(prompt, trig, llm, vintage_for(on_date, llm))))
    return _edge(headline, "", ents, trig, subj, pred, obj)


# ────────────────────────────── LLM-only entity extraction (for comparison) ──────────────────────────────

ENT_SHOTS = ("Headline: Broadcom to Acquire VMware in $61 Billion Deal\nEntities: Broadcom; VMware\n\n"
             "Headline: Pfizer Halts Trial of Obesity Pill After Patients Report Side Effects\nEntities: Pfizer\n\n"
             "Headline: Treasury Yields Climb Before Inflation Print\nEntities: none\n\n")


@torch.no_grad()
def llm_entities(headline, on_date, llm="datedgpt-base", allow_leaky=False):
    """Entities written by the dated LLM itself, few-shot — the approach 0.40 measured at 73 %."""
    _refuse_leaky(llm, allow_leaky)
    year = vintage_for(on_date, llm)
    tok, m = tokenizer(llm, year), model(llm, year)
    enc = tok(ENT_SHOTS + f"Headline: {headline}\nEntities:", return_tensors="pt").to(DEVICE)
    out = m.generate(**enc, max_new_tokens=32, do_sample=False, use_cache=True,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    line = tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).split("\n")[0]
    names = [x.strip(" .") for x in line.split(";")]
    return [n for n in names if n and n.lower() != "none"]


# ────────────────────────────── rule-free extraction: the model chooses the whole triple ──────────────────────────────

RULE_FREE_SHOTS = (
    "Headline: Broadcom to Acquire VMware in $61 Billion Deal\nRelation: acquires\nSubject: Broadcom\nObject: VMware\n\n"
    "Headline: Apple Raised to Buy at Citi; PT $210\nRelation: none\n\n"
    "Headline: Oracle Sues Google Over Use of Java in Android\nRelation: sues\nSubject: Oracle\nObject: Google\n\n"
    "Headline: Indonesia Sells IDR2 Trillion of Bonds; Yield 6%\nRelation: none\n\n"
    "Headline: TSMC to Supply Chips to Tesla Under New Agreement\nRelation: supplies\nSubject: TSMC\nObject: Tesla\n\n"
    "Headline: Morgan Stanley Partners Hire New Head of Research\nRelation: none\n\n"
    "Headline: Temasek Takes 5% Stake in Adyen\nRelation: invests in\nSubject: Temasek\nObject: Adyen\n\n"
    "Headline: Pfizer Shares Fall After Earnings Miss Estimates\nRelation: none\n\n"
    "Headline: Vanguard Cuts Holding in Rivian to 3%\nRelation: sells\nSubject: Vanguard\nObject: Rivian\n\n"
    "Headline: Ford and SK On Sign Battery Plant Joint Venture\nRelation: partners with\nSubject: Ford\nObject: SK On\n\n"
)
RELATIONS = ["none", "acquires", "sells", "invests in", "supplies", "partners with", "sues"]


def relation_free(headline, on_date, llm="datedgpt-base", ner_name="spacy-trf", casing="asis", allow_leaky=False,
                  year=None):
    """No hand-written rules: NER gives the names, the dated model picks the predicate and then the pair.

    `year` overrides the vintage only for a model already marked leaky (a reference, not a measurement).
    """
    _refuse_leaky(llm, allow_leaky)
    if year is not None and LLMS[llm]["status"] != "leaky":
        raise ValueError("a vintage can only be forced on a model already marked leaky")
    year = year or vintage_for(on_date, llm)
    ents = entities(headline, on_date, ner_name, casing)
    stem = f"{RULE_FREE_SHOTS}Headline: {headline}\nRelation:"
    pred = next(iter(choose(stem, RELATIONS, llm, year)))
    if pred == "none" or len(ents) < 2:
        return dict(headline=headline, entities=ents, subject=None, predicate=None, object=None,
                    why="model: none" if pred == "none" else "fewer than two names")
    pairs = [(a, b) for a in ents for b in ents if a != b]
    options = [f"{a}\nObject: {b}" for a, b in pairs]
    best = next(iter(choose(f"{stem} {pred}\nSubject:", options, llm, year)))
    s, o = pairs[options.index(best)]
    return dict(headline=headline, entities=ents, subject=s, predicate=pred, object=o, why="")
