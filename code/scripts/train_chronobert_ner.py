"""Fine-tune ChronoBERT-<year> on CoNLL-2003 into a point-in-time named-entity recogniser.

    python train_chronobert_ner.py 2023          # plain CoNLL
    python train_chronobert_ner.py 2023 --aug    # CoNLL with casing augmentation

Both halves are dated, which is the point of the exercise:

    encoder   manelalab/chrono-bert-v1-<year>1231   pretrained on text up to 31 Dec <year>
    labels    CoNLL-2003 English                      Reuters newswire, Aug 1996 – Aug 1997

so the resulting model has seen nothing written after 31 Dec <year>, and is leakage-free for
any headline dated in <year>+1 or later. The off-the-shelf alternative, spaCy en_core_web_trf,
is roberta-base underneath — clean only for headlines after early 2019, and useless for a
backtest of 2015.

CASING AUGMENTATION (--aug)
    CoNLL is newswire in ordinary casing; Bloomberg headlines are Title Case, and a recogniser
    trained on the first reads every capitalised verb in the second as part of a name ("BAT
    Mulls", "Office AI"). With --aug each training sentence is, at random, left as is (50 %),
    Title-Cased the way Bloomberg does it (35 %), or lowercased (15 %). The sentences are still
    1996-97 Reuters text, so the supervision stays leakage-free; only its typography changes.

Writes data/models/chronobert-ner[-aug]-<year>/ (model, tokenizer, metrics.json). Takes a few minutes
per vintage on Apple silicon.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

EPOCHS, BATCH, LR, MAX_LEN, SEED = 3, 32, 5e-5, 128, 0
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "models"


def conll(split):
    path = hf_hub_download("tner/conll2003", f"dataset/{split}.json", repo_type="dataset")
    return [json.loads(line) for line in open(path)]


SMALL = {"a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at", "by", "as", "with", "from"}


def title_case(tokens):
    """Bloomberg-style: every word capitalised except short function words after the first."""
    out = []
    for i, t in enumerate(tokens):
        if not t[:1].isalpha() or (i and t.lower() in SMALL):
            out.append(t.lower() if i and t.lower() in SMALL else t)
        else:
            out.append(t[:1].upper() + t[1:])
    return out


def augment(row, rng):
    r = rng.random()
    if r < 0.50:
        return row
    tokens = title_case(row["tokens"]) if r < 0.85 else [t.lower() for t in row["tokens"]]
    return {"tokens": tokens, "tags": row["tags"]}


def encode(tok, rows):
    """Word-level tags -> first-subtoken labels; continuation subtokens are ignored (-100)."""
    enc = tok([r["tokens"] for r in rows], is_split_into_words=True, truncation=True,
              max_length=MAX_LEN, padding=True, return_tensors="pt")
    labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    for i, r in enumerate(rows):
        prev = None
        for j, w in enumerate(enc.word_ids(i)):
            if w is not None and w != prev:
                labels[i, j] = r["tags"][w]
            prev = w
    enc["labels"] = labels
    return enc


def spans(tags, id2label):
    """BIO word tags -> {(type, start, end)} exact spans. A stray I- opens a span, as conlleval does."""
    out, cur = set(), None
    for k, t in enumerate(list(tags) + [0]):
        name = id2label[int(t)] if k < len(tags) else "O"
        opens = name.startswith("B-") or (name.startswith("I-") and (cur is None or name[2:] != cur[0]))
        if opens or name == "O":
            if cur:
                out.add((cur[0], cur[1], k))
            cur = (name[2:], k) if opens else None
    return out


@torch.no_grad()
def evaluate(model, tok, rows, id2label, device):
    model.eval()
    tp = fp = fn = 0
    for s in range(0, len(rows), 64):
        chunk = rows[s:s + 64]
        enc = tok([r["tokens"] for r in chunk], is_split_into_words=True, truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        pred = model(**{k: v for k, v in enc.items()}).logits.argmax(-1).cpu()
        for i, r in enumerate(chunk):
            words = [0] * len(r["tokens"])
            prev = None
            for j, w in enumerate(enc.word_ids(i)):
                if w is not None and w != prev:
                    words[w] = int(pred[i, j])
                prev = w
            g, p = spans(r["tags"], id2label), spans(words, id2label)
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9))


def main(year: int, aug: bool = False):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(SEED + 1)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    repo = f"manelalab/chrono-bert-v1-{year}1231"
    label2id = json.load(open(hf_hub_download("tner/conll2003", "dataset/label.json", repo_type="dataset")))
    id2label = {v: k for k, v in label2id.items()}

    train, valid, test = conll("train"), conll("valid"), conll("test")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForTokenClassification.from_pretrained(
        repo, num_labels=len(label2id), id2label=id2label, label2id=label2id).to(device)

    steps = EPOCHS * math.ceil(len(train) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    t0, step = time.time(), 0
    for epoch in range(EPOCHS):
        model.train()
        order = list(range(len(train))); random.shuffle(order)
        for s in range(0, len(order), BATCH):
            rows = [train[i] for i in order[s:s + BATCH]]
            if aug:
                rows = [augment(r, rng) for r in rows]
            batch = encode(tok, rows).to(device)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        dev = evaluate(model, tok, valid, id2label, device)
        print(f"epoch {epoch + 1}: CoNLL dev F1 {dev['f1']:.3f}", flush=True)

    metrics = dict(encoder=repo, labels="CoNLL-2003 (Reuters, 1996-08 .. 1997-08)", casing_augmentation=aug,
                   dev=evaluate(model, tok, valid, id2label, device),
                   test=evaluate(model, tok, test, id2label, device),
                   epochs=EPOCHS, batch=BATCH, lr=LR, seconds=round(time.time() - t0))
    out = OUT_ROOT / (f"chronobert-ner-aug-{year}" if aug else f"chronobert-ner-{year}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), aug="--aug" in sys.argv)
