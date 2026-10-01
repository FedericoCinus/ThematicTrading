"""GLiNER2.5 on the 0.44 headlines: entities + typed relations of schema 0.3 (no events).

    .venv-gliner/bin/python scripts/gliner_044.py                 # base-v1  -> gliner_044.jsonl
    .venv-gliner/bin/python scripts/gliner_044.py multi           # multi-v1 -> gliner_044_multi.jsonl
Separate env: gliner2 needs transformers 4.x, the project uses 5.x.

JointIE decodes one consistent typed graph per headline. It gets the 7 entity types and the 7
predicates with their allowed endpoint types and one-line definitions from guideline 044-v1; no
examples, default thresholds (no tuning on these headlines). Events, modality and drivers are not
produced: those scores are NA for GLiNER, not zero. IMPACTS cannot start from an event here.
Writes data/processed/gliner_044.jsonl in the same record format as the GPT outputs.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch
from gliner2.joint_ie import JointIE, JointIEConfig

CODE = Path(__file__).resolve().parent.parent
GOLD = CODE / "notebooks" / "gold" / "0.44"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
SCHEMA = sys.argv[2] if len(sys.argv) > 2 else "v03"          # v03 (entities/claims) or v04 (nodes/edges)
REPO = f"fastino/gliner2.5-{VARIANT}-v1"
OUT = CODE / "data" / "processed" / ("gliner_044" + ("_v04" if SCHEMA == "v04" else "")
                                     + ("" if VARIANT == "base" else f"_{VARIANT}") + ".jsonl")

ENTITIES = {
    "COMPANY": "A named firm, bank, broker, asset manager, investment fund or news agency.",
    "ORGANIZATION": "A named government, ministry, regulator, central bank, agency, court, university or international body.",
    "PRODUCT": "An identifiable good or service, including traded commodities, drugs, devices, software and AI models.",
    "CONCEPT": "A named technology, process, policy or economic idea.",
    "SECTOR": "A named industry or market.",
    "LOCATION": "A country, region or city named as a noun.",
    "INDICATOR": "A measurable economic variable such as revenue, profit, sales, demand, prices, capacity, output, rates or inflation.",
}
ACT = ["COMPANY", "ORGANIZATION"]
RELATIONS = [
    ("ACQUIRES", ["COMPANY"], ["COMPANY"], "buys, takes control of, bids or offers for a company", {}),
    ("INVESTS_IN", ACT, ["COMPANY", "PRODUCT", "CONCEPT", "SECTOR", "LOCATION"],
     "takes a minority stake in, or declares spending on", {}),
    ("PARTNERS_WITH", ["COMPANY"], ["COMPANY"], "agreement, alliance, joint venture or collaboration with",
     {"symmetric": True}),
    ("SUPPLIES_TO", ["COMPANY"], ACT, "supplier delivers goods or services to a customer", {}),
    ("OFFERS", ACT, ["PRODUCT", "CONCEPT"], "launches, introduces, produces or provides", {}),
    ("ACTIVE_IN", ["COMPANY"], ["SECTOR", "CONCEPT", "PRODUCT", "LOCATION"],
     "the text describes what the company does or is exposed to", {}),
    ("IMPACTS", ["CONCEPT", "PRODUCT", "ORGANIZATION", "INDICATOR"], ["COMPANY", "SECTOR", "PRODUCT", "INDICATOR"],
     "stated cause that lifts or hurts", {}),
]


if SCHEMA == "v04":                    # schema 0.4: no INDICATOR, EXPOSED_TO instead of IMPACTS
    ENTITIES = {k: v for k, v in ENTITIES.items() if k != "INDICATOR"}
    ENTITIES["CONCEPT"] = "A named technology, process, policy, named event or economic idea."
    RELATIONS = [r for r in RELATIONS if r[0] != "IMPACTS"] + [
        ("EXPOSED_TO", ["COMPANY", "SECTOR"], ["PRODUCT", "CONCEPT"], "moves, benefits or suffers because of", {})]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    joint = JointIE.from_pretrained(REPO, device=device)
    load_s = time.time() - t0
    schema = joint.create_schema().entities(ENTITIES)
    for name, head, tail, desc, kw in RELATIONS:
        schema = schema.relation(name, head, tail, desc, **kw)
    schema = schema.no_self_loops()
    cfg = JointIEConfig(optimizer="beam", beam_size=32)

    items = [("real", r["id"], r["headline"]) for r in map(json.loads, (GOLD / "real_50.jsonl").read_text().splitlines())]
    items += [("synthetic", p["id"], p["headline"]) for p in map(json.loads, (GOLD / "synthetic_50.jsonl").read_text().splitlines())]
    rows = []
    t1 = time.time()
    for kind, rid, text in items:
        t = time.time()
        res = joint.extract(text, schema, config=cfg).to_dict()
        ents = [dict(id=e["id"], type=e["type"], mention=e["text"], owner_id=None) for e in res["entities"]]
        claims = [dict(subject_id=r["head"], predicate=r["type"], object_id=r["tail"], event_id=None, product_id=None,
                       modality=None, impact_polarity=None, evidence=text, confidence=r.get("confidence"))
                  for r in res["relations"]]
        if SCHEMA == "v04":
            rows.append(dict(set=kind, id=rid, headline=text,
                             nodes=[dict(id=e["id"], type=e["type"], mention=e["mention"]) for e in ents],
                             edges=[dict(subject_id=c["subject_id"], predicate=c["predicate"], object_id=c["object_id"],
                                         modality=None, evidence=text, confidence=c["confidence"]) for c in claims],
                             feasible=res.get("feasible", True), seconds=round(time.time() - t, 3)))
        else:
            rows.append(dict(set=kind, id=rid, headline=text, entities=ents, events=[], claims=claims,
                             feasible=res.get("feasible", True), seconds=round(time.time() - t, 3)))
    run_s = time.time() - t1
    try:
        from huggingface_hub import model_info
        revision = model_info(REPO).sha
    except Exception:
        revision = None
    meta = dict(set="meta", repo=REPO, revision=revision, device=device, torch=torch.__version__,
                python=platform.python_version(), load_seconds=round(load_s, 1), run_seconds=round(run_s, 1),
                headlines=len(items), schema=SCHEMA, config=dict(optimizer="beam", beam_size=32, thresholds="default"))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [meta] + rows))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
