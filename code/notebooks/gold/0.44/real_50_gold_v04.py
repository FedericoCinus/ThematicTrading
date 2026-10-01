"""Gold for schema 0.4 on the 50 real headlines of 0.44, derived from the 0.3 gold.

    python notebooks/gold/0.44/real_50_gold_v04.py      # -> real_50_gold_v04.jsonl

Mechanical part: events are dropped; INDICATOR is removed from the accepted types (an entity that
was only INDICATOR is dropped); IMPACTS claims are dropped (their only cause, "Earnings", is no longer
a node); 'forecast' becomes 'uncertain'.
Manual part (EDITS below): the rules decided for 0.4 after reading the 0.3 results — ACTIVE_IN from
the declared purpose of a deal, OFFERS for "wins an order to build/supply X", both readings when two
are valid. These rules were written looking at model outputs on these same headlines, so E1 on 0.4
is a development measure, not a clean test.
"""
import json
from pathlib import Path

DIR = Path(__file__).resolve().parent
R, P, U = "reported", "planned", "uncertain"


def E(mention, *types, opt=False):
    return dict(mention=mention, types=list(types), opt=opt)


def C(subjects, predicate, objects, modalities, opt=False):
    return dict(subjects=list(subjects), predicate=predicate, objects=list(objects), modalities=list(modalities), opt=opt)


# per headline: entities to add, entity mentions to make optional, edges to add, edges to make required
EDITS = {
    "r00": dict(add=[C(["HAVELLS"], "ACTIVE_IN", ["KITCHEN APPLIANCES"], [R]),
                     C(["JUMBO GROUP"], "ACTIVE_IN", ["KITCHEN APPLIANCES"], [R]),
                     C(["HAVELLS"], "ACTIVE_IN", ["UAE"], [R], opt=True),
                     C(["JUMBO GROUP"], "ACTIVE_IN", ["UAE"], [R], opt=True)]),
    "r19": dict(entities=[E("ROOFTOP SOLAR", "CONCEPT", opt=True)],
                add=[C(["CAPRI GLOBAL"], "ACTIVE_IN", ["ROOFTOP SOLAR"], [R], opt=True)]),
    # two valid readings: one "Defense AI", or "Defense" + "AI"
    "r25": dict(entities=[E("AI", "CONCEPT", opt=True)], optional=["Defense AI"],
                replace=[C(["Helsing"], "ACTIVE_IN", ["Defense AI", "Defense", "AI"], [R]),
                         C(["Helsing"], "ACTIVE_IN", ["Defense", "AI"], [R], opt=True)]),
    "r31": dict(add=[C(["SMBC"], "ACTIVE_IN", ["Private Credit"], [R], opt=True)]),
    "r39": dict(add=[C(["Telkom Indonesia"], "ACTIVE_IN", ["AI"], [R]), C(["Huawei"], "ACTIVE_IN", ["AI"], [R])]),
    "r40": dict(add=[C(["Adani"], "ACTIVE_IN", ["Defense"], [R]), C(["EDGE"], "ACTIVE_IN", ["Defense"], [R]),
                     *[C(["Adani", "EDGE"], "ACTIVE_IN", ["Manufacturing", "R&D"], [R], opt=True)] * 4]),
    "r41": dict(add=[C(["VGP"], "ACTIVE_IN", ["Autonomous Vehicle"], [R, P]),
                     C(["Verne"], "ACTIVE_IN", ["Autonomous Vehicle"], [R, P])]),
    "r43": dict(add=[C(["Iveco"], "OFFERS", ["Trucks"], [R, P])]),
    "r44": dict(require=["OFFERS"]),
    "r45": dict(require=["OFFERS"]),
    "r46": dict(require=["OFFERS"]),
}


def main():
    old = [json.loads(l) for l in (DIR / "real_50_gold.jsonl").read_text().splitlines()]
    out = []
    for g in old:
        ed = EDITS.get(g["id"], {})
        ents = []
        for e in g["entities"]:
            types = [t for t in e["types"] if t != "INDICATOR"]
            if types:
                ents.append(dict(e, types=types, opt=e["opt"] or e["mention"] in ed.get("optional", [])))
        for e in ed.get("entities", []):
            ents.append(dict(e, id=f"e{len(g['entities']) + len(ents) + 1}"))
        by_m = {e["mention"]: e["id"] for e in ents}
        keep = set(by_m.values())
        edges = []
        if "replace" not in ed:
            for c in g["claims"]:
                if c["predicate"] == "IMPACTS":
                    continue
                subj = [x for x in c["subjects"] if x in keep]
                obj = [x for x in c["objects"] if x in keep]
                if not subj or not obj:
                    continue
                mods = sorted({"uncertain" if m == "forecast" else m for m in c["modalities"] if m}) or [R]
                req = c["predicate"] in ed.get("require", [])
                edges.append(dict(subjects=subj, predicate=c["predicate"], objects=obj, modalities=mods,
                                  opt=c["opt"] and not req))
        for c in ed.get("replace", []) + ed.get("add", []):
            edges.append(dict(c, subjects=[by_m[m] for m in c["subjects"]], objects=[by_m[m] for m in c["objects"]]))
        text = g["headline"]
        for e in ents:
            assert e["mention"] in text, (g["id"], e["mention"])
        out.append(dict(id=g["id"], headline=text, date=g["date"], stratum=g["stratum"], entities=ents, edges=edges,
                        note=g.get("note")))
    (DIR / "real_50_gold_v04.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    n = lambda k, o: sum(1 for r in out for x in r[k] if x["opt"] == o)
    preds = {}
    for r in out:
        for c in r["edges"]:
            if not c["opt"]:
                preds[c["predicate"]] = preds.get(c["predicate"], 0) + 1
    print(f"nodes required {n('entities', False)} (optional {n('entities', True)}); "
          f"edges required {n('edges', False)} (optional {n('edges', True)}); by predicate {preds}")


if __name__ == "__main__":
    main()
