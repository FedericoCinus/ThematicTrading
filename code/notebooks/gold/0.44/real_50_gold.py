"""Gold for the 50 real headlines of 0.44 (guideline 044-v1), written before any model output.

    python notebooks/gold/0.44/real_50_gold.py      # -> real_50_gold.jsonl

One annotator (Claude), to be reviewed by Federico. Where the guideline admits two readings the
gold says so instead of picking one: `types`, `drivers`, `modalities` list every accepted value,
and `opt` marks entities, events, participants or claims that are neither required nor wrong.

Compact notation:
    E(mention, *types, opt=False)                   entity; mention is verbatim
    V(trigger_words, drivers, modalities, parts, opt_parts=(), opt=False)
        trigger_words: any shared word with the model's trigger aligns the event
        parts: mentions (or event ids) that must be ACTOR or TARGET; role label not scored
    C(subjects, predicate, objects, modalities, polarities=(None,), opt=False)
        subjects/objects: accepted ends (mentions or event ids "v1", "v2")
"""
import json
from pathlib import Path

N = None
DIR = Path(__file__).resolve().parent


def E(mention, *types, opt=False):
    return dict(mention=mention, types=list(types), opt=opt)


def V(trigger, drivers, modalities, parts, opt_parts=(), opt=False):
    return dict(trigger=trigger, drivers=list(drivers), modalities=list(modalities),
                parts=list(parts), opt_parts=list(opt_parts), opt=opt)


def C(subjects, predicate, objects, modalities, polarities=(None,), opt=False):
    return dict(subjects=list(subjects), predicate=predicate, objects=list(objects),
                modalities=list(modalities), polarities=list(polarities), opt=opt)


SA, TI, PR, MA, RE, DE, SU = ("STRATEGIC_ACTION", "TECHNOLOGY_INNOVATION", "POLICY_REGULATION", "MACRO",
                              "REVENUE", "DEMAND", "SUPPLY")
GOLD = {
    "r00": dict(E=[E("HAVELLS", "COMPANY"), E("JUMBO GROUP", "COMPANY"), E("KITCHEN APPLIANCES", "PRODUCT"),
                   E("UAE", "LOCATION")],
                V=[V("PARTNERS WITH", [SA], ["reported"], ["HAVELLS", "JUMBO GROUP", "KITCHEN APPLIANCES"], ["UAE"])],
                C=[C(["HAVELLS"], "PARTNERS_WITH", ["JUMBO GROUP"], ["reported"])]),
    "r01": dict(E=[E("EUROAPI", "COMPANY"), E("NET SALES", "INDICATOR")],
                V=[V("SEES", [RE], ["forecast"], ["EUROAPI", "NET SALES"])], C=[]),
    "r02": dict(E=[E("TOPBUILD", "COMPANY"), E("NET SALES", "INDICATOR")],
                V=[V("NET SALES", [RE], ["reported"], ["NET SALES"], ["TOPBUILD"])], C=[],
                note="Result without a verb: the trigger is the variable itself."),
    "r03": dict(E=[E("PBOC", "ORGANIZATION"), E("Swap Facility", "CONCEPT", "PRODUCT")],
                V=[V("Conducts", [MA, PR], ["reported"], ["PBOC", "Swap Facility"])],
                C=[C(["PBOC"], "OFFERS", ["Swap Facility"], ["reported"], opt=True)],
                note="'Conducts' a facility may or may not read as providing it: OFFERS optional."),
    "r04": dict(E=[E("CFIUS", "ORGANIZATION"), E("WAPO", "COMPANY")],
                V=[V("DELIVERED", [PR], ["reported", "uncertain"], ["CFIUS"])], C=[],
                note="The deal is unnamed; WaPo is the source, not a participant."),
    "r05": dict(E=[E("Singapore", "LOCATION"), E("Budget Surplus", "INDICATOR"), E("Handouts", "CONCEPT", opt=True)],
                V=[V("Sees", [MA], ["forecast", "reported"], ["Budget Surplus"], ["Singapore", "Handouts"])], C=[],
                note="'Amid' is not causal: no IMPACTS."),
    "r06": dict(E=[E("Arcelik", "COMPANY"), E("Dividend Payout", "INDICATOR"), E("2023 Net", "INDICATOR", opt=True)],
                V=[V("Proposes", [N, RE, SA], ["planned", "negated"], ["Arcelik", "Dividend Payout"])], C=[]),
    "r07": dict(E=[E("JAPAN WATCHDOG", "ORGANIZATION"), E("CHUGOKU ELEC.", "COMPANY"), E("JAPAN", "LOCATION", opt=True),
                   E("FALSE ADVERTISING", "CONCEPT", opt=True)],
                V=[V("FINES", [PR], ["reported"], ["JAPAN WATCHDOG", "CHUGOKU ELEC."])], C=[]),
    "r08": dict(E=[E("BOJ", "ORGANIZATION"), E("Manufacturing", "SECTOR"), E("Services", "SECTOR"),
                   E("Tankan", "PRODUCT", "CONCEPT", "INDICATOR", opt=True)],
                V=[V("Weakens", [MA], ["reported"], ["Manufacturing"], ["BOJ", "Tankan"]),
                   V("Advances", [MA], ["reported"], ["Services"], ["BOJ", "Tankan"])], C=[]),
    "r09": dict(E=[E("Vodafone Idea", "COMPANY"), E("GMR Airports Infra", "COMPANY")], V=[], C=[],
                note="Futures positioning is a market move: no event."),
    "r10": dict(E=[E("GOLDMAN SACHS", "COMPANY"), E("NET INTEREST INCOME", "INDICATOR")],
                V=[V("NET INTEREST INCOME", [RE], ["reported"], ["NET INTEREST INCOME"], ["GOLDMAN SACHS"])], C=[]),
    "r11": dict(E=[E("LIBERTY RESOURCES ACQUISITION CORP", "COMPANY")],
                V=[V("FILES TO DELAY", [N, PR], ["reported", "planned"], ["LIBERTY RESOURCES ACQUISITION CORP"])], C=[],
                note="'ACQUISITION' is part of a name: no ACQUIRES."),
    "r12": dict(E=[E("1MDB", "COMPANY", "ORGANIZATION"), E("Chapter 15", "CONCEPT", opt=True)],
                V=[V("Put Into Chapter 15", [PR, N], ["reported"], [], ["1MDB", "Chapter 15"]),
                   V("Seek Assets", [N, PR], ["reported", "planned"], [], ["1MDB"], opt=True)], C=[],
                note="The firms and the liquidators are unnamed; 1MDB is only a descriptor."),
    "r13": dict(E=[E("Asia", "LOCATION"), E("Mpox", "CONCEPT"), E("WHO", "ORGANIZATION"), E("Africa", "LOCATION")],
                V=[V("Declaration", [PR], ["reported"], ["WHO"], ["Africa", "Mpox"])], C=[],
                note="'After' alone: no IMPACTS; the share move is not an event."),
    "r14": dict(E=[E("General Dynamics", "COMPANY")],
                V=[V("Names", [N, SA], ["reported", "planned"], ["General Dynamics"])], C=[]),
    "r15": dict(E=[E("NZ", "LOCATION", opt=True), E("PARLIAMENT COMMITTEE", "ORGANIZATION", opt=True)], V=[], C=[],
                note="A person at a committee: nothing to extract."),
    "r16": dict(E=[E("BOE", "ORGANIZATION"), E("INFLATION", "INDICATOR")], V=[], C=[],
                note="Commentary by a person: no event."),
    "r17": dict(E=[E("Bangkok Dusit Medical", "COMPANY"), E("Earnings", "INDICATOR")],
                V=[V("Earnings Rise", [RE], ["reported"], ["Earnings"], ["Bangkok Dusit Medical"])],
                C=[C(["v1", "Earnings"], "IMPACTS", ["Bangkok Dusit Medical"], ["reported"], ["positive"])],
                note="Stock move with a stated cause ('on'): IMPACTS cause -> company."),
    "r18": dict(E=[E("NIPPON ELECTRIC GLASS", "COMPANY")], V=[], C=[], note="Buyback: no event, no ACQUIRES."),
    "r19": dict(E=[E("CAPRI GLOBAL", "COMPANY"), E("ROOFTOP SOLAR FINANCE LOAN PRODUCT", "PRODUCT"),
                   E("ROOFTOP SOLAR", "CONCEPT", opt=True)],
                V=[V("INTRODUCES", [TI, SA], ["reported"], ["CAPRI GLOBAL", "ROOFTOP SOLAR FINANCE LOAN PRODUCT"])],
                C=[C(["CAPRI GLOBAL"], "OFFERS", ["ROOFTOP SOLAR FINANCE LOAN PRODUCT"], ["reported"])]),
    "r20": dict(E=[E("CalSTRS", "COMPANY", "ORGANIZATION"), E("Heidelberg Materials", "COMPANY")],
                V=[V("Backs", [N, SA], ["reported", "planned"], ["CalSTRS", "Heidelberg Materials"])], C=[],
                note="Proxy vote; 'on 27 of 32 Proposals' is not causal."),
    "r21": dict(E=[E("MOODY'S RATINGS", "COMPANY"), E("SOUTHWEST", "COMPANY")], V=[], C=[],
                note="Credit rating action: a rating, no event."),
    "r22": dict(E=[E("FAA", "ORGANIZATION"), E("SPACEX", "COMPANY"), E("CIVIL PENALTIES", "CONCEPT", opt=True)],
                V=[V("PROPOSES", [PR], ["planned", "reported"], ["FAA", "SPACEX"])], C=[]),
    "r23": dict(E=[E("Bloomberg Intelligence", "COMPANY")], V=[], C=[]),
    "r24": dict(E=[E("Crude", "PRODUCT", opt=True), E("Crude Flows", "INDICATOR"), E("Sakhalin", "LOCATION", "PRODUCT")],
                V=[V("Hit Two-Month High", [SU], ["reported"], ["Crude Flows"]),
                   V("Work Ends", [SU], ["reported"], ["Sakhalin"])], C=[],
                note="'Hit' a high is not an effect verb; 'After' alone: no IMPACTS."),
    "r25": dict(E=[E("Helsing", "COMPANY"), E("Defense AI", "CONCEPT", "SECTOR"), E("Defense", "SECTOR", opt=True)],
                V=[V("Valued", [N, SA], ["reported"], ["Helsing"])],
                C=[C(["Helsing"], "ACTIVE_IN", ["Defense AI", "Defense"], ["reported", N])]),
    "r26": dict(E=[E("R&F", "COMPANY"), E("Developer", "SECTOR", opt=True)],
                V=[V("Faces Court Showdown", [N, PR], ["reported", "planned", "forecast", "uncertain"], ["R&F"])],
                C=[C(["R&F"], "ACTIVE_IN", ["Developer"], ["reported", N], opt=True)],
                note="'Developer' is a role noun without the industry word: ACTIVE_IN optional."),
    "r27": dict(E=[E("Remy", "COMPANY"), E("Cognac", "PRODUCT"), E("US", "LOCATION")],
                V=[V("Eyes US Recovery", [DE, RE], ["forecast", "planned"], ["Remy", "US"])],
                C=[C(["Remy"], "ACTIVE_IN", ["Cognac"], ["reported", N])],
                note="'Cognac Maker' refers to Remy; 'as' alone: no IMPACTS."),
    "r28": dict(E=[E("K-Pop", "CONCEPT", "SECTOR", "PRODUCT", opt=True), E("NewJeans", "PRODUCT", "COMPANY", opt=True)],
                V=[V("Wins Ruling", [PR, N], ["reported"], [], ["NewJeans"])], C=[],
                note="The producer is an unnamed person: no ACTIVE_IN."),
    "r29": dict(E=[E("SoftBank", "COMPANY"), E("Elliott", "COMPANY")],
                V=[V("Push", [SA, N], ["reported"], [], ["Elliott", "SoftBank"], opt=True)], C=[],
                note="Buyback: no event and no OFFERS; 'After' alone: no IMPACTS."),
    "r30": dict(E=[E("Raymond", "COMPANY"), E("Real Estate", "SECTOR"), E("New Mumbai Project", "PRODUCT"),
                   E("Mumbai", "LOCATION", opt=True)],
                V=[V("Launches", [TI, SA, N], ["reported"], ["New Mumbai Project"], ["Raymond", "Real Estate"])],
                C=[C(["Raymond"], "ACTIVE_IN", ["Real Estate"], ["reported", N]),
                   C(["Raymond"], "OFFERS", ["New Mumbai Project"], ["reported"], opt=True)],
                note="The launcher is Raymond's unnamed arm: OFFERS by Raymond optional."),
    "r31": dict(E=[E("SMBC", "COMPANY"), E("Private Credit Fund", "PRODUCT"), E("Private Credit", "CONCEPT", opt=True)],
                V=[V("Launches", [TI, SA], ["reported"], ["SMBC", "Private Credit Fund"])],
                C=[C(["SMBC"], "OFFERS", ["Private Credit Fund"], ["reported"])]),
    "r32": dict(E=[E("Anthropic", "COMPANY"), E("AI Model", "PRODUCT"), E("OpenAI", "COMPANY")],
                V=[V("Unveils", [TI], ["reported"], ["Anthropic", "AI Model"], ["OpenAI"])],
                C=[C(["Anthropic"], "OFFERS", ["AI Model"], ["reported"])],
                note="Competing with OpenAI is not a claim."),
    "r33": dict(E=[E("Grupo Bahia", "COMPANY"), E("SAF", "PRODUCT", "CONCEPT"),
                   E("OIL PRODUCTS", "SECTOR", "PRODUCT", opt=True), E("AMERICAS", "LOCATION", opt=True)],
                V=[V("to Invest", [SA, SU], ["planned"], ["Grupo Bahia", "SAF"])],
                C=[C(["Grupo Bahia"], "INVESTS_IN", ["SAF"], ["planned"])]),
    "r34": dict(E=[E("United Arrows", "COMPANY"), E("SM Trust AM", "COMPANY")],
                V=[V("Stake Rises", [SA], ["reported"], ["SM Trust AM", "United Arrows"])],
                C=[C(["SM Trust AM"], "INVESTS_IN", ["United Arrows"], ["reported"])]),
    "r35": dict(E=[E("Abu Dhabi", "LOCATION"), E("Egypt", "LOCATION"), E("Property", "SECTOR", opt=True)],
                V=[V("to Invest", [SA], ["planned"], ["Egypt"], ["Abu Dhabi", "Property"])], C=[],
                note="The consortium is unnamed: no INVESTS_IN."),
    "r36": dict(E=[E("Intuitive Surgical", "COMPANY"), E("Redburn", "COMPANY")], V=[], C=[], note="Rating."),
    "r37": dict(E=[E("Plug Power", "COMPANY"), E("Roth MKM", "COMPANY")], V=[], C=[], note="Rating."),
    "r38": dict(E=[E("Centerbridge", "COMPANY"), E("Banca Progetto", "COMPANY"), E("Oaktree", "COMPANY")],
                V=[V("Agrees to Buy", [SA], ["planned"], ["Centerbridge", "Banca Progetto"], ["Oaktree"])],
                C=[C(["Centerbridge"], "ACQUIRES", ["Banca Progetto"], ["planned"])]),
    "r39": dict(E=[E("Telkom Indonesia", "COMPANY"), E("Huawei", "COMPANY"), E("AI", "CONCEPT")],
                V=[V("Partners With", [TI, SA], ["reported"], ["Telkom Indonesia", "Huawei"], ["AI"])],
                C=[C(["Telkom Indonesia"], "PARTNERS_WITH", ["Huawei"], ["reported"])]),
    "r40": dict(E=[E("Adani", "COMPANY"), E("EDGE", "COMPANY"), E("Defense", "SECTOR", "CONCEPT"),
                   E("Manufacturing", "SECTOR", "CONCEPT", opt=True), E("R&D", "CONCEPT", opt=True)],
                V=[V("Tie Up", [SA, TI], ["reported"], ["Adani", "EDGE"], ["Defense", "Manufacturing", "R&D"])],
                C=[C(["Adani"], "PARTNERS_WITH", ["EDGE"], ["reported"])]),
    "r41": dict(E=[E("VGP", "COMPANY"), E("Verne", "COMPANY"), E("Autonomous Vehicle", "CONCEPT", "PRODUCT")],
                V=[V("Partners With", [SA, TI, SU], ["reported", "planned"], ["VGP", "Verne"], ["Autonomous Vehicle"])],
                C=[C(["VGP"], "PARTNERS_WITH", ["Verne"], ["reported", "planned"])]),
    "r42": dict(E=[E("Ukraine", "LOCATION"), E("Defense", "SECTOR", opt=True)],
                V=[V("to Partner", [SA], ["planned", "forecast", "uncertain"], ["Ukraine"], ["Defense"])], C=[],
                note="Groups unnamed and Ukraine is a country: no PARTNERS_WITH."),
    "r43": dict(E=[E("Iveco", "COMPANY"), E("Italy Army", "ORGANIZATION"), E("Trucks", "PRODUCT"),
                   E("Italy", "LOCATION", opt=True)],
                V=[V("Signs Contract to Supply", [DE, SU, SA], ["reported", "planned"], ["Iveco", "Italy Army", "Trucks"])],
                C=[C(["Iveco"], "SUPPLIES_TO", ["Italy Army"], ["reported", "planned"])]),
    "r44": dict(E=[E("HD Hyundai Mipo", "COMPANY"), E("Ships", "PRODUCT")],
                V=[V("Wins Order to Build", [DE, SU], ["reported"], ["HD Hyundai Mipo", "Ships"])],
                C=[C(["HD Hyundai Mipo"], "OFFERS", ["Ships"], ["planned", "reported"], opt=True)],
                note="Customer unnamed: no SUPPLIES_TO."),
    "r45": dict(E=[E("Boeing", "COMPANY"), E("Max Jet", "PRODUCT")],
                V=[V("Wins Order", [DE, SU], ["reported"], ["Boeing", "Max Jet"])],
                C=[C(["Boeing"], "OFFERS", ["Max Jet"], ["planned", "reported"], opt=True)],
                note="The airline is unnamed: no SUPPLIES_TO."),
    "r46": dict(E=[E("Hanwha Systems", "COMPANY"), E("Radar", "PRODUCT"), E("Saudi", "LOCATION"),
                   E("Arms", "PRODUCT", "SECTOR", opt=True)],
                V=[V("Wins Order", [DE, SU], ["reported"], ["Hanwha Systems", "Radar"], ["Saudi", "Arms"])],
                C=[C(["Hanwha Systems"], "OFFERS", ["Radar"], ["planned", "reported"], opt=True)],
                note="Saudi is a country, not a customer entity: no SUPPLIES_TO."),
    "r47": dict(E=[E("Adidas", "COMPANY"), E("Sambas", "PRODUCT"), E("Yeezys", "PRODUCT"), E("Sales", "INDICATOR"),
                   E("Forecast", "INDICATOR", opt=True)],
                V=[V("Lifts Forecast", [RE], ["reported", "forecast"], ["Adidas"], ["Forecast"]),
                   V("Strong Sales", [DE, RE], ["reported"], [], ["Sales", "Sambas", "Yeezys"], opt=True)],
                C=[C(["Adidas"], "OFFERS", ["Sambas"], ["reported"], opt=True),
                   C(["Adidas"], "OFFERS", ["Yeezys"], ["reported"], opt=True)],
                note="'Amid' is not causal and 'Lifts' here means raises: no IMPACTS."),
    "r48": dict(E=[E("Vedanta", "COMPANY")],
                V=[V("Stake Sale Report", [SA], ["uncertain", "reported"], [], ["Vedanta"])], C=[],
                note="Bond price move is not an event; 'Hits' a high is not an effect verb; 'After' alone."),
    "r49": dict(E=[E("Rocket Lab USA", "COMPANY"), E("Option Trading Volume", "INDICATOR", opt=True)], V=[], C=[],
                note="Option trading volume is market activity: no event."),
}


def main():
    heads = {json.loads(l)["id"]: json.loads(l) for l in (DIR / "real_50.jsonl").read_text().splitlines()}
    assert set(heads) == set(GOLD), set(heads) ^ set(GOLD)
    out = []
    for rid, g in GOLD.items():
        text = heads[rid]["headline"]
        ents = [dict(id=f"e{i + 1}", **e) for i, e in enumerate(g["E"])]
        by_mention = {e["mention"]: e["id"] for e in ents}
        for e in ents:
            assert e["mention"] in text, (rid, e["mention"])
        evs = []
        for i, v in enumerate(g["V"]):
            v = dict(id=f"v{i + 1}", **v)
            v["parts"] = [by_mention[m] for m in v["parts"]]
            v["opt_parts"] = [by_mention[m] for m in v["opt_parts"]]
            evs.append(v)
        ids = {**by_mention, **{v["id"]: v["id"] for v in evs}}
        claims = []
        for c in g["C"]:
            c = dict(c)
            c["subjects"] = [ids[m] for m in c["subjects"]]
            c["objects"] = [ids[m] for m in c["objects"]]
            claims.append(c)
        out.append(dict(id=rid, headline=text, date=heads[rid]["date"], stratum=heads[rid]["stratum"],
                        entities=ents, events=evs, claims=claims, note=g.get("note")))
    (DIR / "real_50_gold.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    n = lambda k, opt: sum(1 for r in out for x in r[k] if x.get("opt", False) == opt)
    print(f"{len(out)} headlines; required entities {n('entities', False)}, events {n('events', False)}, "
          f"claims {n('claims', False)}; optional {n('entities', True)}/{n('events', True)}/{n('claims', True)}")


if __name__ == "__main__":
    main()
