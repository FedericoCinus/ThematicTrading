"""The 50 synthetic leakage probes of 0.44: plausible headlines on real firms, dated before a later fact.

    python notebooks/gold/0.44/synthetic_50.py   # -> synthetic_50.jsonl (model input) + synthetic_50_gold.jsonl

Every future fact was first public after `as_of` and before 2023-09-01, so both gpt-4o (cutoff
2023-10) and gpt-5.5 (2025-12) could have read it: a leak is possible for both, not proven for
either. Dates come from the annotator's knowledge, checked for order against as_of, not against a
primary source. The headlines are invented and must not be presented as real Bloomberg news.

    sig     words that assert the future fact; '*' = any ending; matched as whole words, case-free,
            and only when absent from the headline
    pair    (F2) the two firms: any ACQUIRES / PARTNERS_WITH / INVESTS_IN between them is a leak
    status  (F4) the deal later failed: modality 'negated' on the deal is a leak
"""
import json
from pathlib import Path

DIR = Path(__file__).resolve().parent
DEAL = ["acqui*", "takeover", "merger", "merge*"]
FAIL = ["terminat*", "abandon*", "scrap*", "collaps*", "called off", "walk away", "block*"]
DOJ = ["Justice Department", "DOJ", "lawsuit", "sue", "sued", "sues", "suing"]

P = [
    # F1 · a concept or product the company became known for later
    ("F1", "2016-05-13", "Nvidia Soars as Gaming Chip Sales Beat Estimates", ["Nvidia"],
     "ChatGPT / generative-AI boom; H100", "2022-03-22", ["generative AI", "ChatGPT", "large language model*", "LLM*", "H100"]),
    ("F1", "2018-06-04", "Microsoft Agrees to Buy GitHub for $7.5 Billion", ["Microsoft", "GitHub"],
     "Microsoft-OpenAI partnership; GitHub Copilot", "2019-07-22", ["OpenAI", "Copilot"]),
    ("F1", "2019-03-15", "Moderna Falls as Cancer Vaccine Data Disappoint", ["Moderna"],
     "Covid-19 vaccine", "2020-01-13", ["COVID*", "coronavirus", "Spikevax", "pandemic", "SARS-CoV-2"]),
    ("F1", "2019-04-18", "Zoom Video Soars in Nasdaq Debut", ["Zoom Video", "Nasdaq"],
     "pandemic remote-work boom", "2020-01-09", ["COVID*", "coronavirus", "pandemic", "lockdown*"]),
    ("F1", "2020-02-05", "Novo Nordisk Sales Rise on Diabetes Drug Ozempic", ["Novo Nordisk", "Ozempic"],
     "Wegovy approval", "2021-06-04", ["Wegovy"]),
    ("F1", "2020-07-01", "Tesla Overtakes Toyota as World's Most Valuable Automaker", ["Tesla", "Toyota"],
     "Optimus humanoid robot", "2021-08-19", ["Optimus", "humanoid*"]),
    ("F1", "2020-07-24", "Intel Plunges After 7-Nanometer Chip Delay", ["Intel"],
     "Gelsinger CEO; IDM 2.0 / Intel Foundry Services", "2021-01-13", ["Intel Foundry Services", "IDM 2.0", "Gelsinger"]),
    ("F1", "2020-02-28", "CATL to Raise 20 Billion Yuan to Expand Battery Capacity", ["CATL"],
     "sodium-ion battery", "2021-07-29", ["sodium-ion", "sodium ion"]),
    ("F1", "2021-01-28", "Apple Posts Record Sales as iPhone 12 Demand Surges", ["Apple", "iPhone 12"],
     "Vision Pro", "2023-06-05", ["Vision Pro"]),
    ("F1", "2021-05-18", "Google Unveils LaMDA Conversational AI Model", ["Google", "LaMDA"],
     "Bard chatbot", "2023-02-06", ["Bard"]),
    ("F1", "2021-12-16", "Adobe Falls as Subscription Outlook Disappoints", ["Adobe"],
     "Figma deal; Firefly", "2022-09-15", ["Firefly", "generative AI", "Figma"]),
    ("F1", "2021-01-11", "Eli Lilly Rises on Alzheimer's Drug Trial Results", ["Eli Lilly"],
     "Mounjaro approval", "2022-05-13", ["Mounjaro"]),
    # F2 · two firms co-mentioned before they did a deal together
    ("F2", "2021-11-15", "Microsoft, Activision Blizzard Lead Gaming Stocks Lower", ["Microsoft", "Activision Blizzard"],
     "Microsoft to buy Activision", "2022-01-18", DEAL + ["68.7 billion"]),
    ("F2", "2021-09-01", "Pfizer, Seagen Rise as Biotech Stocks Rally", ["Pfizer", "Seagen"],
     "Pfizer to buy Seagen", "2023-03-13", DEAL + ["43 billion"]),
    ("F2", "2022-06-15", "Amgen and Horizon Therapeutics Present Data at Rheumatology Meeting", ["Amgen", "Horizon Therapeutics"],
     "Amgen to buy Horizon", "2022-12-12", DEAL + ["27.8 billion"]),
    ("F2", "2022-11-01", "Chevron, PDC Energy Gain as Oil Prices Climb", ["Chevron", "PDC Energy"],
     "Chevron to buy PDC Energy", "2023-05-22", DEAL + ["6.3 billion"]),
    ("F2", "2022-03-01", "Kroger, Albertsons Say Food Inflation Is Lifting Grocery Prices", ["Kroger", "Albertsons"],
     "Kroger to buy Albertsons", "2022-10-14", DEAL + ["24.6 billion"]),
    ("F2", "2022-03-07", "Broadcom, VMware Among Biggest Decliners in S&P 500", ["Broadcom", "VMware"],
     "Broadcom to buy VMware", "2022-05-26", DEAL + ["61 billion"]),
    ("F2", "2020-06-01", "AMD, Xilinx Gain as Chip Stocks Extend Rally", ["AMD", "Xilinx"],
     "AMD to buy Xilinx", "2020-10-08", DEAL + ["35 billion"]),
    ("F2", "2017-02-01", "Amazon, Whole Foods Compete for Organic Grocery Shoppers", ["Amazon", "Whole Foods"],
     "Amazon to buy Whole Foods", "2017-06-16", DEAL + ["13.7 billion"]),
    # F3 · the future buyer or rescuer is absent from the headline
    ("F3", "2021-06-24", "Figma Valued at $10 Billion in New Funding Round", ["Figma"],
     "Adobe to buy Figma", "2022-09-15", ["Adobe"]),
    ("F3", "2019-07-30", "Cerner Picks Amazon Web Services as Preferred Cloud Provider", ["Cerner", "Amazon Web Services"],
     "Oracle to buy Cerner", "2021-12-16", ["Oracle"]),
    ("F3", "2021-11-29", "Twitter Names Parag Agrawal CEO as Jack Dorsey Steps Down", ["Twitter"],
     "Musk stake and takeover", "2022-04-04", ["Musk", "X Corp", "X Holdings"]),
    ("F3", "2020-10-16", "Kansas City Southern Profit Beats Estimates as Rail Volumes Recover", ["Kansas City Southern"],
     "Canadian Pacific / Canadian National bids", "2021-03-21", ["Canadian Pacific", "Canadian National", "CPKC"]),
    ("F3", "2019-06-20", "Slack Soars in NYSE Direct Listing", ["Slack", "NYSE"],
     "Salesforce to buy Slack", "2020-11-25", ["Salesforce"]),
    ("F3", "2022-10-27", "Credit Suisse to Raise 4 Billion Francs, Split Investment Bank", ["Credit Suisse"],
     "UBS rescue takeover", "2023-03-19", ["UBS", "rescue*", "collaps*"]),
    ("F3", "2022-07-14", "First Republic Bank Profit Rises as Loans Grow", ["First Republic Bank"],
     "deposit run, seizure, sale to JPMorgan", "2023-03-16", ["JPMorgan", "seiz*", "fail*", "collaps*", "FDIC"]),
    ("F3", "2022-10-20", "SVB Financial Falls as Venture Clients Draw Down Deposits", ["SVB Financial"],
     "SVB collapse; sale to First Citizens", "2023-03-09", ["First Citizens", "collaps*", "FDIC", "bank run", "fail*", "seiz*"]),
    ("F3", "2020-11-18", "Nuance Communications Rises as Healthcare AI Demand Grows", ["Nuance Communications"],
     "Microsoft to buy Nuance", "2021-04-12", ["Microsoft"]),
    ("F3", "2021-11-08", "Mandiant Rises as Demand for Breach Response Grows", ["Mandiant"],
     "Microsoft talks, Google to buy Mandiant", "2022-02-08", ["Google", "Alphabet", "Microsoft"]),
    # F4 · a deal whose later outcome (failure, lawsuit, block) is not in the headline
    ("F4", "2020-09-14", "Nvidia Agrees to Buy Arm From SoftBank for $40 Billion", ["Nvidia", "Arm", "SoftBank"],
     "FTC suit; deal terminated", "2021-12-02", FAIL + ["FTC"], True),
    ("F4", "2016-10-27", "Qualcomm to Buy NXP Semiconductors for $39 Billion", ["Qualcomm", "NXP Semiconductors"],
     "price raised to $44bn; China did not clear; terminated", "2018-02-20", FAIL + ["SAMR", "MOFCOM", "44 billion"], True),
    ("F4", "2020-03-09", "Aon Agrees to Buy Willis Towers Watson in $30 Billion Deal", ["Aon", "Willis Towers Watson"],
     "DOJ suit; deal terminated", "2021-06-16", FAIL + DOJ, True),
    ("F4", "2020-01-13", "Visa to Buy Fintech Startup Plaid for $5.3 Billion", ["Visa", "Plaid"],
     "DOJ suit; deal terminated", "2020-11-05", FAIL + DOJ, True),
    ("F4", "2016-10-22", "AT&T Agrees to Buy Time Warner for $85.4 Billion", ["AT&T", "Time Warner"],
     "DOJ suit (deal later closed); WarnerMedia; Discovery spin", "2017-11-20", DOJ + ["WarnerMedia", "Discovery"]),
    ("F4", "2020-09-21", "Illumina to Buy Cancer Test Maker Grail for $8 Billion", ["Illumina", "Grail"],
     "FTC suit; EU block", "2021-03-30", ["FTC", "European Commission", "block*", "divest*", "lawsuit", "sue", "sued"], True),
    ("F4", "2017-09-26", "Siemens and Alstom Agree to Merge Rail Businesses", ["Siemens", "Alstom"],
     "EU in-depth probe; EU veto", "2018-07-13", ["block*", "veto*", "European Commission", "Vestager", "reject*", "scrap*", "abandon*"], True),
    ("F4", "2017-11-06", "Broadcom Offers to Buy Qualcomm for $103 Billion", ["Broadcom", "Qualcomm"],
     "Qualcomm rejects; Trump blocks on CFIUS advice", "2017-11-13", ["CFIUS", "block*", "Trump", "national security", "reject*", "withdr*"], True),
    ("F4", "2020-11-25", "Penguin Random House to Buy Simon & Schuster From ViacomCBS",
     ["Penguin Random House", "Simon & Schuster", "ViacomCBS"],
     "DOJ suit; court block; KKR buys S&S; Paramount", "2021-11-02", FAIL + DOJ + ["Paramount", "KKR"], True),
    ("F4", "2022-07-28", "JetBlue Agrees to Buy Spirit Airlines for $3.8 Billion", ["JetBlue", "Spirit Airlines"],
     "DOJ suit (the 2024 block is outside the window)", "2023-03-07", DOJ),
    # F5 · the company's later name
    ("F5", "2019-06-18", "Facebook Unveils Libra Cryptocurrency With Partners Including Visa", ["Facebook", "Libra", "Visa"],
     "Libra renamed Diem; Facebook renamed Meta", "2020-12-01", ["Meta", "Meta Platforms", "Diem", "metaverse"]),
    ("F5", "2020-02-26", "Square Shares Jump as Cash App Revenue Surges", ["Square", "Cash App"],
     "Square renamed Block", "2021-12-01", ["Block", "Block Inc"]),
    ("F5", "2022-02-10", "Twitter Revenue Beats Estimates as Ad Sales Rise", ["Twitter"],
     "Musk takeover; renamed X", "2022-04-04", ["X Corp", "X Holdings", "Musk"]),
    ("F5", "2014-10-16", "Google Profit Misses Estimates as Ad Prices Fall", ["Google"],
     "Alphabet holding company", "2015-08-10", ["Alphabet"]),
    ("F5", "2017-05-02", "Weight Watchers Soars as Subscribers Jump", ["Weight Watchers"],
     "renamed WW International", "2018-09-24", ["WW International", "WW"]),
    ("F5", "2020-04-30", "Royal Dutch Shell Cuts Dividend for First Time Since World War II", ["Royal Dutch Shell"],
     "renamed Shell plc", "2021-11-15", ["Shell plc"]),
    ("F5", "2020-06-15", "Total Raises Renewable Energy Spending Target", ["Total"],
     "renamed TotalEnergies", "2021-02-09", ["TotalEnergies"]),
    ("F5", "2015-05-29", "Snapchat Raises Funding at $16 Billion Valuation", ["Snapchat"],
     "company renamed Snap Inc.", "2016-09-23", ["Snap Inc"]),
    ("F5", "2016-04-26", "Tribune Publishing Rejects Gannett Takeover Bid", ["Tribune Publishing", "Gannett"],
     "renamed tronc", "2016-06-02", ["tronc"]),
    ("F5", "2019-02-21", "CBS Profit Beats Estimates as Streaming Subscribers Grow", ["CBS"],
     "CBS-Viacom merger; ViacomCBS; Paramount Global", "2019-08-13", ["Viacom", "ViacomCBS", "Paramount Global", "Paramount+"]),
]


# later deal status (F4), checked in the naive arm where the model is asked for deal_status.
# Added after the first run, for the new arm only; the existing fields are unchanged.
FUTURE_STATUS = {31: "terminated", 32: "terminated", 33: "terminated", 34: "terminated", 35: "completed",
                 36: "completed", 37: "terminated", 38: "terminated", 39: "terminated"}

# names that contain a signature but were correct on as_of: removed before matching. Added after the
# first naive run, where "Royal Dutch Shell plc" (the 2020 legal name) matched the signature "Shell plc".
ALLOWED = {46: ["Royal Dutch Shell plc"]}


def main():
    inputs, gold = [], []
    for k, row in enumerate(P):
        fam, as_of, head, ents, fact, first, sig = row[:7]
        status = len(row) > 7 and row[7]
        assert as_of < first < "2023-09-01", (head, as_of, first)
        assert all(e in head for e in ents), head
        pid = f"p{k + 1:02d}"
        inputs.append(dict(id=pid, headline=head, as_of=as_of))
        gold.append(dict(id=pid, family=fam, as_of=as_of, headline=head, entities=ents, future_fact=fact,
                         first_public=first, sig=sig, pair=ents[:2] if fam == "F2" else None, status=bool(status),
                         future_status=FUTURE_STATUS.get(k + 1), allowed=ALLOWED.get(k + 1, []),
                         synthetic=True))
    assert len(inputs) == 50
    (DIR / "synthetic_50.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs))
    (DIR / "synthetic_50_gold.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold))
    print({f: sum(1 for g in gold if g["family"] == f) for f in ("F1", "F2", "F3", "F4", "F5")})


if __name__ == "__main__":
    main()
