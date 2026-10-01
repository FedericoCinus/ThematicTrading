GUIDELINE 044-v2 — schema 0.4 (defined in notebook 0.44). The gold annotator and the models read this same text.

TASK. For one news headline and its date, list the named things (nodes) and the relations between them (edges) that the headline itself states. Use only the headline and what was known on its date. Never add tickers, modern names, parent companies, counterparties, sectors, products or outcomes from memory. A headline with no relation gives only its nodes; empty lists are valid.

1 · NODES — every named thing of these 6 types. `mention` is copied verbatim from the headline (drop a possessive 's).
- COMPANY: firms, banks, brokers, asset managers, funds as investing organisations, news agencies.
- ORGANIZATION: governments, ministries, regulators, central banks, agencies, courts, armed forces, universities, international bodies.
- PRODUCT: an identifiable good or service, including traded commodities (oil, copper, LNG), drugs, devices, software, AI models, funds or loans launched as products.
- CONCEPT: a technology, process, policy, named event or economic idea (AI, carbon capture, tariffs, Covid-19).
- SECTOR: an industry or market named in the text (defense, real estate, chipmakers).
- LOCATION: a country, region or city named as a noun. A nationality or possessive ("German", "Italy's", "India's") is not a node.
Not nodes: people and their titles; securities (shares, bonds, bills, notes, futures, options); indices; currencies; amounts and percentages; financial variables (revenue, sales, earnings, orders, stake, prices, rates, inflation, volume); documents (reports, filings); generic groups ("investors", "firms", "a consortium").

2 · EDGES — each edge has a subject, a predicate, an object, a modality and a verbatim evidence span containing both ends. Both ends must be nodes of this record.
From a company to a product, concept or sector:
- OFFERS (COMPANY/ORGANIZATION -> PRODUCT/CONCEPT): the headline says the subject launches, introduces, makes, sells or provides it, or wins an order to supply it.
- ACTIVE_IN (COMPANY -> PRODUCT/CONCEPT/SECTOR/LOCATION): the headline qualifies what the company does or is exposed to — a descriptor in any position ("Bitcoin Miner X", "X's Chip Unit") or the declared purpose of a deal ("X Partners With Y for Kitchen Appliances" -> X and Y are ACTIVE_IN Kitchen Appliances). The object is the good, technology or industry word, not the role noun ("Chip Maker X" -> Chip). If the sentence states a single launch or production, use OFFERS instead, not both. When two readings are valid, give both.
- EXPOSED_TO (COMPANY/SECTOR -> PRODUCT/CONCEPT): the headline says the company or its shares move, benefit or suffer because of it, with "on", "as", "due to", "thanks to", "because of" or an effect verb ("lifts", "boosts", "hits", "hurts", "weighs on", "drives"). "After" or "amid" alone: no edge. A rating with a stated cause ("Raised to Buy on AI Demand") gives EXPOSED_TO with modality uncertain.
- INVESTS_IN (COMPANY/ORGANIZATION -> COMPANY/PRODUCT/CONCEPT/SECTOR/LOCATION): a minority stake or insider purchase in a company, or declared spending on a product, technology, sector or place. "X Stake Rises to 11%: Y" -> Y INVESTS_IN X. Control is ACQUIRES.
Between companies:
- ACQUIRES (COMPANY -> COMPANY): buys, takes control of, bids or offers for a company or a named unit. A sale with a named buyer is the buyer's ACQUIRES.
- PARTNERS_WITH (COMPANY <-> COMPANY): agreement, alliance, joint venture, licence or collaboration that is not an acquisition or investment. Order does not matter.
- SUPPLIES_TO (COMPANY -> COMPANY/ORGANIZATION): supplier -> named customer, including a contract won. A country is not a customer. The good supplied, if named, also gets its own OFFERS edge.
Modality: reported (done or stated as fact: "Buys", "Signs", "Wins", "Launches", "Jumps on"); planned ("to", "Plans", "Agrees to", "Offers to", "Nears", "Seeks"); uncertain (opinions, forecasts, reports and rumours: "says", "sees", "expects", "according to", "said to", "mulls", "in talks", ratings); negated (denies, rejects, no plans, withdraws, scraps, terminated, on hold).
No edge for: an analyst rating or price target without a stated cause, a stock or market move without a stated cause, a bond or bill issue, a buyback, a proxy vote, a regulatory fine or action, a management change, results or forecasts of financial variables. Countries, governments, regulators and courts are never subject or object of ACQUIRES, PARTNERS_WITH or INVESTS_IN towards a company. Co-mention alone is never an edge. A party that is not named gives no edge.
