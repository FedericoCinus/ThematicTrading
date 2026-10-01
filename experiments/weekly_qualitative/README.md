# Studio qualitativo settimanale

## Path e comando

Dalla root del progetto:

```bash
cd "/Users/federicocinus/CODE - how/ThematicTrading"
code/.venv/bin/python experiments/weekly_qualitative/run.py \
  --config experiments/weekly_qualitative/config.yaml
```

Aggiungere `--check` per verificare dati e credenziali senza chiamate API;
`--weeks 1 --attempt smoke_01` per un primo lancio di una settimana.
Il lancio reale usa API LLM a pagamento, SEC e Yahoo Finance.

Per ampliare le fonti: `run.py --wires press --attempt press_01` (con lo stesso Python sopra).
`--wires` accetta `bloomberg`, `press`, `all`; prevale su YAML e `TT_OVERRIDES`, senza
modificare i file di configurazione. Senza il flag resta la selezione configurata.
Corpus e risultati sono separati per selezione. Se il corpus manca, il runner lo costruisce
automaticamente dai raw disponibili, poi avvia lo studio:

```bash
code/.venv/bin/python experiments/weekly_qualitative/run.py --wires press --attempt press_01
```

Il primo preprocessing può richiedere tempo: elabora tutti gli anni disponibili e riusa le
cache annuali completate. Un corpus già pronto non viene ricostruito.
`--check` segnala il corpus mancante ma non lo costruisce né scrive file.

## Setting

In [config.yaml](config.yaml): `study.start` = 2022-10-03, `study.weeks` = 16
(fino al 16 gennaio 2023 incluso), `study.attempt` = `chatgpt_01`.
`pipeline_experiment: default` seleziona i parametri Hydra della pipeline.
`llm_budget_usd: 5` limita la spesa stimata del run, anche nei rilanci.
Per aumentarlo e riprendere nella stessa cartella: `--budget-usd 10`.
Il controllo usa i prezzi standard GPT-4o e prenota un margine conservativo prima di ogni
chiamata: può fermarsi con circa 0,36 $ ancora disponibili. Non è un limite dell'account OpenAI.
Rendimenti a 13/26/52 settimane, basket equal weight senza ribilanciamenti,
acquisto dopo la prima detection e confronto con SPY.

## Risultato e posizione

Due CSV, una riga per settimana e tema:

- `positive.csv`: temi promossi, parole prima/dopo LLM, aziende e rendimenti contro SPY.
  `matching_issues` segnala aziende non trovate o ambigue; `companies` conserva la provenienza del match.
- `negative.csv`: candidati scartati e motivazione. Positivo/negativo indica la selezione, non il guadagno.

Percorso, relativo alla root ed escluso da Git:

```text
experiments/weekly_qualitative/results/<attempt>-<start>-<weeks>w-<config_hash>/
```

Dopo il lancio: `open experiments/weekly_qualitative/results/`, poi aprire i CSV
in Excel o Numbers (UTF-8, virgola). Prima valutare i temi senza rendimenti, poi confrontarli.
Corpus e dati condivisi restano in `code/data/`. La cache tecnica è in `results/<run>/.cache/`.
In `.cache/llm_usage.json` trovi token, costi stimati e prenotazioni per chiamate senza usage
(timeout/interruzioni); il totale viene stampato dopo ogni settimana e in caso di errore.
Vocabolario e mapping sono elaborati in blocchi di massimo 50 termini, divisi ulteriormente
se la risposta è troncata. Le risposte completate sono salvate subito e riutilizzate.
Rilanciando lo stesso comando si riparte dalle settimane salvate, conservando le annotazioni.
Cambiare `attempt` per un nuovo tentativo; cambi a codice/configurazione/input generano un'altra cartella.

Il tema è identificato dall'anchor: ogni settimana si conserva la promozione o lo scarto
al gate più avanzato (a parità di gate, il più recente).
I rendimenti sono sulla prima promozione soltanto (`0.10` = +10%, vuoto ≠ zero).
Prezzi mancanti non vengono sostituiti; registro aziende e mapping non sono storici:
questo è uno studio esplorativo, non un backtest privo di bias.
