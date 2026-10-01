# Proposta tecnica: motore di ricerca storico del Web

## 1. Obiettivo

Costruire un sistema che risponda a query del tipo:

``` text
query = "ChatGPT"
cutoff = "2023-01-16 23:59:59"
```

restituendo documenti e pagine che erano pubblicamente disponibili entro
la data indicata, evitando per quanto possibile contaminazioni da
informazioni successive.

L'obiettivo non è ricostruire letteralmente tutto Internet in un
determinato istante --- nessun archivio pubblico possiede una copia
completa del Web --- ma ottenere uno **snapshot informativo
temporalmente coerente e verificabile**.

Esempio d'uso:

> "Che cosa risultava essere ChatGPT il 16 gennaio 2023, utilizzando
> soltanto informazioni verificabili disponibili entro quella data?"

------------------------------------------------------------------------

## 2. Principio fondamentale

Il sistema deve separare due problemi:

1.  **URL discovery** --- trovare le pagine potenzialmente rilevanti per
    una query.
2.  **Temporal validation** --- verificare e recuperare una versione
    della pagina esistente entro il cutoff.

La Wayback Machine è molto utile per il secondo problema, ma non è
sufficiente da sola per il primo: il suo indice è principalmente
orientato a URL e catture storiche, non a una ricerca full-text
dell'intero Web.

Per questo la soluzione proposta è **ibrida**.

------------------------------------------------------------------------

## 3. Architettura generale

``` text
                  QUERY
             "ChatGPT", 16/1/23
                     │
             ┌───────▼───────┐
             │ URL DISCOVERY │
             └───────┬───────┘
                     │
              URL candidati
                     │
             ┌───────▼────────┐
             │ TIME VALIDATION│
             │ snapshot <= T  │
             └───────┬────────┘
                     │
              URL verificati
                     │
             ┌───────▼────────┐
             │ DOWNLOAD       │
             │ historical HTML│
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │ EXTRACT/DEDUPE │
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │ RANKING        │
             └───────┬────────┘
                     │
                     ▼
              DATASET FINALE
```

------------------------------------------------------------------------

## 4. Fonti dati

### 4.1 Internet Archive / Wayback Machine

Fonte principale per recuperare versioni storiche delle pagine.

Per ogni URL candidato si cercano le catture:

``` text
timestamp <= cutoff
statuscode = 200
mimetype = text/html
```

e si seleziona l'ultima cattura valida precedente al cutoff.

Esempio:

``` text
Articolo:
snapshot 2022-12-20
snapshot 2023-01-14
snapshot 2023-02-03

cutoff = 2023-01-16

→ utilizzare snapshot 2023-01-14
→ ignorare snapshot 2023-02-03
```

Questo impedisce il **look-ahead bias**.

### 4.2 Common Crawl

Common Crawl è utile per:

-   discovery di URL;
-   analisi su larga scala;
-   recupero di contenuti WARC/WET;
-   costruzione futura di un indice full-text proprietario.

Non deve però essere interpretato come uno snapshot giornaliero esatto.
I crawl coprono intervalli temporali e possono essere successivi al
cutoff richiesto.

Per recuperare una singola pagina dai WARC non è necessario scaricare
l'intero archivio. Dall'indice si possono ottenere:

``` text
filename
offset
length
```

e utilizzare una HTTP Range Request per scaricare soltanto i byte
necessari.

### 4.3 Fonti complementari

Per aumentare la copertura della discovery:

-   archivi di news;
-   RSS storici;
-   dataset accademici;
-   indici di siti specifici;
-   forum e community;
-   Hacker News;
-   Reddit;
-   Wikipedia storica;
-   siti istituzionali e aziendali.

La selezione dipende dal tipo di query.

------------------------------------------------------------------------

## 5. Pipeline on-demand

Per una query eseguita da zero:

### Fase 1 --- Query expansion

Input:

``` text
ChatGPT
```

Possibili espansioni:

``` text
Chat GPT
OpenAI ChatGPT
OpenAI chatbot
related entities
domain-specific searches
```

Tempo indicativo:

``` text
~1 secondo
```

### Fase 2 --- Historical URL discovery

Si interrogano gli indici disponibili per ottenere URL candidati.

Output indicativo:

``` text
500 – 10.000+ URL candidati
```

Tempo:

``` text
~5–60 secondi
```

### Fase 3 --- Temporal validation

Per ogni URL:

1.  interrogare l'archivio;
2.  trovare le snapshot precedenti al cutoff;
3.  selezionare l'ultima cattura valida;
4.  scartare contenuti non verificabili, se si usa la modalità rigorosa.

Tempo:

``` text
~10–120 secondi
```

Le richieste devono essere parallelizzate con limiti di concorrenza,
retry e backoff.

### Fase 4 --- Download

Scaricare esclusivamente le versioni storiche selezionate.

Tempo:

``` text
~30 secondi – 10+ minuti
```

a seconda della profondità della ricerca.

### Fase 5 --- Estrazione

Conversione:

``` text
HTML
 ↓
main content
 ↓
plain text
```

Possibile utilizzo di librerie come Trafilatura.

Metadati da conservare:

``` text
url
domain
title
publication_date
archive_timestamp
retrieved_at
text
content_hash
verification_status
```

### Fase 6 --- Deduplicazione

Eliminare:

-   copie dello stesso articolo;
-   syndicated content;
-   URL equivalenti;
-   variazioni minime della stessa pagina.

Tecniche:

``` text
exact hash
SimHash
MinHash
canonical URL
```

### Fase 7 --- Ranking

Prima versione:

``` text
BM25
```

Successivamente:

``` text
BM25 + semantic ranking
```

Il ranking non deve modificare il criterio temporale: un documento
temporalmente non valido non può essere recuperato solo perché
semanticamente rilevante.

------------------------------------------------------------------------

## 6. Modello temporale

Una pagina può avere almeno tre date differenti:

``` text
publication_date
archive_timestamp
last_modified
```

Esempio:

``` text
Articolo pubblicato:     2023-01-10
Wayback cattura:         2023-01-14
Articolo modificato:     2023-02-20
Nuova cattura:           2023-02-21
```

Con:

``` text
cutoff = 2023-01-16
```

si utilizza esclusivamente la versione del 14 gennaio.

Caso più difficile:

``` text
Articolo pubblicato:     2023-01-10
Prima cattura Wayback:   2023-01-20
```

È plausibile che la pagina esistesse il 16 gennaio, ma non possediamo
una fotografia verificabile del suo contenuto a quella data.

Il record dovrebbe quindi indicare:

``` text
publication_date = 2023-01-10
first_known_capture = 2023-01-20
verified_at_cutoff = false
```

In modalità temporalmente rigorosa, il documento viene escluso.

------------------------------------------------------------------------

## 7. Output

Formato consigliato per il corpus:

``` text
Parquet
```

con eventuale esportazione:

``` text
JSONL
SQLite
```

Schema indicativo:

``` text
document_id
url
canonical_url
domain
title
publication_date
archive_timestamp
cutoff
text
content_hash
temporal_status
source
retrieved_at
```

Il dataset può successivamente alimentare:

-   ricerca full-text;
-   analisi storiche;
-   RAG;
-   LLM;
-   timeline;
-   confronto della conoscenza disponibile tra date differenti.

------------------------------------------------------------------------

## 8. Prestazioni attese

Stime ingegneristiche indicative per una query on-demand, senza corpus
già scaricato:

  Modalità               Documenti finali           Tempo indicativo
  -------------------- ------------------ --------------------------
  Fast                           100--500                   20--60 s
  Normal                       500--5.000                   1--5 min
  Deep                      5.000--50.000                 5--30+ min
  Web quasi completo               enorme   non realistico per query

I tempi reali dipendono da:

-   numero di URL candidati;
-   disponibilità delle snapshot;
-   velocità degli archivi;
-   rate limiting;
-   concorrenza;
-   dimensione delle pagine;
-   quantità di deduplicazione richiesta.

### Target iniziale consigliato

``` text
~30 s  → primi 100 risultati
~2 min → corpus di buona qualità
~5 min → ricerca approfondita
```

------------------------------------------------------------------------

## 9. Cache

La cache è fondamentale.

Prima query:

``` text
ChatGPT @ 2023-01-16
→ discovery + download + processing
→ ~2 minuti
```

Query successiva:

``` text
"come funzionava ChatGPT?" @ 2023-01-16
```

può riutilizzare gran parte del corpus già recuperato.

Si possono mettere in cache:

``` text
URL discovery
CDX responses
snapshot metadata
raw HTML
extracted text
content hashes
ranking index
```

In questo modo le query successive sullo stesso argomento/periodo
possono scendere da minuti a secondi.

------------------------------------------------------------------------

## 10. MVP

Stack proposto:

``` text
Python
│
├── query.py
│
├── discovery/
│   ├── commoncrawl.py
│   └── historical_sources.py
│
├── archive/
│   └── wayback.py
│
├── extract.py
├── dedupe.py
├── rank.py
├── cache.py
└── output.py
```

Componenti iniziali:

-   Python;
-   asyncio / HTTP client asincrono;
-   Wayback CDX;
-   Common Crawl Index;
-   Trafilatura;
-   SimHash/MinHash;
-   BM25;
-   SQLite o DuckDB per metadata/cache;
-   Parquet per output.

Non è necessario iniziare con Elasticsearch, Spark o un cluster
distribuito.

------------------------------------------------------------------------

## 11. Tempi di sviluppo

Stima per uno sviluppatore esperto:

### Proof of concept --- \~1 giorno

Input:

``` text
("ChatGPT", "2023-01-16")
```

Output:

``` text
lista di pagine storiche + testo
```

### MVP --- \~2--4 giorni

Aggiungere:

-   parallelizzazione;
-   retry;
-   caching;
-   deduplicazione;
-   validazione temporale;
-   ranking;
-   output strutturato.

### Servizio robusto --- \~1--2 settimane

Aggiungere:

-   API;
-   gestione errori completa;
-   rate limiting;
-   persistence;
-   monitoring;
-   job management;
-   ranking migliore;
-   più sorgenti di discovery;
-   test di correttezza temporale.

------------------------------------------------------------------------

## 12. Evoluzione: indice full-text proprietario

Se il sistema dimostra valore, si può eliminare gran parte del costo
on-demand costruendo un indice proprietario.

Pipeline:

``` text
Common Crawl WET/WARC
        ↓
text extraction
        ↓
normalizzazione
        ↓
deduplica
        ↓
temporal metadata
        ↓
OpenSearch / Elasticsearch / Lucene / Tantivy
```

Documento indicizzato:

``` json
{
  "url": "https://example.com/article",
  "title": "Example",
  "text": "...",
  "domain": "example.com",
  "crawl_time": "2023-01-14T12:34:00Z",
  "warc_pointer": "...",
  "content_hash": "..."
}
```

Una query diventerebbe quindi concettualmente:

``` text
text:"ChatGPT"
AND timestamp <= 2023-01-16T23:59:59Z
```

Il costo iniziale di indicizzazione è elevato, ma il costo marginale
delle query diventa molto più basso.

------------------------------------------------------------------------

## 13. Evoluzione consigliata: indice bitemporale/versionato

Non conviene creare uno snapshot completo separato per ogni giorno.

Meglio conservare le versioni dei documenti:

``` text
URL A
├── versione 1: valida 2022-12-01 → 2023-01-13
├── versione 2: valida 2023-01-14 → 2023-02-20
└── versione 3: valida 2023-02-21 → ...
```

Una query al 16 gennaio recupera:

``` text
versione valida al 2023-01-16
```

Questo permette di implementare una sorta di:

> **motore di ricerca con una manopola temporale**

senza duplicare l'intero indice per ogni giornata.

------------------------------------------------------------------------

## 14. Roadmap consigliata

### Fase A --- MVP on-demand

Obiettivo:

``` text
query + cutoff → corpus storico verificato
```

Nessun indice globale proprietario.

### Fase B --- Cache condivisa

Ogni ricerca aumenta progressivamente il patrimonio locale di:

``` text
URL
snapshot
testi
metadata
```

### Fase C --- Indice locale

Indicizzare tutto ciò che è già stato recuperato.

Le query ripetute diventano quasi immediate.

### Fase D --- Pre-indexing selettivo

Indicizzare preventivamente:

-   principali fonti news;
-   domini ad alta qualità;
-   Wikipedia;
-   forum/community rilevanti;
-   dataset Common Crawl selezionati.

### Fase E --- Historical Web Search Engine

Indice full-text versionato su larga scala con:

``` text
search(query, cutoff)
```

come primitiva fondamentale.

------------------------------------------------------------------------

## 15. Conclusione

La prima versione non dovrebbe tentare di scaricare o indicizzare
l'intero Web.

L'approccio consigliato è:

``` text
URL discovery
      ↓
temporal validation
      ↓
selective historical download
      ↓
extraction
      ↓
deduplication
      ↓
ranking
      ↓
cache
```

con un obiettivo operativo iniziale di:

``` text
~30 secondi → primi risultati
~2 minuti   → corpus utile
~5 minuti   → ricerca approfondita
```

Una volta validata l'utilità del sistema, la cache accumulata può
evolvere gradualmente in un **indice full-text storico e versionato**,
riducendo drasticamente la latenza delle query successive.

Il vero problema tecnico non è il download: è ottenere una buona
**discovery storica** mantenendo una rigorosa **validità temporale** dei
documenti.
