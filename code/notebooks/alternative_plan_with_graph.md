Unsupervised Discovery of the Generative AI Theme in Bloomberg Headlines

Objective

The goal is to identify the emerging generative AI theme in Bloomberg headlines around November–December 2022 without defining generative-AI-related keywords in advance.

The method should therefore be:

* unsupervised;
* temporally aware;
* based on words and phrases extracted from the headlines;
* capable of identifying new or rapidly changing themes;
* interpretable after discovery.

The algorithm should not be given keywords such as:

* ChatGPT;
* OpenAI;
* generative AI;
* large language model;
* chatbot;
* text generation.

Instead, it should discover an emerging group of related words, phrases, and headlines. A human can then inspect the discovered group and recognize it as the theme now called generative AI.

The process can be summarized as:

Discover all temporally emerging lexical communities, rank them by burst, novelty, coherence, and persistence, and only afterward inspect the highest-ranked communities to identify the one corresponding to generative AI.

⸻

1. Prepare the Dataset

The minimum required fields are:

date
headline

For example:

2022-11-30 | OpenAI Releases Chatbot Capable of Answering Complex Questions
2022-12-05 | New AI Tool Raises Questions About the Future of Writing

Convert the date column to a standard date format.

import pandas as pd
df["date"] = pd.to_datetime(df["date"])

Remove:

* exact duplicate headlines;
* obvious metadata artifacts;
* empty headlines;
* malformed dates.

Near-duplicate headlines may also need to be removed if Bloomberg republishes similar headlines across different channels or editions.

⸻

2. Create Time Windows

The analysis must include a historical period before November–December 2022.

A suitable setup is:

* historical baseline: January 2021–October 2022;
* discovery period: November–December 2022;
* optional validation period: January–March 2023.

The historical baseline is necessary because the objective is not simply to identify frequent topics. It is to identify topics that became unusually prominent, novel, or interconnected around the target period.

Monthly windows

Monthly windows are easy to interpret:

df["month"] = df["date"].dt.to_period("M")

Weekly windows

Weekly windows may be more appropriate because the emergence of ChatGPT-related coverage was rapid.

df["week"] = df["date"].dt.to_period("W")

A practical strategy is to use:

* weekly windows for discovery;
* monthly windows for presentation and validation.

⸻

3. Extract Candidate Terms and Phrases

For every headline, automatically extract candidate lexical units.

These can include:

* nouns;
* proper nouns;
* named entities;
* noun phrases;
* bigrams;
* trigrams.

Possible outputs might eventually include terms such as:

language model
artificial intelligence
chatbot
text generator
OpenAI
machine-generated text

These must be outputs of the extraction process, not manually supplied inputs.

Recommended preprocessing

Apply:

* lowercasing, except where capitalization helps identify entities;
* punctuation normalization;
* lemmatization;
* singular/plural normalization;
* removal of generic stop words;
* normalization of obvious organization aliases.

Preserve proper nouns and named entities. Emerging organizations, products, or people may be central to the discovery process.

Avoid aggressive preprocessing that removes potentially meaningful words such as:

model
language
machine
answer
write
human

Example placeholder extractor

def extract_phrases(text):
    tokens = [
        token.lower()
        for token in text.split()
        if len(token) > 2
    ]
    return sorted(set(tokens))
df["phrases"] = df["headline"].map(extract_phrases)

For a production implementation, replace this placeholder with a noun-phrase and named-entity extraction pipeline.

Possible tools include:

* spaCy;
* Stanza;
* KeyBERT-style candidate extraction;
* part-of-speech-based n-gram extraction.

The extraction method should remain domain-neutral and should not contain generative-AI-specific dictionaries.

⸻

4. Filter Uninformative Terms

Some words may be frequent but provide little thematic information.

Examples may include:

says
new
company
market
business
report
year

Filtering can be based on corpus statistics rather than manually prepared topic dictionaries.

Possible rules include:

* minimum document frequency;
* maximum document frequency;
* removal of terms appearing across nearly all time periods;
* removal of terms with very low information content;
* removal of publisher-specific boilerplate.

For example, retain terms appearing in at least five headlines:

minimum_document_frequency = 5

The exact threshold should depend on the size of the corpus and the selected time-window granularity.

⸻

5. Build a Keyword Co-occurrence Graph

Construct a separate graph for every time window.

In each graph:

* a node represents a term or phrase;
* an edge connects two terms that appear in the same headline;
* the edge weight represents the strength of their association.

For example, if several headlines contain combinations such as:

OpenAI + chatbot
chatbot + writing
language model + answer
artificial intelligence + essay

these terms begin to form a connected lexical community.

Raw co-occurrence

For terms (a) and (b), define:

[
c_t(a,b)
]

as the number of headlines in time window (t) containing both terms.

Raw co-occurrence is simple but tends to favor highly frequent generic words.

Positive pointwise mutual information

A more informative edge weight is positive pointwise mutual information:

[
\operatorname{PPMI}(a,b)

\max\left(
0,
\log
\frac{P(a,b)}
{P(a)P(b)}
\right)
]

PPMI gives more importance to term pairs that occur together more frequently than expected from their individual frequencies.

This reduces the dominance of generic terms such as:

company
market
technology
business

Example graph-construction code

import networkx as nx
from collections import Counter
from itertools import combinations
from math import log
def build_graph(rows, min_count=5):
    term_counts = Counter()
    pair_counts = Counter()
    n_documents = len(rows)
    for phrases in rows:
        unique_terms = set(phrases)
        term_counts.update(unique_terms)
        pair_counts.update(
            combinations(sorted(unique_terms), 2)
        )
    graph = nx.Graph()
    for term, count in term_counts.items():
        if count >= min_count:
            graph.add_node(
                term,
                count=count,
            )
    for (term_a, term_b), pair_count in pair_counts.items():
        if term_a not in graph or term_b not in graph:
            continue
        probability_ab = pair_count / n_documents
        probability_a = term_counts[term_a] / n_documents
        probability_b = term_counts[term_b] / n_documents
        ppmi = max(
            0,
            log(
                probability_ab
                / (probability_a * probability_b)
            ),
        )
        if ppmi > 0:
            graph.add_edge(
                term_a,
                term_b,
                weight=ppmi,
                count=pair_count,
            )
    return graph

Build one graph for each week or month.

graphs = {}
for period, group in df.groupby("week"):
    graphs[period] = build_graph(
        group["phrases"],
        min_count=5,
    )

⸻

6. Detect Keyword Communities

Apply an unsupervised graph-community detection algorithm to each time-window graph.

Suitable algorithms include:

* Leiden;
* Louvain;
* Infomap.

Each detected community represents a candidate theme.

Louvain example

from networkx.algorithms.community import louvain_communities
communities = louvain_communities(
    graph,
    weight="weight",
    resolution=1.0,
    seed=42,
)

A community discovered in December 2022 might contain terms resembling:

chatbot
OpenAI
language model
write
essay
answer
human
artificial intelligence
Microsoft

The algorithm does not need to call this community “generative AI.” It only needs to discover that these words form a coherent and increasingly prominent group.

Community representation

Represent every community using:

* its highest-frequency terms;
* its highest-centrality terms;
* its strongest internal edges;
* the headlines most closely associated with it;
* its prevalence in each time window.

Useful node-ranking measures include:

* weighted degree;
* PageRank;
* betweenness centrality;
* eigenvector centrality.

⸻

7. Calculate Term Burst Scores

Frequency alone is not sufficient. Artificial intelligence may already have been covered for many years.

The method should identify terms whose frequency increased unexpectedly relative to the historical baseline.

For term (w) in time window (t), define:

[
B(w,t)

\log
\frac{f(w,t)+\alpha}
{\mathbb{E}[f(w,t)\mid\text{historical baseline}]+\alpha}
]

where:

* (f(w,t)) is the frequency of term (w) in window (t);
* the denominator is its expected frequency based on earlier periods;
* (\alpha) is a smoothing parameter.

A simple expectation can be based on the average frequency in previous windows.

A more robust expectation can account for:

* the total number of headlines in each period;
* long-term trends;
* seasonality;
* variance in historical frequency.

Possible statistical models include:

* Poisson regression;
* negative-binomial regression;
* exponentially weighted moving averages;
* Kleinberg burst detection;
* Bayesian change-point models.

⸻

8. Detect Emerging Term Associations

The most important signal may not be the appearance of new words. It may be the formation of new associations between existing words.

Terms such as:

artificial intelligence
model
language
writing
Microsoft

existed before November 2022.

What may have changed is that they began appearing together with terms such as:

chatbot
answer
essay
OpenAI
human

Therefore, compute burst scores for graph edges as well as individual nodes.

For a term pair ((a,b)), define:

[
B(a,b,t)

\log
\frac{c_t(a,b)+\alpha}
{\overline{c}_{\text{baseline}}(a,b)+\alpha}
]

where:

* (c_t(a,b)) is the pair’s co-occurrence count in period (t);
* (\overline{c}_{\text{baseline}}(a,b)) is its historical expected co-occurrence;
* (\alpha) is a smoothing constant.

Edge-burst detection can reveal newly emerging conceptual structures, even when the individual words are not new.

This is particularly useful for discovering a transition such as:

artificial intelligence
        ↓
language model
        ↓
chatbot — writing — answer — essay — OpenAI

⸻

9. Measure Community Coherence

A candidate community should not only be growing. Its terms should also be strongly related to one another.

Possible coherence measures include:

* average internal edge weight;
* graph density;
* weighted graph density;
* ratio of internal to external edge weights;
* average pairwise semantic similarity;
* topic coherence measures such as NPMI.

A simple graph-based coherence score is:

[
\operatorname{Coherence}(C,t)

\frac{
\sum_{a,b \in C} w_t(a,b)
}{
|C|(|C|-1)/2
}
]

where (w_t(a,b)) is the edge weight between terms (a) and (b).

A high score indicates that the terms form a tightly connected theme rather than an arbitrary collection of frequently occurring words.

⸻

10. Measure Community Novelty

A community is novel when its terms or relationships were uncommon during the historical baseline.

Novelty can be measured through:

* the percentage of terms absent from earlier windows;
* low similarity to previous communities;
* the share of internal edges that are new;
* divergence between current and historical term distributions.

A simple vocabulary-novelty score is:

[
\operatorname{VocabularyNovelty}(C,t)

\frac{
|{w \in C : f(w,\text{baseline}) \approx 0}|
}{
|C|
}
]

An edge-novelty score can be defined as:

[
\operatorname{EdgeNovelty}(C,t)

\frac{
|{(a,b)\in E_C : c_{\text{baseline}}(a,b)\approx 0}|
}{
|E_C|
}
]

Edge novelty may be more informative than vocabulary novelty because emerging themes often reorganize existing vocabulary rather than introducing only new words.

⸻

11. Track Communities Across Time

Communities should be matched across adjacent weeks or months.

This allows the system to distinguish:

* established topics;
* growing topics;
* declining topics;
* newly emerging topics;
* topics splitting into subtopics;
* topics merging with other topics;
* temporary noise.

Jaccard similarity

For communities (C_t) and (C_{t-1}), calculate:

[
J(C_t,C_{t-1})

\frac{
|C_t \cap C_{t-1}|
}{
|C_t \cup C_{t-1}|
}
]

A weighted version can account for term importance or frequency.

Alternative matching methods

Community matching can also use:

* cosine similarity between term-weight vectors;
* semantic similarity between community embeddings;
* optimal bipartite matching;
* graph alignment;
* overlap of associated headlines.

The generative AI topic may appear as:

* a completely new community;
* a branch of an older artificial-intelligence community;
* a merger between AI, technology, education, and media-related communities.

The tracking method should allow all three possibilities.

⸻

12. Measure Persistence

A meaningful emerging theme should usually persist for more than one isolated period.

For example, assign a higher score to a community that:

* first appears in late November;
* expands in early December;
* remains visible in late December;
* continues into January.

Assign a lower score to a community that appears in only one week and then disappears.

Persistence can be measured as:

[
\operatorname{Persistence}(C)

\text{number of consecutive windows in which the community is observed}
]

A weighted version can account for changes in community strength over time.

Persistence should not be an absolute requirement because genuinely important events may produce short bursts. It should instead be one component of the final ranking.

⸻

13. Calculate a Composite Emergence Score

Rank every community using a combination of:

1. frequency growth;
2. term burst;
3. edge burst;
4. vocabulary novelty;
5. edge novelty;
6. community coherence;
7. persistence;
8. distinctiveness from previous topics.

A simple formulation is:

[
S(C,t)

\operatorname{Coherence}(C,t)
\times
\operatorname{Burst}(C,t)
\times
\operatorname{Novelty}(C,t)
]

A more flexible weighted score is:

[
S(C,t)

\beta_1 G(C,t)
+
\beta_2 B_{\text{term}}(C,t)
+
\beta_3 B_{\text{edge}}(C,t)
+
\beta_4 N(C,t)
+
\beta_5 H(C,t)
+
\beta_6 P(C,t)
]

where:

* (G(C,t)) is frequency growth;
* (B_{\text{term}}(C,t)) is average term burst;
* (B_{\text{edge}}(C,t)) is average edge burst;
* (N(C,t)) is novelty;
* (H(C,t)) is coherence;
* (P(C,t)) is persistence;
* (\beta_1,\ldots,\beta_6) are weights.

To keep the method unsupervised, the weights should not be selected using a manually labeled generative-AI topic.

Possible unsupervised approaches include:

* equal weights after standardization;
* principal-component analysis;
* rank aggregation;
* anomaly detection over the feature vectors;
* Pareto ranking.

⸻

14. Rank the Emerging Communities

For each week or month:

1. detect all communities;
2. calculate their emergence features;
3. rank them by the composite score;
4. retain the highest-ranked communities;
5. inspect their top terms and representative headlines.

Example output:

Rank	Period	Top terms	Burst	Novelty	Coherence	Persistence
1	2022-W48	chatbot, OpenAI, answer, writing, model	High	High	High	5 weeks
2	2022-W49	oil, price cap, Russia, sanctions	High	Medium	High	8 weeks
3	2022-W50	inflation, rates, Federal Reserve	Medium	Low	High	30 weeks

The human analyst can then recognize the first community as corresponding to the emerging generative AI theme.

The analyst should perform this interpretation only after the ranking is complete.

⸻

15. Retrieve Representative Headlines

For each community, identify the headlines that best represent it.

A simple score is the number or weight of community terms appearing in a headline:

[
R(h,C)

\sum_{w \in h \cap C}
\operatorname{Importance}(w,C)
]

Possible term-importance measures include:

* TF-IDF;
* c-TF-IDF;
* weighted degree;
* PageRank;
* burst score.

Return the top five to twenty headlines for each community.

This makes the results interpretable and helps determine whether the lexical community corresponds to a coherent news theme.

⸻

16. Add Headline-Level Semantic Clustering

The keyword-graph approach is highly interpretable, but it can miss relationships between headlines that use different vocabulary.

A complementary method is to cluster complete headlines semantically.

Pipeline

1. Generate an embedding for each headline.
2. Reduce dimensionality if necessary.
3. Cluster the embeddings.
4. Extract keywords for every cluster.
5. calculate cluster prevalence by week.
6. rank clusters by temporal emergence.

Suitable methods include:

* sentence embeddings;
* UMAP;
* HDBSCAN;
* BERTopic;
* Top2Vec.

Example conceptual pipeline

headline_embeddings = embedding_model.encode(
    df["headline"].tolist()
)
reduced_embeddings = umap_model.fit_transform(
    headline_embeddings
)
cluster_labels = hdbscan_model.fit_predict(
    reduced_embeddings
)
df["cluster"] = cluster_labels

For every cluster, calculate:

* number of headlines per week;
* growth relative to the historical baseline;
* novelty;
* semantic coherence;
* persistence.

Generate cluster keywords using c-TF-IDF or a similar unsupervised method.

⸻

17. Recommended Hybrid Method

The strongest practical design combines two independent views.

View 1: Headline-level semantic clustering

This groups headlines with similar meanings even when they use different words.

Advantages:

* captures synonyms;
* captures paraphrases;
* handles short headlines better than traditional topic models;
* can connect semantically related coverage.

View 2: Keyword co-occurrence communities

This reveals the words and phrases defining each theme.

Advantages:

* interpretable;
* easy to visualize;
* suitable for temporal network analysis;
* exposes newly emerging relationships between terms.

Hybrid procedure

1. Cluster headlines semantically.
2. Build a keyword graph within each semantic cluster.
3. Extract lexical communities and representative phrases.
4. calculate temporal growth for both semantic clusters and lexical communities.
5. retain themes detected by both methods.
6. rank themes by burst, novelty, coherence, and persistence.

Agreement between the two methods provides stronger evidence that the discovered topic is real rather than an artifact of one modeling technique.

⸻

18. Validation Without Using Generative-AI Keywords During Discovery

The discovery stage must remain fully unsupervised.

However, evaluation can occur after discovery.

Qualitative validation

For the highest-ranked communities:

* inspect the top terms;
* inspect representative headlines;
* inspect the first appearance date;
* inspect the community’s development over time;
* determine whether the community is coherent.

Retrospective evaluation

After the unsupervised pipeline has produced its results, it is acceptable to compare them against a retrospectively constructed generative-AI reference set.

Possible evaluation questions include:

* Was the relevant community among the top-ranked emerging themes?
* When was it first detected?
* How many weeks before or after November 30, 2022 was it detected?
* How pure was the community?
* How much relevant coverage did it capture?
* Did it remain distinct from the broader technology or artificial-intelligence theme?

The reference keywords or labels must not influence:

* preprocessing;
* phrase extraction;
* clustering;
* community detection;
* feature construction;
* emergence ranking.

They may only be used after discovery for evaluation.

⸻

19. Robustness Checks

Test whether the result remains stable under different methodological choices.

Time-window robustness

Compare:

* daily;
* weekly;
* biweekly;
* monthly windows.

Frequency-threshold robustness

Compare different minimum term frequencies:

3
5
10
20

Community-resolution robustness

Run the graph algorithm with different resolution parameters.

Preprocessing robustness

Compare:

* tokens only;
* nouns and proper nouns;
* noun phrases;
* named entities;
* unigrams plus bigrams;
* unigrams plus bigrams and trigrams.

Baseline robustness

Compare:

* previous three months;
* previous six months;
* previous twelve months;
* January 2021–October 2022.

Model robustness

Compare:

* Louvain;
* Leiden;
* Infomap;
* HDBSCAN semantic clustering;
* BERTopic.

The generative AI community should remain broadly detectable under several reasonable configurations.

⸻

20. Visualize the Results

Useful visualizations include:

Emerging-topic timeline

Display the prevalence or emergence score of each leading community over time.

Keyword network

For every selected period:

* size nodes by frequency or centrality;
* size edges by association strength;
* group nodes by detected community;
* highlight new nodes and edges.

Community evolution diagram

Show how communities:

* appear;
* disappear;
* merge;
* split;
* grow;
* decline.

Term-burst heatmap

Rows represent terms, columns represent weeks, and values represent burst scores.

Edge-burst table

Show the strongest newly emerging associations, for example:

Term A	Term B	Baseline co-occurrence	Current co-occurrence	Edge burst
chatbot	writing	0	14	High
OpenAI	answer	0	11	High
language model	essay	1	9	High

These examples illustrate the desired output structure. They should not be inserted as predefined target pairs.

⸻

21. Approaches to Avoid

Do not run ordinary LDA only on November–December 2022

This will identify the largest topics in those months, not necessarily the newly emerging topics.

Do not rank terms only by TF-IDF

TF-IDF identifies terms that are distinctive within documents or groups, but it does not necessarily identify coherent or temporally emerging themes.

Do not analyze 2022 as one undifferentiated period

Combining the full year removes the temporal structure needed to detect emergence.

Do not force a fixed number of topics without testing it

A small emerging theme can disappear when the model is forced into a small number of broad topics.

Density-based clustering and graph-community detection are generally more appropriate.

Do not remove all rare terms

An emerging topic often begins with new names, products, organizations, and phrases.

Do not evaluate success based only on the phrase “generative AI”

The terminology may not have been common at the start of the phenomenon.

Success means identifying a coherent, rapidly emerging community corresponding to the phenomenon, even if the community uses different language.

Do not use future information during discovery

Avoid using terminology or labels that became popular after the target period to guide preprocessing or ranking.

⸻

22. End-to-End Algorithm

The complete process is:

1. Load all Bloomberg headlines and publication dates.
2. Remove duplicates, malformed records, and boilerplate.
3. Divide the corpus into weekly or monthly windows.
4. Extract nouns, proper nouns, entities, bigrams, trigrams,
   and noun phrases without using a topic dictionary.
5. Build a term co-occurrence graph for each time window.
6. Weight graph edges using PPMI or another association measure.
7. Detect lexical communities using Leiden, Louvain, or Infomap.
8. Compute term-frequency bursts relative to the historical baseline.
9. Compute edge bursts relative to the historical baseline.
10. Calculate community growth, novelty, coherence,
    distinctiveness, and persistence.
11. Match communities across adjacent time windows.
12. Rank communities using a composite emergence score.
13. Retrieve the top terms and representative headlines
    for the highest-ranked communities.
14. Independently cluster complete headlines using semantic embeddings
    and density-based clustering.
15. Compare the semantic clusters with the lexical communities.
16. Retain themes supported by both views.
17. Inspect the highest-ranked themes and assign human-readable labels.
18. Only after discovery, evaluate whether one of the themes
    corresponds to the emergence of generative AI.

⸻

23. Suggested Output Schema

Store one row per community and time window.

period
community_id
parent_community_id
headline_count
top_terms
top_headlines
frequency_growth
mean_term_burst
mean_edge_burst
vocabulary_novelty
edge_novelty
coherence
persistence
historical_similarity
emergence_score

This structure supports:

* ranking;
* visualization;
* community tracking;
* manual review;
* retrospective evaluation.

⸻

24. Minimal Viable Implementation

A first implementation can use:

* weekly time windows;
* noun phrases and named entities;
* PPMI term graphs;
* Louvain community detection;
* average term burst;
* average edge burst;
* graph-density coherence;
* Jaccard matching across weeks;
* equal-weight rank aggregation.

The first version does not need a complex dynamic-topic model.

A practical minimal pipeline is:

phrase extraction
      ↓
weekly co-occurrence graphs
      ↓
Louvain communities
      ↓
term and edge burst scores
      ↓
community tracking
      ↓
emergence ranking
      ↓
manual interpretation

⸻

25. Final Recommendation

For this problem, the recommended primary approach is:

Temporal keyword co-occurrence networks combined with community detection, term-burst detection, edge-burst detection, and community tracking.

The recommended secondary approach is:

Semantic headline clustering with HDBSCAN or BERTopic, used as an independent validation layer.

The key methodological point is that generative AI should not be searched for directly.

Instead, the system should search for:

* unusually fast-growing lexical communities;
* new relationships between existing terms;
* coherent groups of related headlines;
* themes that persist across consecutive periods;
* communities that differ substantially from their historical predecessors.

The generative AI theme should then emerge as one of the highest-ranked temporal anomalies around November–December 2022.