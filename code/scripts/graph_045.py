"""Example knowledge graph of notebook 0.45 (schema 0.4) and its standalone interactive HTML page.

    from graph_045 import build_graph, to_html
    G = build_graph(news, outs)                       # networkx MultiDiGraph
    Path("grafo.html").write_text(to_html(G, news, title="..."))

Nodes are merged across news only when type and normalised mention coincide: lower case, no punctuation,
no possessive, hyphens as spaces, no company suffix such as Inc or Corp, and — for non-company nodes only —
a naive plural ("Obesity Drugs" = "Obesity-Drug"). There is no entity linking: "AI" and "Artificial
Intelligence" stay two nodes, as do "Novo" and "Novo Nordisk". Every edge keeps the four attributes of
schema 0.4 — date, modality, evidence and the news it comes from — so two news stating the same fact give
two parallel edges.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

import networkx as nx

NODE_STYLE = {  # type -> (Italian label, colour, vis-network shape)
    "COMPANY": ("azienda", "#2563eb", "dot"),
    "PRODUCT": ("prodotto", "#ea580c", "diamond"),
    "CONCEPT": ("concetto", "#9333ea", "star"),
    "SECTOR": ("settore", "#059669", "square"),
    "ORGANIZATION": ("organizzazione", "#64748b", "triangle"),
    "LOCATION": ("luogo", "#b45309", "hexagon"),
}
EDGE_STYLE = {  # predicate -> (Italian label, colour)
    "OFFERS": ("offre", "#ea580c"),
    "ACTIVE_IN": ("attiva in", "#9333ea"),
    "EXPOSED_TO": ("esposta a", "#db2777"),
    "INVESTS_IN": ("investe in", "#16a34a"),
    "ACQUIRES": ("acquisisce", "#dc2626"),
    "PARTNERS_WITH": ("partner di", "#0891b2"),
    "SUPPLIES_TO": ("fornisce a", "#ca8a04"),
}
MODALITY_STYLE = {  # modality -> (Italian label, vis dashes)
    "reported": ("fatto", False),
    "planned": ("piano", [10, 6]),
    "uncertain": ("incerto", [3, 5]),
    "negated": ("negato", [2, 9]),
}
SUFFIX = {"inc", "corp", "corporation", "co", "ltd", "plc", "sa", "ag", "nv", "se", "llc"}


def node_key(mention: str, node_type: str) -> str:
    s = unicodedata.normalize("NFKD", mention).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " ", s)                 # "Weight-Loss" = "Weight Loss"
    words = [w.strip(".-+") for w in re.sub(r"[^a-z0-9&.+-]+", " ", s).split()]
    words = [w for w in words if w]
    if node_type == "COMPANY" and len(words) > 1:
        words = [w for w in words if w not in SUFFIX] or words
    if node_type != "COMPANY":                                          # naive plural: drugs -> drug, chips -> chip
        words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")) else w for w in words]
    return f"{node_type}:{' '.join(words)}"


def build_graph(news, outs) -> nx.MultiDiGraph:
    """news: records with id, date, headline; outs: the schema 0.4 extraction ({nodes, edges}) of each news."""
    G = nx.MultiDiGraph()
    for n, o in zip(news, outs):
        local = {}
        for x in o.get("nodes") or []:
            k = node_key(x["mention"], x["type"])
            local[x["id"]] = k
            if k not in G:
                G.add_node(k, type=x["type"], mentions=Counter(), news=set())
            G.nodes[k]["mentions"][x["mention"]] += 1
            G.nodes[k]["news"].add(n["id"])
        for e in o.get("edges") or []:
            s, t = local.get(e["subject_id"]), local.get(e["object_id"])
            if s is None or t is None or s == t:
                continue
            G.add_edge(s, t, predicate=e["predicate"], modality=e["modality"], date=n["date"],
                       evidence=e["evidence"], news=n["id"], headline=n["headline"])
    for _, d in G.nodes(data=True):
        d["label"] = d["mentions"].most_common(1)[0][0]
    return G


def companies_linked(G: nx.MultiDiGraph, node) -> set:
    """Distinct companies connected to a node by any edge, in either direction."""
    nb = set(G.predecessors(node)) | set(G.successors(node))
    return {m for m in nb if G.nodes[m]["type"] == "COMPANY"}


def to_html(G: nx.MultiDiGraph, news, title: str, subtitle: str = "", about: str = "") -> str:
    """A single self-contained page (vis-network from jsDelivr, data inline). `about` is trusted HTML shown in the
    sidebar ("Come è stato costruito"): extractor, ontology, how the news were selected."""
    dates = {n["id"]: n["date"] for n in news}
    nodes = []
    for k, d in G.nodes(data=True):
        lab, colour, shape = NODE_STYLE[d["type"]]
        comp = len(companies_linked(G, k))
        nodes.append(dict(id=k, label=d["label"], type=d["type"], typeLabel=lab, color=colour, shape=shape,
                          mentions=sorted(d["mentions"]), news=sorted(d["news"]), first=min(dates[i] for i in d["news"]),
                          companies=comp, degree=G.degree(k)))
    edges = []
    for i, (s, t, d) in enumerate(G.edges(data=True)):
        plab, colour = EDGE_STYLE[d["predicate"]]
        mlab, dashes = MODALITY_STYLE.get(d["modality"], (d["modality"], False))
        edges.append(dict(id=f"e{i}", source=s, target=t, predicate=d["predicate"], predLabel=plab, color=colour,
                          modality=d["modality"], modLabel=mlab, dashes=dashes, date=d["date"], evidence=d["evidence"],
                          news=d["news"], headline=d["headline"], symmetric=d["predicate"] == "PARTNERS_WITH"))
    payload = dict(nodes=nodes, edges=edges, news={n["id"]: dict(date=n["date"], headline=n["headline"]) for n in news},
                   nodeStyle={k: dict(label=v[0], color=v[1], shape=v[2]) for k, v in NODE_STYLE.items()},
                   edgeStyle={k: dict(label=v[0], color=v[1]) for k, v in EDGE_STYLE.items()},
                   modalityStyle={k: dict(label=v[0], dashes=v[1]) for k, v in MODALITY_STYLE.items()})
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__ABOUT__", about)
            .replace("__DATA__", data))


TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grafo di esempio</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --fg: #1f2328; --muted: #6b7280; --line: #e5e7eb; --accent: #2563eb;
  --canvas: #fbfbfa; --chip: #f1f5f9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #111317; --panel: #181b21; --fg: #e6e8eb; --muted: #9aa3ae; --line: #2a2f37; --accent: #60a5fa;
    --canvas: #14171c; --chip: #222731;
  }
}
:root[data-theme="dark"] {
  --bg: #111317; --panel: #181b21; --fg: #e6e8eb; --muted: #9aa3ae; --line: #2a2f37; --accent: #60a5fa;
  --canvas: #14171c; --chip: #222731;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body { background: var(--bg); color: var(--fg); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.app { display: grid; grid-template-columns: 300px 1fr; height: 100vh; }
aside { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; padding: 16px; }
main { position: relative; min-width: 0; }
#net { position: absolute; inset: 0; background: var(--canvas); }
h1 { font-size: 17px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 12.5px; margin: 0 0 12px; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 16px 0 6px; }
.row { display: flex; align-items: center; gap: 8px; padding: 3px 0; cursor: pointer; user-select: none; }
.row input { margin: 0; accent-color: var(--accent); }
.row .n { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }
.sw { width: 14px; height: 14px; flex: none; display: inline-block; }
.sw.dot { border-radius: 50%; } .sw.square { border-radius: 2px; }
.sw.diamond { transform: rotate(45deg) scale(.8); border-radius: 2px; }
.sw.star { clip-path: polygon(50% 0, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%); }
.sw.triangle { clip-path: polygon(50% 0, 100% 100%, 0 100%); }
.sw.hexagon { clip-path: polygon(25% 5%, 75% 5%, 100% 50%, 75% 95%, 25% 95%, 0 50%); }
.ln { width: 26px; height: 0; border-top: 3px solid; flex: none; }
.ln.planned { border-top-style: dashed; } .ln.uncertain { border-top-style: dotted; } .ln.negated { border-top: 3px dotted #9ca3af; }
.time { display: flex; align-items: center; gap: 8px; }
.time input[type=range] { flex: 1; accent-color: var(--accent); }
button, input[type=search] {
  font: inherit; color: var(--fg); background: var(--chip); border: 1px solid var(--line); border-radius: 6px; padding: 5px 9px;
}
button { cursor: pointer; } button:hover { border-color: var(--accent); }
input[type=search] { width: 100%; }
#when { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 64px; }
.counts { color: var(--muted); font-size: 12px; margin-top: 6px; }
#details { margin-top: 8px; font-size: 13px; }
#details .k { color: var(--muted); font-size: 12px; }
#details .ev { background: var(--chip); border-radius: 6px; padding: 6px 8px; margin: 4px 0 8px; }
#details ul { list-style: none; padding: 0; margin: 4px 0; }
#details li { padding: 5px 0; border-top: 1px solid var(--line); }
.pill { display: inline-block; border-radius: 999px; padding: 0 7px; font-size: 11.5px; color: #fff; }
.hint { color: var(--muted); font-size: 12px; }
.about { background: var(--chip); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin: 0 0 12px; font-size: 12.5px; }
.about summary { cursor: pointer; font-weight: 600; }
.about p { margin: 6px 0; }
.about code { font-size: 11.5px; background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 0 3px; }
.about .warn { border-left: 3px solid #d97706; padding-left: 7px; }
@media (max-width: 800px) {
  .app { grid-template-columns: 1fr; grid-template-rows: auto 70vh; height: auto; }
  aside { border-right: none; border-bottom: 1px solid var(--line); max-height: none; }
  main { height: 70vh; }
}
</style>
</head>
<body>
<div class="app">
<aside>
  <h1>__TITLE__</h1>
  <p class="sub">__SUBTITLE__</p>
  <details class="about" open><summary>Come è stato costruito</summary>__ABOUT__</details>
  <input id="search" type="search" placeholder="Cerca un nodo (es. Nvidia, AI, Wegovy)">
  <h2>Data della news</h2>
  <div class="time"><button id="play" title="Mostra la crescita mese per mese">▶</button>
    <input id="slider" type="range" min="0" value="0"><span id="when"></span></div>
  <div class="counts" id="counts"></div>
  <h2>Nodi</h2><div id="nodeLegend"></div>
  <label class="row"><input type="checkbox" id="isolated"> mostra i nodi senza archi<span class="n" id="isoN"></span></label>
  <h2>Archi</h2><div id="edgeLegend"></div>
  <h2>Modalità (attributo dell'arco)</h2><div id="modLegend"></div>
  <label class="row"><input type="checkbox" id="edgeLabels"> scrivi il tipo sugli archi</label>
  <h2>Dettaglio</h2>
  <div id="details"><span class="hint">Clicca un nodo o una freccia per vederne gli attributi.</span></div>
</aside>
<main><div id="net" role="img" aria-label="Grafo di conoscenza interattivo"></div></main>
</div>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// months between the first and the last news
const allDates = DATA.edges.map(e => e.date).concat(DATA.nodes.map(n => n.first)).sort();
const months = [];
{ let [y, m] = allDates[0].slice(0, 7).split("-").map(Number); const [Y, M] = allDates.at(-1).slice(0, 7).split("-").map(Number);
  while (y < Y || (y === Y && m <= M)) { months.push(`${y}-${String(m).padStart(2, "0")}`); m++; if (m > 12) { m = 1; y++; } } }
const MONTH_IT = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"];
const slider = document.getElementById("slider");
slider.max = months.length - 1; slider.value = months.length - 1;

const show = { node: Object.fromEntries(Object.keys(DATA.nodeStyle).map(k => [k, true])),
               edge: Object.fromEntries(Object.keys(DATA.edgeStyle).map(k => [k, true])),
               mod: Object.fromEntries(Object.keys(DATA.modalityStyle).map(k => [k, true])) };

function legend(el, entries, kind, swatch) {
  el.innerHTML = entries.map(([k, v, n]) => `<label class="row"><input type="checkbox" data-kind="${kind}" data-key="${k}" checked>
    ${swatch(k, v)}<span>${esc(v.label)}${kind === "node" || kind === "edge" ? ` <span class="hint">${k}</span>` : ""}</span><span class="n">${n}</span></label>`).join("");
}
const count = (arr, f) => arr.reduce((c, x) => (c[f(x)] = (c[f(x)] || 0) + 1, c), {});
const nNode = count(DATA.nodes, n => n.type), nEdge = count(DATA.edges, e => e.predicate), nMod = count(DATA.edges, e => e.modality);
legend(document.getElementById("nodeLegend"), Object.entries(DATA.nodeStyle).map(([k, v]) => [k, v, nNode[k] || 0]), "node",
       (k, v) => `<span class="sw ${v.shape}" style="background:${v.color}"></span>`);
legend(document.getElementById("edgeLegend"), Object.entries(DATA.edgeStyle).map(([k, v]) => [k, v, nEdge[k] || 0]), "edge",
       (k, v) => `<span class="ln" style="border-top-color:${v.color}"></span>`);
legend(document.getElementById("modLegend"), Object.entries(DATA.modalityStyle).map(([k, v]) => [k, v, nMod[k] || 0]), "mod",
       (k, v) => `<span class="ln ${k}" style="border-top-color:${k === "negated" ? "#9ca3af" : css("--fg")}"></span>`);

const nodeById = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));
const nodesDS = new vis.DataSet(DATA.nodes.map(n => ({
  id: n.id, label: n.label, shape: n.shape, size: 10 + 5 * Math.sqrt(n.companies + n.degree / 2),
  color: { background: n.color, border: n.color, highlight: { background: n.color, border: css("--fg") } },
  font: { color: css("--fg"), size: 13, strokeWidth: 3, strokeColor: css("--canvas") },
  title: `${n.label} · ${n.typeLabel} · ${n.news.length} news`,
})));
const edgesDS = new vis.DataSet(DATA.edges.map(e => ({
  id: e.id, from: e.source, to: e.target, dashes: e.dashes, width: e.modality === "negated" ? 1.5 : 2.2,
  color: { color: e.modality === "negated" ? "#9ca3af" : e.color, highlight: e.color, opacity: .9 },
  arrows: { to: { enabled: !e.symmetric, scaleFactor: .6 } },
  title: `${e.predicate} · ${e.modLabel} · ${e.date}`, smooth: { type: "continuous" },
  font: { size: 10, color: css("--muted"), strokeWidth: 0, align: "middle" },
})));

let cutoff = months.at(-1) + "-31";
const edgeById = Object.fromEntries(DATA.edges.map(e => [e.id, e]));
const edgeVisible = e => e.date <= cutoff && show.edge[e.predicate] && show.mod[e.modality] &&
  show.node[nodeById[e.source].type] && show.node[nodeById[e.target].type];
function nodeVisible(n) {
  if (!show.node[n.type] || n.first > cutoff) return false;
  if (document.getElementById("isolated").checked) return true;
  return DATA.edges.some(e => (e.source === n.id || e.target === n.id) && edgeVisible(e));
}
const nodesView = new vis.DataView(nodesDS, { filter: x => nodeVisible(nodeById[x.id]) });
const edgesView = new vis.DataView(edgesDS, { filter: x => edgeVisible(edgeById[x.id]) });

const net = new vis.Network(document.getElementById("net"), { nodes: nodesView, edges: edgesView }, {
  physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -60, springLength: 90, avoidOverlap: .4 },
             stabilization: { iterations: 400 } },
  interaction: { hover: true, tooltipDelay: 120, navigationButtons: false, keyboard: true },
});
net.once("stabilizationIterationsDone", () => { net.setOptions({ physics: { enabled: false } }); net.fit({ animation: false }); });

function refresh() {
  nodesView.refresh(); edgesView.refresh();
  const [y, m] = months[slider.value].split("-");
  document.getElementById("when").textContent = `${MONTH_IT[+m - 1]} ${y}`;
  const vn = nodesView.getIds(), ve = edgesView.getIds();
  const iso = DATA.nodes.filter(n => !DATA.edges.some(e => e.source === n.id || e.target === n.id)).length;
  document.getElementById("isoN").textContent = iso;
  document.getElementById("counts").textContent = `${vn.length} nodi · ${ve.length} archi visibili · news fino a fine ${MONTH_IT[+m - 1]} ${y}`;
}
slider.addEventListener("input", () => { cutoff = months[slider.value] + "-31"; refresh(); });
document.querySelectorAll("aside input[type=checkbox][data-kind]").forEach(cb =>
  cb.addEventListener("change", () => { show[cb.dataset.kind][cb.dataset.key] = cb.checked; refresh(); }));
document.getElementById("isolated").addEventListener("change", refresh);
document.getElementById("edgeLabels").addEventListener("change", ev =>
  edgesDS.update(DATA.edges.map(e => ({ id: e.id, label: ev.target.checked ? e.predLabel : undefined }))));

let timer = null;
document.getElementById("play").addEventListener("click", ev => {
  if (timer) { clearInterval(timer); timer = null; ev.target.textContent = "▶"; return; }
  ev.target.textContent = "❚❚"; slider.value = 0;
  timer = setInterval(() => {
    cutoff = months[slider.value] + "-31"; refresh();
    if (+slider.value >= months.length - 1) { clearInterval(timer); timer = null; ev.target.textContent = "▶"; return; }
    slider.value = +slider.value + 1;
  }, 900);
});

const pill = (text, color) => `<span class="pill" style="background:${color}">${esc(text)}</span>`;
function edgeLine(e, other, dir) {
  return `<li>${dir} ${pill(e.predLabel, e.color)} <b>${esc(nodeById[other].label)}</b>
    <div class="k">${e.date} · ${e.modLabel} · news ${e.news}</div><div class="ev">«${esc(e.evidence)}»</div></li>`;
}
net.on("click", params => {
  const box = document.getElementById("details");
  if (params.nodes.length) {
    const n = nodeById[params.nodes[0]];
    const out = DATA.edges.filter(e => e.source === n.id), inn = DATA.edges.filter(e => e.target === n.id);
    box.innerHTML = `<div>${pill(n.typeLabel, n.color)} <b>${esc(n.label)}</b></div>
      <div class="k">citato in ${n.news.length} news · prima volta ${n.first} · collegato a ${n.companies} aziende</div>
      ${n.mentions.length > 1 ? `<div class="k">scritto come: ${n.mentions.map(esc).join(", ")}</div>` : ""}
      <ul>${out.map(e => edgeLine(e, e.target, "→")).join("")}${inn.map(e => edgeLine(e, e.source, "←")).join("")}</ul>
      ${out.length + inn.length === 0 ? `<div class="hint">Nessun arco: la news lo cita ma non afferma una relazione.</div>` : ""}
      <div class="k">News:</div><ul>${n.news.map(i => `<li class="k">${DATA.news[i].date} · ${esc(DATA.news[i].headline)}</li>`).join("")}</ul>`;
  } else if (params.edges.length) {
    const e = edgeById[params.edges[0]];
    box.innerHTML = `<div><b>${esc(nodeById[e.source].label)}</b> ${pill(e.predLabel, e.color)} <b>${esc(nodeById[e.target].label)}</b></div>
      <div class="k">tipo: ${e.predicate}</div>
      <ul><li><span class="k">data</span><br>${e.date}</li><li><span class="k">modalità</span><br>${e.modLabel}</li>
      <li><span class="k">evidenza</span><div class="ev">«${esc(e.evidence)}»</div></li>
      <li><span class="k">news ${e.news}</span><br>${esc(e.headline)}</li></ul>`;
  }
});
document.getElementById("search").addEventListener("input", ev => {
  const q = ev.target.value.trim().toLowerCase();
  if (!q) { net.unselectAll(); return; }
  const hit = nodesView.get().find(n => n.label.toLowerCase().includes(q));
  if (hit) { net.selectNodes([hit.id]); net.focus(hit.id, { scale: 1.2, animation: { duration: 400 } }); }
});
refresh();
</script>
</body>
</html>
"""
