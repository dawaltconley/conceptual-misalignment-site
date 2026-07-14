# Conceptual Misalignment — Proof of Concept

A Digital Humanities proof of concept that visualizes _conceptual misalignment_
between philosophical terms across languages. The current prototype compares
the classical Chinese concept **仁 (rén)** with its common English translations
— **benevolence** and **humaneness** — by surfacing the different semantic
neighborhoods each term occupies in its respective philosophical tradition.

The site places interactive co-occurrence networks side by side: on one side,
what concepts cluster around 仁 in the _Mengzi_; on the other, what concepts
cluster around "benevolence" or "humaneness" in the Stanford Encyclopedia of
Philosophy. Where the two networks diverge, conceptual misalignment is visible.

---

## How It Works

The project has two phases: an offline NLP pipeline that produces JSON data
files, and a static site that visualizes them.

### Phase 1 — NLP Pipeline (`scripts/`)

The pipeline is run once (or whenever source texts change) to generate the
co-occurrence network data the site consumes.

```
scripts/
├── main.py          # Entry point: orchestrates all fetching and NLP
├── config.py        # Define which terms to analyze (TERMS list)
├── scrape_sep.py    # Fetch and parse Stanford Encyclopedia of Philosophy articles
├── mengzi.py        # Fetch the Mengzi from the Chinese Text Project (ctext.org) API
├── cache.py         # SQLite HTTP cache (7-day TTL via requests-cache)
├── utils.py         # PMI graph construction, cosine similarity, graph pruning
└── nlp/
    ├── english.py   # spaCy tokenizer for English HTML
    └── chinese.py   # CLTK tokenizer for classical Chinese (lzh model)
```

**English text — Stanford Encyclopedia of Philosophy**

For each English translation of a term (e.g. "benevolence"), `scrape_sep.py`
queries the SEP search API and downloads the top-N articles. `nlp/english.py`
tokenizes each article's HTML using [spaCy](https://spacy.io/)
(`en_core_web_sm`) with an HTML-aware tokenizer. Only content words (nouns,
verbs, adjectives, proper nouns) are kept, and each token is reduced to its
lemma.

**Chinese text — the Mengzi**

`mengzi.py` fetches the full text and all 14 books of the Mengzi via the
[Chinese Text Project API](https://api.ctext.org). `nlp/chinese.py` tokenizes
classical Chinese using [CLTK](https://cltk.org/)'s classical Chinese (`lzh`)
model. A domain-specific stopword list removes function words, pronouns, and
common classical particles.

**Building co-occurrence networks**

`utils.py` builds a [NetworkX](https://networkx.org/) graph for each text/source:

1. A vocabulary is built by frequency-filtering tokens (minimum sentence
   frequency threshold).
2. For every pair of vocabulary tokens that appear in the same sentence, a
   **Pointwise Mutual Information (PMI)** score is computed. Only pairs with
   positive PMI (i.e. co-occurring more than chance) become edges.
3. The graph is pruned to the 15 nodes most proximate to the query term
   (prioritizing direct 1-hop neighbors by weight, then 2-hop neighbors by path
   product weight).
4. The resulting subgraph is serialized to [NetworkX node-link JSON
   format](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html).

**Output**

Serialized graphs are written to `src/data/`, where the Astro build can import
them:

```
src/data/
├── sep/
│   ├── benevolence.json      # Networks for each SEP source + combined
│   └── humaneness.json
└── ctext/
    └── 仁.json               # Networks for full Mengzi + each of the 14 books
```

### Phase 2 — Astro Site (`src/`)

The site is a single-page [Astro](https://astro.build/) app with React islands for interactivity.

```
src/
├── pages/index.astro          # Main page: imports JSON, renders both networks
├── components/
│   ├── SEPNetwork.tsx         # Source-switcher UI + layout wrapper
│   ├── Network.tsx            # D3 force-directed graph (canvas edges + DOM nodes)
│   └── HanziNode.tsx          # Chinese character nodes with CC-CEDICT tooltips
└── lib/
    ├── networkx.ts            # Zod schemas for validating the JSON data format
    └── build/
        └── cedict.ts          # Parses CC-CEDICT dictionary at build time
```

**`index.astro`** imports the two JSON files at build time, validates them with
Zod, and passes them as props to the React components. For the Chinese side, it
also pre-builds a dictionary of character definitions from the bundled
CC-CEDICT data (`src/data/cedict_1_0_ts_utf-8_mdbg.txt`).

**`SEPNetwork.tsx`** renders a row of source buttons (one per SEP article, or
one per Mengzi book) and swaps the active network when clicked.

**`Network.tsx`** renders a D3 force-directed layout. Edges are drawn on an
HTML5 Canvas; nodes are absolutely positioned DOM elements that can be dragged.
Edge thickness is proportional to PMI weight.

**`HanziNode.tsx`** wraps Chinese character nodes with a tooltip showing the
CC-CEDICT definition.

---

## Adding New Terms

Edit `scripts/config.py` and add a `Term` to the `TERMS` list:

```python
TERMS: list[Term] = [
    Term('仁', {'benevolence', 'humaneness'}),
    Term('義', {'righteousness', 'justice'}),   # example
]
```

Then re-run the pipeline and update `src/pages/index.astro` to import and
display the new data files.

---

## Building Locally

### Prerequisites

- Node.js 20+ (see `.nvmrc`)
- Python 3.11+

### 1. Install Node dependencies

```bash
npm install
```

### 2. Set up the Python environment

```bash
cd scripts
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The CLTK classical Chinese model will be downloaded automatically on first run.

### 3. Run the NLP pipeline

```bash
cd scripts
python main.py
```

This fetches texts (with SQLite caching at `scripts/.cache/http_cache.sqlite`),
runs NLP, and writes JSON to `src/data/`. Running it again will use cached HTTP
responses and complete in seconds.

The output files are already committed to the repository, so **this step is
optional** if you just want to run the site with the existing data.

### 4. Run the development server

```bash
npm run dev
```

The site will be available at `http://localhost:4321`.

### 5. Build for production

```bash
npm run build
npm run preview   # serves the built output locally
```

---

## Deployment

The static output in `dist/` can be served from any static host.
