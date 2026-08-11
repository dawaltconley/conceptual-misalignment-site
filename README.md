# Mapping Conceptual Misalignment

A Digital Humanities project (dissertation MVP) that visualizes **conceptual
misalignment** between philosophical terms across languages. It compares core
Confucian virtue concepts — **仁 義 禮 智 信** (and more) — as they appear in the
classical Chinese _Mengzi_ against their English renderings in the **Stanford
Encyclopedia of Philosophy (SEP)**. Where a Chinese concept and its English
translations occupy different semantic neighborhoods, the misalignment between a
"thick" source concept and its "thin" translations becomes visible.

The site presents each concept through three complementary lenses:

- **Co-occurrence networks** — PMI graphs of what words cluster _around_ a term in
  each corpus (a _syntagmatic_ / topical view).
- **Similarity networks & embedding scatterplots** — contextual-embedding cosine
  neighborhoods and a PCA/t-SNE map of the whole vocabulary (a _paradigmatic_ /
  substitutability view), with Louvain communities.
- **Cross-lingual alignment** — an interactive Procrustes alignment of the two
  embedding spaces over user-chosen bilingual anchors.

---

## Architecture

Two phases: an offline Python pipeline that produces JSON, and a static Astro
site that visualizes it.

```
scripts/     Python pipeline (embeddings + co-occurrence)  → see scripts/README.md
public/      pipeline output the site loads at runtime (ctext/, sep/, embeddings/)
src/         Astro + React (D3) front end
notes/       methodology memos + verified references
analysis/    optional --artifacts dumps (PNG/CSV), not shipped to the site
```

### Phase 1 — the pipeline (`scripts/`)

One entry point, run from `scripts/`:

```bash
scripts/.venv/bin/python -m main            # both corpora + master index
python -m main --corpus sep --artifacts     # one corpus + analysis dump
```

Everything about _how_ a corpus is processed lives in `config.py` as
`MENGZI_PIPELINE` / `SEP_PIPELINE` (`Pipeline` objects): model, pooling, min
frequency, similarity-graph method, Louvain resolution, stopwords, etc. **The
CLI is small; tune runs by editing `config.py`, not by adding flags.** Full
details — layout, flags, method decisions — are in **`scripts/README.md`**.

Method in brief: contextual embeddings (`GujiRoBERTa` for Chinese,
`roberta-base` for English) are pooled to one vector per word (subword-pool
within an occurrence, then pool across occurrences), mean-centered to counter
anisotropy, and turned into a kNN / quantile cosine graph with Louvain
communities and a PCA-reduced scatter. Chinese tokens/lemmas/POS come from the
gold **Kyoto UD `mengzi.conllu`** treebank; English from a spaCy HTML-tokenized
SEP scrape. PMI co-occurrence networks are built over the same spaCy docs.

**Outputs** (paths anchored in `config.py`):

```
public/ctext/{hanzi}_{source}.json    co-occurrence: Mengzi full + each chapter
public/ctext/{hanzi}_embeds.json      similarity: pruned cosine neighborhood
public/sep/{label}_{source}.json      co-occurrence: SEP combined + each article
public/sep/{label}_embeds.json        similarity network per English rendering
public/embeddings/{mengzi,sep}.json   PCA scatter (all vocab + communities)
src/data/terms.json                   master index: term → sources → file paths
```

### Phase 2 — the site (`src/`)

A single-page [Astro](https://astro.build/) app with React islands and D3.
`src/pages/index.astro` reads `src/data/terms.json` (validated with Zod) and
renders four sections:

- **`TermNetwork`** (`kind="cooccurrence"` / `"similarity"`) — two dropdowns
  (Chinese term + English rendering) driving side-by-side **`MultiNetwork`**s.
- **`EmbeddingScatter`** — the per-corpus PCA/t-SNE scatter, colored by community.
- **`AlignmentScatter`** — the cross-lingual Procrustes alignment.

`MultiNetwork` renders `Network` (D3 force layout — canvas edges, draggable DOM
nodes, edge width ∝ weight); Chinese nodes get CC-CEDICT tooltips via
`HanziNode` (dictionary built at Astro build time from bundled CC-CEDICT). Zod
schemas live in `src/lib/{terms,networkx,embeddings}.ts`.

---

## Adding / changing terms

Edit `TERMS` in `scripts/config.py` — a list of `Term`s, each with English
`Rendering`s (fnmatch-glob stems, kept disjoint so no token maps to two labels):

```python
Term('義', (
    Rendering('righteousness', 'righteous*'),
    Rendering('justice', 'justness'),
    Rendering('morality', 'moral*'),
)),
```

Then re-run the pipeline; the site reads whatever `terms.json` points to (no code
change needed). Currently active: **仁 義 禮 智 信** (others are present but
commented out).

---

## Building locally

**Prerequisites:** Node 20+ (`.nvmrc`), Python 3.11+. The pipeline uses a
ROCm build of PyTorch and a GPU; embeddings on CPU are slow but possible.

```bash
npm install                      # Node deps
scripts/init-env.sh              # ROCm torch + editable Python install + spaCy model
```

`init-env.sh` creates `scripts/.venv`, installs torch from the ROCm index first,
then the project (`pip install -e "..[dev]"` from the root `pyproject.toml`), then
`en_core_web_sm`.

```bash
cd scripts && .venv/bin/python -m main    # run the pipeline (HTTP-cached; optional —
                                          # output is committed under public/)
npm run dev                               # dev server → http://localhost:4321
npm run build                             # production build → dist/
npm run lint                              # astro check + stylelint + pyright + prettier
```

The `dist/` output is a static bundle; deploy to any static host.

---

## Repo conventions

- Run the pipeline as `scripts/.venv/bin/python -m main` **from the repo root**
  (or `python -m main` from `scripts/`); output paths are absolute.
- `notes/` holds methodology memos with verified references (anisotropy,
  register-dominant clustering, community ordering, glyph normalization, …).
- `segpos/` + `scripts/cli/segment.py` are a separate XunziALLM segmentation
  experiment, independent of the main pipeline.
