# CLAUDE.md

Orientation for agents working in this repo. See `README.md` for the project
overview and `scripts/README.md` for pipeline internals.

## What this is

Dissertation MVP "Mapping Conceptual Misalignment": compares Confucian virtue
concepts (仁 義 禮 智 信 …) in the classical-Chinese _Mengzi_ against their English
SEP renderings, via contextual embeddings + PMI co-occurrence. Astro/React/D3
front end reads JSON produced by a Python pipeline. **MVP deadline 2026-08-15 —
favor working end-to-end over completeness.**

## Two lenses (keep them straight)

- **Co-occurrence / PMI** networks = _syntagmatic_ (words appearing together →
  topical relatedness). `cooccurrence/`.
- **Embedding similarity + scatter** = _paradigmatic_ (substitutable → same-type).
  `embeddings/`. Whole-vocab embedding clustering is **register/POS-dominated**,
  not topical — see `notes/claude/embedding-communities-and-semantics.md`.

## Running things

- Pipeline: **`scripts/.venv/bin/python -m main`** (run from repo root — no `cd`
  needed; output paths are absolute). Flags: `--corpus {mengzi,sep,all}`,
  `--per-term N`, `--artifacts`, `--master-only`.
- Dev server: `npm run dev` (→ `http://localhost:4321`). Build: `npm run build`.
- Lint/typecheck: `npm run lint` (astro check + stylelint + **pyright** + prettier);
  Python only: `npm run lint:python`.
- Env is ROCm GPU (torch device `cuda`); rebuild via `scripts/init-env.sh`.

## Where logic lives (`scripts/`)

- `main.py` — the one pipeline entry point (`run_mengzi` / `run_sep` /
  `build_master` + CLI + timing).
- `config.py` — `TERMS`/`Rendering`s, stopwords, and the `MENGZI_PIPELINE` /
  `SEP_PIPELINE` `Pipeline` objects that hold **all run tuning** (model, pooling,
  min_freq, sim-graph method, resolution, …). Prefer editing these over adding CLI flags.
- `models.py` — dataclasses (`Pipeline`, `Source/Rendering/Term`, `NetworkData`,
  `Embeddings`) + JSON serialization.
- `corpus/` (fetch/parse; only place that hits the network), `embeddings/`
  (`model`, `vectors`, `occurrences`, `analyze`, `layouts`), `cooccurrence/`,
  `graph/`, `segmentation/` + `cli/` (separate XunziALLM experiment).

Front end: `src/pages/index.astro` → `TermNetwork` (cooccurrence/similarity
dropdowns) + `EmbeddingScatter` + `AlignmentScatter`; Zod schemas in
`src/lib/{terms,networkx,embeddings}.ts`.

## Conventions & gotchas

- **Don't hand-edit `public/**`or`src/data/terms.json`\*\* — the pipeline
  regenerates them, and they're often mid-experiment.
- **Treat `config.py` as the user's** — it's frequently staged / mid-edit; don't
  change `TERMS`, pipeline params, or stopwords unless asked.
- Chinese source of record is the **Kyoto UD `scripts/data/mengzi.conllu`** (gold
  tokens/lemmas/POS), **not CLTK**. Node id = lemma; display `form` = common modern
  glyph (敎→教).
- Method defaults: subword-mean → cross-occurrence-max pooling, mean-center
  (anisotropy), kNN cosine graph, neglog sim-transform, Louvain communities. The
  pooling modes and Louvain `resolution` are `Pipeline` knobs; `sim_network`
  "threshold" cuts at a **quantile** of the similarity distribution (rank-based).
- `run_sep` order is **parse → coverage guard → embeddings → co-occurrence →
  export**: the PMI lens shares the embedding lens's variant merge, which needs
  vectors first. Both halves are revertible (`Pipeline.merge_similarity` /
  `merge_cooccurrence`); the lens crossing is argued in
  `notes/claude/derivational-variant-merging.md`.
- The scatter's default view is a **precomputed t-SNE** (`Embeddings.layouts`, one
  per `Pipeline.tsne_sources` x `tsne_perplexities`; the **first is the default**).
  PCA and the client-side t-SNE remain options. `tsne_sources` picks the vectors:
  `reduced` (the export — what the client also has) or `full` (untruncated; only
  marginally more faithful — the note measures it, and PCA-50 turns out to cost
  very little). `reduced` layouts
  retune from the artifact via `tools/relayout.py` in seconds; `full` needs the
  analysis matrix a run caches in `scripts/.cache/vectors/`.
  See `notes/claude/precomputed-tsne-layouts.md`.
- Importing `config`/`main` triggers a (cached) ctext fetch at module scope — noisy
  but harmless. Pyright's `models.Pipeline` / `corpus.sep.SEP_CORPUS` /
  `vectors.reduce_vectors` false-positives came from it resolving `models` in a
  **sibling clone**; `[tool.pyright]` in `pyproject.toml` now pins the venv. If they
  reappear, check which interpreter pyright picked (`pyright --verbose`).

## Active threads (as of 2026-08-01)

- **`min_freq` is overloaded**: shared between the embedding vocab and the
  per-source co-occurrence networks, so a high value (set to thin the scatter)
  starves co-occurrence (null graphs despite real occurrences). Fix in flight:
  a **separate `cooccurrence_min_freq`**, and thin the scatter by relevance
  (top-N by nearest-target similarity) instead of frequency.
- SEP community register-domination — to try (per-pipeline): **HDBSCAN** option.
  **Debiasing** (all-but-the-top / whitening) is **done**: `Pipeline.debias`
  (`abtt`/`whiten`) + `scripts/tools/debias_diagnostics.py`; see
  `notes/claude/frequency-gradient-and-debiasing.md`.
- Active terms: **仁 義 禮 智 信** (others commented out in `config.py`).

## Memory & notes

- `notes/claude` — methodology memos with verified reference lists. The parent
  `notes/` directory and any other subdirectories are reserved for human
  writing.
- Persistent agent memory lives under
  `~/.claude/projects/-home-dawaltco-Code-itp-conceptual-misalignment-site/memory/`
  (indexed by `MEMORY.md`); update it for durable, non-obvious facts.
