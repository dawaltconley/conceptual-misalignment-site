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
  (`model`, `vectors`, `occurrences`, `analyze`), `cooccurrence/`, `graph/`,
  `segmentation/` + `cli/segment.py` (XunziALLM word segmentation; **not** part of
  a pipeline run — a manual step, run against a served model, whose output
  `corpus/recombine.py` reads if present).

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
- The treebank is **one character per token**, so `corpus/recombine.py` merges
  subword tokens into words at load time (天下, 諸侯, 杞柳) — every consumer
  downstream sees words without knowing it happened. Boundaries come from UD
  `compound`/`flat`/`fixed`, a segmenter lexicon, and a curated override file;
  `conj`/`nmod` are **excluded** and a group containing 仁義禮智信 is never merged,
  so target counts are stable. See `notes/multi-character-tokenization.md`.
- Method defaults: subword-mean → cross-occurrence-max pooling, mean-center
  (anisotropy), kNN cosine graph, neglog sim-transform, Louvain communities. The
  pooling modes and Louvain `resolution` are `Pipeline` knobs; `sim_network`
  "threshold" cuts at a **quantile** of the similarity distribution (rank-based).
- `run_sep` order is **parse → coverage guard → embeddings → co-occurrence →
  export**: the PMI lens shares the embedding lens's variant merge, which needs
  vectors first. Both halves are revertible (`Pipeline.merge_similarity` /
  `merge_cooccurrence`); the lens crossing is argued in
  `notes/claude/derivational-variant-merging.md`.
- Importing `config`/`main` triggers a (cached) ctext fetch at module scope — noisy
  but harmless. Pyright may flag `models.Pipeline` / `corpus.sep.SEP_CORPUS` /
  `vectors.reduce_vectors` as unknown — **stale false-positives**; they run fine.

## Active threads (as of 2026-08-12)

- **Multi-character tokenization** is on `feat/chinese-retokenization`, not yet
  merged to `dev`. Open calls for the user are logged in
  `notes/open-decisions.md` — read that first if you pick this up.
- **`min_freq` / `cooccurrence_min_freq` want re-tuning.** The separate
  `cooccurrence_min_freq` knob is **done**, but merging moved the frequency
  distribution both are cut from (embedding vocab 602 → 571), so neither value is
  tuned to anything now. Thinning the scatter by relevance (top-N by
  nearest-target similarity) rather than frequency is still unbuilt.
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
