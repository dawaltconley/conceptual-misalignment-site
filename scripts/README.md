# scripts/

Data pipeline for *Mapping Conceptual Misalignment*. One pipeline (`main.py`)
builds every JSON artifact the website reads. Run every command with `scripts/`
as the working directory (modules resolve sibling packages via `sys.path`);
output paths are absolute (anchored in `config.py`), so outputs land in the right
place regardless.

## The pipeline

```bash
# Everything: both corpora + the master index
.venv/bin/python -m main

# One corpus at a time
.venv/bin/python -m main --corpus mengzi        # Chinese (Mengzi / GujiRoBERTa)
.venv/bin/python -m main --corpus sep           # English (SEP / roberta-base)

# Just rebuild the master index from files already on disk (no model, no fetches)
.venv/bin/python -m main --master-only
```

The CLI is deliberately small — it only chooses *what* to run:

- `--corpus {mengzi,sep,all}` — which corpus (default `all`).
- `--per-term N` — SEP articles fetched per English rendering (default 12; caps
  corpus size / embedding memory).
- `--artifacts` — also dump the PNG/CSV analysis into `analysis/`.
- `--master-only` — just rebuild `src/data/terms.json` from the manifests.
- `--prune` — after each corpus run, delete files in its output directory that
  the run didn't write (leftovers from terms since removed from `TERMS`).
  Destructive, so opt-in; review the `git status public/` diff. Can't be used
  with `--master-only`, which writes nothing to compare against.

Everything about *how* each corpus is processed lives in `config.py`, as the
`MENGZI_PIPELINE` / `SEP_PIPELINE` `Pipeline` objects: `model`, `min_freq`,
similarity-graph method (`sim_network` = `knn`/`threshold`) with its `knn_k` /
`threshold`, `sim_transform` (default `neglog`), `center`, `reduce_to_dims`,
`max_network_nodes`, `batch_size`, `content_pos`, `stopwords`, and the output
directory. Change a run's behavior by editing those, not by passing flags.

### Outputs (all consumed by the site; paths from `config.py`)

- **Co-occurrence** — one file per (term, source): `public/ctext/{hanzi}_{source}.json`
  (Mengzi: full corpus + each chapter) and `public/sep/{label}_{source}.json`
  (English: combined search + each article). Each a `models.NetworkData`.
- **Similarity** — one file per term (pruned cosine neighborhood over the whole
  corpus): `public/ctext/{hanzi}_embeds.json` / `public/sep/{label}_embeds.json`.
- **Embeddings** — one PCA-reduced dataset per corpus: `public/embeddings/{mengzi,sep}.json`
  (`models.Embeddings`), with Louvain communities from the same cosine graph.
- **Manifest** — one per corpus: `public/ctext/index.json` / `public/sep/index.json`
  (`models.CorpusIndex`). Everything above goes through an `output.CorpusWriter`,
  which writes the file *and* records its `Source` — provenance, occurrence count,
  and web path — in the manifest, in one call. So a path is derived once, from
  `config.PUBLIC`, and never parsed back out of a filename.
- **Master index** — `src/data/terms.json`: every term → its sources → the paths
  above, composed from the two manifests alone. A corpus that hasn't been run has
  no manifest and gets empty sides (with a warning); a corpus that *has* keeps its
  side across a run of the other, so `--corpus sep` doesn't drop the Mengzi terms.

## Layout

```
config.py            Term/Rendering taxonomy, MENGZI_PIPELINE/SEP_PIPELINE configs, stopwords, absolute path constants
main.py              the pipeline entrypoint (run_mengzi / run_sep / build_master + CLI)
models.py            Source/Rendering/Term + Pipeline config + NetworkData/Embeddings/Vector + CorpusIndex/TermIndex + JSON serialization
output.py            CorpusWriter — the one place a name becomes a path; writes each file and records it in the corpus manifest

data/                inputs, no code
  mengzi.conllu      UD Kyoto treebank (gold tokens/lemmas/POS) — the Chinese source of record
  mengzi/            per-chapter Mengzi plaintext .txt (1A … 7B)

corpus/              "get the source text" layer (the only code that hits the network)
  build.py           build_chinese_corpus / build_english_corpus
  mengzi.py          ctext.org fetch + local chapter loader
  sep.py             Stanford Encyclopedia search / scrape
  inpho.py           InPhO topic filter (is_chinese_philosophy)
  conllu.py          load a UD .conllu into spaCy Docs (no model needed)
  parse.py           chapters/articles → spaCy Docs (conllu loader; en_core_web_sm html tokenizer)
  cache.py           shared requests cache (scripts/.cache)

embeddings/          the transformer semantic space
  occurrences.py     content_key / build_vocab / build_segments over spaCy Docs (both corpora)
  model.py           Embedder — final-layer per-occurrence span vectors
  vectors.py         max-pool across occurrences + mean-center + PCA reduce for export
  analyze.py         cosine / kNN graph, Louvain communities, sim transforms, --artifacts dump

cooccurrence/        PMI co-occurrence networks
  pmi.py             PMI + graph construction (corpus-agnostic string lists)
  pmi_spacy.py       spaCy-Doc front end (lemma-id nodes + display-form attr)

graph/               shared graph utilities
  prune.py           prune_to_neighborhood, proximity_score
  serialize.py       save_graph_json

segmentation/        preprocessing: Mengzi → segpos/  (XunziALLM) — separate from the main pipeline
  seg.py segpos.py utils.py seg_data.json

cli/                 auxiliary runnable tools
  segment.py         segment Mengzi chapters with XunziALLM
  find_punctuation.py  inventory punctuation in a text file
```

## Auxiliary tools

```bash
# Segment + POS-tag the full Mengzi with XunziALLM (feeds segmentation/ → segpos/).
# Positional: `seg` = segmentation only, `segpos` also tags POS. Always processes
# the whole corpus (no --input); --arch api needs an OpenAI-compatible server up.
# See `-m cli.segment --help` for --model / --api-base / --grammar / --limit.
.venv/bin/python -m cli.segment seg --output ../segpos/mengzi.jsonl --arch api

# One-off — inventory punctuation in a text file
.venv/bin/python -m cli.find_punctuation <path>
```
