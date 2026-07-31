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

Useful flags: `--network {knn,threshold}` (default `knn`), `--knn-k`, `--threshold`;
`--sim-transform {none,neglog,poslog}` (default `neglog` — reweights cosine before
the similarity graph); `--per-term N` (SEP articles per rendering — caps corpus
size / embedding memory); `--min-freq`, `--max-nodes`, `--no-center`; `--artifacts`
(also dump the PNG/CSV analysis into `analysis/`).

### Outputs (all consumed by the site; paths from `config.py`)

- **Co-occurrence** — one file per (term, source): `public/ctext/{hanzi}_{source}.json`
  (Mengzi: full corpus + each chapter) and `public/sep/{label}_{source}.json`
  (English: combined search + each article). Each a `models.NetworkData`.
- **Similarity** — one file per term (pruned cosine neighborhood over the whole
  corpus): `public/ctext/{hanzi}_embeds.json` / `public/sep/{label}_embeds.json`.
- **Embeddings** — one PCA-reduced dataset per corpus: `public/embeddings/{mengzi,sep}.json`
  (`models.Embeddings`), with Louvain communities from the same cosine graph.
- **Master index** — `src/data/terms.json`: every term → its sources → the paths above.

## Layout

```
config.py            Term / Rendering taxonomy + all absolute path constants
main.py              the pipeline entrypoint (run_mengzi / run_sep / build_master + CLI)
models.py            Source/Rendering/Term + NetworkData/Embeddings/Vector + JSON serialization

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
  vectors.py         max-pool across occurrences + mean-center
  analyze.py         cosine / kNN graph, Louvain communities, sim transforms, --artifacts dump

cooccurrence/        PMI co-occurrence networks
  pmi.py             PMI + graph construction (corpus-agnostic string lists)
  pmi_spacy.py       spaCy-Doc front end (lemma-id nodes + display-form attr)

graph/               shared graph utilities
  prune.py           prune_to_neighborhood, proximity_score
  serialize.py       save_graph_json

text/                language primitives
  chinese.py         classical-Chinese function-word STOPWORDS

segmentation/        preprocessing: Mengzi → segpos/  (XunziALLM) — separate from the main pipeline
  seg.py segpos.py utils.py seg_data.json

cli/                 auxiliary runnable tools
  segment.py         segment Mengzi chapters with XunziALLM
  find_punctuation.py  inventory punctuation in a text file
```

## Auxiliary tools

```bash
# Segment Mengzi chapters with XunziALLM (feeds segmentation/ → segpos/)
.venv/bin/python -m cli.segment seg --input data/mengzi/1A.txt \
    --output ../segpos/chapters/1A.jsonl --unit line --arch api

# One-off — inventory punctuation in a text file
.venv/bin/python -m cli.find_punctuation <path>
```
