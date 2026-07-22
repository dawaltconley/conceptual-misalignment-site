# scripts/

Analysis pipelines for *Mapping Conceptual Misalignment*. Run every command with
`scripts/` as the working directory (modules resolve sibling packages via
`sys.path`); output paths are absolute (anchored in `config.py`), so outputs land
in the right place regardless.

## Layout

```
config.py            Term / Rendering taxonomy + all absolute path constants
paths → config.py    MENGZI_DIR (input), DATA/PUBLIC/SEP/CTEXT/ANALYSIS/SEGPOS (outputs)

data/                inputs, no code
  mengzi/            per-chapter Mengzi source .txt (1A … 7B)

corpus/              "get the source text" layer (the only code that hits the network)
  cache.py           shared requests cache (scripts/.cache)
  mengzi.py          ctext.org fetch + local chapter loader
  sep.py             Stanford Encyclopedia search/scrape
  inpho.py           InPhO topic filter (is_chinese_philosophy)

nlp/                 language primitives
  chinese.py english.py text.py (is_cjk)

graph/               graph utilities shared by BOTH methods
  prune.py           prune_to_neighborhood, proximity_score
  serialize.py       save_graph_json

cooccurrence/        METHOD 1 — PMI / co-occurrence networks → public/{sep,ctext}
  pmi.py             PMI + co-occurrence/cosine graph construction

embeddings/          METHOD 2 — transformer semantic spaces → analysis/
  model.py vectors.py analyze.py occurrences.py sep_occurrences.py

segmentation/        preprocessing: Mengzi → segpos/  (XunziALLM)
  seg.py segpos.py utils.py seg_data.json

cli/                 runnable entrypoints
  cooccurrence.py    embed_mengzi.py  embed_sep.py  segment.py  find_punctuation.py
```

## Entrypoints

```bash
# Method 1 — PMI co-occurrence networks for the site
.venv/bin/python -m cli.cooccurrence

# Method 2 — Chinese (Mengzi / GujiRoBERTa) semantic space
.venv/bin/python -m cli.embed_mengzi [--dry-run] --center --network knn --knn-k 8 --max-nodes 25

# Method 2 — English (SEP / roberta-base) semantic space
.venv/bin/python -m cli.embed_sep --per-term 12 --min-freq 75 --center --network knn --knn-k 8 --max-nodes 25

# Preprocessing — segment Mengzi chapters with XunziALLM
.venv/bin/python -m cli.segment seg --input data/mengzi/1A.txt --output ../segpos/chapters/1A.jsonl --unit line --arch api

# One-off — inventory punctuation in a text file
.venv/bin/python -m cli.find_punctuation <path>
```
