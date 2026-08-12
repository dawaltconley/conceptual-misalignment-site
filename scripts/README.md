# scripts/

Data pipeline for _Mapping Conceptual Misalignment_. One pipeline (`main.py`)
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

The CLI is deliberately small — it only chooses _what_ to run:

- `--corpus {mengzi,sep,all}` — which corpus (default `all`).
- `--per-term N` — SEP articles fetched per English rendering (default 12; caps
  corpus size / embedding memory).
- `--artifacts` — also dump the PNG/CSV analysis into `analysis/`.
- `--master-only` — just rebuild `src/data/terms.json` from the manifests.
- `--prune` — after each corpus run, delete files in its output directory that
  the run didn't write (leftovers from terms since removed from `TERMS`).
  Destructive, so opt-in; review the `git status public/` diff. Can't be used
  with `--master-only`, which writes nothing to compare against.
- `--allow-empty-renderings` — downgrade the empty-rendering abort to a warning
  (see `renderings.py`).

`run_sep`'s phases run **parse → coverage guard → embeddings → co-occurrence →
similarity/export**. Co-occurrence is last because it shares the embedding
lens's derivational-variant merge, which cannot exist until there are vectors
to gate it on (`Pipeline.merge_cooccurrence`; see
`notes/derivational-variant-merging.md`). The guard sits right after parsing so
a dead rendering still fails the run before the GPU work, not after.

Everything about _how_ each corpus is processed lives in `config.py`, as the
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
  which writes the file _and_ records its `Source` — provenance, occurrence count,
  and web path — in the manifest, in one call. So a path is derived once, from
  `config.PUBLIC`, and never parsed back out of a filename.
- **Master index** — `src/data/terms.json`: every term → its sources → the paths
  above, composed from the two manifests alone. A corpus that hasn't been run has
  no manifest and gets empty sides (with a warning); a corpus that _has_ keeps its
  side across a run of the other, so `--corpus sep` doesn't drop the Mengzi terms.

## Layout

```
config.py            Term/Rendering taxonomy, MENGZI_PIPELINE/SEP_PIPELINE configs, stopwords, absolute path constants
main.py              the pipeline entrypoint (run_mengzi / run_sep / build_master + CLI)
renderings.py        coverage guard — a Rendering that matches no token aborts the run (see notes/spacy-lemma-exceptions.md)
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
  parse.py           chapters/articles → spaCy Docs (conllu loader; en_core_web_sm html tokenizer) + lemma exceptions + multi-token rendering merge
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

# Which Renderings matched, and which lost occurrences to a lemmatizer error?
# (`mores` -> lemma `more` silently zeroed 禮's "mores" for entire runs.) The
# pipeline aborts on a rendering that matches *nothing*; this also reports
# partial losses. -> analysis/sep/rendering_diagnostics.md
.venv/bin/python tools/rendering_diagnostics.py --per-term 12
```

---

## The pipeline, step by step

What follows is the whole run in plain language. Both corpora go through the same
stages; where the Chinese and English halves differ, that's called out. Every
number mentioned here (`min_freq`, `knn_k`, the merge threshold, …) is a field on
the `Pipeline` object in `config.py`.

### 1. Decide what we're looking for

`TERMS` in `config.py` is the whole target list: each Chinese term paired with
the English words translators use for it. A `Rendering` is one such English word
plus the glob patterns that count as it — `Rendering('humaneness', 'humane*')`
means every token lemmatizing to `humane`, `humanely`, `humaneness` is one thing
called "humaneness". Patterns are kept disjoint, because matching is first-wins:
if two renderings could claim the same token, the second one silently becomes a
dead node.

### 2. Fetch the text

**Chinese.** The Mengzi's fourteen chapters come from local `.txt` files under
`scripts/data/mengzi/`; ctext.org is queried only for each chapter's canonical
URL and metadata, and those requests are HTTP-cached in `scripts/.cache`.

**English.** For each rendering, the pipeline searches the SEP, pages through
results, and scrapes each article's preamble + main text (bibliography and
"related entries" stripped, and parenthetical citations like "(Smith 1998, 22)"
removed so they don't pollute the word counts). Two filters run over the
results:

- **Topic filter** — articles whose InPhO topic model gives them more than 25%
  "Chinese philosophy" are skipped. The point of the project is to see what these
  English words mean in _general_ Anglophone philosophy, so an SEP article about
  Mengzi would be circular. Skipped articles are still fetched and their
  occurrences counted, reported separately as `chinesePhilosophyOccurrences`.
- **Frequency filter** — an article mentioning the rendering fewer than
  `cooccurrence_min_freq` times is dropped; it contributes noise, not context.

`--per-term N` caps how many articles survive per rendering (default 12), which
is the main lever on corpus size and GPU memory. Articles are deduplicated, so
one article found by two renderings is parsed once.

### 3. Parse into words

Everything becomes a spaCy `Doc`, so the rest of the pipeline has exactly one
input shape.

**Chinese** does not get parsed by a model at all. The tokens, lemmas, and POS
tags are read straight out of the gold Kyoto UD treebank
(`scripts/data/mengzi.conllu`) and loaded into a blank `Doc` — a hand-annotated
source of record beats anything a classical-Chinese tagger would guess.

**English** runs `en_core_web_sm`, plus two corrections:

- **Lemma exceptions** (`scripts/lemmas/english.conf`) patch known spaCy
  mistakes, keyed on surface form. spaCy strips the final _s_ from discipline
  names (`ethics` → `ethic`) and back-forms words that aren't English
  (`species` → `specie`); the table maps each surface back to itself. Keying on
  the surface rather than the bad lemma matters, because the bad lemma is often
  a real word too — most occurrences of the lemma `aesthetic` really are the
  adjective.
- **Phrase merging** collapses multi-word renderings (`social norms`) into a
  single token whose lemma is the rendering's label, because a pattern is
  matched against one token's lemma and could otherwise never fire.

**Coverage guard.** Right after parsing, before any GPU work, the run checks
that every configured rendering matched at least one token. A rendering that
matches nothing aborts the run with a diagnosis, because the failure is
otherwise invisible: it writes a null network for every file it owns, which
looks identical to a term the corpus simply doesn't discuss. (禮's `mores` was
dead this way for an unknown number of runs — spaCy lemmatizes it to `more`.)

### 4. Decide what counts as a word

One function, `content_key`, answers this for both lenses and both languages, so
a node means the same thing everywhere. Per token:

- if it matches a target rendering → it keys under that rendering's **label**
  (so `humane` and `humaneness` pool onto one node called "humaneness");
- otherwise, if it's a content word — alphabetic, not punctuation, not a
  stopword, POS in `content_pos` (NOUN/VERB/ADJ) → it keys under its **lemma**;
- otherwise it's dropped.

### 5. Turn words into vectors

- **Vocabulary.** Keep words occurring at least `min_freq` times across the whole
  corpus (5 for Mengzi, 20 for SEP), optionally bounded by document frequency.
  Target terms are always kept regardless of how rare they are.
- **Segments.** Each source's sentences are greedily packed into chunks that fit
  the model's 512-token window, never crossing a document boundary. Each chunk
  records the character span of every vocabulary word inside it.
- **Encode.** The transformer (`GujiRoBERTa` for Chinese, `roberta-base` for
  English) runs over each chunk, and for every recorded span we take the
  final-layer hidden states of the tokens overlapping it and average them. That
  gives one vector per _occurrence_ — this word, in this sentence.
- **Pool.** Occurrences of the same word are folded into a single vector by an
  element-wise max. This happens streaming, as batches come off the GPU, so
  memory stays proportional to the vocabulary rather than to the corpus.

The result is one vector per word: its average behavior across every context it
appears in.

### 6. Straighten the space

Raw transformer vectors are **anisotropic** — they all point roughly the same
direction (mean cosine to the centroid ≈ 0.84 for GujiRoBERTa), so every pair of
words looks similar and no cosine threshold means anything. Two fixes:

- **Center** — subtract the vocabulary centroid, which restores a meaningful
  zero and roughly doubles the spread of cosine values.
- **Debias** (`abtt`) — project out the top principal components, because word
  _frequency_ loads onto them (Mu & Viswanath 2018). Without this, the leading
  axis of the scatter plots how common a word is rather than what it means.

Everything downstream — the similarity graph, the merge decision, the exported
scatter — is measured in this one corrected space.

### 7. Merge derivational variants (English only)

The English vocabulary keys on a lemma, which is inflectional only: `inspires`
and `inspired` collapse to `inspire`, but `inspire` and `inspiration` remain two
nodes sitting on top of each other. About a quarter of the SEP vocabulary is
tangled this way. So:

- **Propose** candidates from Open English WordNet's derivation relations, plus
  `-ed`/`-ing` forms whose base verb is also in the vocabulary.
- **Gate** them on cosine in the analysis space, using complete linkage — every
  pair inside a merged family must clear `merge_threshold`. Both halves are
  load-bearing. Morphology alone would fuse `know`/`knowledge`; similarity alone
  would fuse `monotonic`/`nonmonotonic` and `priori`/`posteriori`, whose vectors
  are near-identical _because_ they're opposites. (Hence: strip suffixes, never
  prefixes.)
- **Name** the survivor — prefer a noun, then the most frequent, then the
  shortest. The corpus says `ethical` more often than `ethics`, but `ethics` is
  what a reader looks for.

Target renderings never merge. The merge is then re-applied from the original
accumulators, and the space is rebuilt so the graph, communities, and scatter are
all computed over the merged vocabulary.

This is also why **`run_sep` builds its co-occurrence networks after the
embeddings** rather than before, unlike `run_mengzi`: the PMI lens reuses this
merge so a node means the same word in both views, and the merge can't be decided
until vectors exist. Both halves are individually revertible
(`merge_similarity` / `merge_cooccurrence`), because this is the one point where
a paradigmatic criterion reaches into the syntagmatic graphs — see
`notes/claude/derivational-variant-merging.md`.

### 8. Build the similarity network (paradigmatic lens)

Cosine similarity between every pair of word vectors, reweighted by `-ln(1 - s)`
to spread out the crowded high-similarity end. Then edges: by default each word
keeps its `knn_k` nearest neighbors (an edge exists if _either_ endpoint ranks
the other). kNN is rank-defined, so it survives the anisotropy that makes an
absolute cutoff meaningless; the alternative `threshold` mode is also
rank-defined, cutting at a quantile of the similarity distribution rather than at
a fixed cosine value.

Isolated nodes are dropped, **Louvain** finds communities, and each node picks up
its weighted degree, PageRank, and eigenvector centrality.

Then, per term, the graph is **pruned to a neighborhood**: the term plus its
strongest ~15 neighbors by edge weight, topped up with the best two-hop
neighbors (scored by the product of the two edge weights) if there's room. That
subgraph is what the site draws.

### 9. Build the co-occurrence networks (syntagmatic lens)

A separate, much simpler question: what words show up in the same _sentence_ as
this term?

Each sentence is reduced to its list of node keys (same `content_key` as above,
same variant merge for English), sentences with fewer than two keys are dropped,
and then for every pair we compute **PMI** — how much more often two words
co-occur than chance would predict, given how often each appears on its own.
Only positive-PMI edges are kept, isolates are dropped, and the graph is pruned
to the term's neighborhood exactly as in step 8.

This runs once per (term, source): the whole Mengzi and each of its 14 chapters;
the combined SEP search and each individual article. Nodes carry a `form`
attribute — the most common surface glyph seen for that key — so the reader sees
教 even though the graph keys on the treebank's 敎.

Note that `cooccurrence_min_freq` (3) is deliberately separate from the embedding
`min_freq`: it applies _per source_, so a single chapter or article would produce
an empty graph under the whole-corpus floor.

### 10. Export

- **Scatter** — the vectors are mean-centered, L2-normalized (direction only),
  and PCA'd down to `reduce_to_dims` (50) for the browser, one file per corpus
  with community ids and the per-node statistics attached.
- **Networks** — one JSON per pruned graph.
- **Manifest** — every file goes through a `CorpusWriter`, which writes it _and_
  records its provenance, occurrence count, and web path in that corpus's
  `index.json`, in one call. A path is derived once and never parsed back out of
  a filename.
- **Master index** — `build_master` composes `src/data/terms.json` from the two
  manifests alone. Because each corpus owns its own manifest, running
  `--corpus sep` leaves the Mengzi side of the index intact.

Optionally, `--artifacts` additionally dumps PNGs and CSVs (PCA/t-SNE plots,
cosine heatmaps, k-means, per-target cohesion) to `analysis/`, and `--prune`
deletes output files the run didn't write — leftovers from terms since removed
from `TERMS`.

Each phase is timed, and the run prints a breakdown at the end.
