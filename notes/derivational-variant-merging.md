# Merging derivational variants on the SEP scatter

Why the scatter plots `inspire` and `inspiration` as two points, why that is
worth fixing, and why the fix needs *both* a morphological resource and an
embedding-similarity gate — neither alone is safe.

Companion to [[spacy-lemma-exceptions]] (the lemmatizer errors found along the
way) and [[embedding-communities-and-semantics]] (why the space is
register-dominated in the first place).

## The problem

The SEP vocabulary keys on a spaCy **lemma**, and lemmatization is inflectional
only. `inspires`/`inspired`/`inspiring` already collapse to `inspire`, but the
*derivational* step does not: `inspire` and `inspiration` stay two nodes, land
on top of each other, and clutter the view. Roughly a quarter of the ~3950-entry
SEP vocabulary is entangled this way.

There is a consistency argument too. The pipeline already does exactly this
merge for target terms, by hand: `Rendering('wisdom', 'wisdom*', 'wise*')` and
`Rendering('benevolence', 'benevolen*')` are curated derivational families. The
non-target vocabulary being unmerged is an inconsistency, not a principled
default.

## Why the merge is cheap and exact

SEP pools with `occurrence_pooling="max"`. Element-wise max is associative over
the occurrence set, so folding two *pooled* vectors gives exactly the vector
that pooling the union of their occurrences would have given. `mean` is equally
exact given the occurrence counts, which `Pooler` already tracks. So the merge
needs no re-embedding — `Pooler.merge` folds accumulators and the space is
rebuilt from there.

It sits between `pool()` and the graph in `main.run_sep`: late enough that the
candidate vectors exist (the gate needs them), early enough that centering,
debias, the kNN graph, Louvain, `norm` and the PCA export are all computed over
the merged vocabulary.

The gate measures cosine in the **centred/debiased analysis space**, never on
raw pooled vectors — those sit in an anisotropic band where every cosine is
~0.84 and a threshold means nothing (see [[anisotropy-and-network-construction]]).

## Both halves of the criterion are load-bearing

**Morphology alone is not enough.** `know`/`knowledge`/`knowable` share a root
and not a meaning. (As it happens OEWN records no derivational link for
`knowledge` at all, so it is excluded before the gate even runs — but the
stemmer variant does group it, and there the gate is what saves it.)

**Similarity alone is worse.** The highest-cosine pairs in the SEP space that
are *not* morphologically related are:

| cosine | pair |
|---|---|
| 0.935 | `monotonic` / `nonmonotonic` |
| 0.928 | `posteriori` / `priori` |

Negation and antonym pairs. Their vectors are near-identical *precisely because*
they are contraries — they occur in identical frames. A pure-cosine rule would
fuse them, which would be flatly wrong. **Corollary: strip suffixes only, never
prefixes.** Any prefix rule would recreate this failure by construction.

Clustering uses **complete** linkage, not single: single linkage lets a chain of
adjacent pairs drag a distant member in, which is the `know`-`knowable`-
`knowledge` failure again.

## Choosing the resource

Princeton WordNet 3.0 is frozen (2011). **Open English Wordnet** is its
maintained successor, annually released and citable, and the `wn` package
exposes the relation directly as `sense.get_related('derivation')`. Measured
against the SEP vocabulary:

| resource | in-vocab derivational pairs |
|---|---|
| OEWN 2025 | 1083 |
| nltk WordNet 3.0 | 1047 |
| shared | 1044 (39 OEWN-only, 3 nltk-only) |

OEWN is a strict improvement, so the pipeline pins `oewn:2025`.

Note that `spacy-wordnet` is *not* the entry point for this — it exposes
synsets, lemmas and Wordnet-Domains, but not derivational relations.

## The stemmer: measured complementary, shipped unwired

A hand-rolled suffix stripper is **not** redundant with the lexicon. At cosine
0.70 the union of the two yields 231 merges, of which:

- the lexicon alone misses **31%** — almost all `-ism`/`-ist` philosophy jargon
  that no general lexicon carries (`compatibilism`/`compatibilist`,
  `retributivism`/`retributivist`, `fideism`/`fideist`);
- the stemmer alone misses **26%** — irregular derivations no suffix rule can
  reach (`belief`/`believe`, `democracy`/`democratic`, `conceive`/`conception`,
  `fail`/`failure`, `tragedy`/`tragic`).

The stemmer's cost is over-merging: `virtual`/`virtue`/`virtuous` share a stem,
and `nature`/`naturalism`/`naturalist`/`naturalize` group as one family. The
cosine gate cleans most of that up, but it is hand-tuned machinery to defend in
a way a released lexicon is not. So `families.stem_key` / `stem_pairs` ship
**exported but unimported** — available as a second candidate source, not
currently wired in.

## Participial forms

A third candidate source, `families.participial_pairs`, links `-ed`/`-ing`
entries back to their base verb. This looked at first like a lemmatizer bug to
fix outright, but it is not: spaCy tags these plain `JJ`, not `VBN`/`VBG`, so
nothing at the tag level separates a genuine artifact (`devoted` → `devote`)
from a lexicalized adjective (`determined`, `concerned`) or from a deverbal noun
that must not collapse (`building`, `beginning`, `accounting`, `bearing`). 104
vocabulary entries have the shape. They are therefore proposed as gated
candidates like everything else, and the cosine gate is the discriminator.

## The circularity caveat — state it, don't hide it

Gating merges on embedding similarity is mildly circular if the merged plot is
then used as evidence about similarity structure. The defence is that the merge
**cannot manufacture proximity**: it only collapses points already within
`merge_threshold` of each other, so no pair is drawn together that was not
already together. It changes the unit of analysis from "word type" to "lexeme
family" and declutters the display; it does not create structure. Any claim read
off the merged scatter should be one that would survive on the unmerged scatter
with the variants overplotted.

## Calibrating the threshold — measure in the right space

The first attempt picked `merge_threshold` by sweeping cosine on the **exported**
vectors (`public/embeddings/sep.json`). That was wrong, and wrong by a lot. The
export is PCA-reduced to `reduce_to_dims` (50) and the truncation discards
low-variance dimensions, which is precisely where unrelated words differ — so
cosine reads much higher there than in the 768-d space the gate actually uses:

| threshold | merges on the 50-d export | merges in the real analysis space |
|---|---|---|
| 0.70 | 231 | 31 |

Same threshold, same candidates, 7× difference in outcome. The lesson generalizes
past this feature: **any threshold tuned against the artifact is not the
threshold the pipeline applies.**

The fix is `dump_family_candidates` in `main.run_sep`, which writes every
candidate family's pairwise cosines *in the analysis space* to
`analysis/{corpus}/family_candidates.csv` at merge time. Those pairs are exactly
what complete linkage consumes, so `tools/family_diagnostics.py` can replay any
threshold exactly without re-running the embedder (~18 min a run).

### What the real space looks like

771 candidate families over 1871 words — **47% of the vocabulary** is
morphologically entangled. Within-family cosine: median 0.412, p75 0.517, p90
0.593, max 0.863.

| threshold | merges | words absorbed | resulting vocab |
|---|---|---|---|
| 0.30 | 701 | 843 | 3108 |
| 0.40 | 592 | 660 | 3291 |
| **0.45** | **508** | **559** | **3392** |
| 0.50 | 379 | 409 | 3542 |
| 0.60 | 129 | 136 | 3815 |
| 0.70 | 29 | 31 | 3920 |

### The anchor: what "no target pair exceeds 0.335" means

A cosine threshold is meaningless in the abstract. 0.45 sounds low if you are
used to raw sentence-embedding similarities, which cluster around 0.8+; this
space is mean-centred *and* debiased with all-but-the-top, which deliberately
strips the common direction that inflates those numbers. So "is 0.45 high?" can
only be answered against something measured in **this** space.

The project supplies a natural yardstick — the English words the five Chinese
terms get translated as:

| term | renderings |
|---|---|
| 仁 | benevolence, humaneness, humanity |
| 義 | righteousness, justice, meaning, morality |
| 禮 | ritual, propriety, etiquette, ~~social norms~~ |
| 智 | knowledge, wisdom, intelligence |
| 信 | trustworthiness, faith, sincerity |

`social norms` never reaches the embedding vocabulary (it is also reported
absent from the similarity graph each run), leaving **16 targets** in the space.

Every one of these is a word the project treats as a **distinct concept** —
telling them apart is the research question. Their 120 pairwise cosines
(16 × 15 / 2) are already computed in the analysis space and written to
`analysis/sep/cosine_targets.csv` on an `--artifacts` run. The distribution:

| | cosine | pair |
|---|---|---|
| max | **0.335** | justice / propriety |
| 2nd | 0.322 | benevolence / humaneness |
| 3rd | 0.296 | justice / morality |
| median | 0.117 | — |
| min | −0.120 | — |

"No target pair exceeds 0.335" means: **across all 120 pairs, the two most
similar words the project considers different concepts sit at 0.335.** The
runner-up is more pointed still — `benevolence` and `humaneness` are two English
renderings of the *same character* 仁, the pair with the strongest possible claim
to being near-synonyms, and they only reach 0.322.

That gives the threshold a floor with an argument behind it. Setting
`merge_threshold` above ~0.40 guarantees that **every merge the pipeline makes is
tighter than any pair of words the project itself declines to identify.** If the
gate ever merged something at 0.30, that merge would be looser than
`benevolence`/`humaneness` — words nobody would want collapsed into one point.
At 0.45 there is comfortable margin above that line.

The comparison also shows the candidates are not marginal: within-family
candidate pairs have a **median of 0.412**, i.e. the typical morphological
relative is already more similar than *all but two* of the 120 distinct-concept
pairs.

Two honest limits on the argument:

- It is a **necessary, not sufficient** condition. Clearing 0.335 does not prove
  two words are one lexeme; it only rules out merges looser than the project's
  own "clearly different concepts" band. The morphological gate does the actual
  work of establishing they are the same word — this just bounds how loose the
  cosine half is allowed to be.
- The 16 targets are a small and non-random sample, chosen for being
  *translations of related virtue terms*, so they are probably more similar to
  each other than 16 arbitrary words would be. That biases the yardstick
  *conservative* (a ceiling drawn from unusually-related words is, if anything,
  too high), which is the safe direction for a floor.

0.45 was chosen because the floor stays clean there (`famous`/`fame`,
`equal`/`equality`, `crime`/`criminal`, `grow`/`growth`, `sympathy`/`sympathize`)
and it captures `inspire`/`inspiration` at 0.4714 — the pair that prompted the
whole investigation. At 0.40 the floor starts admitting doubtful merges
(`temper`/`temperance`, `set`/`setting`).

### Complete linkage splits some strong pairs, by design

51 candidate pairs at 0.45 clear the floor and still do not merge, because a
*third* family member fails. `{tolerance, tolerant, tolerate, toleration}` is the
clean example: `tolerance`/`tolerant` is 0.604 and merges, after which
`toleration` cannot join (`tolerant`/`toleration` = 0.4985, just under). So
`tolerance`/`toleration` at 0.576 stays split. This is the guard doing its job —
it is what stops `know`-`knowable`-`knowledge` chaining — but it means lowering
the threshold does not necessarily capture a given pair. `family_diagnostics.py`
reports these separately from pairs that are simply below the floor.

## Result of the shipped run (τ = 0.45)

`3951 -> 3393` nodes: **507 merges absorbing 558 words, 14.1% of the vocabulary**.
No target merged, and no absorbed word survives as a stray node. Communities
dropped 21 → 20 and graph edges 21056 → 18206, as expected from a smaller vocab.

The motivating case resolved: `inspire` now carries `inspiration` (0.4714).

The counter-example resolved too, and by the intended mechanism. Family 34 is
`{know, knowable, knower, knowing, known}`; the gate split it into
`know ← knowing, known` and `knower ← knowable`, leaving `know`/`knowable` apart
at 0.4196. `knowledge` is a target and never entered the running.

### Open judgment call: which member wins the label

The surviving label is the most frequent member, which means an adjective can
beat the noun a reader would look for. Four of the discipline names protected in
[[spacy-lemma-exceptions]] get re-folded into their adjective:

| merged as | absorbing |
|---|---|
| `ethical` | `ethics` |
| `mathematical` | `mathematics` |
| `metaphysical` | `metaphysics` |
| `semantic` | `semantics` |

(Eleven others — `aesthetics`, `politics`, `physics`, `economics`, `linguistics`,
`pragmatics`, `hermeneutics`, `metaethics`, `metasemantics`, `mechanics`,
`robotics` — survive as their own nodes.)

This is not a regression of the lemma fix. Before it, spaCy silently pooled
`ethics` into `ethic`, a *different* word, with no record. Now `ethics` is
explicitly grouped with a genuine relative, the vector is a correct pooling of
both, and the tooltip names what was absorbed. But `ethical ← ethics` still reads
oddly on a philosophy scatter, where `ethics` is the more salient term. If that
matters, bias `merge_map`'s `rank` toward nouns rather than raw frequency.

## Knobs

- `Pipeline.merge_variants` — off by default; on for SEP. English-only, since
  candidates come from an English lexicon and classical Chinese has no
  derivational suffixes.
- `Pipeline.merge_threshold` — cosine floor, default 0.70.
- Targets are never merged, and never become a merge's surviving label.
- `scripts/tools/family_diagnostics.py` sweeps the threshold against the shipped
  artifact and lists both the merges and the near-misses the gate rejected.

## References

- Fellbaum, C. (ed.) (1998). *WordNet: An Electronic Lexical Database.* MIT Press.
- McCrae, J. P., Rademaker, A., Bond, F., Rudnicka, E., Fellbaum, C. (2019).
  *English WordNet 2019 — An Open-Source WordNet for English.* Proceedings of
  the 10th Global WordNet Conference. https://aclanthology.org/2019.gwc-1.31/
- Open English WordNet. https://en-word.net/
- `wn` (Goodman & Bond), the library used here. https://github.com/goodmami/wn
