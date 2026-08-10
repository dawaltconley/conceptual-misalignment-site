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
