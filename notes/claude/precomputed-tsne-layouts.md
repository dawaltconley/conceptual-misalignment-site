# Precomputing the t-SNE layouts

Why the embedding scatter now opens on a t-SNE the pipeline computed, and what
that costs. Companion to [[embedding-communities-and-semantics]] (what the
communities on that map do and don't mean) and [[client-side-alignment]] (the
alignment view this supersedes for the MVP).

## The problem

The scatter had two projections, both derived in the browser:

- **PCA** — free. The exported vectors are PCA-reduced and variance-ordered, so
  columns 0/1 *are* the 2-D principal-component coordinates.
- **t-SNE** — run client-side in a web worker over the full 50-D exported vector,
  500 iterations, streaming each frame back so the map visibly converges.

PCA was the default because it was instant. But PCA is the weaker view of this
space by a wide margin: it is a linear projection, so it flattens exactly the
local neighborhood structure the project is asking about, and after debiasing
([[frequency-gradient-and-debiasing]]) the leading components carry little that
reads as topical. t-SNE separates the communities legibly; PCA shows a blob.

So the useful view was the one nobody saw first, and reaching it cost several
seconds of animation per page load — twice, since the page shows two corpora.

## What changed

The perplexities worth looking at are computed once in the pipeline
(`embeddings/layouts.py`) and shipped as plain x/y in the embedding artifact's
`layouts` array. The scatter reads them directly; the first is the default view.
PCA stays as an option, and the client-side t-SNE stays for a perplexity nobody
precomputed.

**What gets precomputed, after the measurements below: the 768-d layouts only.**
The reduced (50-d) ones are exactly what the browser can compute for itself, so
shipping them too is dead weight — the pipeline precomputes what the client
*can't* reach, and the client covers the rest. In the UI this is one slider:
by default it snaps to the precomputed perplexities (ticked under the track,
instant, 768-d), and ticking "recompute" frees it to any value, computed here
over the downloaded 50-d vectors. The dimensionality is printed beside the value
so the swap is visible rather than inferred.

Tuning lives on the `Pipeline` (`config.py`), like every other run parameter:
`tsne_sources` and `tsne_perplexities` (one layout per pair), `tsne_epsilon`
(learning rate), `tsne_iterations`, `tsne_seed`.

Cost is small: 4 layouts add ~60 KB to `mengzi.json` and ~400 KB to `sep.json`,
against 0.8 MB / 4.3 MB of vectors already being shipped, and a 4-perplexity
sweep takes ~4 s (Mengzi) / ~15 s (SEP) on CPU from the reduced vectors.

### Three decisions worth recording

**Which vectors a layout embeds is a knob, and the default is arguable.**
`Pipeline.tsne_sources` takes `reduced` (the exported PCA-reduced matrix),
`full` (the untruncated analysis space, in the export's own center + L2
preprocessing — the matrix its PCA is fitted on, minus the truncation), or both.
The case for `reduced`: PCA-to-~50-then-t-SNE is van der Maaten's own
recommendation, it denoises and makes the neighbor search tractable, and it is
the only thing the *client* has — so a precomputed layout and a live one are
looking at the same numbers rather than being two different maps under one name.
The case for `full` is measured below, and it is strong.

**`init="pca"`, fixed seed.** t-SNE is non-convex; random init makes the global
arrangement seed-dependent, so a rerun of the pipeline would silently redraw the
map. PCA init largely removes that (Kobak & Berens, 2019) and the seed pins the
rest, so the layout is a stable object the writing can refer to.

**Coordinates are keyed by node id, not parallel to `nodes`.** An array would be
one node-ordering change away from putting every word at another word's
coordinates — a bug that looks like a *finding*. A lookup fails safe: a node the
layout doesn't cover is not plotted.

## Does the truncation cost anything?

A little, consistently — but less than a first look suggests, and the two obvious
ways to measure it disagree about how much. Ground truth throughout is the
**768-d analysis space**: where the similarity graph, the communities and every
reported cosine actually live. Mengzi, 602 words.

**Strict top-10 overlap** — of each word's 10 nearest neighbors in the full
space, how many are still among its 10 nearest on the map:

| space read                       | neighbors retained |
| -------------------------------- | -----------------: |
| exported 50-d vectors (no t-SNE) |              0.550 |
| t-SNE over the 50-d vectors      |              0.218 |
| t-SNE over the full 768-d space  |              0.312 |

**Trustworthiness** (sklearn's standard projection-quality metric, which
penalizes intruders by how far they were in the original space rather than
demanding an exact set match):

| space read                       |   k=5 |  k=10 |  k=20 |
| -------------------------------- | ----: | ----: | ----: |
| exported 50-d vectors (no t-SNE) | 0.925 | 0.909 | 0.889 |
| t-SNE over the 50-d vectors      | 0.837 | 0.760 | 0.692 |
| t-SNE over the full 768-d space  | 0.866 | 0.781 | 0.706 |

The same on SEP (3277 words, computed entirely within one run so nothing is
mixed):

| space read                       | top-10 |   k=5 |  k=10 |  k=20 |
| -------------------------------- | -----: | ----: | ----: | ----: |
| exported 50-d vectors (no t-SNE) |  0.551 | 0.991 | 0.988 | 0.983 |
| t-SNE over the 50-d vectors      |  0.233 | 0.919 | 0.862 | 0.808 |
| t-SNE over the full 768-d space  |  0.318 | 0.931 | 0.869 | 0.805 |

**The two metrics disagree, and the strict one is the misleading one.** Set
overlap says the 50-d export keeps only ~55% of the exact top-10 neighbors on
both corpora; trustworthiness says 0.909 (Mengzi) and 0.988 (SEP), i.e. the words
that fall out of the top 10 were near neighbors anyway. The truncation *shuffles
the exact ordering* far more than it moves anything far. On SEP in particular,
768 -> 50 is very nearly lossless.

So the answer to "does the truncation cost anything" is: **surprisingly little**,
and the full-vector layout is a marginal improvement rather than a different
picture — +0.02 trustworthiness at k=10 on Mengzi, +0.007 on SEP, and at k=20 on
SEP it is *slightly worse* (0.805 vs 0.808). The 768-d Mengzi map does look
tidier on screen (the five targets land together), but one corpus and one
eyeball is not evidence.

Two caveats on the numbers. The ground truth is the space the `full` layout
starts from, so some advantage is baked in — the framing is still the right one
(does the picture represent the space the analysis is conducted in?) but the
comparison is not neutral. And these are scratch-script numbers, not a committed
tool; rebuild from `embeddings.vectors.load_analysis_matrix` plus
`sklearn.manifold.trustworthiness`.

### Where the truncation does bite: short projections

Only the `full` source is unit-norm when t-SNE sees it. Both sources get the same
L2 step — `unit_vectors` runs inside `reduce_vectors`, before its PCA — but the
projection then shortens every row by a word-specific amount. On the Mengzi the
exported rows keep a mean of 0.657 of their unit length, ranging 0.37 to 0.79:
that residual is *how much of a word's direction the top 50 PCs actually
captured*. (Not renormalizing afterwards is deliberate. Rescaling a 0.37 row back
to 1.0 would assert a precision the projection doesn't have.)

That residual is not noise, and it is not uniform:

- it correlates **+0.75** (Mengzi) / **+0.69** (SEP) with `strength` — words in
  dense regions of the similarity graph are the ones PCA captures well;
- on the Mengzi it correlates **−0.34** with `doc_freq`: the most widespread words
  are among the *worst* captured, which fits [[frequency-gradient-and-debiasing]]
  — `abtt` removes the dominant directions those words lived on, leaving a
  residual spread thinly across many dimensions;
- **信 sits in the 1st percentile** (0.479, 6th-worst of 602) while 仁/禮/智/義
  are 34th–86th. So one of the five virtues is represented markedly worse than
  its peers in the exported space.

The failure mode is not what you would guess. 信's own top-10 neighbors survive
truncation about as well as the other targets' (6/10, versus 5/10 for 禮 and 義).
What happens instead is that **poorly-captured words crowd into everyone else's
neighborhoods**: 信 appears in the exported top-10 of 仁, 禮 and 義 and in the
768-d top-10 of none of them. Across the whole Mengzi vocabulary, retained norm
versus the change in how often a word is listed as someone else's neighbor is
**spearman −0.657**. A short projection leaves a noisy direction, and after
normalization that noise lands plausibly close to many things.

Scope: this affects the **exported vectors** — the client-side PCA and live t-SNE
views, and any `reduced` layout. It does *not* touch the pipeline's similarity
graph, communities, or reported cosines, which are computed before truncation.

This is the sharpest argument for keeping `full` around: not that the average
layout is better (it barely is), but that it is the only way to check whether a
suspicious adjacency on the scatter is real or a short-projection artifact.

**What to do with that.** It settled the division of labour. Since the browser
can recompute any 50-d layout on demand, precomputing those buys nothing, and the
`full` sweep — modest as its gain is — is the one thing shipping can add. So
`tsne_sources = ("full",)`, and the reduced layouts are computed client-side via
the "recompute" checkbox. The comparison stays available (tick the box and the
same perplexity is recomputed at 50-d), it just isn't shipped.

That also means the aggregate numbers above aren't really what `full` is for.
Where it earns its place is the spurious-neighbour check below: when an adjacency
on the scatter looks surprising, the 768-d layout answers "is this the space, or
is this the truncation?"

## Consequences to keep in mind

**The doc-frequency filter now hides points rather than shaping the layout.** The
SEP scatter is mounted with `minDocFreq=0.2 / maxDocFreq=0.95`; a precomputed
layout was computed with those words present, so filtering removes points from a
map whose geometry they helped set. The live t-SNE, by contrast, lays out only
the surviving subset. This is the better default — the map stays put as you
filter, and the neighborhood structure is the corpus's, not the filter's — but it
does mean the two t-SNE options are not the same picture. To have the thinning
shape the layout, thin the *vocabulary* instead, via `Pipeline.min_doc_freq` /
`max_doc_freq`, which the export already supports.

**Perplexity 5 is not a useful default.** At 602 / 3283 nodes it reads as
near-uniform noise: too local to show community structure. The sweep therefore
leads with 30. This is a display choice, not a finding — the low-perplexity
layouts are still there to switch to, and their fragmentation is itself
informative about how little global structure survives debiasing.

**Precomputed and live t-SNE will not agree point-for-point** even at equal
perplexity: different implementations (scikit-learn vs `@keckelt/tsne`), different
iteration counts (1000 vs 500), and learning rates that are not on a shared scale
despite both being called epsilon. They should agree about *neighborhoods*, which
is all t-SNE ever claims.

## Retuning without a pipeline run

`scripts/tools/relayout.py` rebuilds the layouts from an artifact already on disk
(`public/embeddings/{corpus}.json`), since they only ever see the exported
vectors. Everything else in the file passes through untouched, so it is safe
against a mid-experiment artifact — it cannot change what the scatter plots, only
where the t-SNE view puts it.

```
scripts/.venv/bin/python scripts/tools/relayout.py                  # both corpora
scripts/.venv/bin/python scripts/tools/relayout.py --perplexity 12  # try a value
scripts/.venv/bin/python scripts/tools/relayout.py --source full    # 768-d only
```

`full` layouts need the untruncated matrix, which is not in the artifact, so a
pipeline run caches it under the gitignored `scripts/.cache/vectors/{corpus}.npz`
(~2 MB Mengzi, ~10 MB SEP). `relayout` checks the cached labels against the
artifact's and refuses the `full` sweep on a mismatch, rather than pairing one
run's coordinates with another's vocabulary. With no cache the `full` sweep is
skipped with a warning and the `reduced` one still runs.

That check is not theoretical: re-running SEP produced 3277 words against the
shipped artifact's 3283, and the differences (`account`/`accounting`,
`discussion`/`discuss`) are **variant-merge flips, not corpus drift** — pairs
sitting near the 0.45 cosine gate, which tiny embedding differences push across.
Worth knowing on its own: the SEP vocabulary is not bit-stable across runs, so
anything that pairs a stored vocabulary with a fresh one needs a guard like this.
The shipped SEP artifact therefore carries `reduced` layouts only; giving it
`full` ones means a real `--corpus sep` run (~63 min, and it rewrites
`public/sep/`).

One honest caveat for the `reduced` sweep: the artifact stores vectors rounded to
5 decimals, so a relayout embeds the *rounded* matrix while a pipeline run embeds
the unrounded one. The maps are visually the same and the neighborhoods
identical, but they are not bit-identical — don't expect a full run to reproduce
a relayout exactly. (The `full` sweep reads the cache, which is unrounded
float32, so it doesn't have this problem.)

## References

- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*,
  9, 2579–2605.
- Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
  transcriptomics. *Nature Communications*, 10, 5416.
  https://www.nature.com/articles/s41467-019-13056-x
- Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to Use t-SNE Effectively.
  *Distill*. https://distill.pub/2016/misread-tsne/
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*,
  12, 2825–2830. (`sklearn.manifold.TSNE`, Barnes-Hut.)
