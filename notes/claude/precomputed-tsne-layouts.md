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

Yes, quite a lot. Take each word's 10 nearest neighbors in the **768-d analysis
space** — the space where the similarity graph, the communities and every
reported cosine actually live — and ask what fraction of them are still among its
10 nearest neighbors on the map. Mengzi, 602 words, perplexity 30:

| space read                       | neighbors retained |
| -------------------------------- | -----------------: |
| exported 50-d vectors (no t-SNE) |              0.550 |
| t-SNE over the 50-d vectors      |              0.218 |
| t-SNE over the full 768-d space  |              0.312 |

Two things to take from this. First, **the PCA truncation alone discards about
half the neighborhood structure** before t-SNE is even involved — the 50-d export
is a much lossier object than "50 principal components of a 768-d space" sounds.
Second, embedding the full space **recovers a good part of what the shipped
layout was losing** (+43% relative), so the two maps are not cosmetic variants of
each other.

The honest caveat: this metric uses the full space as ground truth, which is the
space the `full` layout starts from, so some advantage is baked in. But that
framing is the right one — the question is whether the picture on screen
represents the space the analysis is conducted in, and by that standard the
truncated layout is measurably worse. The numbers come from a scratch script, not
a committed tool; rerun it with `embeddings.vectors.load_analysis_matrix` if you
want it against SEP or another k.

This is an argument for making `full` the default view (put it first in
`tsne_sources`). It is not an argument for dropping `reduced`: the live
client-side t-SNE can only ever see the exported vectors, so keeping a `reduced`
layout is what makes "precomputed vs live" a fair comparison.

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
