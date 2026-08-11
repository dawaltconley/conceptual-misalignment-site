# Anisotropy in the term vectors, and how it shapes the similarity network

Why the first cosine-similarity networks connected almost every character at
uniformly high weight, what causes it, and the four remedies we compared. This
documents a methodological decision for the dissertation: how the semantic
network in `scripts/embed/` is constructed and why.

## The symptom

The initial network (`analysis/task1/`, and reproduced as `analysis/baseline/`:
raw pooled vectors, cosine edges at threshold 0.75) was almost a complete graph.
In the pruned per-target neighborhood for 仁 (target + its top 15 neighbors)
**every** pair of nodes was connected — density 1.00 — and every edge weight sat
in a narrow 0.85–0.90 band. The network said "everything is strongly related to
everything," which is not a usable semantic map.

## The diagnosis: anisotropy / representation degeneration

This is not a bug in the pipeline; it is a well-documented geometric property of
contextual embeddings from BERT-family models. The vectors do not fill their
768-dimensional space — they occupy a narrow cone, all pointing in a similar
direction. Because cosine similarity measures angle, any two vectors in a narrow
cone look similar regardless of meaning.

Measured on our own pooled term vectors (705 words):

- **Mean cosine of each word to the corpus-mean direction = 0.84.** Every vector
  is strongly aligned with one shared common component.
- **Pairwise cosine distribution: mean 0.706, std 0.074, range [0.22, 0.94].**
  There is no meaningful zero; two *random* words already sit at ~0.71.
- At threshold 0.75, **29% of all word pairs** clear the bar — hence the near-
  complete graph among frequent words.

Ethayarajh (2019) first quantified this "anisotropy" for BERT, ELMo, and GPT-2,
showing that in the upper layers randomly sampled words have expected cosine
similarity far above zero. Gao et al. (2019) trace it to a "representation
degeneration problem" in which token embeddings are pushed into a narrow cone
during training. Timkey and van Schijndel (2021) sharpen the practical warning:
a *handful* of "rogue" dimensions (often 1–3) with large magnitude and variance
dominate cosine similarity, so the raw cosine can be "all bark and no bite" —
driven by a few outlier dimensions rather than by the semantic content that
actually matters to the model. **An absolute cosine threshold is therefore the
wrong knob:** the usable spread is only ~0.07 wide and offset far from zero, so
no cutoff cleanly separates structure — it just toggles between "keep everything"
and "keep almost nothing."

## The four remedies we compared

All numbers below are from fresh runs on the `segpos/chapters/` corpus (713
pooled words, 2 active targets 仁 義). Commands are the `-m embed.run` flags that
produced each folder under `analysis/`.

### 1. Baseline — raw cosine + threshold (`analysis/baseline/`)
The pathology, kept as a reference. Median node degree **240**; the 仁
neighborhood is a complete graph (density 1.00).

### 2. Mean-centering — subtract the corpus centroid (`analysis/mean-centering/`)
`--center --threshold 0.45`. The single most-cited fix: subtract the mean vector
from every embedding before computing cosine, removing the dominant common
direction. This is the first step of Mu and Viswanath's (2018) "All-but-the-Top"
postprocessing (they additionally project out the top few principal components;
we found that over-corrected on our small vocabulary and shrank the spread back
down, so we mean-center only). Standardization / whitening variants (Timkey and
van Schijndel, 2021; Li et al., 2020) attack the same problem.

Effect on the pairwise cosine distribution: **mean 0.706 → −0.001, std 0.074 →
0.147, range opens to [−0.62, +0.89].** A real zero returns and the spread
roughly doubles, so a threshold becomes meaningful again. At threshold 0.45 the
network drops to 234 nodes / 1,430 edges, median degree **5**, 28 Louvain
communities. The 仁 neighborhood density falls from 1.00 to ~0.75 — better, but
still dense, because pruning selects a target's most-similar words and in this
space those are also fairly similar *to each other*. Centering attenuates the
saturation; it does not fully escape it.

Note that mean-centering is a pure translation, so it is **invariant** for
Euclidean- and PCA-based artifacts: `pca.png`, `tsne.png` (Euclidean metric,
PCA init), `kmeans.csv`, and the `variance` column of `cohesion_variance.csv` are
unchanged. Only the cosine-derived artifacts move (cosine heatmap/CSVs, the
`cohesion` column, and the network).

### 3. Relative neighborhoods — k-nearest-neighbor graph (`analysis/relative-neighborhoods/`)
`--network knn --knn-k 8`. Instead of a global threshold, keep each node's `k`
most-similar neighbors by rank (union kNN: an edge if *either* endpoint ranks the
other in its top-k). Because it is **relative per node**, kNN is immune to the
global offset that makes an absolute threshold meaningless — it always surfaces
the k most-distinguishing relationships regardless of the absolute scale. This
produced the visually cleanest graphs (per-target density ~0.30–0.34, median
degree 10).

Honest caveat: kNN's tidiness is partly *imposed*, not *discovered*. It forces a
roughly uniform degree (~k) on every node, so it hides real variation in local
density — if a term genuinely has 25 strong neighbors kNN shows 8; if it has 3 it
pads to 8 with weaker ones. So a kNN graph answers "what are this term's nearest
neighbors and how do they interlink" cleanly, but you cannot read "how tightly
does this region cluster" off it. (kNN / shared-nearest-neighbor graphs are a
standard construction for high-dimensional data; the local-structure emphasis is
the same reason t-SNE is read for neighborhoods, not global distances — van der
Maaten and Hinton, 2008.)

### 4. Reversed log transform — DISCARDED
`--sim-transform poslog`. Wu and Wang (2025, p. 5) describe their log-transformed
cosine as "attenuating the saturation effects commonly observed in high-
similarity ranges." The pipeline's existing transform is `-ln(1 - s)`, which is
**convex** (slope `1/(1-s)` → ∞ as s → 1) and therefore *expands* the crowded
high-similarity range — the de-crowding reading of "attenuate saturation." We
tested the opposite direction, a concave `ln(1 + s)` that *compresses* the high
range, in case "attenuate" meant "dampen."

Two findings led us to discard it:
1. **A log transform is monotonic**, so at a matched percentile it does not change
   which edges exist — only their weights (and hence weighted-Louvain communities
   and the visual scale). It cannot fix a saturation problem that is fundamentally
   about the *offset*, which is what mean-centering addresses.
2. Empirically, `ln(1 + s)` compressed the edge weights into an even narrower
   high band ([0.50, 0.66]) — the opposite of de-crowding. This supports reading
   Wu and Wang's transform as the convex, expanding `-ln(1 - s)` the pipeline
   already had, i.e. our existing transform is likely aligned with their intent,
   not reversed. (Confirm against their p.5 figure: does it spread high
   similarities apart or pull them together?)

The `--sim-transform {none,neglog,poslog}` machinery remains in `analyze.py`
(`neglog` = the original `-ln(1-s)`, kept as the `log_transform` alias), but no
analysis run uses `poslog`.

### Best of both — centered kNN (`analysis/centered-knn/`)
`--center --network knn --knn-k 8 --max-nodes 25`. Run the rank-based kNN graph
*on the mean-centered vectors*. Mean-centering is not a monotone transform of the
cosine matrix, so it genuinely re-ranks neighbors on a de-anisotropized space;
the surviving kNN edges then carry **meaningful weights** instead of the
compressed anisotropy floor. This is the recommended construction.

## Comparison

Full network (713 pooled words):

| variant | flags | nodes | edges | median deg | communities | edge-weight range |
|---|---|---|---|---|---|---|
| baseline | raw, thr 0.75 | 677 | 75,216 | 240 | 8 | 0.75–0.94 |
| mean-centering | `--center`, thr 0.45 | 234 | 1,430 | 5 | 28 | 0.45–0.89 |
| relative-neighborhoods | `--network knn` k=8 | 713 | 5,300 | 10 | 7 | 0.59–0.94 |
| **centered-knn** | `--center --network knn` k=8 | 713 | 4,752 | 10 | 6 | **0.13–0.89** |

Per-target neighborhood density (target + top-N neighbors; how complete the
sub-graph is — lower is more legible):

| variant | 仁 density | 義 density |
|---|---|---|
| baseline | 1.00 | 1.00 |
| mean-centering | 0.73 | 0.78 |
| relative-neighborhoods | 0.34 | 0.30 |
| centered-knn (N=25) | 0.29 | 0.29 |

Two things make **centered-knn** the strongest:

- **Meaningful weights.** Raw kNN keeps clean topology but its weights are still
  crushed into the anisotropy band [0.59, 0.94], so every tie looks equally
  strong. Centering first opens the range to [0.13, 0.89], so a weak neighbor is
  visibly distinct from a strong one.
- **Full coverage, no isolates.** A threshold graph drops any node with no
  above-threshold edge (mean-centering keeps only 234 of 713 nodes; the rest
  render grey in the scatter plots). kNN gives every node its top-k edges, so all
  713 nodes are placed in a community — the t-SNE has no grey mass.

It stays just as legible at 25 neighbors (density 0.29) as raw kNN was at 15.

## Practical notes

- The target's own degree in a pruned per-target graph is capped by `--max-nodes`,
  not by the threshold: `prune_to_neighborhood` keeps the target + its top
  `max_nodes` neighbors, so 仁 shows 15 (or 25) spokes as long as it has that many
  neighbors at all. The threshold only controls the neighbor-to-neighbor edges.
- Relevant code: `vectors.center_matrix`; `analyze.build_knn_graph`,
  `apply_sim_transform`, `build_and_save_networks` (`method` / `knn_k` /
  `sim_transform` / `max_nodes`); CLI flags `--center`, `--network`, `--knn-k`,
  `--sim-transform`, `--max-nodes` in `run.py`.

## References

- Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?
  Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *EMNLP-IJCNLP*,
  55–65. https://aclanthology.org/D19-1006/
- Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.-Y. (2019).
  Representation Degeneration Problem in Training Natural Language Generation
  Models. *ICLR*. https://arxiv.org/abs/1907.12009
- Mu, J., & Viswanath, P. (2018). All-but-the-Top: Simple and Effective
  Postprocessing for Word Representations. *ICLR*. https://arxiv.org/abs/1702.01417
- Timkey, W., & van Schijndel, M. (2021). All Bark and No Bite: Rogue Dimensions
  in Transformer Language Models Obscure Representational Quality. *EMNLP*,
  4527–4546. https://aclanthology.org/2021.emnlp-main.372/
- Li, B., Zhou, H., He, J., Wang, M., Yang, Y., & Li, L. (2020). On the Sentence
  Embeddings from Pre-trained Language Models. *EMNLP*, 9119–9130.
  https://aclanthology.org/2020.emnlp-main.733/
- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE.
  *Journal of Machine Learning Research*, 9, 2579–2605.
- Wu & Wang (2025). [Semantic-space method adapted for this project; verify the
  full author names / title from the article.] *npj Heritage Science*.
  https://doi.org/10.1038/s40494-025-01893-7
