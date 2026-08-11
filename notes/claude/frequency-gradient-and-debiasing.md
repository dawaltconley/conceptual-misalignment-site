# The frequency gradient on the scatter — spotting it, diagnosing it, and debiasing it

Conversation summary (2026-08-02). Companion to [[embedding-communities-and-semantics]] and
[[register-dominant-clustering]] (which explain *why* whole-vocab embedding structure is
register/frequency-dominated) and [[anisotropy-and-network-construction]] (the centering step).
This note is the practical episode: we noticed a gradient on the scatter, worked out what it was,
built a tool to measure it, and fixed it with **all-but-the-top debiasing**. Verified reference
list at the bottom. Written to be readable while still learning the concepts, so terms are
defined inline.

## What we noticed

Coloring the embedding scatter by each word's **document frequency** (how many documents a word
appears in) showed a **clear directional gradient on the PCA map** — df rose smoothly as you moved
along one axis — and a **weaker** version of the same on the t-SNE map. Then, coloring by
**strength** and **PageRank** (two measures of how central/well-connected a word is in the
similarity network) showed a *different* shape: the strongest words sat at **both** the left and
right extremes, with the weak ones bunched in the **middle** — a two-sided pattern, not a
one-directional slope.

Two quick definitions:
- **PCA** (Principal Component Analysis): a *linear* projection that finds the directions of
  greatest spread in the data and flattens onto the top two. It preserves the big, global shape.
- **t-SNE**: a *non-linear* projection that tries to keep each point near its true neighbors,
  deliberately sacrificing the global shape to make local clusters legible.

## Why this is expected (in plain terms)

Word **frequency is one of the biggest things that varies** in a contextual-embedding space — not
because frequency is meaningful, but because of *how the vectors are shaped*:

- Contextual vectors are **anisotropic**: instead of pointing every which way, they crowd into a
  narrow cone (Ethayarajh 2019; Mimno & Thompson 2017). Frequency is baked into *where* in that
  cone a word lands (Gao et al. 2019, the "representation degeneration" problem).
- **Centering** (subtracting the average vector — our anisotropy fix) removes *one* shared
  direction, but frequency lives across **several** of the top directions, so a gradient survives.

Because PCA keeps the **global** high-variance directions, and frequency *is* a high-variance
direction, coloring by df paints a clean slope along a PCA axis. t-SNE throws the global shape away
in favor of local neighborhoods, so the same slope gets folded and scrambled — hence "weaker on
t-SNE." **The discrepancy is a property of the two methods, not a sign either is wrong.**

## The two readings of the two-sided (strength / PageRank) pattern

A one-directional gradient (df) is a *signed* direction: it increases one way. A two-sided pattern
(high at both ends, low in the middle) is a *magnitude*: it's about being **far from the center**
in either direction. That can come from two very different situations, and they need different
responses:

1. **Radial / "norm" reading.** Central words are simply **far from the centroid** (their vectors
   have larger *norm* = length = distance from the average word). Project that onto one axis and
   "far from center in any direction" looks like "high at both ends." If this is the cause,
   **debiasing should flatten it** — it's the same frequency/anisotropy nuisance.
2. **Bipolar-contrast reading.** The axis genuinely separates **two communities** (say two
   registers, or two parts of speech), and the hubs are each community's core (the two poles),
   while the weak middle is bridge/ambiguous words. If this is the cause, **debiasing will *not*
   remove it** — it's real signal, not an artifact.

The whole question was: *which one is it?* — because it decides whether the pattern is noise to
remove or structure to keep.

## The diagnostic tool

We wrote `scripts/tools/debias_diagnostics.py`, which reads the shipped artifact
(`public/embeddings/{corpus}.json`) and reports three things. It uses **Spearman ρ**, a
correlation that ranges from −1 to +1 and works on *ranks* (so it catches any monotonic
relationship, not just a straight line):

- **A. Metric vs. position.** For each word-metric, ρ against the *signed* first component (PC1) —
  catches a **directional** gradient — vs. against the *absolute* value |PC1| and the **radius**
  (distance from the 2-D center) — catches a **two-sided/magnitude** pattern.
- **B. Radial vs. bipolar.** Splits the map into angular wedges and checks whether strength tracks
  radius in **every** wedge (⇒ radial) or only in the left–right wedges (⇒ two real poles); plus a
  |PC1|-vs-|PC2| symmetry check.
- **C. Pole cross-tab.** Splits words into left / middle / right bands and tabulates which
  **communities** land in each — are the two poles *different* communities (bipolar) or the *same*
  ones (radial)?

### One field we had to add: `norm`

The scatter vectors in the artifact are **L2-normalized before PCA** — i.e. every vector is scaled
to length 1, keeping only its *direction* and throwing away its *length*. But length (distance from
the centroid) is exactly the quantity reading #1 hinges on. So the pipeline now also exports a
per-word **`norm`** field (the vector's length in the centered/debiased space) purely so the
diagnostic can test the radial hypothesis. **`norm` is an output, not a setting** — it's written on
every run regardless of any debias choice.

## The baseline measurement (before debiasing)

Reading the columns by magnitude (PCA's sign is arbitrary, so `+0.90` and `−0.90` are equally
"strong"):

**Mengzi**
| metric | PC1 | \|PC1\| | radius | reading |
|---|---|---|---|---|
| doc_freq | **+0.90** | +0.10 | +0.08 | DIRECTIONAL |
| strength | +0.00 | **+0.87** | **+0.91** | TWO-SIDED |
| pagerank | −0.20 | **+0.76** | **+0.83** | TWO-SIDED |
| eigenvector | **+0.86** | +0.25 | +0.22 | DIRECTIONAL |

**SEP**
| metric | PC1 | \|PC1\| | radius | reading |
|---|---|---|---|---|
| doc_freq | **+0.84** | +0.17 | +0.20 | DIRECTIONAL |
| strength | −0.14 | **+0.88** | **+0.88** | TWO-SIDED |
| pagerank | −0.12 | **+0.74** | **+0.77** | TWO-SIDED |
| eigenvector | **−0.93** | +0.04 | +0.03 | DIRECTIONAL |

So df (and eigenvector centrality) rode a **signed** axis — the frequency gradient — while strength
and PageRank were strongly **two-sided**.

## The fix: all-but-the-top debiasing

We added a `debias` knob to the pipeline (`Pipeline.debias` in `models.py`; the maths in
`embeddings.vectors.debias_matrix`), applied right after centering:

- **`abtt`** — *all-but-the-top* (Mu & Viswanath 2018). After centering, it finds the top few
  principal directions (the ones carrying frequency/register) and **projects them out** — deletes
  those axes from every vector. `debias_k` sets how many to remove (default `max(1, D/100)` ≈ the
  single top axis here).
- **`whiten`** — PCA-whitening (Su et al. 2021): rescales every direction to equal variance so
  *no* direction can dominate. A stronger, blunter alternative.
- **`none`** — unchanged (the default; existing behavior).

Setting `debias="abtt"` in `config.py` and regenerating was the entire fix. (You set only that;
everything else, including `norm`, followed automatically.)

## After `abtt` — the result

**Mengzi**
| metric | PC1 | \|PC1\| | radius | reading |
|---|---|---|---|---|
| doc_freq | −0.03 | −0.11 | −0.16 | weak / none |
| norm | −0.07 | +0.12 | +0.14 | weak / none |
| strength | +0.10 | +0.25 | +0.34 | two-sided (mild) |
| pagerank | +0.08 | +0.26 | +0.34 | two-sided (mild) |
| eigenvector | −0.09 | +0.24 | +0.27 | weak / none |

**SEP**
| metric | PC1 | \|PC1\| | radius | reading |
|---|---|---|---|---|
| doc_freq | −0.03 | −0.08 | −0.04 | weak / none |
| norm | +0.02 | −0.06 | −0.02 | weak / none |
| strength | +0.10 | +0.19 | +0.27 | weak / none |
| pagerank | +0.05 | +0.18 | +0.26 | weak / none |
| eigenvector | +0.36 | +0.14 | +0.16 | directional (residual) |

**Before → after, side by side:**
| | doc_freq vs PC1 | strength \|PC1\| / radius | verdict |
|---|---|---|---|
| Mengzi before | +0.90 | +0.87 / +0.91 | directional + bipolar |
| Mengzi after | **−0.03** | +0.25 / +0.34 | gone / mild |
| SEP before | +0.84 | +0.88 / +0.88 | directional + bipolar |
| SEP after | **−0.03** | +0.19 / +0.27 | gone / weak-radial |

## What we learned

- The **document-frequency gradient collapsed** (0.90 → ~0 on both corpora): it *was* the
  frequency-dominated top principal component — precisely what all-but-the-top removes.
- The **strength/PageRank two-sidedness mostly dissolved** too (|PC1| from ~0.87 down to ~0.2).
  Since removing the frequency directions took most of it with it, that pattern was **mostly the
  norm/frequency nuisance (reading #1), not a genuine two-community contrast (reading #2)** — had
  it been real register structure, abtt wouldn't have flattened it. The tool's post-fix verdict
  agrees: SEP reads **RADIAL**, Mengzi **MIXED/weak**.
- Practically, the scatter — **especially the t-SNE view** — is now far more coherent, because the
  points are no longer arranged mostly by how common each word is.

## Caveats & knobs

- **abtt with default `debias_k` removes only the single top axis.** If a gradient creeps back,
  raise `debias_k` to 2–3 (remove more directions) or try `debias="whiten"`, and rerun the tool to
  compare.
- One **residual** to watch: SEP `eigenvector` is still directional (+0.36) after abtt — a leftover
  register-ish axis. The pole cross-tab in `analysis/sep/debias_diagnostics.md` says whether it's a
  real community split worth keeping.
- Debiasing changes the *geometry* the similarity graph and communities are built from; re-check
  community coherence after changing it (see the "to try" list in [[embedding-communities-and-semantics]]).

## References (verified via search, 2026-08-02)

- Mu, J., Viswanath, P. (2018). *All-but-the-Top: Simple and Effective Postprocessing for Word
  Representations.* ICLR. https://arxiv.org/abs/1702.01417
- Ethayarajh, K. (2019). *How Contextual are Contextualized Word Representations? Comparing the
  Geometry of BERT, ELMo, and GPT-2 Embeddings.* EMNLP-IJCNLP. https://aclanthology.org/D19-1006/
- Gao, J., He, D., Tan, X., Qin, T., Wang, L., Liu, T.-Y. (2019). *Representation Degeneration
  Problem in Training Natural Language Generation Models.* ICLR. https://arxiv.org/abs/1907.12009
- Mimno, D., Thompson, L. (2017). *The strange geometry of skip-gram with negative sampling.*
  EMNLP, 2873–2878. https://aclanthology.org/D17-1308/
- Su, J., Cao, J., Liu, W., Ou, Y. (2021). *Whitening Sentence Representations for Better Semantics
  and Faster Retrieval.* arXiv:2103.15316. https://arxiv.org/abs/2103.15316
- van der Maaten, L., Hinton, G. (2008). *Visualizing Data using t-SNE.* Journal of Machine
  Learning Research 9, 2579–2605. https://jmlr.org/papers/v9/vandermaaten08a.html
