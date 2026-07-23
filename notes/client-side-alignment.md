# Client-side interactive alignment of the Chinese & English spaces

How Task 2 (quantifying conceptual misalignment between the Mengzi's 仁/義 and their SEP
English renderings) is approached, and why it runs **in the browser** rather than as a
fixed offline computation. This is a design/methodology memo for the dissertation.

## The core idea

We have two monolingual semantic spaces, both 768-D:

- **Chinese** — GujiRoBERTa over the *Mengzi* (`analysis/centered-knn/`), targets 仁 義.
- **English** — roberta-base over SEP (`analysis/sep-centered-knn/`), targets
  benevolence / humaneness / righteousness / justice.

Their axes are unrelated (different models, different languages), so cross-space distances
are meaningless until the spaces are **aligned**. The standard tool is an orthogonal
**Procrustes** map fit on a set of bilingual **anchor** pairs — translation pairs assumed
to name "the same" concept in both corpora — after which the residual distance between a
virtue and its rendering measures misalignment.

**The pivot:** *which terms are legitimate anchors is exactly the question this project
studies.* Baking one anchor set into a single offline alignment would pre-decide the thing
under investigation. So instead the alignment is **interactive and client-side**: the user
selects anchor pairs and watches the shared space re-form live. Python only **exports
precomputed artifacts**; there is no alignment "answer" shipped, only the raw materials for
the user to explore. A Python server buys nothing here — it would just add latency to the
loop we want to feel instant.

## What the alignment actually computes

Given the user's chosen anchor pairs — English anchor matrix `A` (k×d) and Chinese `B`
(k×d) — orthogonal Procrustes finds the rotation/reflection `R` minimizing `‖A R − B‖`:
`R = U Vᵀ` where `U Σ Vᵀ = SVD(Aᵀ B)` (Schönemann, 1966). Aligned English is then `E R`.

Two preprocessing steps, standard in cross-lingual embedding alignment and doubly
justified by our anisotropy work: **mean-center** each space and **L2-normalize** each
vector before fitting. The orthogonality constraint (rather than a free linear map) is the
right choice: it preserves within-space distances and angles, so alignment can only
*rotate* the English cloud onto the Chinese one, not distort it to force anchors together —
the finding across the cross-lingual literature that orthogonal maps generalize better than
unconstrained ones (Mikolov et al., 2013; Xing et al., 2015; Smith et al., 2017).

## Why a single anchor pair does not work

Aligning on one pair (e.g. "assume 仁 ≡ benevolence, fit everything else") constrains
exactly **one direction** of the 768; the rotation in the remaining 767 dimensions is
mathematically unconstrained (`Aᵀ B` is rank 1). Benevolence lands on 仁 *by construction*
and every other cross-space position is arbitrary. So per-pair alignments are not a
meaningful shared space. More generally, the alignment is only trustworthy in the subspace
the anchors span; **few anchors → unreliable alignment**. Rather than hide this, the tool
surfaces it (an "N anchors selected" cue; sparse-anchor states read as exploratory), which
turns the fragility into part of the argument the user can feel.

## What gets precomputed offline (Python → `src/data/alignment/`)

A new exporter (`cli/export_alignment.py`) emits:

1. **Reduced vectors, both spaces.** Mean-center + L2-normalize, then PCA each space to
   ~50 dims **independently** (Procrustes tolerates each space having its own orthonormal
   basis; it only needs matching dimensionality). Reducing dims (a) makes the client-side
   SVD a trivial 50×50 and (b) fixes the otherwise-underdetermined 768-D Procrustes. Ship
   as a compact `Float32Array`/`.bin` (or JSON): `{labels, is_target, vecs[N×50]}` per
   corpus — ~250 KB total, versus ~4 MB for the raw 768-D vectors.
2. **Fixed Chinese 2-D PCA frame** — the static backdrop the aligned English terms drop
   into: the Chinese 2-D coordinates *plus* the 50→2 projection axes, so English can be
   projected into the **same** frame after alignment. (A precomputed Chinese **t-SNE**
   layout is a later export, for the t-SNE view.)
3. **Candidate anchor pairs** — from CC-CEDICT (`src/data/cedict_…mdbg.txt`): every
   Mengzi-vocab hanzi whose gloss intersects a SEP-vocab English term yields a candidate
   `(hanzi ↔ english)` pair. The target virtues are excluded (you cannot anchor on the
   thing you are measuring). The anchor menu thus builds itself from the data.

## Visualizations

**PCA view (build first — cheaper, simpler).** On each anchor change: fit Procrustes in
50-D, map English `E R`, project onto the **fixed** Chinese PCA axes. Only the English
terms move; the Chinese map is a stationary reference. Deterministic and instant.

**t-SNE view (follow-on — better local fidelity).** PCA can flatten the local
intra-corpus neighborhoods that matter; t-SNE preserves them (van der Maaten & Hinton,
2008) but is too slow/unstable to recompute live. The fix is to **precompute the Chinese
t-SNE once** (fixed backdrop) and place each aligned English term by **out-of-sample
"landmark" embedding**: drop it at the similarity-weighted average (softmax over the
top-k) of its nearest Chinese neighbors' precomputed 2-D coordinates. Cheap per point,
stable (the map underneath never moves), and it updates live because the high-D neighbors
change with the alignment.

Caveat for both views: English terms are placed *within the Chinese neighborhood
structure*, so the main view reads "does benevolence fall among 仁's Mengzi neighbors?" —
misalignment against the Chinese map — rather than English's own local structure. Showing
English's intrinsic neighborhoods too would be a separate, symmetric two-map view
(post-MVP).

## A Procrustes-free alternative to keep in mind

**Relative representations** (Moschella et al., 2023): re-describe every term by its vector
of *cosines to the anchor set*. Because within-space angles are preserved across encoders,
two spaces expressed in "similarity-to-anchors" coordinates become directly comparable
**without any rotation fit**. It is cheap, client-friendly, and a natural future toggle —
but, like Procrustes, it needs several anchors to carry signal (one anchor collapses it to
a 1-D shadow). Worth exposing as an alternate alignment mode once the Procrustes view lands.

## References

- Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem.
  *Psychometrika*, 31(1), 1–10.
- Mikolov, T., Le, Q. V., & Sutskever, I. (2013). Exploiting Similarities among Languages
  for Machine Translation. arXiv:1309.4168.
- Xing, C., Wang, D., Liu, C., & Lin, Y. (2015). Normalized Word Embedding and Orthogonal
  Transform for Bilingual Word Translation. *NAACL-HLT*, 1006–1011.
- Smith, S. L., Turban, D. H. P., Hamblin, S., & Hammerla, N. Y. (2017). Offline Bilingual
  Word Vectors, Orthogonal Transformations and the Inverted Softmax. *ICLR*.
  https://arxiv.org/abs/1702.03859
- Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., & Rodolà, E. (2023).
  Relative representations enable zero-shot latent space communication. *ICLR*.
  https://arxiv.org/abs/2209.15430
- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9,
  2579–2605.
