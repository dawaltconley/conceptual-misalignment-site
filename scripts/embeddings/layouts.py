"""Precomputed 2-D projections of the exported vectors.

The scatter plots used to project in the browser: PCA for free (the export's
columns are variance-ordered) and t-SNE live in a web worker. t-SNE turned out to
be the more legible view of the space by a wide margin, but recomputing it per
page load costs seconds of animation before anything is readable — so the
perplexities worth looking at are computed here, once, and shipped as plain x/y.
The client-side run stays available for a perplexity nobody precomputed.

Everything runs over the **exported** (PCA-reduced) matrix, not the full analysis
space: PCA-to-50-then-t-SNE is van der Maaten's own recommendation (it denoises
and makes the neighbor search tractable), and it keeps a precomputed layout
comparable with one the client computes from the same vectors it downloaded.

See ``notes/claude/precomputed-tsne-layouts.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models import Layout, Pipeline


def tsne_layouts(labels: list[str], matrix: np.ndarray,
                 p: "Pipeline") -> "list[Layout]":
    """One :class:`models.Layout` per ``Pipeline.tsne_perplexities`` entry.

    ``matrix`` is the reduced matrix as exported (rows parallel to ``labels``).
    Returns ``[]`` when the corpus is too small to embed or no perplexities are
    configured, which is a valid export — the client falls back to PCA.
    """
    from models import Layout

    n = len(labels)
    if n < 3 or not p.tsne_perplexities:
        return []

    layouts: list[Layout] = []
    for requested in p.tsne_perplexities:
        # sklearn requires perplexity < n_samples; the effective number of
        # neighbors is 3*perplexity, so clamp there to keep the layout honest
        # rather than let a small corpus silently run a degenerate embedding.
        perplexity = float(max(2.0, min(requested, (n - 1) / 3)))
        coords = _tsne(matrix, perplexity, p)
        if perplexity != requested:
            print(f"layout     : perplexity {requested:g} -> {perplexity:g} "
                  f"(clamped for {n} nodes)")
        layouts.append(Layout(
            id=f"tsne-p{requested:g}",
            method="tsne",
            label=f"t-SNE · perplexity {requested:g}",
            params={
                "perplexity": perplexity,
                "epsilon": p.tsne_epsilon,
                "iterations": p.tsne_iterations,
                "seed": p.tsne_seed,
                "dims": int(matrix.shape[1]),
            },
            # 3 decimals: the coordinates only ever feed a screen-space scale,
            # and full precision would add megabytes across a perplexity sweep.
            coords={label: [round(float(x), 3), round(float(y), 3)]
                    for label, (x, y) in zip(labels, coords)},
        ))
    print(f"layouts    : {len(layouts)} t-SNE ({n} nodes) — perplexity "
          f"{', '.join(f'{q:g}' for q in p.tsne_perplexities)}")
    return layouts


def _tsne(matrix: np.ndarray, perplexity: float, p: "Pipeline") -> np.ndarray:
    from sklearn.manifold import TSNE
    return np.asarray(TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=p.tsne_epsilon,  # pyright:ignore -- stub says str-only
        max_iter=p.tsne_iterations,
        # PCA init (over random) makes the global arrangement reproducible and
        # far less seed-dependent — Kobak & Berens (2019).
        init="pca",
        random_state=p.tsne_seed,
    ).fit_transform(matrix))
