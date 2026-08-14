"""Precomputed 2-D projections of the exported vectors.

The scatter plots used to project in the browser: PCA for free (the export's
columns are variance-ordered) and t-SNE live in a web worker. t-SNE turned out to
be the more legible view of the space by a wide margin, but recomputing it per
page load costs seconds of animation before anything is readable — so the
perplexities worth looking at are computed here, once, and shipped as plain x/y.
The client-side run stays available for a perplexity nobody precomputed.

Which vectors a layout embeds is a ``Pipeline`` knob (``tsne_sources``), because
the two defensible answers disagree:

- ``reduced`` — the **exported** PCA-reduced matrix. PCA-to-~50-then-t-SNE is van
  der Maaten's own recommendation (it denoises and makes the neighbor search
  tractable), and it is the only option the *client* has, so a precomputed layout
  and a live one are then looking at the same numbers.
- ``full`` — the untruncated analysis space, in the export's own preprocessing
  (mean-centered + L2-normalized: exactly the matrix the export's PCA is fitted
  on). Nothing is thrown away, so this is the layout to check when you want to
  know whether the 50-d truncation was costing structure.

Configuring both ships both, and the scatter's picker can flip between them.

See ``notes/claude/precomputed-tsne-layouts.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from models import Layout, Pipeline

TsneSource = Literal["reduced", "full"]


def tsne_layouts(labels: list[str], matrices: "Mapping[str, np.ndarray]",
                 p: "Pipeline") -> "list[Layout]":
    """A :class:`models.Layout` per (``tsne_sources`` x ``tsne_perplexities``).

    ``matrices`` supplies whichever sources the caller has — ``reduced`` (the
    matrix as exported) and/or ``full`` (the pre-export analysis matrix). Rows
    must be parallel to ``labels``. A configured source that isn't supplied is
    skipped with a warning rather than failing the run: ``tools/relayout.py``
    can only offer ``full`` when a pipeline run has cached it.

    Order is the config's order, so the **first** layout — first source, first
    perplexity — is what the scatter opens on. Returns ``[]`` when the corpus is
    too small to embed or nothing is configured, which is a valid export: the
    client falls back to PCA.
    """
    from models import Layout

    n = len(labels)
    if n < 3 or not p.tsne_perplexities or not p.tsne_sources:
        return []

    # Only name the source when there's an ambiguity to resolve — a one-source
    # export shouldn't make the reader wonder what the other one was.
    show_source = len(p.tsne_sources) > 1
    layouts: list[Layout] = []
    for source in p.tsne_sources:
        matrix = matrices.get(source)
        if matrix is None:
            print(f"layout     : no {source!r} vectors available — skipped")
            continue
        # `reduced` must stay exactly what the client downloaded. `full` gets the
        # export's own center + L2 step, leaving truncation as the only variable.
        if source == "full":
            from embeddings.vectors import unit_vectors
            matrix = unit_vectors(matrix)
        dims = int(matrix.shape[1])
        seen: set[float] = set()
        for requested in p.tsne_perplexities:
            # sklearn requires perplexity < n_samples; the effective number of
            # neighbors is 3*perplexity, so clamp there to keep the layout honest
            # rather than let a small corpus silently run a degenerate embedding.
            perplexity = float(max(2.0, min(requested, (n - 1) / 3)))
            if perplexity != requested:
                print(f"layout     : perplexity {requested:g} -> {perplexity:g} "
                      f"(clamped for {n} nodes)")
            # A corpus small enough to clamp collapses several requests onto one
            # perplexity; ship it once rather than offer the same map under three
            # different (and now wrong) labels. Everything below names the
            # *effective* value for the same reason.
            if perplexity in seen:
                continue
            seen.add(perplexity)
            coords = _tsne(matrix, perplexity, p)
            prefix = "tsne" if source == "reduced" else f"tsne-{source}"
            label = f"t-SNE · perplexity {perplexity:g}"
            layouts.append(Layout(
                id=f"{prefix}-p{perplexity:g}",
                method="tsne",
                label=f"{label} · {dims}-d" if show_source else label,
                params={
                    "perplexity": perplexity,
                    "epsilon": p.tsne_epsilon,
                    "iterations": p.tsne_iterations,
                    "seed": p.tsne_seed,
                    "source": source,
                    "dims": dims,
                },
                # 3 decimals: the coordinates only ever feed a screen-space
                # scale, and full precision would add megabytes across a sweep.
                coords={word: [round(float(x), 3), round(float(y), 3)]
                        for word, (x, y) in zip(labels, coords)},
            ))
        print(f"layouts    : {len(seen)} t-SNE from {source} vectors "
              f"({n} nodes x {dims} dims) — perplexity "
              f"{', '.join(f'{q:g}' for q in sorted(seen))}")
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
