"""Recompute the precomputed scatter layouts on an artifact already on disk.

``public/embeddings/{corpus}.json`` ships a ``layouts`` array — one t-SNE per
``Pipeline.tsne_perplexities`` — which the site plots directly (see
``embeddings/layouts.py``). A full pipeline run regenerates them, but a full run
also re-fetches the corpora and re-embeds on the GPU, which is minutes of work to
change one number.

The layouts only ever see the *exported* (PCA-reduced) vectors, so they can be
rebuilt from the artifact alone — that is what this does. Everything else in the
file (nodes, communities, metrics, provenance) is passed through untouched, so
this is safe to run against a mid-experiment artifact: it cannot change what the
scatter plots, only where the t-SNE view puts it.

Run:  scripts/.venv/bin/python scripts/tools/relayout.py [--corpus mengzi]
      scripts/.venv/bin/python scripts/tools/relayout.py --perplexity 12 --perplexity 40
      scripts/.venv/bin/python scripts/tools/relayout.py --source reduced --source full

``--perplexity`` / ``--source`` (both repeatable) override
``Pipeline.tsne_perplexities`` / ``tsne_sources`` for the run without touching
``config.py`` — for trying a value before committing to it.

``--source full`` embeds the untruncated analysis matrix, which is not in the
artifact; it comes from the cache a pipeline run writes
(``embeddings.vectors.cache_analysis_matrix``). Without that cache — or if it
belongs to an older vocabulary — the full sweep is skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import config
from embeddings.layouts import tsne_layouts
from embeddings.vectors import load_analysis_matrix

PIPELINES = {"mengzi": config.MENGZI_PIPELINE, "sep": config.SEP_PIPELINE}


def relayout(corpus: str, perplexities: tuple[float, ...] | None,
             sources: tuple[str, ...] | None) -> Path:
    """Rewrite ``{corpus}.json``'s ``layouts`` in place; returns the path."""
    p = PIPELINES[corpus]
    if perplexities:
        p = replace(p, tsne_perplexities=perplexities)
    if sources:
        p = replace(p, tsne_sources=sources)
    path = config.EMBEDDINGS / f"{corpus}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    labels = [n["id"] for n in nodes]
    reduced = np.array([n["vec"] for n in nodes], dtype=float)
    print(f"\n=== {corpus} === {len(labels)} nodes x {reduced.shape[1]} dims "
          f"-> {path}")

    matrices: dict[str, np.ndarray] = {"reduced": reduced}
    # The full analysis matrix isn't shipped, so `full` layouts can only be
    # rebuilt from the cache a pipeline run leaves behind — and only if that run
    # produced *these* nodes. A stale cache would pair one run's coordinates with
    # another's vocabulary, so check the labels rather than trusting the file.
    if "full" in p.tsne_sources:
        cached = load_analysis_matrix(corpus)
        if cached is None:
            print("  no cached analysis matrix — run the pipeline for this "
                  "corpus to enable 'full' layouts")
        elif cached[0] != labels:
            print(f"  cached analysis matrix is stale ({len(cached[0])} labels "
                  f"vs {len(labels)} in the artifact) — skipping 'full'")
        else:
            matrices["full"] = cached[1]

    data["layouts"] = [asdict(l) for l in tsne_layouts(labels, matrices, p)]
    # Same dump options as models._save_json, so re-running this leaves the rest
    # of the file byte-identical rather than reflowing an artifact under review.
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=[*PIPELINES, "all"], default="all")
    ap.add_argument("--perplexity", type=float, action="append", default=None,
                    dest="perplexities",
                    help="Override Pipeline.tsne_perplexities (repeatable).")
    ap.add_argument("--source", action="append", default=None, dest="sources",
                    choices=["reduced", "full"],
                    help="Override Pipeline.tsne_sources (repeatable). 'full' "
                         "needs the analysis matrix a pipeline run caches.")
    args = ap.parse_args()

    corpora = list(PIPELINES) if args.corpus == "all" else [args.corpus]
    for corpus in corpora:
        if not (config.EMBEDDINGS / f"{corpus}.json").exists():
            print(f"skipping {corpus}: no artifact yet (run the pipeline first)")
            continue
        relayout(corpus, tuple(args.perplexities or ()),
                 tuple(args.sources or ()))


if __name__ == "__main__":
    main()
