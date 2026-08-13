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

``--perplexity`` (repeatable) overrides ``Pipeline.tsne_perplexities`` for the run
without touching ``config.py`` — for trying a value before committing to it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import config
from embeddings.layouts import tsne_layouts

PIPELINES = {"mengzi": config.MENGZI_PIPELINE, "sep": config.SEP_PIPELINE}


def relayout(corpus: str, perplexities: tuple[float, ...] | None) -> Path:
    """Rewrite ``{corpus}.json``'s ``layouts`` in place; returns the path."""
    p = PIPELINES[corpus]
    if perplexities:
        p = replace(p, tsne_perplexities=perplexities)
    path = config.EMBEDDINGS / f"{corpus}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    labels = [n["id"] for n in nodes]
    matrix = np.array([n["vec"] for n in nodes], dtype=float)
    print(f"\n=== {corpus} === {len(labels)} nodes x {matrix.shape[1]} dims "
          f"-> {path}")

    data["layouts"] = [asdict(l) for l in tsne_layouts(labels, matrix, p)]
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
    args = ap.parse_args()

    corpora = list(PIPELINES) if args.corpus == "all" else [args.corpus]
    for corpus in corpora:
        if not (config.EMBEDDINGS / f"{corpus}.json").exists():
            print(f"skipping {corpus}: no artifact yet (run the pipeline first)")
            continue
        relayout(corpus, tuple(args.perplexities or ()))


if __name__ == "__main__":
    main()
