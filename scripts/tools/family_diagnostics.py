"""Audit the derivational variant merge — what merged, what nearly did, and at
which cosine threshold you would want to draw the line.

``Pipeline.merge_threshold`` is the one number in the variant merge with no
principled value: too high and the clutter stays, too low and stem-mates that
genuinely drifted apart (``know``/``knowledge``) get fused. This replays the
whole sweep so the threshold can be chosen by reading the merges.

Reads ``analysis/{corpus}/family_candidates.csv``, which the pipeline writes at
merge time: every candidate family's pairwise cosines **in the real analysis
space**. That file is what makes the sweep trustworthy. Sweeping against
``public/embeddings/{corpus}.json`` instead would mislead — the artifact stores
PCA-reduced vectors (``Pipeline.reduce_to_dims``, 50 by default) and the
truncation inflates cosine badly: 0.70 picks 231 merges on the reduced export
and only 31 on the space the gate actually sees.

Run:  scripts/.venv/bin/python scripts/tools/family_diagnostics.py [--corpus sep]
      scripts/.venv/bin/python scripts/tools/family_diagnostics.py --thresholds 0.3,0.4,0.5
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import config

DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)


def load_candidates(corpus: str) -> tuple[dict[int, list[str]], dict[tuple[str, str], float]]:
    """``(families, cosine)`` from the pipeline's real-analysis-space dump."""
    path = config.ANALYSIS / corpus / "family_candidates.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run `python -m main --corpus {corpus}` with "
            f"Pipeline.merge_variants enabled first")
    members: dict[int, set[str]] = defaultdict(set)
    cosine: dict[tuple[str, str], float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fam, a, b = int(row["family"]), row["a"], row["b"]
            members[fam] |= {a, b}
            cosine[(a, b) if a < b else (b, a)] = float(row["cosine"])
    return {k: sorted(v) for k, v in members.items()}, cosine


def counts_from_artifact(corpus: str) -> dict[str, int]:
    """Doc frequency per node, to pick each family's surviving label."""
    path = config.EMBEDDINGS / f"{corpus}.json"
    if not path.exists():
        return {}
    nodes = json.loads(path.read_text(encoding="utf-8"))["nodes"]
    return {n["id"]: n.get("doc_freq", 0) for n in nodes}


def complete_linkage(family: list[str], cos, threshold: float) -> list[list[str]]:
    """Split so every pair inside a cluster clears ``threshold`` (mirrors families.py)."""
    clusters = [[w] for w in family]
    while True:
        best, bi, bj = threshold, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                worst = min(cos(a, b) for a in clusters[i] for b in clusters[j])
                if worst >= best:
                    best, bi, bj = worst, i, j
        if bi < 0:
            return clusters
        clusters[bi] += clusters.pop(bj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="sep")
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated cosine floors to sweep")
    ap.add_argument("--show", type=float, default=None,
                    help="list every merge at this threshold (default: the median swept)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    taus = (tuple(float(t) for t in args.thresholds.split(","))
            if args.thresholds else DEFAULT_THRESHOLDS)
    fams, cosine = load_candidates(args.corpus)
    counts = counts_from_artifact(args.corpus)

    def cos(a: str, b: str) -> float:
        return cosine.get((a, b) if a < b else (b, a), -1.0)

    n_words = len({w for f in fams.values() for w in f})
    sims = np.array(list(cosine.values()))
    print(f"corpus     : {args.corpus}")
    print(f"candidates : {len(fams)} families, {n_words} words, "
          f"{len(cosine)} within-family pairs")
    print(f"cosine     : median {np.median(sims):.3f}  "
          f"p75 {np.percentile(sims, 75):.3f}  p90 {np.percentile(sims, 90):.3f}  "
          f"max {sims.max():.3f}")

    lines = [f"# Variant-merge diagnostics — {args.corpus}", "",
             f"{len(fams)} candidate families over {n_words} words "
             f"({len(cosine)} within-family pairs). Cosines are the real "
             f"analysis-space values the gate saw.", "",
             f"Within-family cosine: median {np.median(sims):.3f}, "
             f"p75 {np.percentile(sims, 75):.3f}, "
             f"p90 {np.percentile(sims, 90):.3f}, max {sims.max():.3f}.", "",
             "## Threshold sweep", "",
             "| cosine | merges | words absorbed |", "|---|---|---|"]

    print("\nthreshold sweep")
    for tau in taus:
        merges = [c for f in fams.values()
                  for c in complete_linkage(f, cos, tau) if len(c) > 1]
        absorbed = sum(len(c) - 1 for c in merges)
        print(f"  >= {tau:.2f}: {len(merges):4d} merges, {absorbed:4d} words absorbed")
        lines.append(f"| {tau:.2f} | {len(merges)} | {absorbed} |")

    show = args.show if args.show is not None else sorted(taus)[len(taus) // 2]
    merges = [c for f in fams.values()
              for c in complete_linkage(f, cos, show) if len(c) > 1]

    def worst(cluster):
        return min(cos(a, b) for i, a in enumerate(cluster) for b in cluster[i + 1:])

    merges.sort(key=worst, reverse=True)
    print(f"\nevery merge at cosine >= {show} ({len(merges)}):")
    lines += ["", f"## Merges at cosine >= {show}", "",
              "| weakest pair | surviving label | absorbed |", "|---|---|---|"]
    for cluster in merges:
        primary = min(cluster, key=lambda w: (-counts.get(w, 0), len(w), w))
        rest = sorted(w for w in cluster if w != primary)
        print(f"  {worst(cluster):.3f}  {primary} <- {', '.join(rest)}")
        lines.append(f"| {worst(cluster):.3f} | {primary} | {', '.join(rest)} |")

    merged_pairs = {(a, b) if a < b else (b, a)
                    for c in merges for i, a in enumerate(c) for b in c[i + 1:]}
    # Two different reasons a candidate pair does not merge, worth telling apart:
    # below the floor, or above it but split because a THIRD family member fails
    # complete linkage (`tolerance`/`toleration` at 0.576 loses to `tolerant`).
    # Only the first kind is fixed by lowering the threshold.
    rejected = sorted(((s, a, b) for (a, b), s in cosine.items()
                       if (a, b) not in merged_pairs), reverse=True)
    by_linkage = [r for r in rejected if r[0] >= show]
    by_floor = [r for r in rejected if r[0] < show]

    print(f"\nsplit by complete linkage despite clearing {show} "
          f"({len(by_linkage)}) — a third family member failed:")
    for sim, a, b in by_linkage[:10]:
        print(f"  {sim:.3f}  {a} / {b}")
    print(f"\nclosest pairs below the floor (top 15 of {len(by_floor)}):")
    for sim, a, b in by_floor[:15]:
        print(f"  {sim:.3f}  {a} / {b}")

    lines += ["", f"## Not merged at cosine >= {show}", "",
              f"**Split by complete linkage** ({len(by_linkage)}) — the pair "
              f"clears the floor, but a third member of its family does not, so "
              f"the cluster could not absorb it. Lowering the threshold does not "
              f"necessarily fix these.", "", "| cosine | a | b |", "|---|---|---|"]
    for sim, a, b in by_linkage[:40]:
        lines.append(f"| {sim:.3f} | {a} | {b} |")
    lines += ["", f"**Below the floor** ({len(by_floor)}) — the near-misses to "
              f"read when choosing a threshold.", "",
              "| cosine | a | b |", "|---|---|---|"]
    for sim, a, b in by_floor[:60]:
        lines.append(f"| {sim:.3f} | {a} | {b} |")

    out = args.out or (config.ANALYSIS / args.corpus / "family_diagnostics.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
