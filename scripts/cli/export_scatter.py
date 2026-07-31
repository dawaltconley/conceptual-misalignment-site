"""Export one reduced-vector dataset per corpus for the client-side scatter view.

For each corpus we ship a *single* PCA-reduced vector per node (not precomputed 2-D
coords); the browser derives both projections from it — PCA is free (the reduced
columns are variance-ordered, so columns 0/1 are the 2-D PCA) and t-SNE is run
client-side over the full reduced vector with a tunable perplexity.

Preprocessing (mean-center + L2-normalize, then PCA to `--dims`) matches the
cross-lingual alignment prep, so this same artifact carries over to Task 2.

Run from ``scripts/``:

    python -m cli.export_scatter                 # both corpora, 50 dims
    python -m cli.export_scatter --dims 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from config import ANALYSIS, EMBEDDINGS
from embeddings import vectors

# (output name, analysis run dir) — the centered-kNN runs are the chosen method.
CORPORA = [
    ("mengzi", ANALYSIS / "centered-knn"),
    ("sep", ANALYSIS / "sep-centered-knn"),
]


def community_map(analysis_dir: Path) -> dict[str, int]:
    """Read the Louvain community per node from the run's full network JSON.

    Nodes absent from the network (isolates dropped during graph build) are not
    present here; callers default them to -1 (grey), matching the analysis PNGs.
    """
    net = json.loads((analysis_dir / "networks" /
                     "full.json").read_text("utf-8"))
    return {str(n["id"]): int(n["community"]) for n in net["nodes"]}


def reduce_vectors(matrix: np.ndarray, dims: int) -> np.ndarray:
    """Mean-center + L2-normalize, then PCA to `dims` (variance-ordered columns)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.clip(norms, 1e-12, None)
    n_components = min(dims, unit.shape[1], unit.shape[0])
    return PCA(n_components=n_components, random_state=0).fit_transform(unit)


def export_corpus(name: str, analysis_dir: Path, dims: int, out_dir: Path) -> Path:
    labels, matrix, is_target = vectors.load_vectors(analysis_dir)
    reduced = reduce_vectors(matrix, dims)
    comm = community_map(analysis_dir)

    nodes = [
        {
            "id": lbl,
            "target": bool(tgt),
            "community": comm.get(lbl, -1),
            "vec": [round(float(v), 5) for v in row],
        }
        for lbl, tgt, row in zip(labels, is_target, reduced)
    ]
    payload = {"corpus": name, "dims": reduced.shape[1], "nodes": nodes}

    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(
        payload, ensure_ascii=False), encoding="utf-8")
    n_targets = sum(n["target"] for n in nodes)
    print(f"{name:7s}: {len(nodes)} nodes ({n_targets} targets), "
          f"dims={reduced.shape[1]} -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, default=50,
                   help="PCA target dimensionality of the shipped vectors.")
    p.add_argument("--out", type=Path, default=EMBEDDINGS,
                   help="Output directory (default: public/embeddings).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, analysis_dir in CORPORA:
        export_corpus(name, analysis_dir, args.dims, args.out)


if __name__ == "__main__":
    main()
