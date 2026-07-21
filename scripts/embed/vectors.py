"""Max-pool per-occurrence vectors into one vector per word, and persist.

Pooling follows Wu & Wang: mean over subwords *within* an occurrence (done in
model.py), then element-wise **max** *across* occurrences of the word, which
keeps the dominant sense and suppresses subword-fragmentation noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def max_pool(by_word: dict[str, list[np.ndarray]]) -> tuple[list[str], np.ndarray]:
    """Element-wise max-pool each word's occurrence vectors.

    Returns ``(labels, matrix)`` where ``matrix[i]`` is the pooled vector for
    ``labels[i]``. Words are sorted for stable ordering.
    """
    labels = sorted(by_word)
    matrix = np.stack([np.max(np.stack(by_word[w]), axis=0) for w in labels])
    return labels, matrix


def center_matrix(
    matrix: np.ndarray, mean: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center vectors to counter embedding anisotropy.

    GujiRoBERTa vectors share a strong common direction (mean |cos| to the
    corpus centroid ~0.84), inflating every pairwise cosine into a narrow high
    band. Subtracting the centroid restores a meaningful zero and roughly
    doubles the cosine spread. ``mean`` defaults to the vocab centroid; pass an
    existing mean to project other vectors (e.g. occurrence stacks) into the
    same centered space. Returns ``(centered, mean)``.
    """
    if mean is None:
        mean = matrix.mean(axis=0)
    return matrix - mean, mean  # pyright:ignore -- bad typing


def save_vectors(
    out_dir: Path,
    labels: list[str],
    matrix: np.ndarray,
    targets: set[str],
    by_word: dict[str, list[np.ndarray]],
    mean: np.ndarray | None = None,
) -> None:
    """Persist pooled vectors + per-target occurrence stacks.

    - ``vectors.npz``: ``labels`` (str array), ``matrix`` (N x H),
      ``is_target`` (bool array). When the run was mean-centered, ``matrix`` is
      already centered and ``mean`` (the subtracted vocab centroid) is stored
      too, for reproducibility / re-projecting other vectors.
    - ``occurrences_targets.npz``: one entry per target term holding its stacked
      (n_occurrences x H) occurrence vectors, for cohesion / variance analysis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "labels": np.array(labels, dtype=object),
        "matrix": matrix,
        "is_target": np.array([lbl in targets for lbl in labels]),
    }
    if mean is not None:
        arrays["mean"] = mean
    np.savez(out_dir / "vectors.npz", **arrays)
    target_stacks = {
        w: np.stack(by_word[w]) for w in targets if by_word.get(w)
    }
    np.savez(out_dir / "occurrences_targets.npz", **target_stacks)


def load_vectors(out_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load ``(labels, matrix, is_target)`` from ``vectors.npz``."""
    data = np.load(out_dir / "vectors.npz", allow_pickle=True)
    return list(data["labels"]), data["matrix"], data["is_target"]


def load_target_occurrences(out_dir: Path) -> dict[str, np.ndarray]:
    """Load per-target occurrence stacks from ``occurrences_targets.npz``."""
    data = np.load(out_dir / "occurrences_targets.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}
