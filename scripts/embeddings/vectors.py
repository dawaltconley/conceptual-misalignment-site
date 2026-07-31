"""Max-pool per-occurrence vectors into one vector per word.

Pooling follows Wu & Wang: mean over subwords *within* an occurrence (done in
model.py), then element-wise **max** *across* occurrences of the word, which
keeps the dominant sense and suppresses subword-fragmentation noise.
"""

from __future__ import annotations

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
