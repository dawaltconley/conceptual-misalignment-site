"""Streaming max-pool of per-occurrence vectors into one vector per word.

Pooling follows Wu & Wang: mean over subwords *within* an occurrence (model.py),
then element-wise **max** *across* occurrences of the word. Element-wise max is an
associative reduction, so it is folded **online** (a running max per word) as the
embedder streams occurrences — peak memory is O(vocab), not O(occurrences).
"""

from __future__ import annotations

import numpy as np


class Pooler:
    """Fold streamed ``(word, occurrence_vector)`` events into a max-pooled matrix.

    Keeps a running element-wise max per word (O(vocab)); additionally retains the
    full occurrence stacks for the small ``keep`` set (the target terms) so their
    cohesion / variance can still be measured. Feed it with :meth:`add`, then read
    :meth:`matrix` / :meth:`stacks`.
    """

    def __init__(self, keep: set[str] | frozenset[str] = frozenset()):
        self._max: dict[str, np.ndarray] = {}
        self._keep = set(keep)
        self._stacks: dict[str, list[np.ndarray]] = {}

    def add(self, word: str, vec: np.ndarray) -> None:
        cur = self._max.get(word)
        self._max[word] = vec if cur is None else np.maximum(cur, vec)
        if word in self._keep:
            self._stacks.setdefault(word, []).append(vec)

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """``(labels, matrix)`` where ``matrix[i]`` is the pooled vector for
        ``labels[i]``. Words are sorted for stable ordering."""
        labels = sorted(self._max)
        return labels, np.stack([self._max[w] for w in labels])

    def stacks(self) -> dict[str, np.ndarray]:
        """Per-``keep`` word: its stacked ``(n_occurrences x H)`` occurrence
        vectors (for cohesion / variance analysis)."""
        return {w: np.stack(v) for w, v in self._stacks.items()}


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
