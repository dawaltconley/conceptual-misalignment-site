"""Streaming pool of per-occurrence vectors into one vector per word.

Default follows Wu & Wang: mean over subwords *within* an occurrence (model.py),
then element-wise **max** *across* occurrences of the word. The cross-occurrence
reduction is configurable (``mean``/``max``/``none``) and every mode folds
**online** as the embedder streams occurrences — peak memory is O(vocab), not
O(occurrences). (``max`` is an associative running max; ``mean`` a running
sum+count; ``none`` keeps the first occurrence.)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Pooling = Literal["mean", "max", "none"]


class Pooler:
    """Fold streamed ``(word, occurrence_vector)`` events into a pooled matrix.

    ``mode`` sets the cross-occurrence reduction (``max``/``mean``/``none``), kept
    as a running accumulator per word (O(vocab)). Additionally retains the full
    occurrence stacks for the small ``keep`` set (the target terms) so their
    cohesion / variance can still be measured. Feed it with :meth:`add`, then read
    :meth:`matrix` / :meth:`stacks`.
    """

    def __init__(self, mode: Pooling = "max",
                 keep: set[str] | frozenset[str] = frozenset()):
        self._mode = mode
        self._acc: dict[str, np.ndarray] = {}   # running max / sum / first-seen
        self._count: dict[str, int] = {}        # occurrence count (for mean)
        self._keep = set(keep)
        self._stacks: dict[str, list[np.ndarray]] = {}

    def add(self, word: str, vec: np.ndarray) -> None:
        cur = self._acc.get(word)
        if cur is None:
            # float64 accumulator for mean so a long running sum doesn't lose precision
            self._acc[word] = vec.astype(np.float64) if self._mode == "mean" else vec
            self._count[word] = 1
        elif self._mode == "max":
            self._acc[word] = np.maximum(cur, vec)
        elif self._mode == "mean":
            self._acc[word] = cur + vec
            self._count[word] += 1
        # "none": the first occurrence wins — nothing to fold.
        if word in self._keep:
            self._stacks.setdefault(word, []).append(vec)

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """``(labels, matrix)`` where ``matrix[i]`` is the pooled vector for
        ``labels[i]``. Words are sorted for stable ordering."""
        labels = sorted(self._acc)
        rows = [self._acc[w] / self._count[w] if self._mode == "mean" else self._acc[w]
                for w in labels]
        return labels, np.stack(rows)

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


def reduce_vectors(matrix: np.ndarray, dims: int) -> np.ndarray:
    """Mean-center + L2-normalize, then PCA to `dims` (variance-ordered columns)."""
    from sklearn.decomposition import PCA
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.clip(norms, 1e-12, None)
    n_components = min(dims, unit.shape[1], unit.shape[0])
    return PCA(n_components=n_components, random_state=0).fit_transform(unit)
