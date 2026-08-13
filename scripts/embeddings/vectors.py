"""Streaming pool of per-occurrence vectors into one vector per word.

Default follows Wu & Wang: mean over subwords *within* an occurrence (model.py),
then element-wise **max** *across* occurrences of the word. The cross-occurrence
reduction is configurable (``mean``/``max``/``none``) and every mode folds
**online** as the embedder streams occurrences — peak memory is O(vocab), not
O(occurrences). (``max`` is an associative running max; ``mean`` a running
sum+count; ``none`` keeps the first occurrence.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

Pooling = Literal["mean", "max", "none"]
DebiasMethod = Literal["none", "abtt", "whiten"]


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

    def merge(self, alias: dict[str, str]) -> None:
        """Fold each aliased word's accumulator into its surviving label.

        Exact, not an approximation: every mode's accumulator is associative
        over the occurrence set, so folding two *pooled* accumulators gives the
        same result as having pooled the union of their occurrences in the first
        place. ``max`` is an element-wise max of running maxima, ``mean`` a sum
        of running sums with summed counts, and ``none`` keeps whichever
        first-seen vector belongs to the surviving label.
        """
        for word, primary in alias.items():
            vec = self._acc.pop(word, None)
            count = self._count.pop(word, 0)
            stack = self._stacks.pop(word, None)
            if vec is None:
                continue
            current = self._acc.get(primary)
            if current is None:
                self._acc[primary] = vec
                self._count[primary] = count
            elif self._mode == "max":
                self._acc[primary] = np.maximum(current, vec)
                self._count[primary] += count
            elif self._mode == "mean":
                self._acc[primary] = current + vec
                self._count[primary] += count
            else:               # "none": the surviving label's own vector wins
                self._count[primary] += count
            if stack:
                self._stacks.setdefault(primary, []).extend(stack)

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


def debias_matrix(
    matrix: np.ndarray, method: DebiasMethod, k: int | None = None
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Remove the dominant (frequency/register-correlated) directions from *centered*
    embeddings, so the leading axes stop encoding word frequency instead of meaning.

    Word frequency loads onto the top principal components of a contextual-embedding
    matrix (Mu & Viswanath 2018, "All-but-the-Top"); centering fixes the anisotropy
    *offset* but not this *direction*, which is why a document-frequency coloring
    still shows a clean gradient along a PCA axis. Two removals:

    - ``abtt`` (all-but-the-top): project out the top ``k`` principal components.
      ``k`` defaults to ``max(1, D // 100)`` (Mu & Viswanath's rule of thumb).
    - ``whiten``: PCA-whitening — rotate to the principal axes and rescale each to
      unit variance, so *no* direction dominates the variance (dissolves the gradient
      entirely rather than just deleting a few axes). ``k`` optionally truncates to
      the top-``k`` axes kept before whitening (``None`` keeps all).

    Assumes ``matrix`` is already mean-centered (see :func:`center_matrix`); pair it
    with ``center=True``. Returns ``(debiased, project)`` where ``project`` applies the
    *same* linear map to other centered vectors (e.g. the target occurrence stacks),
    so every downstream measurement stays in one space.
    """
    from sklearn.decomposition import PCA
    n_samples, dim = matrix.shape
    if method == "abtt":
        k = max(1, dim // 100) if k is None else k
        k = max(1, min(k, n_samples, dim))
        comps = np.asarray(
            PCA(n_components=k, random_state=0).fit(matrix).components_)  # (k, D)

        def project(x: np.ndarray) -> np.ndarray:
            return x - (x @ comps.T) @ comps
        return project(matrix), project
    if method == "whiten":
        n = min(k or dim, n_samples, dim)
        pca = PCA(n_components=n, whiten=True, random_state=0).fit(matrix)
        return pca.transform(matrix), pca.transform
    raise ValueError(f"unknown debias method {method!r}; "
                     "choose from 'none', 'abtt', 'whiten'")


def cache_analysis_matrix(corpus: str, labels: list[str],
                          matrix: np.ndarray) -> "Path":
    """Stash a run's full analysis matrix (centered/debiased, pre-export) so the
    layouts can be rebuilt from it later without re-embedding.

    Only the reduced vectors are shipped, so nothing else on disk can reconstruct
    the untruncated space — and re-deriving it costs a GPU pass over the whole
    corpus. This is a cache, not an artifact: it lives under the gitignored
    ``scripts/.cache`` and is regenerated by any pipeline run.
    """
    from config import VECTORS_CACHE
    VECTORS_CACHE.mkdir(parents=True, exist_ok=True)
    path = VECTORS_CACHE / f"{corpus}.npz"
    np.savez_compressed(path, labels=np.array(labels, dtype=object),
                        matrix=matrix.astype(np.float32))
    return path


def load_analysis_matrix(corpus: str) -> "tuple[list[str], np.ndarray] | None":
    """The cached ``(labels, matrix)`` for ``corpus``, or ``None`` if no run has
    written one (or it predates the cache)."""
    from config import VECTORS_CACHE
    path = VECTORS_CACHE / f"{corpus}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return [str(l) for l in data["labels"]], data["matrix"]


def unit_vectors(matrix: np.ndarray) -> np.ndarray:
    """Mean-center + L2-normalize (direction only) — the space the export's PCA is
    fitted on. Split out from :func:`reduce_vectors` so the precomputed t-SNE can
    embed the *untruncated* version of exactly that space (``Pipeline.tsne_sources``
    = ``full``), leaving PCA truncation as the only difference between the two."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.clip(norms, 1e-12, None)


def reduce_vectors(matrix: np.ndarray, dims: int) -> np.ndarray:
    """Mean-center + L2-normalize, then PCA to `dims` (variance-ordered columns)."""
    from sklearn.decomposition import PCA
    unit = unit_vectors(matrix)
    n_components = min(dims, unit.shape[1], unit.shape[0])
    return PCA(n_components=n_components, random_state=0).fit_transform(unit)
