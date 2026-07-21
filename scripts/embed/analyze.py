"""Within-corpus semantic-space analysis of the pooled term vectors.

Produces (into ``out_dir``): PCA & t-SNE scatter PNGs, K-means cluster
assignments, a cohesion/variance table, a cosine-similarity matrix (raw +
log-transformed) with heatmap, and a cosine-similarity network with Louvain
communities serialized in the site's node-link JSON schema.
"""

from __future__ import annotations
from itertools import cycle
from utils import prune_to_neighborhood, save_graph_json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from matplotlib import font_manager
import matplotlib.pyplot as plt

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

# Render CJK glyphs (default DejaVu Sans has none) using system Noto Sans CJK.
_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    font_manager.fontManager.addfont(_CJK_FONT)
    plt.rcParams["font.family"] = "Noto Sans CJK TC"
    plt.rcParams["axes.unicode_minus"] = False
except (FileNotFoundError, RuntimeError):
    pass  # fall back to default font (CJK glyphs will be boxes)


EPS = 1e-12


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    return cosine_similarity(vectors)


def neglog_transform(sim: np.ndarray) -> np.ndarray:
    """Convex stretch that *expands* the high-similarity range: ``-ln(1 - sim)``.

    Diagonal (self-similarity 1.0) would diverge, so it is set to NaN. The slope
    ``1/(1 - sim)`` grows without bound as similarity approaches 1, spreading out
    the crowded high-similarity region (de-crowding reading of Wu & Wang's
    log-transformed cosine).
    """
    out = -np.log(np.clip(1.0 - sim, EPS, None))
    np.fill_diagonal(out, np.nan)
    return out


def poslog_transform(sim: np.ndarray) -> np.ndarray:
    """Concave stretch that *compresses* the high-similarity range: ``ln(1 + sim)``.

    The reversed counterpart of :func:`neglog_transform`: slope ``1/(1 + sim)``
    shrinks toward 1, so high similarities are damped and low ones expanded
    (dampening reading of "attenuating saturation effects"). Diagonal set to NaN
    to match the other transforms.
    """
    out = np.log1p(sim)
    np.fill_diagonal(out, np.nan)
    return out


# Backwards-compatible alias: the original single transform was ``-ln(1 - sim)``.
log_transform = neglog_transform

_SIM_TRANSFORMS = {
    "none": None,
    "neglog": neglog_transform,
    "poslog": poslog_transform,
}


def apply_sim_transform(sim: np.ndarray, name: str) -> np.ndarray:
    """Dispatch a named similarity transform; ``none`` returns ``sim`` unchanged.

    Both transforms are strictly monotone, so at a matched percentile the edge
    *set* is unchanged; what shifts is the edge *weights* (and hence any
    weight-based step like weighted Louvain) and the visual scale.
    """
    if name not in _SIM_TRANSFORMS:
        raise ValueError(
            f"unknown sim transform {name!r}; choose from {sorted(_SIM_TRANSFORMS)}")
    fn = _SIM_TRANSFORMS[name]
    return sim if fn is None else fn(sim)


def write_matrix_csv(path: Path, labels: list[str], mat: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for lbl, row in zip(labels, mat):
            w.writerow(
                [lbl] + [f"{v:.4f}" if np.isfinite(v) else "" for v in row])


# ---------------------------------------------------------------------------
# Cohesion & variance (per target, from occurrence stacks)
# ---------------------------------------------------------------------------

def cohesion_variance(target_occ: dict[str, np.ndarray]) -> list[dict]:
    """Per-target spread of its occurrence vectors around their centroid.

    - ``n``: occurrence count.
    - ``cohesion``: mean cosine distance (1 - cos) to the centroid (lower = tighter).
    - ``variance``: mean squared Euclidean distance to the centroid.
    """
    rows: list[dict] = []
    for word, stack in sorted(target_occ.items()):
        centroid = stack.mean(axis=0, keepdims=True)
        cos_to_c = cosine_similarity(stack, centroid).ravel()
        cohesion = float((1.0 - cos_to_c).mean())
        variance = float(((stack - centroid) ** 2).sum(axis=1).mean())
        rows.append(
            {
                "term": word,
                "n": int(stack.shape[0]),
                "cohesion": round(cohesion, 4),
                "variance": round(variance, 4),
            }
        )
    return rows


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Dimensionality-reduction plots
# ---------------------------------------------------------------------------

def _scatter(coords, labels, is_target, title, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    for (x, y), lbl, tgt in zip(coords, labels, is_target):
        ax.scatter(x, y, s=90 if tgt else 30,
                   c="crimson" if tgt else "steelblue",
                   zorder=3 if tgt else 2, alpha=0.9 if tgt else 0.6)
        ax.annotate(lbl, (x, y), fontsize=13 if tgt else 9,
                    fontweight="bold" if tgt else "normal",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pca_plot(vectors, labels, is_target, path: Path) -> None:
    coords = PCA(n_components=2, random_state=0).fit_transform(vectors)
    _scatter(coords, labels, is_target,
             "PCA of virtue + neighbor vectors", path)


def tsne_plot(vectors, labels, is_target, path: Path) -> None:
    perplexity = max(2, min(30, len(labels) - 1))
    coords = TSNE(
        n_components=2, random_state=0, perplexity=perplexity, init="pca"
    ).fit_transform(vectors)
    _scatter(coords, labels, is_target,
             f"t-SNE (perplexity={perplexity})", path)


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------

def kmeans_assignments(vectors, labels, is_target, k: int) -> list[dict]:
    k = max(1, min(k, len(labels)))
    km = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10  # pyright:ignore -- bad typing
    ).fit(vectors)
    clusters = km.labels_ if not km.labels_ is None else cycle([0])
    return [
        {"term": lbl, "is_target": bool(t), "cluster": int(c)}
        for lbl, t, c in zip(labels, is_target, clusters)
    ]


# ---------------------------------------------------------------------------
# Cosine-similarity network + Louvain
# ---------------------------------------------------------------------------

def build_cosine_graph(
    labels: list[str], sim: np.ndarray, threshold: float
) -> nx.Graph:
    """Nodes = words; edges where the (possibly transformed) similarity >= threshold."""
    G = nx.Graph()
    G.add_nodes_from(labels)
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                G.add_edge(labels[i], labels[j], weight=float(sim[i, j]))
    return G


def build_knn_graph(labels: list[str], sim: np.ndarray, k: int) -> nx.Graph:
    """Nodes = words; edge (a, b) if either ranks the other in its top-``k`` (union kNN).

    Relative-neighborhood alternative to a global threshold: each node keeps its
    ``k`` most-similar neighbors, which is robust to the anisotropy offset that
    makes an absolute cutoff meaningless. Edge weight is the similarity. (Mutual
    kNN — edge only if *both* rank each other — is a stricter variant.)
    """
    G = nx.Graph()
    G.add_nodes_from(labels)
    n = len(labels)
    k = max(1, min(k, n - 1))
    ranked = sim.copy()
    np.fill_diagonal(ranked, -np.inf)  # never pick self
    for i in range(n):
        for j in np.argsort(ranked[i])[::-1][:k]:
            w = ranked[i, j]
            if np.isfinite(w):
                G.add_edge(labels[i], labels[int(j)], weight=float(w))
    return G


def annotate_communities(G: nx.Graph) -> int:
    """Run Louvain and store each node's community id as a ``community`` attr."""
    if G.number_of_edges() == 0:
        for node in G:
            G.nodes[node]["community"] = 0
        return 1
    communities = nx.community.louvain_communities(G, weight="weight", seed=0)
    for cid, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["community"] = cid
    return len(communities)


def build_and_save_networks(
    labels: list[str],
    vectors: np.ndarray,
    is_target: np.ndarray,
    threshold: float,
    out_dir: Path,
    max_nodes: int = 15,
    method: str = "threshold",
    knn_k: int = 8,
    sim_transform: str = "none",
) -> int:
    """Build the similarity network, detect communities, and serialize JSON.

    ``method`` selects edge construction: ``"threshold"`` keeps pairs with
    similarity >= ``threshold``; ``"knn"`` keeps each node's top-``knn_k``
    neighbors (relative neighborhoods). ``sim_transform`` optionally reweights
    the cosine matrix first (see :func:`apply_sim_transform`). Writes the full
    network plus one pruned neighborhood per target term. Returns the number of
    Louvain communities found.
    """
    sim = apply_sim_transform(cosine_similarity(vectors), sim_transform)
    if method == "knn":
        G = build_knn_graph(labels, sim, knn_k)
    elif method == "threshold":
        G = build_cosine_graph(labels, sim, threshold)
    else:
        raise ValueError(f"unknown network method {method!r}")
    G.remove_nodes_from(list(nx.isolates(G)))
    n_comms = annotate_communities(G)

    net_dir = out_dir / "networks"
    net_dir.mkdir(parents=True, exist_ok=True)
    save_graph_json(G, net_dir / "full.json")

    targets = [lbl for lbl, t in zip(labels, is_target) if t]
    for term in targets:
        pruned = prune_to_neighborhood(G, term, max_nodes)
        save_graph_json(pruned, net_dir / f"{term}.json")
    return n_comms


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def heatmap(mat: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(
        figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
    data = np.where(np.isfinite(mat), mat, np.nan)
    im = ax.imshow(data, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_analysis(
    labels: list[str],
    matrix: np.ndarray,
    is_target: np.ndarray,
    target_occ: dict[str, np.ndarray],
    out_dir: Path,
    threshold: float,
    kmeans_k: int,
    method: str = "threshold",
    knn_k: int = 8,
    sim_transform: str = "none",
) -> dict:
    """Run every analysis and write artifacts; return a small summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_labels = [lbl for lbl, t in zip(labels, is_target) if t]

    # Similarity among target terms only. cosine_targets.csv is always the raw
    # cosine (reference); cosine_targets_log.csv is the -ln(1-s) view (kept for
    # continuity). The heatmap uses this run's sim_transform so it shares a scale
    # with the network edges below.
    tgt_idx = [i for i, t in enumerate(is_target) if t]
    tgt_vecs = matrix[tgt_idx]
    sim = cosine_matrix(tgt_vecs)
    write_matrix_csv(out_dir / "cosine_targets.csv", target_labels, sim)
    write_matrix_csv(out_dir / "cosine_targets_log.csv",
                     target_labels, log_transform(sim))
    heat = apply_sim_transform(sim, sim_transform)
    heat_title = "Cosine similarity (virtues)"
    if sim_transform != "none":
        heat_title += f" [{sim_transform}]"
    heatmap(heat, target_labels, heat_title, out_dir / "cosine_heatmap.png")

    # Cohesion / variance.
    cv = cohesion_variance(target_occ)
    write_rows_csv(out_dir / "cohesion_variance.csv", cv)

    # Dimensionality-reduction plots (all nodes).
    pca_plot(matrix, labels, is_target, out_dir / "pca.png")
    if len(labels) >= 3:
        tsne_plot(matrix, labels, is_target, out_dir / "tsne.png")

    # K-means.
    km = kmeans_assignments(matrix, labels, is_target, kmeans_k)
    write_rows_csv(out_dir / "kmeans.csv", km)

    # Network + Louvain.
    n_comms = build_and_save_networks(
        labels, matrix, is_target, threshold, out_dir,
        method=method, knn_k=knn_k, sim_transform=sim_transform)

    return {
        "n_terms": len(labels),
        "n_targets": len(target_labels),
        "targets": target_labels,
        "cohesion_variance": cv,
        "louvain_communities": n_comms,
        "threshold": threshold,
        "method": method,
        "knn_k": knn_k,
        "sim_transform": sim_transform,
    }
