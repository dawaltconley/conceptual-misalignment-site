"""Within-corpus semantic-space analysis of the pooled term vectors.

Produces (into ``out_dir``): PCA & t-SNE scatter PNGs, K-means cluster
assignments, a cohesion/variance table, a cosine-similarity matrix (raw +
log-transformed) with heatmap, and a cosine-similarity network with Louvain
communities serialized in the site's node-link JSON schema.
"""

from __future__ import annotations
from typing import Literal, Callable
from itertools import cycle
from graph.prune import prune_to_neighborhood
from graph.serialize import save_graph_json
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

type Method = Literal["threshold", "knn"]
type SimTransform = Literal["none", "neglog", "poslog"]

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

_SIM_TRANSFORMS: dict[SimTransform, Callable[[np.ndarray]] | None] = {
    "none": None,
    "neglog": neglog_transform,
    "poslog": poslog_transform,
}


def apply_sim_transform(sim: np.ndarray, name: SimTransform) -> np.ndarray:
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

_NO_COMMUNITY_COLOR = "#cccccc"  # nodes absent from the network (isolates)


def _community_colors(communities: list[int]) -> dict[int, tuple]:
    """Map each community id to a distinct color (grey reserved for ``-1``)."""
    uniq = sorted({c for c in communities if c >= 0})
    cmap = plt.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(uniq)}


def _scatter(coords, labels, is_target, communities, title, path: Path) -> None:
    """Scatter colored by Louvain community; targets marked with an edged star."""
    palette = _community_colors(communities)
    fig, ax = plt.subplots(figsize=(11, 9))
    for (x, y), lbl, tgt, comm in zip(coords, labels, is_target, communities):
        color = palette.get(comm, _NO_COMMUNITY_COLOR)
        ax.scatter(x, y, s=200 if tgt else 34, c=[color],
                   marker="*" if tgt else "o",
                   edgecolors="black" if tgt else "none",
                   linewidths=1.4 if tgt else 0.0,
                   zorder=3 if tgt else 2, alpha=0.95 if tgt else 0.7)
        ax.annotate(lbl, (x, y), fontsize=13 if tgt else 8,
                    fontweight="bold" if tgt else "normal",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pca_plot(vectors, labels, is_target, communities, path: Path) -> None:
    coords = PCA(n_components=2, random_state=0).fit_transform(vectors)
    _scatter(coords, labels, is_target, communities,
             "PCA of virtue + neighbor vectors (color = community)", path)


def tsne_plot(vectors, labels, is_target, communities, path: Path) -> None:
    perplexity = max(2, min(30, len(labels) - 1))
    coords = TSNE(
        n_components=2, random_state=0, perplexity=perplexity, init="pca"
    ).fit_transform(vectors)
    _scatter(coords, labels, is_target, communities,
             f"t-SNE (perplexity={perplexity}, color = community)", path)


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
    labels: list[str], sim: np.ndarray, quantile: float
) -> nx.Graph:
    """Nodes = words; keep the edges whose similarity is in the top ``1 - quantile``.

    ``quantile`` is a percentile cutoff in ``[0, 1)`` over the *distribution* of
    off-diagonal similarities, not an absolute cosine value: ``0.9`` keeps the
    strongest 10% of pairwise edges. Being rank-defined, the surviving edge set is
    invariant to any monotone ``sim_transform`` (an absolute cutoff is not — see
    :func:`apply_sim_transform`) and immune to the anisotropy offset that makes a
    fixed cosine threshold meaningless. Raise it to sparsify the graph (more nodes
    fall out as isolates); lower it to densify.
    """
    G = nx.Graph()
    G.add_nodes_from(labels)
    n = len(labels)
    upper = sim[np.triu_indices(n, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return G
    cutoff = float(np.quantile(upper, quantile))
    for i in range(n):
        for j in range(i + 1, n):
            w = sim[i, j]
            if np.isfinite(w) and w >= cutoff:
                G.add_edge(labels[i], labels[j], weight=float(w))
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


def annotate_communities(G: nx.Graph, resolution: float = 1.0) -> int:
    """Run Louvain and store each node's community id as a ``community`` attr.

    ``resolution`` > 1 favors more, smaller communities (splits the frequency/
    register hub); < 1 favors fewer, larger ones."""
    if G.number_of_edges() == 0:
        for node in G:
            G.nodes[node]["community"] = 0
        return 1
    communities = nx.community.louvain_communities(
        G, weight="weight", seed=0, resolution=resolution)
    for cid, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["community"] = cid
    return len(communities)


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
# Similarity network + Louvain communities (build; caller saves via save_network)
# ---------------------------------------------------------------------------


def build_networks(
    labels: list[str],
    matrix: np.ndarray,
    *,
    method: Method = "knn",
    quantile: float = 0.9,
    knn_k: int = 8,
    sim_transform: SimTransform = "none",
    resolution: float = 1.0,
) -> tuple[nx.Graph, int, dict[str, int]]:
    """Build the similarity graph over the full vocab + detect communities.

    The single cosine-graph builder for both the JSON pipeline (``main`` reads
    ``(G, community_map)``) and the ``--artifacts`` dump (``run_analysis`` also
    uses ``n_communities`` and serializes via :func:`save_network`).
    ``method`` selects edge construction ("knn" top-``knn_k`` neighbors, vs
    "threshold" keeping edges above the ``quantile`` of the similarity distribution
    — both rank-defined and so robust to the anisotropy offset that makes an
    absolute cosine cutoff meaningless); ``sim_transform`` optionally reweights the
    cosine matrix first (monotone, so it changes edge weights + communities, not
    which edges appear). Returns ``(G, n_communities, community_map)``; isolated
    nodes are dropped (absent from the map, treated as community ``-1``).
    """
    sim = apply_sim_transform(cosine_similarity(matrix), sim_transform)
    if method == "knn":
        G = build_knn_graph(labels, sim, knn_k)
    elif method == "threshold":
        G = build_cosine_graph(labels, sim, quantile)
    else:
        raise ValueError(f"unknown network method {method!r}")

    G.remove_nodes_from(list(nx.isolates(G)))
    n_comms = annotate_communities(G, resolution)
    community_map = {n: int(G.nodes[n]["community"]) for n in G}
    edge = f"knn k={knn_k}" if method == "knn" else f"top {1 - quantile:.0%} (q={quantile})"
    tr = "" if sim_transform == "none" else f", {sim_transform}"
    res = "" if resolution == 1.0 else f", res={resolution}"
    print(f"cosine     : {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges ({edge}{tr}); {n_comms} communities{res}")
    return G, n_comms, community_map


def save_network(
    network: nx.Graph,
    filepath: Path,
    term: str | None = None,
    max_nodes: int = 15
) -> None:
    output = network
    if term is not None:
        output = prune_to_neighborhood(network, term, max_nodes)
        if output is None:
            print(f"[{term}] term not found...")
    if output is not None:
        save_graph_json(output, filepath)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_analysis(
    labels: list[str],
    matrix: np.ndarray,
    is_target: np.ndarray,
    target_occ: dict[str, np.ndarray],
    out_dir: Path,
    quantile: float,
    kmeans_k: int,
    method: Method = "threshold",
    knn_k: int = 8,
    sim_transform: SimTransform = "none",
    resolution: float = 1.0,
    max_nodes: int = 15,
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

    # Network + Louvain first, so the scatter plots can color by community.
    sim_network, n_comms, community_map = build_networks(
        labels, matrix, method=method, quantile=quantile, knn_k=knn_k,
        sim_transform=sim_transform, resolution=resolution)

    # save the similarity networks, pruned to terms
    net_dir = out_dir / "networks"
    net_dir.mkdir(parents=True, exist_ok=True)
    save_network(sim_network, net_dir / "full.json", max_nodes=max_nodes)
    for term in target_labels:
        save_network(sim_network, net_dir /
                     f"{term}.json", term, max_nodes=max_nodes)

    communities = [community_map.get(lbl, -1) for lbl in labels]

    # Dimensionality-reduction plots (all nodes), colored by community.
    pca_plot(matrix, labels, is_target, communities, out_dir / "pca.png")
    if len(labels) >= 3:
        tsne_plot(matrix, labels, is_target, communities, out_dir / "tsne.png")

    # K-means.
    km = kmeans_assignments(matrix, labels, is_target, kmeans_k)
    write_rows_csv(out_dir / "kmeans.csv", km)

    return {
        "n_terms": len(labels),
        "n_targets": len(target_labels),
        "targets": target_labels,
        "cohesion_variance": cv,
        "louvain_communities": n_comms,
        "quantile": quantile,
        "resolution": resolution,
        "method": method,
        "knn_k": knn_k,
        "sim_transform": sim_transform,
    }
