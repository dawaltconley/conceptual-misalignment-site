import json
import math
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Character utilities
# ---------------------------------------------------------------------------

def is_cjk(s: str) -> bool:
    """Return True if s is non-empty and consists entirely of CJK characters."""
    return bool(s) and all("\u4e00" <= c <= "\u9fff" for c in s)


# ---------------------------------------------------------------------------
# PMI
# ---------------------------------------------------------------------------

def pmi(
    pair_cooc: Counter[tuple[str, str]],
    sent_freq: Counter[str],
    n_sents: int,
    a: str,
    b: str,
) -> float:
    """Pointwise mutual information between two terms across sentences.

    Returns -inf when the terms never co-occur or either has zero sentence
    frequency. Returns 0.0 when co-occurrence equals the chance expectation.
    Keys in pair_cooc must be sorted (min, max) pairs.
    """
    key = (min(a, b), max(a, b))
    cooc = pair_cooc.get(key, 0)
    if cooc == 0 or sent_freq[a] == 0 or sent_freq[b] == 0:
        return float("-inf")
    return math.log((cooc * n_sents) / (sent_freq[a] * sent_freq[b]))


def count_pair_cooccurrences(
    sent_token_lists: list[list[str]],
    vocab: set[str],
) -> Counter[tuple[str, str]]:
    """Count sentences in which each pair of vocab tokens co-occurs.

    Each pair is stored as (lexically-smaller, lexically-larger) so lookups
    are order-independent. A token appearing multiple times in one sentence
    counts only once for that sentence.
    """
    pair_cooc: Counter[tuple[str, str]] = Counter()
    for sent in sent_token_lists:
        unique = sorted({t for t in sent if t in vocab})
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                pair_cooc[(a, b)] += 1
    return pair_cooc


def build_pmi_graph(
    sent_node_lists: list[list[str]],
    nodes: set[str],
) -> tuple[nx.Graph, Counter[tuple[str, str]], Counter[str], int]:
    """Build a PMI-weighted co-occurrence graph, keeping only positive PMI edges.

    Returns (G, pair_cooc, sent_freq, n_sents) so the caller can compute
    additional per-term PMI values without re-scanning the sentences.
    """
    n_sents = len(sent_node_lists)
    sent_freq: Counter[str] = Counter(
        tok for sent in sent_node_lists for tok in set(sent)
    )
    pair_cooc = count_pair_cooccurrences(sent_node_lists, nodes)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (a, b), _ in pair_cooc.items():
        score = pmi(pair_cooc, sent_freq, n_sents, a, b)
        if score > 0:
            G.add_edge(a, b, weight=score)

    return G, pair_cooc, sent_freq, n_sents


# ---------------------------------------------------------------------------
# Cosine similarity graph
# ---------------------------------------------------------------------------

def build_cosine_similarity_graph(
    sent_node_lists: list[list[str]],
    node_list: list[str],
    threshold: float,
) -> nx.Graph:
    """Build a graph whose edge weights are cosine similarities of co-occurrence vectors.

    Each node's context vector counts how many sentences it shares with every
    other node. Edges below *threshold* are omitted.
    """
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    M = np.zeros((N, N), dtype=float)
    for sent in sent_node_lists:
        in_sent = [t for t in sent if t in idx]
        for a in in_sent:
            for b in in_sent:
                if a != b:
                    M[idx[a], idx[b]] += 1.0

    sim_matrix = cosine_similarity(M)

    S = nx.Graph()
    S.add_nodes_from(node_list)
    for i, a in enumerate(node_list):
        for j, b in enumerate(node_list):
            if i < j and sim_matrix[i, j] >= threshold:
                S.add_edge(a, b, weight=float(sim_matrix[i, j]))

    return S


# ---------------------------------------------------------------------------
# Graph pruning
# ---------------------------------------------------------------------------

def proximity_score(G: nx.Graph, term: str, node: str) -> float:
    """Score a node by weighted proximity to term (1-hop weight or best 2-hop product)."""
    if G.has_edge(term, node):
        return float(G[term][node]["weight"])
    return max(
        (G[term][mid]["weight"] * G[mid][node]["weight"]
         for mid in nx.common_neighbors(G, term, node)
         if G.has_edge(term, mid)),
        default=0.0,
    )


def prune_to_neighborhood(G: nx.Graph, term: str, max_nodes: int) -> nx.Graph:
    """Return a subgraph of G containing term and its top max_nodes neighbours.

    1-hop neighbors (direct edges) are always prioritised by edge weight.
    Any remaining capacity is filled by the highest-scoring 2-hop neighbors.
    If term is not in G, returns G unchanged.
    """
    if term not in G:
        return G

    one_hop = sorted(
        G.neighbors(term),
        key=lambda n: G[term][n]["weight"],
        reverse=True,
    )
    keep = {term} | set(one_hop[:max_nodes])

    remaining = max_nodes - (len(keep) - 1)
    if remaining > 0:
        ego = nx.ego_graph(G, term, radius=2)
        two_hop = sorted(
            (n for n in ego.nodes() if n != term and n not in keep),
            key=lambda n: proximity_score(ego, term, n),
            reverse=True,
        )
        keep |= set(two_hop[:remaining])

    return G.subgraph(keep).copy()


# ---------------------------------------------------------------------------
# Filter sentence nodes
# ---------------------------------------------------------------------------

def filter_to_sent_node_lists(
    sent_token_lists: list[list[str]],
    term: str,
    min_freq: int,
    stopwords: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[list[str]], set[str]]:
    """Frequency-filter a token vocabulary, then return sentences restricted to that vocab."""
    freq = Counter(tok for sent in sent_token_lists for tok in sent)
    nodes: set[str] = {
        t for t, c in freq.items()
        if c >= min_freq and t not in stopwords
    }
    nodes.add(term)

    sent_node_lists = [
        [t for t in sent if t in nodes]
        for sent in sent_token_lists
    ]
    return [s for s in sent_node_lists if len(s) >= 2], nodes


# ---------------------------------------------------------------------------
# Build networks
# ---------------------------------------------------------------------------

def build_cooccurrence_network(
    sent_node_lists: list[list[str]],
    nodes: set[str],
    term: str,
    max_nodes: int = 15,
) -> nx.Graph:
    G, _, _, _ = build_pmi_graph(sent_node_lists, nodes)
    G.remove_nodes_from(list(nx.isolates(G)))
    return prune_to_neighborhood(G, term, max_nodes)


def build_similarity_network(
    sent_node_lists: list[list[str]],
    nodes: set[str],
    term: str,
    max_nodes: int = 15,
    sim_threshold: float = 0.7,
) -> nx.Graph:
    S = build_cosine_similarity_graph(
        sent_node_lists, sorted(nodes), sim_threshold)
    S.remove_nodes_from(list(nx.isolates(S)))
    return prune_to_neighborhood(S, term, max_nodes)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_graph_json(G: nx.Graph, path: Path) -> None:
    """Write a NetworkX graph to a node-link JSON file."""
    path.write_text(json.dumps(nx.node_link_data(G),
                    indent=2), encoding="utf-8")
