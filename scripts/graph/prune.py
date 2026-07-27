"""Graph pruning shared by both pipelines: reduce a graph to a term's neighborhood."""

import networkx as nx


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


def prune_to_neighborhood(G: nx.Graph, term: str, max_nodes: int) -> nx.Graph | None:
    """Return a subgraph of G containing term and its top max_nodes neighbours.

    1-hop neighbors (direct edges) are always prioritised by edge weight.
    Any remaining capacity is filled by the highest-scoring 2-hop neighbors.
    If term is not in G, returns None.
    """
    if term not in G:
        return None

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
