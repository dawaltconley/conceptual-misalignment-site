"""Test every community-node-ordering metric from notes/community-legend-ordering.md.

For a given corpus it loads the shipped artifacts —

  * ``public/embeddings/{corpus}.json``     (id / target / community / 50-d PCA vec)
  * ``analysis/{corpus}/networks/full.json`` (the weighted similarity graph)
  * ``scripts/data/{corpus}.conllu``         (lemma frequencies; Mengzi only)

— computes each candidate ordering per node, and reports (a) how much the orderings
actually disagree (Spearman rank-correlation matrix, within-community) and (b) the
top-k of each community under every metric, so the orderings can be eyeballed
side-by-side. Writes a Markdown report next to the artifacts and prints a digest.

Run:  .venv/bin/python scripts/tools/community_ordering.py [--corpus mengzi] [--top 12]

Metrics (grouped as in the note):
  A. importance   — prototypicality, proximity_to_virtue, frequency,
                    strength (weighted degree), pagerank, eigenvector
  B. distinctive  — silhouette, coreness (in-community edge-weight fraction)
The one listed metric this cannot cover is TF-IDF specificity: it needs per-community
co-occurrence counts, which are not in these artifacts (it is reported as skipped).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import spearmanr

import config

# Higher score = listed earlier, for every metric below (so they are comparable).
IMPORTANCE = ["prototypicality", "proximity_to_virtue", "frequency",
              "strength", "pagerank", "eigenvector"]
DISTINCTIVE = ["silhouette", "coreness"]
METRICS = IMPORTANCE + DISTINCTIVE


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_embeddings(path: Path):
    """-> (ids, unit_vecs NxD, community dict, target set). Vectors are L2-normalized
    so dot products are true cosines (the PCA-reduced vecs are not unit-norm)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    ids = [n["id"] for n in nodes]
    vecs = np.array([n["vec"] for n in nodes], dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.clip(norms, 1e-12, None)
    community = {n["id"]: int(n["community"]) for n in nodes}
    targets = {n["id"] for n in nodes if n.get("target")}
    return ids, unit, community, targets


def load_graph(path: Path) -> nx.Graph:
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges")


def load_frequencies(path: Path) -> collections.Counter:
    """Count the LEMMA column of a .conllu (skip comments / multiword ranges)."""
    freq: collections.Counter = collections.Counter()
    if not path.exists():
        return freq
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 2 and "-" not in cols[0]:
                freq[cols[2]] += 1
    return freq


# --------------------------------------------------------------------------- #
# Metrics  (each returns {node_id: score}, higher = rank first)
# --------------------------------------------------------------------------- #

def community_centroids(ids, unit, community) -> dict[int, np.ndarray]:
    """Unit-normalized mean vector of each community."""
    by_comm: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    for i, node in enumerate(ids):
        by_comm[community[node]].append(unit[i])
    cents = {}
    for c, rows in by_comm.items():
        m = np.mean(rows, axis=0)
        cents[c] = m / np.clip(np.linalg.norm(m), 1e-12, None)
    return cents


def vec_metrics(ids, unit, community, targets):
    """prototypicality, proximity_to_virtue, silhouette — all from vecs."""
    idx = {n: i for i, n in enumerate(ids)}
    cents = community_centroids(ids, unit, community)

    # Per-community virtue anchor: its target(s) if any, else the centroid.
    tgt_by_comm: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    for t in targets:
        tgt_by_comm[community[t]].append(unit[idx[t]])
    anchors = {}
    for c, cen in cents.items():
        if tgt_by_comm[c]:
            a = np.mean(tgt_by_comm[c], axis=0)
            anchors[c] = a / np.clip(np.linalg.norm(a), 1e-12, None)
        else:
            anchors[c] = cen

    proto, prox, sil = {}, {}, {}
    for i, node in enumerate(ids):
        c = community[node]
        proto[node] = float(unit[i] @ cents[c])
        prox[node] = float(unit[i] @ anchors[c])
        others = [float(unit[i] @ cents[oc]) for oc in cents if oc != c]
        sil[node] = proto[node] - (max(others) if others else 0.0)
    return proto, prox, sil


def graph_metrics(G: nx.Graph, community: dict[int, int]):
    """strength, pagerank, eigenvector, coreness — all from the weighted graph."""
    strength = {n: float(d) for n, d in G.degree(weight="weight")}
    pagerank = nx.pagerank(G, weight="weight")
    try:
        eigen = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:                       # disconnected / convergence
        eigen = {n: float("nan") for n in G}

    coreness = {}
    for n in G:
        inside = total = 0.0
        for nbr, d in G[n].items():
            w = d.get("weight", 1.0)
            total += w
            if community.get(nbr) == community.get(n):
                inside += w
        coreness[n] = (inside / total) if total else 0.0
    return strength, pagerank, eigen, coreness


def collect(ids, unit, community, targets, G, freq) -> dict[str, dict]:
    proto, prox, sil = vec_metrics(ids, unit, community, targets)
    strength, pagerank, eigen, coreness = graph_metrics(G, community)
    return {
        "prototypicality": proto,
        "proximity_to_virtue": prox,
        "frequency": {n: float(freq.get(n, 0)) for n in ids},
        "strength": strength,
        "pagerank": pagerank,
        "eigenvector": eigen,
        "silhouette": sil,
        "coreness": coreness,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def within_community_spearman(ids, community, scores) -> np.ndarray:
    """Mean (community-size-weighted) Spearman rank-corr between every metric pair,
    computed *within* each community (the legend ranks within a community)."""
    comms = collections.defaultdict(list)
    for n in ids:
        comms[community[n]].append(n)
    m = len(METRICS)
    acc = np.zeros((m, m))
    wsum = 0.0
    for members in comms.values():
        if len(members) < 3:
            continue
        cols = np.array([[scores[met][n] for met in METRICS] for n in members])
        if np.allclose(cols.std(axis=0), 0):
            continue
        rho, _ = spearmanr(cols)
        rho = np.atleast_2d(np.asarray(rho, dtype=float))
        acc += np.nan_to_num(rho) * len(members)
        wsum += len(members)
    return acc / wsum if wsum else acc


def ranked(members, score) -> list[str]:
    return sorted(members, key=lambda n: score[n], reverse=True)


def build_report(ids, community, targets, scores, top: int) -> str:
    comms = collections.defaultdict(list)
    for n in ids:
        comms[community[n]].append(n)

    out = ["# Community node-ordering — metric comparison\n"]
    corr = within_community_spearman(ids, community, scores)
    out.append("## Within-community Spearman correlation between orderings\n")
    out.append("How redundant the metrics are (1.0 = identical ranking). "
               "Low/negative pairs give genuinely different legends.\n")
    head = "| |" + "|".join(m[:6] for m in METRICS) + "|"
    out.append(head)
    out.append("|" + "---|" * (len(METRICS) + 1))
    for i, met in enumerate(METRICS):
        row = f"| **{met[:12]}** |" + \
            "|".join(f"{corr[i, j]:+.2f}" for j in range(len(METRICS))) + "|"
        out.append(row)
    out.append("")

    for c in sorted(comms):
        members = comms[c]
        tset = [n for n in members if n in targets]
        out.append(f"## Community {c} — {len(members)} nodes"
                   + (f"  (targets: {' '.join(tset)})" if tset else ""))
        cols = {met: ranked(members, scores[met])[:top] for met in METRICS}
        out.append("| rank |" + "|".join(METRICS) + "|")
        out.append("|---|" + "---|" * len(METRICS))
        for r in range(min(top, len(members))):
            cells = []
            for met in METRICS:
                n = cols[met][r]
                mark = "*" if n in targets else ""
                cells.append(f"{mark}{n}{mark}")
            out.append(f"| {r + 1} |" + "|".join(cells) + "|")
        out.append("")
    return "\n".join(out)


def print_digest(corr: np.ndarray) -> None:
    print("\nWithin-community Spearman (metric redundancy):")
    print("        " + " ".join(f"{m[:5]:>5}" for m in METRICS))
    for i, met in enumerate(METRICS):
        print(f"{met[:7]:>7} " +
              " ".join(f"{corr[i, j]:+.2f}"[:5].rjust(5) for j in range(len(METRICS))))
    # Flag the most-distinct pairs (candidates for a useful two-way toggle).
    pairs = []
    for i in range(len(METRICS)):
        for j in range(i + 1, len(METRICS)):
            pairs.append((corr[i, j], METRICS[i], METRICS[j]))
    pairs.sort()
    print("\nMost divergent metric pairs (distinct orderings):")
    for rho, a, b in pairs[:5]:
        print(f"  {rho:+.2f}  {a}  vs  {b}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="mengzi", choices=["mengzi", "sep"])
    ap.add_argument("--top", type=int, default=12,
                    help="rows per community in the report (default 12)")
    ap.add_argument("--out", type=Path, default=None,
                    help="report path (default analysis/<corpus>/community_ordering.md)")
    args = ap.parse_args()

    emb_path = config.EMBEDDINGS / f"{args.corpus}.json"
    graph_path = config.ANALYSIS / args.corpus / "networks" / "full.json"
    conllu = config.MENGZI_CONLLU if args.corpus == "mengzi" else Path("/nonexistent")
    for p in (emb_path, graph_path):
        if not p.exists():
            raise SystemExit(f"missing artifact: {p} — run `python -m main "
                             f"--corpus {args.corpus} --artifacts` first")

    ids, unit, community, targets = load_embeddings(emb_path)
    G = load_graph(graph_path)
    freq = load_frequencies(conllu)
    if not freq:
        print(f"note: no frequencies for {args.corpus} (frequency metric will be 0); "
              "TF-IDF specificity is skipped either way (needs co-occurrence counts).")

    scores = collect(ids, unit, community, targets, G, freq)

    n_comm = len(set(community.values()))
    print(f"corpus={args.corpus}  nodes={len(ids)}  communities={n_comm}  "
          f"targets={len(targets)}")
    print_digest(within_community_spearman(ids, community, scores))

    out = args.out or (config.ANALYSIS / args.corpus / "community_ordering.md")
    out.write_text(build_report(ids, community, targets, scores, args.top),
                   encoding="utf-8")
    print(f"\nfull per-community tables -> {out}")


if __name__ == "__main__":
    main()
