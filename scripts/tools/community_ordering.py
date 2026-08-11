"""Test every community-node-ordering metric from notes/community-legend-ordering.md.

For a given corpus it loads the shipped artifacts —

  * ``public/embeddings/{corpus}.json``     (id / target / community / 50-d PCA vec,
    plus the per-node ``strength`` / ``pagerank`` / ``eigenvector`` / ``doc_freq``
    the pipeline already computed on that run's similarity graph)
  * ``analysis/{corpus}/networks/full.json`` (the weighted similarity graph —
    **optional**, needed only for ``coreness``)
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

**Every metric except ``coreness`` comes from the embeddings artifact alone**, so it
cannot go out of sync with itself. ``networks/full.json`` is only written under
``--artifacts``, so on any ordinary pipeline run it is left behind by the newer
``{corpus}.json`` — which is what used to crash this tool with a ``KeyError`` on the
first node the stale graph had never heard of. Now a mismatched or absent graph
simply drops ``coreness`` from the report, with the reason printed.

A metric whose values are all identical (e.g. ``frequency`` when there is nothing to
read it from) is dropped too, rather than shown as a column of zeros that also drags
a row of 0.00 through the correlation matrix.
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
    """-> (ids, unit_vecs NxD, community, targets, node_scores). Vectors are
    L2-normalized so dot products are true cosines (the PCA-reduced vecs are not
    unit-norm).

    ``node_scores`` are the graph-derived fields the pipeline already wrote onto
    each node from *this run's* similarity graph, so they need no second artifact
    and cannot disagree with the communities they are reported against.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    ids = [n["id"] for n in nodes]
    vecs = np.array([n["vec"] for n in nodes], dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.clip(norms, 1e-12, None)
    community = {n["id"]: int(n["community"]) for n in nodes}
    targets = {n["id"] for n in nodes if n.get("target")}
    node_scores = {
        field: {n["id"]: float(n.get(field) or 0.0) for n in nodes}
        for field in ("strength", "pagerank", "eigenvector", "doc_freq", "freq")
    }
    return ids, unit, community, targets, node_scores


def load_graph(path: Path, ids: list[str]) -> tuple[nx.Graph | None, str]:
    """The weighted similarity graph, or ``(None, reason)``.

    Only ``coreness`` needs it, and it is written solely by ``--artifacts`` — so it
    is routinely older than the embeddings artifact beside it. Comparing node sets
    is what catches that: a graph from another run describes another vocabulary
    (different variant merges, different ``min_freq``), and scoring one run's
    communities with the other's edges is meaningless even where the ids overlap.
    """
    if not path.exists():
        return None, f"{path} not found (written only by --artifacts)"
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.node_link_graph(data, edges="edges")
    missing = set(ids) - set(G)
    extra = set(G) - set(ids)
    if missing or extra:
        return None, (
            f"{path.name} is from a different run than the embeddings "
            f"({len(missing)} nodes missing from the graph, {len(extra)} extra) — "
            f"re-run with --artifacts to refresh it")
    return G, ""


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


def coreness_metric(G: nx.Graph, community: dict[str, int]) -> dict[str, float]:
    """Fraction of a node's edge weight that stays inside its own community."""
    coreness = {}
    for n in G:
        inside = total = 0.0
        for nbr, d in G[n].items():
            w = d.get("weight", 1.0)
            total += w
            if community.get(nbr) == community.get(n):
                inside += w
        coreness[n] = (inside / total) if total else 0.0
    return coreness


def frequency_metric(ids, node_scores, freq) -> tuple[dict[str, float], str]:
    """The best available "how common is this word", and where it came from.

    The artifact's ``freq`` is the right answer for both corpora — a raw
    occurrence count, merged consistently with the communities it is reported
    against. The conllu is the legacy Mengzi path, kept as a fallback for
    artifacts written before ``freq`` existed; ``doc_freq`` is the last resort
    and a poor one, since it saturates at the corpus's document count and leaves
    most of the vocabulary tied.
    """
    exported = node_scores.get("freq", {})
    if len(set(exported.values())) > 1:
        return dict(exported), "artifact freq (corpus occurrences)"
    if freq:
        return {n: float(freq.get(n, 0)) for n in ids}, "conllu lemma counts"
    return dict(node_scores["doc_freq"]), "doc_freq (saturating — re-run to export freq)"


def collect(ids, unit, community, targets, node_scores, freq,
            G: nx.Graph | None) -> dict[str, dict]:
    proto, prox, sil = vec_metrics(ids, unit, community, targets)
    frequency, _ = frequency_metric(ids, node_scores, freq)
    scores = {
        "prototypicality": proto,
        "proximity_to_virtue": prox,
        "frequency": frequency,
        "strength": node_scores["strength"],
        "pagerank": node_scores["pagerank"],
        "eigenvector": node_scores["eigenvector"],
        "silhouette": sil,
    }
    if G is not None:
        scores["coreness"] = coreness_metric(G, community)
    # A metric with no variation ranks nothing; keeping it would only add a column
    # of ties and a row of 0.00 to the correlation matrix.
    return {met: s for met, s in scores.items() if len(set(s.values())) > 1}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def within_community_spearman(ids, community, scores, metrics) -> np.ndarray:
    """Mean (community-size-weighted) Spearman rank-corr between every metric pair,
    computed *within* each community (the legend ranks within a community)."""
    comms = collections.defaultdict(list)
    for n in ids:
        comms[community[n]].append(n)
    m = len(metrics)
    acc = np.zeros((m, m))
    wsum = 0.0
    for members in comms.values():
        if len(members) < 3:
            continue
        cols = np.array([[scores[met][n] for met in metrics] for n in members])
        if np.allclose(cols.std(axis=0), 0):
            continue
        rho, _ = spearmanr(cols)
        rho = np.atleast_2d(np.asarray(rho, dtype=float))
        acc += np.nan_to_num(rho) * len(members)
        wsum += len(members)
    return acc / wsum if wsum else acc


def ranked(members, score) -> list[str]:
    return sorted(members, key=lambda n: score[n], reverse=True)


def build_report(ids, community, targets, scores, metrics, top: int,
                 skipped: dict[str, str]) -> str:
    comms = collections.defaultdict(list)
    for n in ids:
        comms[community[n]].append(n)

    out = ["# Community node-ordering — metric comparison\n"]
    for met, why in skipped.items():
        out.append(f"> **{met} skipped** — {why}\n")
    corr = within_community_spearman(ids, community, scores, metrics)
    out.append("## Within-community Spearman correlation between orderings\n")
    out.append("How redundant the metrics are (1.0 = identical ranking). "
               "Low/negative pairs give genuinely different legends.\n")
    head = "| |" + "|".join(m[:6] for m in metrics) + "|"
    out.append(head)
    out.append("|" + "---|" * (len(metrics) + 1))
    for i, met in enumerate(metrics):
        row = f"| **{met[:12]}** |" + \
            "|".join(f"{corr[i, j]:+.2f}" for j in range(len(metrics))) + "|"
        out.append(row)
    out.append("")

    for c in sorted(comms):
        members = comms[c]
        tset = [n for n in members if n in targets]
        out.append(f"## Community {c} — {len(members)} nodes"
                   + (f"  (targets: {' '.join(tset)})" if tset else ""))
        cols = {met: ranked(members, scores[met])[:top] for met in metrics}
        out.append("| rank |" + "|".join(metrics) + "|")
        out.append("|---|" + "---|" * len(metrics))
        for r in range(min(top, len(members))):
            cells = []
            for met in metrics:
                n = cols[met][r]
                mark = "*" if n in targets else ""
                cells.append(f"{mark}{n}{mark}")
            out.append(f"| {r + 1} |" + "|".join(cells) + "|")
        out.append("")
    return "\n".join(out)


def print_digest(corr: np.ndarray, metrics: list[str]) -> None:
    print("\nWithin-community Spearman (metric redundancy):")
    print("        " + " ".join(f"{m[:5]:>5}" for m in metrics))
    for i, met in enumerate(metrics):
        print(f"{met[:7]:>7} " +
              " ".join(f"{corr[i, j]:+.2f}"[:5].rjust(5) for j in range(len(metrics))))
    # Flag the most-distinct pairs (candidates for a useful two-way toggle).
    pairs = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            pairs.append((corr[i, j], metrics[i], metrics[j]))
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
    # Only the embeddings artifact is required; the graph is coreness-only.
    if not emb_path.exists():
        raise SystemExit(f"missing artifact: {emb_path} — run "
                         f"`python -m main --corpus {args.corpus}` first")

    ids, unit, community, targets, node_scores = load_embeddings(emb_path)
    G, graph_why = load_graph(graph_path, ids)
    freq = load_frequencies(conllu)
    _, freq_source = frequency_metric(ids, node_scores, freq)
    print(f"note: frequency metric from {freq_source}. TF-IDF specificity is "
          "skipped regardless (needs per-community co-occurrence counts).")
    if G is None:
        print(f"note: coreness skipped — {graph_why}")

    scores = collect(ids, unit, community, targets, node_scores, freq, G)
    metrics = [m for m in METRICS if m in scores]
    skipped = {m: (graph_why if m == "coreness" and G is None
                   else "no variation in this run's artifact")
               for m in METRICS if m not in scores}

    n_comm = len(set(community.values()))
    print(f"corpus={args.corpus}  nodes={len(ids)}  communities={n_comm}  "
          f"targets={len(targets)}  metrics={len(metrics)}/{len(METRICS)}")
    print_digest(within_community_spearman(ids, community, scores, metrics),
                 metrics)

    out = args.out or (config.ANALYSIS / args.corpus / "community_ordering.md")
    out.write_text(
        build_report(ids, community, targets, scores, metrics, args.top, skipped),
        encoding="utf-8")
    print(f"\nfull per-community tables -> {out}")


if __name__ == "__main__":
    main()
