from typing import TYPE_CHECKING, Literal, Self
from dataclasses import dataclass, asdict, field

if TYPE_CHECKING:
    from networkx import Graph
    from pathlib import Path
    from numpy import ndarray
    from _typeshed import DataclassInstance
    from embeddings.analyze import Method as SimMethod, SimTransform

Pooling = Literal["mean", "max", "none"]


def _json_default(o: object):
    """Serialize the non-JSON values that survive ``asdict``: numpy arrays/scalars
    and live ``Source`` objects (reduced to a small, JSON-safe subset — never
    their full ``.text``)."""
    import numpy as np
    if isinstance(o, np.ndarray):
        return np.round(o, 5).tolist()
    if isinstance(o, np.generic):
        return round(float(o), 5)
    if hasattr(o, "title") and hasattr(o, "url"):  # Source-like
        return {
            "id": getattr(o, "id", None),
            "url": getattr(o, "url", None),
            "title": getattr(o, "title", None),
            "description": getattr(o, "description", None),
        }
    raise TypeError(
        f"Object of type {type(o).__name__} is not JSON serializable")


def _sorted_node_link(data: dict | None) -> dict | None:
    """Sort a node-link graph's node array by id, so the serialized JSON lists
    nodes in a readable, stable order (ordering only, not data). Edge order is left
    as-is — the graphs are not made fully reproducible across runs."""
    if not data:
        return data
    data["nodes"].sort(key=lambda n: str(n.get("id")))
    return data


def _save_json(data: "DataclassInstance", filepath: "Path") -> None:
    import json
    with filepath.open("w", encoding="utf-8") as file:
        json.dump(asdict(data), file, ensure_ascii=False,
                  default=_json_default)


@dataclass
class Source:
    """A source (chapter / article / whole corpus / corpus stand-in). ``occurrences``
    (the term's count in this source) and ``data`` (web path to this source's dataset)
    are context-dependent: unset on the per-corpus embedding source and on a per-file
    ``NetworkData.source``; the master index fills both in."""
    id: str
    url: str
    title: str
    description: str | None
    # kw_only so subclasses (corpus.sep.SEP*) can add positional fields after these
    occurrences: int | None = field(default=None, kw_only=True)
    data: str | None = field(default=None, kw_only=True)

    @classmethod
    def from_sourcelike(cls, s: Self, *, occurrences: int | None = None,
                        data: str | None = None) -> Self:
        return cls(s.id, s.url, s.title, s.description,
                   occurrences=occurrences, data=data)


@dataclass
class TermData:
    label: str
    variants: list[str] = field(default_factory=list)


@dataclass
class NetworkData:
    term: TermData
    source: Source
    network: object | None

    def __init__(self, term: TermData, source: Source, network: "Graph | None",
                 *, occurrences: int = 0):
        from networkx import node_link_data
        self.term = term
        self.source = Source.from_sourcelike(source, occurrences=occurrences)
        self.network = _sorted_node_link(
            node_link_data(network) if network is not None else None)

    def save_json(self, filepath: "Path") -> None:
        _save_json(self, filepath)


@dataclass
class Vector:
    id: str
    target: bool
    community: int
    vec: "ndarray"
    doc_freq: int = 0
    """How many documents (sources) in the corpus this word appears in."""
    strength: float = 0.0
    """Weighted degree in the similarity graph (sum of incident edge weights);
    high = a dense/tight region of the space. 0 for nodes absent from the graph."""
    pagerank: float = 0.0
    """PageRank on the weighted similarity graph — global importance/centrality.
    0 for nodes absent from the graph."""


def _weighted_degree(graph: "Graph | None") -> dict[str, float]:
    """Per-node sum of incident edge weights (0 for graph-absent nodes)."""
    if graph is None:
        return {}
    return {n: float(d) for n, d in graph.degree(weight="weight")}


def _pagerank(graph: "Graph | None") -> dict[str, float]:
    """PageRank per node on the weighted graph (0 for graph-absent nodes)."""
    if graph is None or graph.number_of_edges() == 0:
        return {}
    import networkx as nx
    return {n: float(v) for n, v in nx.pagerank(graph, weight="weight").items()}


@dataclass
class Embeddings:
    source: Source
    dims: int
    documents: int
    """Total number of documents (sources) in the corpus, for relative doc-freq."""
    nodes: list[Vector]

    @classmethod
    def from_matrix(cls, source: Source, labels: list[str], matrix: "ndarray", targets: set[str] | frozenset[str] = set(), communities: dict[str, int] = {}, doc_freq: dict[str, int] = {}, documents: int = 0, graph: "Graph | None" = None):
        strength = _weighted_degree(graph)
        pagerank = _pagerank(graph)
        nodes: list[Vector] = []
        for label, row in zip(labels, matrix):
            vector = Vector(label, label in targets,
                            communities.get(label, -1), row,
                            doc_freq.get(label, 0),
                            strength.get(label, 0.0),
                            pagerank.get(label, 0.0))
            nodes.append(vector)
        return cls(Source.from_sourcelike(source), int(matrix.shape[1]),
                   documents, nodes)

    def save_json(self, filepath: "Path") -> None:
        _save_json(self, filepath)


class Rendering:
    """One English rendering of a term: a canonical label + its word family.

    ``patterns`` are ``fnmatch`` globs (or exact forms) matched against a token's
    lowercased lemma, so morphological variants pool into one node (e.g.
    ``humane*`` -> humane / humaneness / humanely, but not humanity/human).
    ``pos`` optionally restricts matches to given spaCy coarse POS tags, for
    disambiguating forms like adjectival vs. adverbial ``just`` (unused for now).
    """

    def __init__(
        self,
        *patterns: str,
        pos: tuple[str, ...] | None = None,
    ):
        self.label = patterns[0]
        self.patterns = tuple(patterns)
        self.pos = frozenset(pos) if pos else None

    def matches(self, lemma: str, pos: str | None = None) -> bool:
        from fnmatch import fnmatchcase
        if pos and self.pos and pos not in self.pos:
            return False
        return any(fnmatchcase(lemma, p) for p in self.patterns)


class Term:
    def __init__(self, hanzi: str, renderings: tuple[Rendering, ...]):
        self.hanzi = hanzi
        self.renderings = tuple(renderings)

    @property
    def english(self) -> tuple[str, ...]:
        """Rendering labels only (backward-compatible with the old tuple API)."""
        return tuple(r.label for r in self.renderings)


@dataclass(kw_only=True)
class Pipeline:
    min_freq: int = 5
    """Minimum corpus frequency for a word to enter the *embedding* vocabulary
    (governs the scatter/similarity vocab and vector stability). Applied over the
    whole corpus, so it can be moderately high."""

    cooccurrence_min_freq: int = 3
    """Minimum frequency for a word to enter a *co-occurrence* network. Applied
    PER SOURCE — a single article or chapter — so it must stay low; reusing the
    (whole-corpus) min_freq here starves per-article graphs to null. Kept separate
    for that reason."""

    max_doc_freq: float | None = None
    """Document-frequency cap on the embedding vocabulary (drops ubiquitous, generic
    words that a frequency floor keeps). sklearn ``max_df`` convention: <= 1.0 is a
    fraction of the corpus's documents, > 1 an absolute document count; None disables
    it. Targets are always kept regardless."""

    center: bool = True
    """Whether to center embeddings by subtracting their centroid. Used to fix
    anisotropy."""

    sim_network: "SimMethod" = "knn"
    """The kind of similarity network produced: a global quantile threshold
    (threshold) or relative neighborhoods (knn)."""

    quantile: float = 0.9
    """For sim_network="threshold": the percentile cutoff in [0, 1) on the
    similarity distribution — 0.9 keeps the strongest 10% of edges. Rank-defined,
    so it is invariant to sim_transform and robust to anisotropy (unlike an
    absolute cosine cutoff). Raise it to sparsify the graph / drop more nodes."""

    knn_k: int = 8
    """K-nearest-neighbors for a knn graph: keep each node's `k` most-similar
    neighbors by rank (union kNN: an edge if *either* endpoint ranks the other
    in its top-k)."""

    sim_transform: "SimTransform" = "neglog"
    """The type of log transform to apply to cosine similarity values.
    See notes/anisotropy-and-network-construction.md"""

    resolution: float = 1.0
    """Louvain resolution for community detection. >1 yields more, smaller/narrower
    communities (splits the frequency/register hub that otherwise absorbs most
    target terms); <1 yields fewer, larger ones."""

    reduce_to_dims: int = 50
    """Number of dimensions to keep when serializing vectors to json. Affects
    the size of the export."""

    max_network_nodes: int = 15
    """Cap on the number of nodes included in the similarity and co-occurrence
    networks."""

    model: str
    """The LLM used for calculating embeddings."""

    batch_size: int = 32
    """Number of embeddings to calcualte at a given time; affects memory and
    speed of the pipeline."""

    subword_pooling: "Pooling" = "mean"
    """How to combine an occurrence's subword-token vectors into one occurrence
    vector: mean, max, or none (the first / word-initial piece — the BPE root,
    which pools a word family like symbol/symbolic/symbolize). Mostly relevant in
    English; GujiRoBERTa tokens are single characters, so it rarely applies there."""

    occurrence_pooling: "Pooling" = "max"
    """How to combine a word's per-occurrence vectors into its single type vector:
    max (default, per Wu & Wang), mean (smoother, less swayed by outlier contexts),
    or none (the first occurrence only). All fold online, so memory stays O(vocab)."""

    content_pos: frozenset[str] | None = None
    """Parts-of-speech to include. If None, POS is not filtered."""

    stopwords: frozenset[str] = frozenset()
    """Words to exclude from results."""

    out_dir: "Path"
