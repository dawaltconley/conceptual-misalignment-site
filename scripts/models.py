from typing import TYPE_CHECKING, Literal, Self
from dataclasses import dataclass, asdict, field

if TYPE_CHECKING:
    from collections import Counter
    from networkx import Graph
    from pathlib import Path
    from numpy import ndarray
    from _typeshed import DataclassInstance
    from embeddings.analyze import Method as SimMethod, SimTransform
    from embeddings.vectors import DebiasMethod

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


def _save_json(data: "DataclassInstance", filepath: "Path",
               *, indent: int | None = None) -> None:
    import json
    with filepath.open("w", encoding="utf-8") as file:
        json.dump(asdict(data), file, ensure_ascii=False, indent=indent,
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
    """The other words this term matched — the same list its node carries, scoped
    to whatever the enclosing file is: one article/chapter on a per-source
    co-occurrence file, the whole corpus on a similarity file and in the manifest
    (and so in ``src/data/terms.json``)."""


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
    freq: int = 0
    """Raw occurrence count over the whole corpus — how often the word is actually
    said, as opposed to how widely it is spread (``doc_freq``). The two answer
    different questions and rank differently: ``doc_freq`` saturates at the corpus's
    document count, so most of the vocabulary ties, while occurrences are
    heavy-tailed (log or rank them for display). Summed across a variant merge,
    which is exact for occurrences but not for document frequencies — see
    ``embeddings.occurrences.content_frequencies``."""
    norm: float = 0.0
    """L2 norm in the centered/debiased analysis space (≈ distance from the corpus
    centroid). The exported ``vec`` is L2-normalized (direction only), so this is the
    only place the radial 'how far out on the manifold' signal survives — the
    discriminating variable for the norm-vs-bipolar debias diagnostics."""
    strength: float = 0.0
    """Weighted degree in the similarity graph (sum of incident edge weights);
    high = a dense/tight region of the space. 0 for nodes absent from the graph."""
    pagerank: float = 0.0
    """PageRank on the weighted similarity graph — global importance/centrality.
    0 for nodes absent from the graph."""
    eigenvector: float = 0.0
    """Eigenvector centrality on the weighted similarity graph — importance weighted
    by neighbors' importance. 0 for graph-absent nodes (or if it fails to converge)."""
    variants: list[str] = field(default_factory=list)
    """The other words this node stands for, from either of the two things that
    make one label cover several: the derivational merge
    (``Pipeline.merge_variants`` — ``inspiration`` carrying ``['inspirational',
    'inspire']``), or, for a target, the lemmas its rendering's globs matched
    (``wisdom`` carrying ``['wise', 'wisely']`` — see
    ``embeddings.occurrences.matched_lemmas``). Targets never merge, so the two
    sources never both apply to one node. Empty for every unmerged non-target,
    and for a target the corpus happened to use only under its own label."""


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


def _eigenvector(graph: "Graph | None") -> dict[str, float]:
    """Eigenvector centrality per node (0 for graph-absent nodes; empty on failure)."""
    if graph is None or graph.number_of_edges() == 0:
        return {}
    import networkx as nx
    try:
        cent = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:  # convergence failure / disconnected components
        return {}
    return {n: float(v) for n, v in cent.items()}


@dataclass
class Embeddings:
    source: Source
    dims: int
    documents: int
    """Total number of documents (sources) in the corpus, for relative doc-freq."""
    nodes: list[Vector]

    @classmethod
    def from_matrix(cls, source: Source, labels: list[str], matrix: "ndarray", targets: set[str] | frozenset[str] = set(), communities: dict[str, int] = {}, doc_freq: dict[str, int] = {}, documents: int = 0, graph: "Graph | None" = None, norms: dict[str, float] = {}, variants: dict[str, list[str]] = {}, freq: "dict[str, int] | Counter[str]" = {}):
        strength = _weighted_degree(graph)
        pagerank = _pagerank(graph)
        eigenvector = _eigenvector(graph)
        nodes: list[Vector] = []
        for label, row in zip(labels, matrix):
            vector = Vector(label, label in targets,
                            communities.get(label, -1), row,
                            doc_freq.get(label, 0),
                            int(freq.get(label, 0)),
                            norms.get(label, 0.0),
                            strength.get(label, 0.0),
                            pagerank.get(label, 0.0),
                            eigenvector.get(label, 0.0),
                            list(variants.get(label, ())))
            nodes.append(vector)
        return cls(Source.from_sourcelike(source), int(matrix.shape[1]),
                   documents, nodes)

    def save_json(self, filepath: "Path") -> None:
        _save_json(self, filepath)


@dataclass
class TermIndex:
    """What one corpus run produced for one term label: its whole-corpus counts
    and the ``Source`` (provenance + ``data`` web path + per-source
    ``occurrences``) of every file written for it."""
    term: TermData
    total_occurrences: int = 0
    """The term's occurrences across the analyzed corpus. The master index adds
    ``chinese_philosophy_occurrences`` on top for its grand total."""
    chinese_philosophy_occurrences: int = 0
    """Occurrences inside the SEP articles excluded as Chinese philosophy —
    fetched for counting only, never fed to the embedding corpus. 0 for Mengzi."""
    cooccurrence: list[Source] = field(default_factory=list)
    similarity: list[Source] = field(default_factory=list)


@dataclass
class CorpusIndex:
    """A manifest of one corpus run — everything it wrote and where the site can
    fetch it. Written to ``<Pipeline.out_dir>/index.json`` by
    ``output.CorpusWriter``, and the *only* thing ``main.build_master`` reads for
    that corpus: paths are recorded at write time rather than re-derived from
    filenames, and one corpus's manifest survives a run of the other."""
    corpus: str
    """``"mengzi"`` / ``"sep"`` — the corpus key used in the master index."""
    source: Source
    """Corpus-level provenance (the Mengzi, or the combined SEP)."""
    embeddings: Source | None = None
    """The corpus's PCA-reduced embedding dataset (``/embeddings/{corpus}.json``)."""
    terms: dict[str, TermIndex] = field(default_factory=dict)
    """Keyed by term label — hanzi for Mengzi, ``Rendering.label`` for SEP."""

    def save_json(self, filepath: "Path") -> None:
        # Terms are accumulated from a set, so sort them for a stable diff — the
        # manifests are checked in. Source lists keep their insertion order
        # (whole corpus first, then chapters/articles), which is meaningful.
        self.terms = dict(sorted(self.terms.items()))
        _save_json(self, filepath, indent=2)

    @classmethod
    def load(cls, filepath: "Path") -> "CorpusIndex | None":
        """Rehydrate a manifest, or ``None`` if the corpus has never been run."""
        import json
        if not filepath.exists():
            return None
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        emb = raw.get("embeddings")
        return cls(
            corpus=raw["corpus"],
            source=Source(**raw["source"]),
            embeddings=Source(**emb) if emb else None,
            terms={label: TermIndex(
                term=TermData(**t["term"]),
                total_occurrences=t["total_occurrences"],
                chinese_philosophy_occurrences=t["chinese_philosophy_occurrences"],
                cooccurrence=[Source(**s) for s in t["cooccurrence"]],
                similarity=[Source(**s) for s in t["similarity"]],
            ) for label, t in raw["terms"].items()},
        )


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

    min_doc_freq: float | None = None
    """Document-frequency floor on the embedding vocabulary (drops words confined to
    a handful of documents, which a corpus-wide frequency floor alone can miss — e.g.
    a word repeated many times within one source). Same sklearn ``min_df`` convention
    as ``max_doc_freq``: <= 1.0 is a fraction of the corpus's documents, > 1 an
    absolute document count; None disables it. Targets are always kept regardless."""

    merge_variants: bool = False
    """Collapse derivational variants of one word into a single vocabulary entry
    (``inspire``/``inspiration``), so the scatter plots one point per word rather
    than one per lemma. English-only: candidates come from Open English Wordnet,
    and classical Chinese has no derivational suffixes, so leave it off for
    Mengzi. Target renderings are never merged. See ``embeddings.families``.

    Master switch: it decides whether the merge is *computed* at all. The two
    flags below decide which lens it is then *applied* to."""

    merge_similarity: bool = True
    """Apply the variant merge to the embedding lens — pooled vectors, cosine
    graph, communities, scatter export. Off (with ``merge_variants`` on) computes
    the merge for co-occurrence only and leaves the scatter one point per lemma;
    the exported ``variants`` field then carries target matches only, since the
    ordinary nodes are not in fact merged."""

    merge_cooccurrence: bool = True
    """Apply the variant merge to the PMI lens, so a node means the same word in
    both networks. Only has an effect where the merge runs (``merge_variants``,
    i.e. SEP today — ``run_mengzi`` computes no alias).

    Separable from ``merge_similarity`` on purpose: the merge is *gated* on
    embedding cosine, a paradigmatic criterion, so switching it on here lets one
    lens shape the other. See ``notes/derivational-variant-merging.md`` — the
    argument is that the candidates are morphological and cosine only vetoes
    them, but if that stops convincing, this is the revert."""

    merge_threshold: float = 0.45
    """Cosine floor for ``merge_variants``, applied in the centred/debiased
    analysis space with complete linkage — every pair inside a merged family must
    clear it. Guards the merge against collapsing stem-mates that drifted apart
    (``know``/``knowledge``); raise it to merge only near-duplicates.

    Calibrate against ``analysis/{corpus}/family_candidates.csv`` via
    ``tools/family_diagnostics.py``, NOT against the exported vectors — those are
    PCA-reduced and the truncation inflates cosine badly (0.70 picks 231 merges on
    the export and 31 in the real space). A useful anchor: the *most* similar pair
    of distinct target renderings, ``benevolence``/``humaneness``, sits at 0.322,
    and no target pair exceeds 0.335 — so a floor above ~0.40 keeps every merge
    tighter than any two words the project treats as different concepts."""

    center: bool = True
    """Whether to center embeddings by subtracting their centroid. Used to fix
    anisotropy."""

    debias: "DebiasMethod" = "none"
    """Nuisance-direction removal applied *after* centering (so pair with
    ``center=True``). Centering fixes the anisotropy offset but leaves word frequency
    loaded onto the top PCs — the reason a doc-frequency coloring still shows a clean
    PCA gradient. ``abtt`` (all-but-the-top) projects out the top ``debias_k``
    components; ``whiten`` PCA-whitens so no direction dominates the variance;
    ``none`` leaves the space untouched. See ``embeddings.vectors.debias_matrix``."""

    debias_k: int | None = None
    """Components for ``debias``: for ``abtt`` the number to project out (default
    ``max(1, D // 100)``, Mu & Viswanath's rule); for ``whiten`` the number of top
    axes to keep before whitening (``None`` keeps all). Ignored when ``debias`` is
    ``none``."""

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

    merge_deps: frozenset[str] | None = None
    """UD dependency relations treated as *word formation*: a token attached to its
    head by one of these is part of the same word, and the two are recombined into
    a single token before anything else sees the corpus (天 + 下 -> 天下). Chinese
    only — the treebank annotates one character per token, so without this the
    vocabulary is characters rather than words. ``frozenset({"compound", "flat",
    "fixed"})`` is the vetted set; ``conj``/``nmod`` are deliberately absent (they
    would merge 仁義). None leaves tokenization exactly as the source has it.
    See ``corpus.recombine`` and ``notes/multi-character-tokenization.md``."""

    merge_lexicon: "Path | None" = None
    """A segmentation JSONL (from ``python -m cli.segment --source conllu``) whose
    word boundaries are merged alongside ``merge_deps``. This is what supplies the
    bisyllabic words the treebank labels ``nmod`` and the relations therefore miss
    — 諸侯, 天子, 大夫, 聖人. Requires ``merge_deps`` to be set. The file is optional:
    generating it is a manual step, so a missing one is a no-op, never an error."""

    out_dir: "Path"
