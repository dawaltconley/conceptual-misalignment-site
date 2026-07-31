from typing import TYPE_CHECKING
from dataclasses import dataclass, asdict, field

if TYPE_CHECKING:
    from networkx import Graph
    from pathlib import Path
    from numpy import ndarray
    from _typeshed import DataclassInstance


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


def _source_ref(source: object) -> dict | None:
    """Reduce any ``Source`` to a small, JSON-safe subset. Done at construction
    time (not via ``asdict``/``_json_default``) so it is uniform whether the source
    is a plain class (``Chapter``) or a dataclass (``SEP``/``SEPSearch``) — for the
    latter ``asdict`` would otherwise recurse into every field (dumping ``.text`` /
    ``.articles``) and skip the ``id``/``text`` *properties*."""
    if source is None:
        return None
    return {
        "id": getattr(source, "id", None),
        "url": getattr(source, "url", None),
        "title": getattr(source, "title", None),
        "description": getattr(source, "description", None),
    }


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
                  indent=2, default=_json_default)


@dataclass
class Source:
    id: str
    url: str
    title: str
    description: str


class SourceData:
    source: Source


@dataclass
class TermData:
    label: str
    occurrences: int
    variants: list[str] = field(default_factory=list)


@dataclass
class NetworkData:
    term: TermData
    source: dict | None  # reduced via _source_ref at construction
    network: object | None

    def __init__(self, term: TermData, source: Source, network: "Graph | None"):
        from networkx import node_link_data
        self.term = term
        self.source = _source_ref(source)
        self.network = _sorted_node_link(
            node_link_data(network) if network is not None else None)

    def save_json(self, filepath: "Path") -> None:
        _save_json(self, filepath)


@dataclass
class NLPSource(Source):
    id: str
    url: str
    title: str
    description: str
    cooccurrence: object | None

    def __init__(self, source: Source, /, cooccurrence: "Graph | None" = None):
        from networkx import node_link_data
        self.id = source.id
        self.url = source.url
        self.title = source.title
        self.description = source.description
        self.cooccurrence = cooccurrence and node_link_data(cooccurrence)


# @dataclass
# class TermData:
#     term: str
#     sources: list[NLPSource]
#     stems: list[str] | None = None
#
#     def save_json(self, filepath: "Path") -> None:
#         _save_json(self, filepath)


@dataclass
class Vector:
    id: str
    target: bool
    community: int
    vec: "ndarray"


@dataclass
class Embeddings:
    source: dict | None  # reduced via _source_ref at construction
    dims: int
    nodes: list[Vector]

    @classmethod
    def from_matrix(cls, source: Source, labels: list[str], matrix: "ndarray", targets: set[str] | frozenset[str] = set(), communities: dict[str, int] = {}):
        nodes: list[Vector] = []
        for label, row in zip(labels, matrix):
            vector = Vector(label, label in targets,
                            communities.get(label, -1), row)
            nodes.append(vector)
        return cls(_source_ref(source), int(matrix.shape[1]), nodes)

    def save_json(self, filepath: "Path") -> None:
        _save_json(self, filepath)


def reduce_vectors(matrix: "ndarray", dims: int) -> "ndarray":
    """Mean-center + L2-normalize, then PCA to `dims` (variance-ordered columns)."""
    import numpy as np
    from sklearn.decomposition import PCA
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.clip(norms, 1e-12, None)
    n_components = min(dims, unit.shape[1], unit.shape[0])
    return PCA(n_components=n_components, random_state=0).fit_transform(unit)


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
