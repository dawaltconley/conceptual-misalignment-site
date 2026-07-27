from typing import TYPE_CHECKING, Protocol
from dataclasses import dataclass, asdict

if TYPE_CHECKING:
    from networkx import Graph
    from pathlib import Path


class Source(Protocol):
    url: str
    title: str
    description: str


@dataclass
class NLPSource(Source):
    url: str
    title: str
    description: str
    co_occurance: object

    def __init__(self, source: Source, /, co_occurance: "Graph"):
        from networkx import node_link_data
        self.url = source.url
        self.title = source.title
        self.description = source.description
        self.co_occurance = node_link_data(co_occurance)


@dataclass
class TermData:
    term: str
    sources: list[NLPSource]
    stems: list[str] | None = None

    def save_json(self, filepath: "Path") -> None:
        import json
        serialized = json.dumps(asdict(self), indent=2)
        filepath.write_text(serialized, encoding="utf-8")


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
