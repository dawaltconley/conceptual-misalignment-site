from fnmatch import fnmatchcase
from pathlib import Path

DATA = Path("../src/data")
DATA.mkdir(exist_ok=True)

PUBLIC = Path("../public")

SEP = PUBLIC / "sep"
SEP.mkdir(exist_ok=True)

CTEXT = PUBLIC / "ctext"
CTEXT.mkdir(exist_ok=True)


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


TERMS: list[Term] = [
    Term('仁', (
        Rendering('benevolence', 'benevolen*'),
        Rendering('humaneness', 'humane*'),
    )),
    Term('義', (
        Rendering('righteousness', 'righteous*'),
        Rendering('justice', 'justness'),
    )),
]

# All canonical English target labels across every term.
ENGLISH_LABELS: set[str] = {r.label for t in TERMS for r in t.renderings}


def match_rendering(lemma: str, pos: str) -> str | None:
    """Return the canonical rendering label a (lemma, POS) belongs to, or None."""
    for term in TERMS:
        for r in term.renderings:
            if r.matches(lemma, pos):
                return r.label
    return None
