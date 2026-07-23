from fnmatch import fnmatchcase
from pathlib import Path

# Absolute anchors so the pipelines no longer depend on the current working
# directory. Everything is resolved relative to this file's location.
_SCRIPTS = Path(__file__).resolve().parent   # .../scripts
_ROOT = _SCRIPTS.parent                       # repo root

# --- Inputs (live inside scripts/) ---
MENGZI_DIR = _SCRIPTS / "data" / "mengzi"     # per-chapter Mengzi source .txt

# --- Outputs (live at the repo root) ---
DATA = _ROOT / "src" / "data"
DATA.mkdir(parents=True, exist_ok=True)

PUBLIC = _ROOT / "public"

SEP = PUBLIC / "sep"
SEP.mkdir(parents=True, exist_ok=True)

CTEXT = PUBLIC / "ctext"
CTEXT.mkdir(parents=True, exist_ok=True)

ANALYSIS = _ROOT / "analysis"     # embedding-pipeline artifacts
SEGPOS = _ROOT / "segpos"         # Xunzi segmentation output

EMBEDDINGS = PUBLIC / "embeddings"    # client-loaded scatter datasets
EMBEDDINGS.mkdir(parents=True, exist_ok=True)


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
