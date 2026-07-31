from pathlib import Path
from lib import Term, Rendering

# Absolute anchors so the pipelines no longer depend on the current working
# directory. Everything is resolved relative to this file's location.
_SCRIPTS = Path(__file__).resolve().parent   # .../scripts
_ROOT = _SCRIPTS.parent                       # repo root

# --- Inputs (live inside scripts/) ---
MENGZI_DIR = _SCRIPTS / "data" / "mengzi"     # per-chapter Mengzi source .txt
MENGZI_CONLLU = _SCRIPTS / "data" / "mengzi.conllu"

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
