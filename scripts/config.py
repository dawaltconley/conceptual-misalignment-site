from pathlib import Path

DATA = Path("../src/data")
DATA.mkdir(exist_ok=True)

SEP = DATA / "sep"
SEP.mkdir(exist_ok=True)

CTEXT = DATA / "ctext"
CTEXT.mkdir(exist_ok=True)


class Term:
    def __init__(self, hanzi: str, translations: set[str]):
        self.hanzi = hanzi
        self.english = tuple(translations)


TERMS: list[Term] = [
    Term('仁', {'benevolence', 'humaneness'})
]
