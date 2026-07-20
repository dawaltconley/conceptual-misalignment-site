from pathlib import Path

DATA = Path("../src/data")
DATA.mkdir(exist_ok=True)

PUBLIC = Path("../public")

SEP = PUBLIC / "sep"
SEP.mkdir(exist_ok=True)

CTEXT = PUBLIC / "ctext"
CTEXT.mkdir(exist_ok=True)


class Term:
    def __init__(self, hanzi: str, translations: tuple[str, ...]):
        self.hanzi = hanzi
        self.english = translations


TERMS: list[Term] = [
    Term('仁', ('benevolence', 'humaneness')),
    Term('義', ('righteousness', 'justice')),
]
