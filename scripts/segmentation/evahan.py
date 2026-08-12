"""Few-shot examples drawn from the EvaHan 2022 gold data.

The few-shot examples pin the segmentation convention, so where they come from
decides what the model does. The three hand-written ones in :mod:`segmentation.segpos`
were placeholders; EvaHan's gold files are a far better source for one specific
reason — **they segment exactly the words the Kyoto treebank's word-formation
relations miss**: 諸侯 (163), 天下 (109), 大夫 (68), 寡人 (62), 君子 (27), 天子 (25),
聖人 (10), 庶人 (4). That set is the whole motivation for consulting a segmenter at
all (see ``notes/multi-character-tokenization.md``), so demonstrating it is worth
more than any general gain in tagging accuracy.

Format is identical to this project's: ``word/TAG`` separated by spaces, and every
tag in ``segpos.TAGSET`` appears in the gold data (``f`` was added to TAGSET for
these files).

Two things to be careful about, both handled by :func:`select`:

- **Genre.** ``testa`` is 左傳; ``testb`` reads as 資治通鑑. Both are narrative
  history, thick with personal and place names, whereas the Mengzi is
  philosophical dialogue. Picking examples by tag diversity alone yields
  齊威王/藺相如-dense passages that bias the model toward name merging, so the
  default scoring rewards *common-noun* compounds and dialogue particles instead.
- **Leakage.** Sample shots from one file and evaluate on the other
  (``tools/eval_segmentation.py``), never both from the same one.
"""

from __future__ import annotations

from pathlib import Path

EVAHAN_DIR = Path("/home/dawaltco/Code/itp/ACDS")
TESTA = EVAHAN_DIR / "EvaHan_testa_gold.txt"      # 左傳
TESTB = EVAHAN_DIR / "EvaHan_testb_gold.txt"      # 資治通鑑

PUNCTUATION = set("，。？！；：、「」『』（）《》〈〉·…—“”\"'‘’")

# Particles that mark reported speech and argument — what the Mengzi is made of,
# as against EvaHan's default register of campaign narrative.
DIALOGUE_MARKERS = ("曰", "也", "矣", "乎", "之", "者")


class Shot:
    """One gold line, in both the punctuated and unpunctuated views."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    @property
    def words(self) -> list[str]:
        return [w for w, _ in self.pairs]

    @property
    def text(self) -> str:
        return "".join(self.words)

    def tagged(self, with_pos: bool = True) -> str:
        if with_pos:
            return " ".join(f"{w}/{t}" for w, t in self.pairs)
        return "/".join(self.words) + "/"

    def strip_punctuation(self) -> "Shot":
        return Shot([(w, t) for w, t in self.pairs
                     if not all(c in PUNCTUATION for c in w)])


def load(path: str | Path) -> list[Shot]:
    """Read an EvaHan gold file into one :class:`Shot` per line."""
    shots: list[Shot] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        pairs = [(tok.rpartition("/")[0], tok.rpartition("/")[2])
                 for tok in line.split() if "/" in tok]
        if pairs:
            shots.append(Shot(pairs))
    return shots


def _score(shot: Shot) -> float:
    """Rank a candidate by how much it looks like the text we actually tag.

    Rewards multi-character *common* nouns (the behaviour we want demonstrated)
    and dialogue particles; penalises proper-noun density, which is where EvaHan's
    register diverges most from the Mengzi's.
    """
    common = sum(1 for w, t in shot.pairs if len(w) > 1 and t in {"n", "t", "f"})
    proper = sum(1 for _, t in shot.pairs if t in {"nr", "ns"})
    dialogue = sum(1 for w, _ in shot.pairs if w in DIALOGUE_MARKERS)
    variety = len({t for _, t in shot.pairs})
    return 3.0 * common + 1.5 * dialogue + 0.5 * variety - 2.0 * proper


def select(
    path: str | Path = TESTA,
    n: int = 12,
    *,
    tagset: set[str] | frozenset[str] | None = None,
    min_chars: int = 18,
    max_chars: int = 55,
    unpunctuated: bool = True,
    with_pos: bool = True,
) -> list[tuple[str, str]]:
    """The ``n`` best-scoring gold lines, as ``(input, tagged)`` few-shot pairs.

    ``tagset`` drops any line using a tag the prompt cannot emit — the GBNF
    grammar only permits tags in ``segpos.VALID_TAGS``, so demonstrating one
    outside it teaches something the model is then forbidden to produce. Pass
    ``with_pos=False`` for the segmentation-only process.
    """
    candidates: list[Shot] = []
    for shot in load(path):
        if tagset is not None and any(t not in tagset for _, t in shot.pairs):
            continue
        view = shot.strip_punctuation() if unpunctuated else shot
        if not (min_chars <= len(view.text) <= max_chars):
            continue
        if not any(len(w) > 1 and t in {"n", "t", "f"} for w, t in view.pairs):
            continue
        candidates.append(view)

    candidates.sort(key=_score, reverse=True)
    return [(s.text, s.tagged(with_pos)) for s in candidates[:n]]
