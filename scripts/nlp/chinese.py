from collections import defaultdict

from spacy.tokens import Doc

from utils import is_cjk

# UD UPOS tags kept as "content words" — mirrors nlp/english.py::CONTENT_POS.
CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

# Word-formation dependency relations: a token attached to its head by one of
# these forms part of a single multi-character word and is merged into one token
# (e.g. 天 --compound--> 下  =>  天下). Coordination (conj), modification (nmod),
# etc. are intentionally excluded, so 仁義 (義 --conj--> 仁) stays two tokens.
#
# Note: the UD Classical Chinese model is not fully consistent — it labels some
# bisyllabic words (天子, 諸侯) as `nmod`, which this set does NOT merge. Broaden
# with care (adding `nmod` over-merges genuine modifiers); prefer a curated
# lexicon for known compounds the parser mislabels.
MERGE_DEPS = {"compound", "flat", "fixed"}

# Function words / high-frequency particles dropped from the co-occurrence
# vocabulary. Most are now redundant with the POS filter above (they tag as
# PART/ADP/PRON), but several are content-POS verbs/pronouns (有, 為, 曰, 謂, 得,
# 能 ...) that we still want to exclude. Applied downstream in main.py.
STOPWORDS: set[str] = {
    "之", "也", "乎", "矣", "焉", "哉", "邪", "耳", "已",
    "而", "則", "以", "且", "雖", "若", "如", "猶", "亦", "故", "乃", "夫",
    "我", "吾", "汝", "其", "此", "彼", "是",
    "有", "無", "為", "曰", "謂", "不", "非", "所", "者", "於", "豈",
    "然", "得", "能", "可", "將", "及", "皆", "未", "與",
}

PUNCTUATION: set[str] = {
    "、", "。", "《", "》", "「", "」", "『", "』", "！", "，", "：", "；", "？"
}


def strip_punct(text: str) -> str:
    return "".join([char for char in text if char not in PUNCTUATION])


_nlp = None


def get_nlp():
    """Lazily load and memoize the SuPar-Kanbun (spaCy) classical-Chinese pipeline."""
    global _nlp
    if _nlp is None:
        import suparkanbun

        _nlp = suparkanbun.load(Danku=True)
    return _nlp


def _merge_word_formation(doc: Doc) -> Doc:
    """Merge tokens joined to their head by a word-formation relation into one token.

    Groups tokens connected (in either direction) by a ``MERGE_DEPS`` relation via
    union-find, then retokenizes each contiguous group into a single token. The
    merged token inherits the span's syntactic-root attributes (POS, dep, head), so
    e.g. 天下 becomes one NOUN token. Non-contiguous groups are left untouched.
    """
    n = len(doc)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for token in doc:
        if token.dep_ in MERGE_DEPS and token.head.i != token.i:
            union(token.i, token.head.i)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    with doc.retokenize() as retok:
        for idxs in groups.values():
            lo, hi = min(idxs), max(idxs)
            # Only merge a contiguous run with no non-member tokens interleaved.
            if len(idxs) > 1 and len(idxs) == hi - lo + 1:
                retok.merge(doc[lo:hi + 1])
    return doc


def tag_segment(segment: str) -> Doc:
    """Tag one classical-Chinese segment and merge word-formation compounds.

    Returns the retokenized spaCy ``Doc`` (used by both the co-occurrence
    tokenizer below and the diagnostic segpos dump).
    """
    nlp = get_nlp()
    return _merge_word_formation(nlp(segment))


def tokenize_classical_chinese(text: str) -> list[list[str]]:
    """Per-segment content-word token lists via SuPar-Kanbun (spaCy, UD POS).

    Word-formation compounds (e.g. 天下) are merged into a single token before filtering,
    so co-occurrence is computed over whole words. Each surviving token's surface
    form (``token.text``, traditional characters as in the source) is kept when it
    is all-CJK and its UD ``pos_`` is a content tag.
    """

    doc = tag_segment(strip_punct(text))
    sents = list(doc.sents)
    return [
        [
            token.text
            for token in sent
            if not token.is_punct
            and is_cjk(token.text)
            and token.pos_ in CONTENT_POS
        ]
        for sent in sents
    ]
