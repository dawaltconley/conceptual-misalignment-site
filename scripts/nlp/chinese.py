from utils import is_cjk
from nlp.sentences import split_sentences

# UD UPOS tags kept as "content words" — mirrors nlp/english.py::CONTENT_POS.
CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

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

_nlp = None


def get_nlp():
    """Lazily load and memoize the SuPar-Kanbun (spaCy) classical-Chinese pipeline."""
    global _nlp
    if _nlp is None:
        import suparkanbun

        _nlp = suparkanbun.load()
    return _nlp


def tokenize_classical_chinese(text: str) -> list[list[str]]:
    """Per-sentence content-word token lists via SuPar-Kanbun (spaCy, UD POS).

    SuPar-Kanbun treats its whole input string as a single sentence, so we split
    on sentence-final punctuation first and tag each sentence independently. Each
    token's surface form (``token.text``, traditional characters as they appear in
    the source) is kept when it is all-CJK and its UD ``pos_`` is a content tag.
    """
    nlp = get_nlp()
    sent_token_lists: list[list[str]] = []
    for sentence in split_sentences(text):
        doc = nlp(sentence)
        tokens = [
            token.text
            for token in doc
            if is_cjk(token.text) and token.pos_ in CONTENT_POS
        ]
        if tokens:
            sent_token_lists.append(tokens)
    return sent_token_lists
