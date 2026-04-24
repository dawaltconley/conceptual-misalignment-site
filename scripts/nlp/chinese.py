from utils import is_cjk
from collections import defaultdict

STOPWORDS: set[str] = {
    "之", "也", "乎", "矣", "焉", "哉", "邪", "耳", "已",
    "而", "則", "以", "且", "雖", "若", "如", "猶", "亦", "故", "乃", "夫",
    "我", "吾", "汝", "其", "此", "彼", "是",
    "有", "無", "為", "曰", "謂", "不", "非", "所", "者", "於", "豈",
    "然", "得", "能", "可", "將", "及", "皆", "未", "與",
}


def tokenize_classical_chinese(text: str) -> list[list[str]]:
    """Per-sentence CJK token lists via CLTK's classical Chinese (lzh) model."""
    from cltk import NLP as CLTK_NLP

    cltk_nlp = CLTK_NLP(language_code="lzh")
    doc = cltk_nlp.analyze(text=text)

    sent_tokens: dict[int, list[str]] = defaultdict(list)
    for w in doc.words:
        if w.string and w.index_sentence and is_cjk(w.string):
            sent_tokens[w.index_sentence].append(w.string)

    return [sent_tokens[i] for i in sorted(sent_tokens.keys())]
