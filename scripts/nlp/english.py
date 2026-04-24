CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

_nlp = None


def _get_nlp():
    import spacy
    from spacy_html_tokenizer import create_html_tokenizer
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        _nlp.tokenizer = create_html_tokenizer()(_nlp)
    return _nlp


def tokenize_english_html(html: str) -> list[list[str]]:
    """Per-sentence lemma lists from an HTML string via spaCy (content words only)."""
    nlp = _get_nlp()
    doc = nlp(html)
    return [
        [
            token.lemma_.lower()
            for token in sent
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.is_alpha
            and token.pos_ in CONTENT_POS
        ]
        for sent in doc.sents
    ]
