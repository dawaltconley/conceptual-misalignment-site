from typing import TYPE_CHECKING
from config import MENGZI_CONLLU

if TYPE_CHECKING:
    from spacy.tokens import Doc
    from corpus.sep import SEP
    from corpus.mengzi import Chapter as MengziChapter

_nlp = None


def _get_en_nlp():
    import spacy
    from spacy_html_tokenizer import create_html_tokenizer
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        _nlp.tokenizer = create_html_tokenizer()(_nlp)
    return _nlp


_sep_article_docs: dict[str, "Doc"] = {}


def parse_sep_article(sep: "SEP") -> "Doc":
    global _sep_article_docs
    if sep.id in _sep_article_docs:
        return _sep_article_docs[sep.id]
    nlp = _get_en_nlp()
    _sep_article_docs[sep.id] = nlp(sep.text)
    return _sep_article_docs[sep.id]


_mengzi_chapter_docs: dict[str, "Doc"] = {}


def parse_mengzi_chapter(chapter: "MengziChapter") -> "Doc":
    """Doesn't actually parse the `.text` attribute in the chapter; parses the contents of a conllu file.
    This makes it possible to swap in a real spaCy-compatible parser (suparkanbun) later without changing the API."""
    from corpus.conllu import load_conllu
    global _mengzi_chapter_docs
    if chapter.title not in _mengzi_chapter_docs:
        conllu = load_conllu(MENGZI_CONLLU)
        for parsed in conllu:
            _mengzi_chapter_docs[parsed.title] = parsed.doc
    if chapter.title not in _mengzi_chapter_docs:
        raise KeyError(f"Couldn't load chapter from conllu: {chapter.title}")
    return _mengzi_chapter_docs[chapter.title]
