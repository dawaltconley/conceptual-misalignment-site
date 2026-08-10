from typing import TYPE_CHECKING
from config import ENGLISH_LEMMAS, MENGZI_CONLLU

if TYPE_CHECKING:
    from spacy.tokens import Doc
    from corpus.sep import SEP
    from corpus.mengzi import Chapter as MengziChapter

_nlp = None


def _get_en_nlp():
    import spacy
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def verb_lemma(word: str) -> str | None:
    """The verb lemma of ``word`` per spaCy's own rules, or ``None``.

    Used to link participial vocabulary entries back to their base verb
    (``devoted`` -> ``devote``). spaCy tags participial adjectives ``JJ``, so
    asking the pipeline directly returns the inflected form; this forces the
    ``VERB`` reading and re-runs the rule lemmatizer over the same tables.
    """
    nlp = _get_en_nlp()
    doc = nlp(word)
    if len(doc) != 1:
        return None
    token = doc[0]
    token.pos_ = "VERB"
    lemmas = nlp.get_pipe("lemmatizer").rule_lemmatize(token)
    return lemmas[0].lower() if lemmas else None


def _apply_lemma_exceptions(doc: "Doc") -> "Doc":
    """Patch en_core_web_sm's lemma mistakes in place, keyed on surface form.

    Two error classes dominate the SEP corpus (both measured, see
    ``scripts/lemmas/english.conf``): discipline names in ``-ics`` lose their
    final s and collapse into an unrelated adjective (``ethics``/``aesthetics``
    /``semantics`` -> ``ethic``/``aesthetic``/``semantic``), and a handful of
    words get a back-formed base that is not English at all (``species`` ->
    ``specie``, ``senses`` -> ``sens``). Fixing it here rather than in
    ``content_key`` means the embedding and co-occurrence lenses both get the
    correction, and nothing has to thread a table through six call sites.
    """
    for token in doc:
        lemma = ENGLISH_LEMMAS.get(token.text.lower())
        if lemma is not None:
            token.lemma_ = lemma
    return doc


def parse_sep_article(sep: "SEP") -> "Doc":
    nlp = _get_en_nlp()
    return _apply_lemma_exceptions(nlp(sep.text))


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
