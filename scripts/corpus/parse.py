from typing import TYPE_CHECKING
from config import MENGZI_CONLLU, MERGE_OVERRIDES
from corpus.recombine import MergeConfig, MergeReport, load_overrides

if TYPE_CHECKING:
    from spacy.tokens import Doc
    from models import Pipeline
    from corpus.sep import SEP
    from corpus.mengzi import Chapter as MengziChapter

_nlp = None


def _get_en_nlp():
    import spacy
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def parse_sep_article(sep: "SEP") -> "Doc":
    nlp = _get_en_nlp()
    return nlp(sep.text)


def mengzi_merge_config(
    p: "Pipeline", targets: frozenset[str]
) -> MergeConfig | None:
    """The word-recombination settings for a Mengzi run, or ``None`` when
    ``merge_deps`` is unset (leave the treebank's one-character tokens alone).

    The pipeline supplies the relations and the stopword list; ``targets`` is
    passed so a merge can never swallow a term we are measuring, and the curated
    override file — which need not exist — is layered on top."""
    if p.merge_deps is None:
        return None
    return MergeConfig(
        deps=p.merge_deps,
        stopwords=p.stopwords,
        targets=targets,
        overrides=load_overrides(MERGE_OVERRIDES),
    )


# Keyed by the merge settings as well as the chapter, since the same chapter
# tokenizes differently under a different MergeConfig and two pipelines can share
# one process.
_mengzi_chapter_docs: dict[MergeConfig | None, dict[str, "Doc"]] = {}
_mengzi_merge_reports: dict[MergeConfig | None, MergeReport] = {}


def parse_mengzi_chapter(
    chapter: "MengziChapter", merge: MergeConfig | None = None
) -> "Doc":
    """Doesn't actually parse the `.text` attribute in the chapter; parses the contents of a conllu file.
    This makes it possible to swap in a real spaCy-compatible parser (suparkanbun) later without changing the API.

    ``merge`` recombines subword tokens into whole words (天 + 下 -> 天下); see
    :mod:`corpus.recombine`. The whole file is read on the first call, so the
    report for a given config is complete once any chapter has been requested —
    :func:`mengzi_merge_report` returns it."""
    from corpus.conllu import load_conllu
    docs = _mengzi_chapter_docs.get(merge)
    if docs is None:
        docs = _mengzi_chapter_docs[merge] = {}
        report = _mengzi_merge_reports[merge] = MergeReport()
        for parsed in load_conllu(MENGZI_CONLLU, merge=merge, report=report):
            docs[parsed.title] = parsed.doc
    if chapter.title not in docs:
        raise KeyError(f"Couldn't load chapter from conllu: {chapter.title}")
    return docs[chapter.title]


def mengzi_merge_report(merge: MergeConfig | None = None) -> MergeReport | None:
    """What :func:`parse_mengzi_chapter` merged under ``merge`` — ``None`` until
    a chapter has been parsed with that config."""
    return _mengzi_merge_reports.get(merge)
