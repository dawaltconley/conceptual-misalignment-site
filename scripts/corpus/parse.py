from fnmatch import translate
from typing import TYPE_CHECKING, Sequence
from config import ENGLISH_LEMMAS, MENGZI_CONLLU, MERGE_OVERRIDES, TERMS
from corpus.recombine import MergeConfig, MergeReport, load_overrides

if TYPE_CHECKING:
    from spacy.matcher import Matcher
    from spacy.tokens import Doc
    from models import Pipeline
    from corpus.sep import SEP
    from corpus.mengzi import Chapter as MengziChapter
    from models import Rendering

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


# ---------------------------------------------------------------------------
# Multi-word renderings → one token
# ---------------------------------------------------------------------------
# A ``Rendering`` is matched against *one token's lemma*
# (``models.Rendering.matches``), so a pattern spanning more than one token can
# never match, however the text is written: ``social norm*`` for 禮 and
# ``human nature`` for 性 are dead on arrival, and so is ``heart-mind*`` for 心,
# because the English tokenizer splits hyphens. Unlike the lemma errors in
# ``lemmas/english.conf``, no exception table reaches this — the fix has to
# change what counts as a token.
#
# So: find the phrases with a ``Matcher`` and merge each one into a single token
# whose LEMMA is the rendering's label. Downstream nothing changes — the merged
# token matches the label pattern like any other lemma, carries the phrase as its
# surface ``form`` for display, and keeps exact ``token.idx`` offsets (merging
# does not alter ``doc.text``), so ``build_segments`` and the embedder see the
# whole phrase as one span, which is what you want embedded anyway.


def _token_spec(piece: str, attr: str) -> dict:
    """One Matcher token spec, as a glob if ``piece`` has wildcards."""
    if any(c in piece for c in "*?["):
        # fnmatch's regex is right-anchored (``\Z``) but not left-anchored, and
        # spaCy's REGEX predicate uses re.search — so anchor it explicitly, or
        # ``norm*`` would also match "abnormal".
        return {attr: {"REGEX": "^" + translate(piece)}}
    return {attr: piece}


def _token_specs(pattern: str, tokenizer, attr: str) -> list[dict] | None:
    """Matcher specs for ``pattern``, or ``None`` if it fits in one token.

    Each whitespace-separated chunk is run through the *real* tokenizer, so a
    pattern is caught whenever the tokenizer would split it — hyphens included —
    not only when it contains a space. A trailing wildcard stays attached to the
    last piece (``heart-mind*`` -> ``heart`` ``-`` ``mind*``).
    """
    specs: list[dict] = []
    for chunk in pattern.split():
        stem = chunk.rstrip("*")
        glob = chunk[len(stem):]
        pieces = [t.text for t in tokenizer(stem)] or [stem]
        specs += [_token_spec(p + (glob if i == len(pieces) - 1 else ""), attr)
                  for i, p in enumerate(pieces)]
    return specs if len(specs) > 1 else None


def spans_multiple_tokens(pattern: str) -> bool:
    """Would ``pattern`` have to cover more than one token to match?

    The one definition of "multi-token", shared with ``renderings`` so the
    coverage guard classifies a failure the same way the merger does — and by
    the real tokenizer, so ``heart-mind*`` counts even without a space in it.
    """
    return _token_specs(pattern, _get_en_nlp().tokenizer, "LEMMA") is not None


_matcher: "Matcher | None" = None


def _get_phrase_matcher(renderings: "Sequence[Rendering] | None" = None):
    """A Matcher keyed on rendering label, holding every multi-token pattern.

    Two variants per pattern — one over LEMMA, one over LOWER — because the
    single-token path matches lemmas but a phrase's inflection usually sits on
    its head (``social norms`` lemmatizes to ``social norm``, and both should
    hit ``social norm*``). Matching either way is strictly more forgiving than
    the single-token rule, never less.
    """
    global _matcher
    if renderings is None and _matcher is not None:
        return _matcher
    from spacy.matcher import Matcher
    nlp = _get_en_nlp()
    rends = renderings if renderings is not None else [
        r for term in TERMS for r in term.renderings]
    matcher = Matcher(nlp.vocab)
    for rendering in rends:
        variants = [specs for pattern in rendering.patterns
                    for attr in ("LEMMA", "LOWER")
                    if (specs := _token_specs(pattern, nlp.tokenizer, attr))]
        if variants:
            matcher.add(rendering.label, variants)
    if renderings is None:
        _matcher = matcher
    return matcher


def merge_phrases(doc: "Doc",
                  renderings: "Sequence[Rendering] | None" = None) -> "Doc":
    """Merge each multi-token rendering occurrence into one token, in place.

    The merged token's lemma is the rendering's label, which is always
    ``patterns[0]`` and therefore matches the rendering by construction. Every
    other attribute is inherited from the span's syntactic head, so the phrase
    keeps a sensible POS (``social norms`` NOUN, ``care for`` VERB) rather than
    one hardcoded here.
    """
    from spacy.util import filter_spans
    matcher = _get_phrase_matcher(renderings)
    if not len(matcher):
        return doc
    # A phrase never spans a sentence break; a match that does is the matcher
    # reaching across punctuation, and merging it would corrupt the sentence
    # boundaries build_segments packs on.
    label_of = {doc[start:end]: doc.vocab.strings[match_id]
                for match_id, start, end in matcher(doc)
                if not any(t.is_sent_start for t in doc[start + 1:end])}
    with doc.retokenize() as retokenizer:
        for span in filter_spans(list(label_of)):  # longest wins, no overlaps
            retokenizer.merge(span, attrs={"LEMMA": label_of[span]})
    return doc


_sep_article_docs: dict[str, "Doc"] = {}


def parse_sep_article(sep: "SEP") -> "Doc":
    global _sep_article_docs
    if sep.id in _sep_article_docs:
        return _sep_article_docs[sep.id]
    nlp = _get_en_nlp()
    parsed = merge_phrases(_apply_lemma_exceptions(nlp(sep.text)))
    _sep_article_docs[sep.id] = parsed
    return parsed


def unparse_sep_article(sep: "SEP") -> None:
    del _sep_article_docs[sep.id]


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
