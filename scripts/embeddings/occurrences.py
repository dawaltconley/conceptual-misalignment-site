"""Unified occurrence extraction for BOTH corpora — one code path over spaCy Docs.

Consolidates ``occurrences.py`` (Chinese, Xunzi JSONL passages) and
``sep_occurrences.py`` (English, SEP spaCy docs) now that both corpora arrive as
spaCy ``Doc``s (English via ``en_core_web_sm``; Chinese via the CoNLL-U loader in
``corpus/conllu.py`` today, suparkanbun later). Because the Doc *is* the parse,
there is no per-corpus tokenizer, sentence splitter, or char-offset math left:

- sentence splitting  → ``doc.sents``
- content filtering    → ``token`` attributes (``is_alpha``/``is_stop``/``pos_``)
- target matching      → the caller's ``match_fn`` (glob families for English,
                         exact hanzi for Chinese — both ``(lemma, pos) -> label?``)
- char spans           → ``token.idx`` (a segment's text is a slice of ``doc.text``,
                         so offsets are exact and original spacing is preserved)

The one primitive that differs per token — "is this a word we keep, and under
what key?" — is :func:`content_key`; ``build_vocab`` and ``build_segments`` both
run off it. The only real per-corpus knob is ``content_pos`` (English restricts to
NOUN/VERB/ADJ/PROPN; classical Chinese may want ``None`` = keep all POS, matching
the old CJK-non-stopword behaviour) and the ``stopwords`` set (English uses spaCy's
``is_stop``; the blank-vocab Chinese Doc has none, so pass the classical set).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, NamedTuple
from models import Source

from spacy.tokens import Doc, Token


class SourceDoc(NamedTuple):
    """A source unit: a stable id + its parsed Doc. (``corpus.conllu.ChapterDoc``
    and any ``(id, doc)`` wrapper for a SEP article both satisfy this.)"""

    source: Source
    doc: Doc


class MatchFn(Protocol):
    """Map a token's ``(lemma, pos)`` to a canonical target label, or ``None``."""

    def __call__(self, lemma: str, pos: str | None = None) -> str | None:
        ...


@dataclass(frozen=True)
class Span:
    """One occurrence of a word of interest — char offsets into its segment."""

    # the canonical KEY (family label or lemma), not necessarily the surface
    word: str
    start: int
    end: int


@dataclass
class Segment:
    """A model input unit: contiguous sentences from one source, under the cap."""

    text: str
    source_id: str
    spans: list[Span] = field(default_factory=list)


def content_key(
    token: Token,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = set(),
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> str | None:
    """The one shared decision: this token's node key, or ``None`` to drop it.

    A target family member re-keys to its canonical label (so ``humane`` pools
    under ``humaneness``); otherwise a content word keys by its lemma; otherwise
    it is skipped (stopword, punctuation, non-content POS).
    """
    lemma = token.lemma_.lower()
    label = match_fn(lemma, token.pos_) if match_fn else None
    if label is not None:
        return label
    is_content = (
        token.is_alpha
        and not token.is_punct
        and not token.is_space
        and not token.is_stop
        and lemma not in stopwords
        and (content_pos is None or token.pos_ in content_pos)
    )
    return lemma if is_content else None


def content_frequencies(
    sources: Iterable[SourceDoc],
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = set(),
    stopwords: frozenset[str] | set[str] = frozenset(),
    alias: Mapping[str, str] | None = None,
) -> Counter[str]:
    """Corpus occurrence count per content key.

    ``alias`` re-keys each hit to its surviving label, for recounting after a
    variant merge (see ``embeddings.families``). Occurrences sum exactly under a
    merge, unlike document frequencies, but going through the same path keeps the
    two counts consistent.
    """
    freq: Counter[str] = Counter()
    for source in sources:
        for token in source.doc:
            key = content_key(token, match_fn, content_pos, stopwords)
            if key is not None:
                freq[alias.get(key, key) if alias else key] += 1
    return freq


def matched_lemmas(
    doc: Doc, label: str, match_fn: MatchFn | None = None,
) -> Counter[str]:
    """Which lemmas ``label`` absorbed in ``doc``, and how often each.

    A target label is an abstraction over a word family: ``Rendering('wisdom',
    'wisdom*', 'wise*')`` re-keys ``wise`` and ``wisely`` to ``wisdom`` in
    :func:`content_key`, which returns the label and drops the lemma. This is the
    same decision, kept: the counter's total is the label's occurrence count, and
    its keys are the words that produced it.

    Deliberately *unfiltered* by ``content_pos``/``stopwords``, because
    ``content_key`` tests ``match_fn`` before the content filter — a matched
    target token is kept whatever its POS. Without ``match_fn`` (Chinese, where a
    hanzi is a target because its lemma *is* the label) the rule is exact-lemma,
    so the counter has at most the one key.

    One family member yields nothing to report: a multi-word rendering is merged
    into a single token whose LEMMA is the label itself (``corpus.parse.
    merge_phrases``), so it counts under the label and contributes no variant.
    """
    if match_fn is None:
        return Counter(t.lemma_ for t in doc if t.lemma_ == label)
    return Counter(
        lemma for t in doc
        if match_fn(lemma := t.lemma_.lower(), t.pos_) == label)


def variant_list(counts: Counter[str], label: str) -> list[str]:
    """The exported shape of :func:`matched_lemmas`: the other words the label
    stands for, never including the label itself.

    Nothing is filtered: the list is exactly what was counted, so it inherits the
    corpus's tokenization noise along with its words. SEP footnote markers glue
    onto the word they follow, and ``knowledge*`` then matches lemmas like
    ``knowledge.[24`` — ~1% of a term's occurrences, but each one distinct, so in
    an alphabetical list they would outnumber the real variants.

    Hence the order: most frequent first, well-formed words ahead of debris at
    equal counts, alphabetical last. Deterministic for a fixed corpus (the JSON
    still diffs cleanly), and it puts the words actually carrying the node first
    while leaving the artifacts visible, where they belong — they are a fact about
    the corpus, and they are being counted and embedded whether or not they are
    printed.
    """
    def is_word(w: str) -> bool:
        """A real word form, not the label with punctuation stuck to it."""
        return w.replace("-", "").replace("'", "").isalpha()

    return [word for word, _ in
            sorted(((w, n) for w, n in counts.items() if w != label),
                   key=lambda item: (-item[1], not is_word(item[0]), item[0]))]


def dominant_pos(
    sources: Iterable[SourceDoc],
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = set(),
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> dict[str, str]:
    """The most frequent part of speech for each content key.

    Type-level, because a vocabulary entry *is* a type: ``study`` may be tagged
    ``NOUN`` in some occurrences and ``VERB`` in others, and a merge needs one
    answer for the whole key rather than whichever token happened to be sampled.
    Ties break toward the POS seen first, which is stable for a fixed corpus.

    Used to prefer the noun when naming a merged family (see
    ``embeddings.families.merge_map``).
    """
    tally: dict[str, Counter[str]] = {}
    for source in sources:
        for token in source.doc:
            key = content_key(token, match_fn, content_pos, stopwords)
            if key is not None:
                tally.setdefault(key, Counter())[token.pos_] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in tally.items()}


def build_vocab(
    sources: Iterable[SourceDoc],
    min_freq: int,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = set(),
    stopwords: frozenset[str] | set[str] = frozenset(),
    alias: Mapping[str, str] | None = None,
) -> set[str]:
    """Content-word vocabulary: keys occurring >= ``min_freq``.

    ``alias`` re-keys each hit to its surviving label *before* the floor is
    applied, which is the only correct order under a variant merge: ``inspire``
    (6) and ``inspiration`` (7) each fail a floor of 10 that their merged node
    (13) clears, so filtering first would make the merge lose nodes instead of
    combining them.
    """
    freq = content_frequencies(
        sources, match_fn, content_pos, stopwords, alias)
    return {k for k, c in freq.items() if c >= min_freq}


def document_frequencies(
    sources: Iterable[SourceDoc],
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = set(),
    stopwords: frozenset[str] | set[str] = frozenset(),
    alias: Mapping[str, str] | None = None,
) -> Counter[str]:
    """Document frequency per content key: how many distinct sources (documents)
    each key appears in (each source counted once, regardless of within-doc count).

    ``alias`` re-keys each hit to its surviving label. Document frequency must be
    *recomputed* this way after a variant merge rather than summed — one article
    can contain both ``inspire`` and ``inspiration``, and summing would count it
    twice.
    """
    df: Counter[str] = Counter()
    for source in sources:
        keys = {
            alias.get(key, key) if alias else key
            for token in source.doc
            if (key := content_key(token, match_fn, content_pos, stopwords))
            is not None
        }
        df.update(keys)
    return df


def build_segments(
    sources: Sequence[SourceDoc],
    words_of_interest: set[str],
    *,
    sent_len_fn: Callable[[list[str]], list[int]],
    max_tokens: int,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = set(),
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> list[Segment]:
    """Greedily pack each source's sentences into <=cap segments; record spans.

    A segment never crosses a source boundary. Its text is a *slice of the source
    Doc's text* (contiguous sentences), so character offsets and original spacing
    carry over verbatim — no separator to guess, no cumulative-offset math.
    ``sent_len_fn`` maps sentence texts to model token counts (special tokens
    excluded); the budget reserves 2 slots for ``[CLS]``/``[SEP]``.
    """
    per_source = [(src, list(src.doc.sents)) for src in sources]
    lens = sent_len_fn([s.text for _, sents in per_source for s in sents])
    budget = max_tokens - 2

    def emit(src: SourceDoc, batch: list) -> None:
        if batch:
            segments.append(_segment(src, batch, match_fn, content_pos,
                                     stopwords, words_of_interest))

    segments: list[Segment] = []
    k = 0
    for src, sents in per_source:
        batch: list = []  # accumulated sentences for the current segment
        cur_len = 0
        for sent in sents:
            slen = lens[k]
            k += 1
            if batch and cur_len + slen > budget:
                emit(src, batch)
                batch = []
                cur_len = 0
            batch.append(sent)
            cur_len += slen
        emit(src, batch)
    return segments


def _segment(
    src: SourceDoc,
    sents: list,
    match_fn: MatchFn | None,
    content_pos: set[str] | None,
    stopwords: frozenset[str] | set[str],
    words_of_interest: set[str],
) -> Segment:
    """Build one Segment from contiguous sentences of ``src.doc``."""
    start = sents[0].start_char
    end = sents[-1].end_char
    text = src.doc.text[start:end]
    spans: list[Span] = []
    for sent in sents:
        for token in sent:
            key = content_key(token, match_fn, content_pos, stopwords)
            if key is not None and key in words_of_interest:
                a = token.idx - start
                spans.append(Span(key, a, a + len(token.text)))
    return Segment(text=text, source_id=src.source.id, spans=spans)
