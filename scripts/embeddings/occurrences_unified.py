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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from spacy.tokens import Doc, Token

CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}


class SourceDoc(Protocol):
    """A source unit: a stable id + its parsed Doc. (``corpus.conllu.ChapterDoc``
    and any ``(id, doc)`` wrapper for a SEP article both satisfy this.)"""

    id: str
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
    match_fn: MatchFn,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> str | None:
    """The one shared decision: this token's node key, or ``None`` to drop it.

    A target family member re-keys to its canonical label (so ``humane`` pools
    under ``humaneness``); otherwise a content word keys by its lemma; otherwise
    it is skipped (stopword, punctuation, non-content POS).
    """
    lemma = token.lemma_.lower()
    label = match_fn(lemma, token.pos_)
    if label is not None:
        return label
    is_content = (
        token.is_alpha
        and not token.is_punct
        and not token.is_space
        and not token.is_stop
        and token.text not in stopwords
        and (content_pos is None or token.pos_ in content_pos)
    )
    return lemma if is_content else None


def build_vocab(
    sources: Iterable[SourceDoc],
    match_fn: MatchFn,
    targets: Iterable[str],
    min_freq: int,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> set[str]:
    """Content-word vocabulary: keys occurring >= ``min_freq``, plus the targets."""
    freq: Counter[str] = Counter()
    for source in sources:
        for token in source.doc:
            key = content_key(token, match_fn, content_pos, stopwords)
            if key is not None:
                freq[key] += 1
    vocab = {k for k, c in freq.items() if c >= min_freq}
    vocab |= set(targets)
    return vocab


def build_segments(
    sources: Sequence[SourceDoc],
    words_of_interest: set[str],
    match_fn: MatchFn,
    sent_len_fn: Callable[[list[str]], list[int]],
    max_tokens: int,
    content_pos: set[str] | None = CONTENT_POS,
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
    match_fn: MatchFn,
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
    return Segment(text=text, source_id=src.id, spans=spans)
