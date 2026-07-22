"""Fetch and prepare the SEP English corpus for the roberta-base semantic space.

The English analogue of ``occurrences.py``. Where the Chinese side loads
pre-segmented passages, here we pull Stanford Encyclopedia of Philosophy
articles for each English target term (via ``scrape_sep.search_sep``), strip the
HTML, and use spaCy for sentence splitting + lemmatization. Occurrences are keyed
by **lemma** (so ``virtues`` and ``virtue`` pool into one node), and each span is
the surface token's character range within its segment.

Sentences are packed greedily into ``Segment``s under roberta-base's token cap
without crossing article boundaries, mirroring the Chinese ``build_segments`` so
the downstream ``model.embed`` / ``vectors`` / ``analyze`` code is reused as-is.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from bs4 import BeautifulSoup

from config import Rendering
from embed.occurrences import Segment, Span
from inpho import is_chinese_philosophy
from scrape_sep import search_sep

# A matcher maps a (lemma, POS) to a canonical target label, or None. In practice
# this is ``config.match_rendering`` (glob-based term families).


class MatchFn(Protocol):
    def __call__(self, lemma: str, pos: str | None = None) -> str | None:
        ...


CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}


@dataclass
class Doc:
    """One cleaned SEP article: plain text plus provenance."""

    doc_id: str          # article URL (stable id, also the boundary key)
    title: str
    text: str


@dataclass
class ParsedSentence:
    """A sentence's text plus its content-word / target tokens as lemma+span."""

    doc_id: str
    text: str
    # (lemma, start, end) char offsets *within this sentence's text*.
    toks: list[tuple[str, int, int]]


def _clean_html(html: str) -> str:
    """Strip tags and collapse whitespace; spaCy re-segments the plain text."""
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())


def fetch_corpus(
    terms: list[Rendering],
    per_term: int = 12,
) -> list[Doc]:
    """Search SEP for each term, dedupe articles by URL, and clean the HTML.

    ``terms`` are the search queries (the rendering labels). Non-Chinese-
    philosophy articles only (matches ``main.py``) so the terms carry their
    Western philosophical sense, the target of the thick/thin contrast.
    """
    docs: dict[str, Doc] = {}
    for term in terms:
        articles = search_sep(
            term, per_term,
            pre_filter=lambda url: not is_chinese_philosophy(url),
        )
        for a in articles:
            if a.url not in docs:  # first term to surface an article keeps it
                docs[a.url] = Doc(a.url, a.title, _clean_html(a.text))
    return list(docs.values())


def parse_docs(docs: list[Doc], nlp, match_fn: MatchFn) -> list[ParsedSentence]:
    """Run spaCy over each doc; collect content-word + target tokens per sentence.

    Each token is keyed by ``match_fn(lemma, pos)`` (its canonical target label,
    e.g. ``humane`` -> ``humaneness``) if it belongs to a term family; otherwise
    by its lemma if it is a content word (alpha, non-stop, POS in ``CONTENT_POS``).
    Re-keying family members here means they pool + count under the label
    downstream with no further changes.
    """
    sentences: list[ParsedSentence] = []
    for doc in docs:
        parsed = nlp(doc.text)
        for sent in parsed.sents:
            base = sent.start_char
            toks: list[tuple[str, int, int]] = []
            for t in sent:
                lemma = t.lemma_.lower()
                label = match_fn(lemma, t.pos_)
                is_content = (
                    t.is_alpha and not t.is_stop and not t.is_punct
                    and not t.is_space and t.pos_ in CONTENT_POS
                )
                if label:
                    key = label
                elif is_content:
                    key = lemma
                else:
                    continue
                toks.append((key, t.idx - base, t.idx - base + len(t.text)))
            if toks:
                sentences.append(ParsedSentence(doc.doc_id, sent.text, toks))
    return sentences


def build_vocab(
    sentences: list[ParsedSentence],
    targets: frozenset[str] | set[str],
    min_freq: int,
) -> set[str]:
    """Content-word vocabulary: frequent lemmas + the targets (always kept)."""
    freq = Counter(lemma for s in sentences for lemma, _, _ in s.toks)
    vocab = {w for w, c in freq.items() if c >= min_freq}
    vocab |= set(targets)
    return vocab


def build_segments(
    sentences: list[ParsedSentence],
    words_of_interest: set[str],
    sent_len_fn: Callable[[list[str]], list[int]],
    max_tokens: int,
) -> list[Segment]:
    """Greedily pack sentences into <=cap segments that never cross articles.

    Spans are recorded for any token whose lemma is in ``words_of_interest``,
    offset to its position in the packed segment text. Mirrors the Chinese
    ``occurrences.build_segments`` (sentences are joined with a single space).
    """
    lens = sent_len_fn([s.text for s in sentences])
    budget = max_tokens - 2  # reserve <s> / </s>

    segments: list[Segment] = []
    cur: Segment | None = None
    cur_len = 0
    cur_doc: str | None = None

    def flush() -> None:
        nonlocal cur, cur_len
        if cur is not None and cur.text:
            segments.append(cur)
        cur = None
        cur_len = 0

    for s, slen in zip(sentences, lens):
        if cur is None or s.doc_id != cur_doc or (cur.text and cur_len + slen > budget):
            flush()
            cur = Segment(text="", chapter=s.doc_id, pid=len(segments))
            cur_len = 0
            cur_doc = s.doc_id
        sep = " " if cur.text else ""
        base = len(cur.text) + len(sep)
        for lemma, a, b in s.toks:
            if lemma in words_of_interest:
                cur.spans.append(Span(lemma, base + a, base + b))
        cur.text += sep + s.text
        cur_len += slen
    flush()
    return segments
