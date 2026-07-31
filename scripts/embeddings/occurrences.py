"""Load per-chapter passage segmentation and pack it into embedding segments.

Input: one JSONL file per chapter (from ``xunzi/run.py --unit line``), each line
a passage record ``{chapter, id, passage, tokens}`` where ``tokens`` is the
Xunzi word-segmentation of the *whole passage*. Because Xunzi tags passages as
units, sentence splitting happens here, not upstream.

Packing respects source boundaries: sentences are packed greedily into
``Segment``s that stay under the model's token cap, and a segment never spans two
passages (hence never two chapters). A passage that fits under the cap becomes a
single segment; an over-long one splits into several at sentence boundaries.

As with the sentence corpus, ``"".join(tokens) == passage`` per record, so a
word's char span is a simple cumulative offset that maps onto the HuggingFace
tokenizer's ``offset_mapping``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from config import SEGPOS
from nlp.text import is_cjk
from nlp.chinese import STOPWORDS

DEFAULT_CORPUS_DIR = SEGPOS / "chapters"

# Sentence-final punctuation, plus closers that get pulled back onto the
# sentence they close (mirrors xunzi.run.split_sentences at token granularity).
_TERMINALS = frozenset("。！？；")
_CLOSERS = frozenset("」』）】》)")


@dataclass(frozen=True)
class Span:
    """One occurrence of a word of interest, char offsets into its segment."""

    word: str
    start: int
    end: int


@dataclass
class Passage:
    """A source passage: one chapter line, segmented as a whole by Xunzi."""

    chapter: str
    pid: int
    text: str
    tokens: list[str]


@dataclass
class Segment:
    """A model input unit: sentences packed within one passage, under the cap."""

    text: str
    chapter: str
    pid: int
    spans: list[Span] = field(default_factory=list)


def load_passages(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Passage]:
    """Read every ``*.jsonl`` chapter file (book order) into passages."""
    passages: list[Passage] = []
    for path in sorted(corpus_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                passages.append(
                    Passage(d["chapter"], d["id"], d["passage"], d["tokens"])
                )
    return passages


def build_vocab(
    passages: list[Passage],
    targets: set[str],
    min_freq: int,
) -> set[str]:
    """Content-word vocabulary: frequent CJK non-stopword tokens + the targets."""
    freq = Counter(
        tok
        for p in passages
        for tok in p.tokens
        if is_cjk(tok) and tok not in STOPWORDS
    )
    vocab = {tok for tok, c in freq.items() if c >= min_freq}
    vocab |= targets
    return vocab


def split_sentence_tokens(tokens: list[str]) -> list[list[str]]:
    """Split a passage's token list into sentence token-lists.

    A sentence ends at a run of terminal punctuation, absorbing any immediately
    following closing quotes/brackets. Trailing text with no terminal
    punctuation forms a final sentence.
    """
    sentences: list[list[str]] = []
    cur: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        cur.append(tokens[i])
        if tokens[i] in _TERMINALS:
            while i + 1 < n and tokens[i + 1] in _CLOSERS:
                i += 1
                cur.append(tokens[i])
            sentences.append(cur)
            cur = []
        i += 1
    if cur:
        sentences.append(cur)
    return sentences


def build_segments(
    passages: list[Passage],
    words_of_interest: set[str],
    sent_len_fn: Callable[[list[str]], list[int]],
    max_tokens: int,
) -> list[Segment]:
    """Greedily pack each passage's sentences into segments under the token cap.

    ``sent_len_fn`` maps a list of sentence texts to their model token counts
    (batched, special tokens excluded); the budget reserves 2 slots for
    ``[CLS]``/``[SEP]``. Segments never cross passage boundaries.
    """
    per_passage = [split_sentence_tokens(p.tokens) for p in passages]
    flat_texts = ["".join(s) for sents in per_passage for s in sents]
    lens = sent_len_fn(flat_texts)

    budget = max_tokens - 2
    segments: list[Segment] = []
    k = 0
    for p, sents in zip(passages, per_passage):
        cur = Segment(text="", chapter=p.chapter, pid=p.pid)
        cur_len = 0
        for sent_toks in sents:
            slen = lens[k]
            k += 1
            if cur.text and cur_len + slen > budget:
                segments.append(cur)
                cur = Segment(text="", chapter=p.chapter, pid=p.pid)
                cur_len = 0
            base = len(cur.text)
            offset = 0
            for tok in sent_toks:
                if tok in words_of_interest:
                    cur.spans.append(
                        Span(tok, base + offset, base + offset + len(tok)))
                offset += len(tok)
            cur.text += "".join(sent_toks)
            cur_len += slen
        if cur.text:
            segments.append(cur)
    return segments
