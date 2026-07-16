"""Extract term / vocabulary occurrences and pack sentences into passages.

The corpus is word-segmented and, verified across all 2890 lines,
``"".join(tokens) == sentence``. That lets us compute each token's character
span by simple cumulative offsets, which then map exactly onto the HuggingFace
tokenizer's ``offset_mapping`` (also character offsets into the raw text).

Sentences are packed greedily into passages that stay under the model's token
cap, so each term is embedded *with* its neighboring-sentence context (via
self-attention across the packed sequence) rather than in isolation. Span
coordinates are re-based into passage coordinates during packing.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from utils import is_cjk
from nlp.chinese import STOPWORDS

DEFAULT_CORPUS = Path("../segpos/full-seg-2/mengzi.segpos.jsonl")


@dataclass(frozen=True)
class Span:
    """One occurrence of a word of interest, with char offsets into its passage."""

    word: str
    start: int
    end: int


@dataclass
class Passage:
    """One packed passage: concatenated sentence text + every occurrence in it."""

    text: str
    spans: list[Span] = field(default_factory=list)
    sent_ids: list[int] = field(default_factory=list)


def load_sentences(path: Path = DEFAULT_CORPUS) -> list[dict]:
    """Read the segpos JSONL into a list of ``{id, sentence, tokens}`` dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_vocab(
    sentences: list[dict],
    targets: set[str],
    min_freq: int,
) -> set[str]:
    """Content-word vocabulary for the neighborhood network.

    Keeps CJK tokens (dropping punctuation / non-CJK) that are not stopwords and
    occur at least ``min_freq`` times, plus the target terms unconditionally.
    """
    freq = Counter(
        tok
        for sent in sentences
        for tok in sent["tokens"]
        if is_cjk(tok) and tok not in STOPWORDS
    )
    vocab = {tok for tok, c in freq.items() if c >= min_freq}
    vocab |= targets
    return vocab


def sentence_spans(tokens: list[str], words_of_interest: set[str]) -> list[Span]:
    """Spans (relative to the sentence) of every standalone word of interest.

    A word only counts as an occurrence when it is a *standalone segmented
    token* (``token == word``), which excludes compound / proper-name uses.
    """
    spans: list[Span] = []
    offset = 0
    for tok in tokens:
        if tok in words_of_interest:
            spans.append(Span(tok, offset, offset + len(tok)))
        offset += len(tok)
    return spans


def build_passages(
    sentences: list[dict],
    words_of_interest: set[str],
    sent_token_lens: list[int],
    max_tokens: int,
) -> list[Passage]:
    """Greedily pack consecutive sentences into passages under the token cap.

    Every sentence (occurrence-bearing or not) is included so that packed
    context is faithful to the source text. ``sent_token_lens[i]`` is the model
    token count of ``sentences[i]`` (special tokens excluded); the budget
    reserves 2 slots for ``[CLS]``/``[SEP]``. A lone sentence longer than the
    budget becomes its own (over-cap) passage and is truncated at embed time.
    """
    budget = max_tokens - 2
    passages: list[Passage] = []
    cur = Passage(text="")
    cur_len = 0

    for sent, tlen in zip(sentences, sent_token_lens):
        if cur.text and cur_len + tlen > budget:
            passages.append(cur)
            cur, cur_len = Passage(text=""), 0
        base = len(cur.text)
        for sp in sentence_spans(sent["tokens"], words_of_interest):
            cur.spans.append(Span(sp.word, base + sp.start, base + sp.end))
        cur.text += sent["sentence"]
        cur.sent_ids.append(sent["id"])
        cur_len += tlen

    if cur.text:
        passages.append(cur)
    return passages
