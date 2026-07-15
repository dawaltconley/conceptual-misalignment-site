"""Extract term / vocabulary occurrences (with char spans) from the segpos corpus.

The corpus is word-segmented and, verified across all 2890 lines,
``"".join(tokens) == sentence``. That lets us compute each token's character
span by simple cumulative offsets, which then map exactly onto the HuggingFace
tokenizer's ``offset_mapping`` (also character offsets into the raw sentence).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from utils import is_cjk
from nlp.chinese import STOPWORDS

DEFAULT_CORPUS = Path("../segpos/full-seg-2/mengzi.segpos.jsonl")


@dataclass(frozen=True)
class Span:
    """One occurrence of a word of interest inside a sentence."""

    word: str
    start: int  # char offset into `sentence`
    end: int


@dataclass
class SentenceSpans:
    """A sentence plus every word-of-interest occurrence found in it."""

    sent_id: int
    sentence: str
    spans: list[Span]


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


def find_occurrences(
    sentences: list[dict],
    words_of_interest: set[str],
) -> list[SentenceSpans]:
    """For each sentence, locate every occurrence of a word of interest.

    A word only counts when it is a *standalone segmented token* (``token ==
    word``), which excludes compound / proper-name uses (e.g. 仁 inside a name).
    Sentences with no matches are dropped.
    """
    results: list[SentenceSpans] = []
    for sent in sentences:
        offset = 0
        spans: list[Span] = []
        for tok in sent["tokens"]:
            if tok in words_of_interest:
                spans.append(Span(tok, offset, offset + len(tok)))
            offset += len(tok)
        if spans:
            results.append(SentenceSpans(sent["id"], sent["sentence"], spans))
    return results
