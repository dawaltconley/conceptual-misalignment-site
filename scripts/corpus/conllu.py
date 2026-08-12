"""Load a pre-annotated UD CoNLL-U treebank into spaCy ``Doc``s — no model needed.

The ``.conllu`` already carries tokens, lemmas, UPOS, heads and deprels, so there
is nothing to tokenize or parse: a bare ``Vocab`` is enough. We build one ``Doc``
per ``# newdoc`` (a Mengzi book/chapter), with sentence boundaries from the CoNLL
sentences, so the result drops into the corpus pipeline as a source unit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

from corpus.recombine import (
    MergeConfig, MergeReport, load_lexicon, merge_doc)


class ChapterDoc(NamedTuple):
    id: str      # the `# newdoc id`, e.g. "KR1h0001_001"
    title: str   # the chapter's `_title` sentence text, e.g. "梁惠王上"
    doc: Doc     # one chapter: sentences marked; token.pos_/lemma_/dep_/head set


class ConlluUnit(NamedTuple):
    """A span of consecutive treebank sentences, with where its tokens live.

    Written for feeding the treebank's own text to a segmenter: because the unit
    records ``token_start``/``n_tokens`` — indices into the same chapter Doc
    :func:`load_conllu` builds — a segmentation of ``text`` maps straight back
    onto token indices. No alignment against another edition is involved.
    """

    doc_id: str                 # the `# newdoc id`, e.g. "KR1h0001_001"
    chapter: str                # the chapter's title, e.g. "梁惠王上"
    par: int                    # paragraph ordinal within the chapter (0-based)
    sent_ids: tuple[str, ...]   # the `# sent_id`s this unit covers, in order
    token_start: int            # index of the first token within the chapter Doc
    n_tokens: int
    text: str                   # the tokens' forms, concatenated (unpunctuated)


def iter_units(
    path: str | Path,
    *,
    max_chars: int = 120,
    skip_titles: bool = True,
) -> Iterator[ConlluUnit]:
    """Yield the treebank's text as units of at most ``max_chars`` characters.

    Sentences are packed greedily and a unit never crosses a ``# newpar``
    boundary, so each one is a contiguous stretch of a single passage — the
    treebank's sentences are far too short to segment in isolation (median 5
    characters), while whole paragraphs run to 1313. A single sentence longer
    than ``max_chars`` becomes its own unit rather than being split.
    """
    path = Path(path)
    doc_id = title = ""
    par = -1
    token_i = 0                       # running token index within the chapter
    buf: list[tuple[str, str, int]] = []   # (sent_id, text, n_tokens)

    def flush() -> Iterator[ConlluUnit]:
        nonlocal buf
        if buf:
            n = sum(b[2] for b in buf)
            yield ConlluUnit(
                doc_id, title, par, tuple(b[0] for b in buf),
                token_i - n, n, "".join(b[1] for b in buf),
            )
            buf = []

    for comments, rows in _iter_sentences(path):
        if (newdoc := comments.get("newdoc id")) is not None:
            yield from flush()
            doc_id, par, token_i = newdoc, -1, 0
        if "newpar text" in comments or "newpar" in comments:
            yield from flush()
            par += 1

        sent_id = comments.get("sent_id", "")
        if sent_id.endswith("_title"):
            title = comments.get("text", "")
            if skip_titles:
                continue

        text = "".join(row[1] for row in rows)
        if buf and sum(len(b[1]) for b in buf) + len(text) > max_chars:
            yield from flush()
        buf.append((sent_id, text, len(rows)))
        token_i += len(rows)

    yield from flush()


def _iter_sentences(path: Path) -> Iterator[tuple[dict[str, str], list[list[str]]]]:
    """Yield ``(comments, rows)`` per CoNLL-U sentence (blank-line separated)."""
    comments: dict[str, str] = {}
    rows: list[list[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if rows:
                    yield comments, rows
                comments, rows = {}, []
            elif line.startswith("#"):
                key, _, val = line[1:].strip().partition("=")
                comments[key.strip()] = val.strip()
            else:
                cols = line.split("\t")
                if "-" not in cols[0] and "." not in cols[0]:  # skip MWT / empty nodes
                    rows.append(cols)
    if rows:
        yield comments, rows


def load_conllu(
    path: str | Path,
    vocab: Vocab | None = None,
    skip_titles: bool = True,
    merge: MergeConfig | None = None,
    report: MergeReport | None = None,
) -> Iterator[ChapterDoc]:
    """Yield one :class:`ChapterDoc` per ``# newdoc``.

    ``skip_titles`` drops each chapter's ``_title`` sentence from the body (its
    text is still returned as ``ChapterDoc.title``); set False to keep it inline.

    ``merge`` recombines each chapter's subword tokens into whole words before it
    is yielded (see :mod:`corpus.recombine`), so no consumer downstream has to
    know the treebank annotates one character per token. ``None`` leaves the
    tokenization exactly as the file has it. Pass ``report`` to collect what the
    merge did across every chapter.
    """
    path = Path(path)
    vocab = vocab or spacy.blank("xx").vocab  # bare container; no language model
    # Keyed by chapter title, matching `ChapterDoc.title`. Empty when no lexicon
    # is configured or the file is absent — the segmentation is a manual step.
    lexicon = load_lexicon(merge.lexicon_path) if merge is not None else {}

    doc_id = title = ""
    cols: dict[str, list] = {}

    def reset() -> None:
        cols.clear()
        for k in ("words", "spaces", "lemmas", "pos", "tags",
                  "morphs", "heads", "deps", "sent_starts"):
            cols[k] = []

    def flush() -> ChapterDoc | None:
        if not cols.get("words"):
            return None
        doc = Doc(
            vocab,
            words=cols["words"], spaces=cols["spaces"], lemmas=cols["lemmas"],
            pos=cols["pos"], tags=cols["tags"], morphs=cols["morphs"],
            heads=cols["heads"], deps=cols["deps"], sent_starts=cols["sent_starts"],
        )
        if merge is not None:
            doc = merge_doc(doc, merge, report, lexicon.get(title, ()))
        return ChapterDoc(doc_id, title, doc)

    reset()
    for comments, rows in _iter_sentences(path):
        newdoc = comments.get("newdoc id")
        if newdoc is not None:
            out = flush()
            if out is not None:
                yield out
            reset()
            doc_id, title = newdoc, ""

        is_title = comments.get("sent_id", "").endswith("_title")
        if is_title:
            title = comments.get("text", "")
            if skip_titles:
                continue

        base = len(cols["words"])  # this sentence's offset in the growing chapter Doc
        for i, row in enumerate(rows):
            _, form, lemma, upos, xpos, feats, head, deprel, _, misc = row
            cols["words"].append(form)
            cols["lemmas"].append(form if lemma == "_" else lemma)
            cols["pos"].append("" if upos == "_" else upos)
            cols["tags"].append("" if xpos == "_" else xpos)
            cols["morphs"].append("" if feats == "_" else feats)
            cols["deps"].append("dep" if deprel == "_" else deprel)
            # CoNLL head is 1-indexed within the sentence (0 = root); spaCy wants
            # the head's absolute index in the Doc, with root pointing to itself.
            h = 0 if head in ("_", "0") else int(head)
            cols["heads"].append(base + i if h == 0 else base + h - 1)
            cols["spaces"].append("SpaceAfter=No" not in misc)
            cols["sent_starts"].append(i == 0)

    out = flush()
    if out is not None:
        yield out
