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


class ChapterDoc(NamedTuple):
    id: str      # the `# newdoc id`, e.g. "KR1h0001_001"
    title: str   # the chapter's `_title` sentence text, e.g. "梁惠王上"
    doc: Doc     # one chapter: sentences marked; token.pos_/lemma_/dep_/head set


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
) -> Iterator[ChapterDoc]:
    """Yield one :class:`ChapterDoc` per ``# newdoc``.

    ``skip_titles`` drops each chapter's ``_title`` sentence from the body (its
    text is still returned as ``ChapterDoc.title``); set False to keep it inline.
    """
    path = Path(path)
    vocab = vocab or spacy.blank("xx").vocab  # bare container; no language model

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
