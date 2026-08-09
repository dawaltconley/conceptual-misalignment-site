"""Where a corpus run's JSON goes, and the record of what it wrote.

``CorpusWriter`` is the only code that turns a name into a path. Each ``add_*``
writes the file *and*, in the same call, records the resulting ``Source`` — the
file's provenance plus its ``data`` web path and occurrence count — into a
``CorpusIndex``. That manifest is saved as ``<out_dir>/index.json`` and is the
sole input ``main.build_master`` reads for the corpus, so a path is never
derived twice and never parsed back out of a filename.

It also means one corpus's record survives a run of the other: ``--corpus sep``
rewrites ``public/sep/index.json`` and leaves ``public/ctext/index.json`` alone,
so the Mengzi side of the master index is preserved.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from config import EMBEDDINGS, PUBLIC
from models import (
    CorpusIndex,
    Embeddings,
    NetworkData,
    Pipeline,
    Source,
    TermData,
    TermIndex,
)

if TYPE_CHECKING:
    from networkx import Graph


def web_path(path: Path) -> str:
    """The site URL for a file written under ``config.PUBLIC`` — ``public/ctext/
    仁_1A.json`` -> ``/ctext/仁_1A.json``. Derived, so the filesystem layout and
    the URL layout can never drift apart."""
    return "/" + path.resolve().relative_to(PUBLIC).as_posix()


class CorpusWriter:
    """Writes one corpus run's JSON and accumulates its manifest.

    Construct one per corpus runner, call the ``add_*`` / ``set_*`` methods as the
    run produces data, and finish with :meth:`save_index`.
    """

    INDEX_NAME = "index.json"

    def __init__(self, p: Pipeline, corpus: str, source: Source) -> None:
        self.out_dir: Path = p.out_dir
        self.corpus = corpus
        self.index = CorpusIndex(corpus, Source.from_sourcelike(source))

    # --- files -------------------------------------------------------------

    def add_cooccurrence(self, label: str, source: Source, file_id: str,
                         network: "Graph | None", occurrences: int) -> None:
        """One PMI network for ``label`` over ``source`` (a chapter, an article,
        or a whole-corpus stand-in) -> ``{label}_{file_id}.json``."""
        data = NetworkData(TermData(label), source, network,
                           occurrences=occurrences)
        web = self._write(data, f"{label}_{file_id}.json")
        self._term(label).cooccurrence.append(replace(data.source, data=web))

    def add_similarity(self, label: str, source: Source,
                       network: "Graph | None", occurrences: int) -> None:
        """``label``'s pruned cosine neighborhood over the whole corpus ->
        ``{label}_embeds.json``."""
        data = NetworkData(TermData(label), source, network,
                           occurrences=occurrences)
        web = self._write(data, f"{label}_embeds.json")
        self._term(label).similarity.append(replace(data.source, data=web))

    def add_embeddings(self, embeddings: Embeddings) -> Path:
        """The corpus's PCA-reduced dataset -> ``public/embeddings/{corpus}.json``
        (shared by every term, so it is recorded once on the manifest itself)."""
        path = EMBEDDINGS / f"{self.corpus}.json"
        embeddings.save_json(path)
        self.index.embeddings = replace(embeddings.source,
                                        data=web_path(path))
        return path

    # --- counts ------------------------------------------------------------

    def set_total(self, label: str, occurrences: int) -> None:
        """``label``'s occurrence count across the analyzed corpus."""
        self._term(label).total_occurrences = occurrences

    def set_chinese_philosophy(self, label: str, occurrences: int) -> None:
        """``label``'s occurrence count inside the excluded Chinese-philosophy
        SEP articles (counted, but kept out of the analyzed corpus)."""
        self._term(label).chinese_philosophy_occurrences = occurrences

    # --- manifest ----------------------------------------------------------

    def save_index(self) -> Path:
        path = self.out_dir / self.INDEX_NAME
        self.index.save_json(path)
        print(f"index      : {len(self.index.terms)} terms -> {path}")
        return path

    # --- internals ---------------------------------------------------------

    def _term(self, label: str) -> TermIndex:
        return self.index.terms.setdefault(label, TermIndex(TermData(label)))

    def _write(self, data: NetworkData, filename: str) -> str:
        path = self.out_dir / filename
        data.save_json(path)
        return web_path(path)
