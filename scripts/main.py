"""The pipeline entrypoint: emit every JSON file the website needs, from one place.

Run from ``scripts/`` as ``python -m main`` (see ``--help`` for flags). Outputs are
shaped by the ``models.py`` dataclasses (``NetworkData`` / ``Embeddings``), and the
frontend Zod schemas are kept in sync with them.

Per corpus it writes three kinds of JSON:

- co-occurrence: one ``NetworkData`` file per (term, source) — the full corpus and
  each chapter/article — under ``public/ctext`` (or ``public/sep``).
- similarity:    one ``NetworkData`` file per term, the term's pruned cosine
  neighborhood over the whole corpus.
- embeddings:    one ``Embeddings`` file per corpus (PCA-reduced vectors).

Everything a run writes goes through an ``output.CorpusWriter``, which records it
in that corpus's ``index.json`` manifest; ``build_master`` composes
``src/data/terms.json`` from the two manifests alone.

Optional ``analysis/`` artifacts (PNGs/CSVs) are decoupled behind ``--artifacts``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
from numpy import ndarray
from networkx import Graph
from spacy.tokens import Doc

from config import TERMS, CTEXT, SEP, DATA, ANALYSIS
from models import (
    CorpusIndex, Embeddings, Pipeline, Source, TermData, TermIndex)
from output import CorpusWriter
from corpus.sep import SEP_CORPUS
from corpus.build import build_chinese_corpus, build_english_corpus
from corpus.parse import parse_sep_article, parse_mengzi_chapter
from embeddings import analyze, vectors
from embeddings.analyze import Method as SimMethod
from embeddings.model import Embedder
from embeddings.occurrences import (
    MatchFn,
    Segment,
    SourceDoc,
    build_segments,
    build_vocab,
    document_frequencies,
)
from cooccurrence import pmi_spacy
from graph.prune import prune_to_neighborhood


class Stopwatch:
    """Record named phase durations and print a per-run timing summary.

    Call :meth:`lap` at each phase boundary (it measures the time since the
    previous lap), then :meth:`summary` at the end for a breakdown + total.
    """

    def __init__(self) -> None:
        self._start = self._last = time.perf_counter()
        self._laps: list[tuple[str, float]] = []

    def lap(self, label: str) -> None:
        now = time.perf_counter()
        self._laps.append((label, now - self._last))
        self._last = now

    def summary(self, title: str) -> None:
        total = time.perf_counter() - self._start
        print(f"timing [{title}] : {total:.1f}s total")
        for label, dt in self._laps:
            print(f"  {label:<16} {dt:7.1f}s  {dt / total:4.0%}")


# ---------------------------------------------------------------------------
# Co-occurrence (one NetworkData per (term, source))
# ---------------------------------------------------------------------------

def _to_set(frz: frozenset | None) -> set | None:
    return None if frz is None else set(frz)


def get_cooccurrence(
    p: Pipeline,
    term: str,
    sources: list[SourceDoc],
    *,
    match_fn: MatchFn | None = None,
) -> Graph | None:
    """The term's PMI co-occurrence neighborhood over ``sources`` (spaCy Docs)."""
    return pmi_spacy.build_cooccurrence_network(
        sources, term, p.cooccurrence_min_freq,
        max_nodes=p.max_network_nodes,
        match_fn=match_fn,
        content_pos=_to_set(p.content_pos),
        stopwords=p.stopwords or set(),
    )


def count_occurrences(doc, label: str, match_fn: MatchFn | None) -> int:
    """How many tokens in ``doc`` belong to ``label`` (via ``match_fn`` if given,
    else exact-lemma). Used for the ``TermData.occurrences`` display count."""
    if match_fn is not None:
        return sum(1 for t in doc if match_fn(t.lemma_.lower(), t.pos_) == label)
    return sum(1 for t in doc if t.lemma_ == label)


def save_cooccurrence(
    p: Pipeline, w: CorpusWriter, label: str, network_sources: list[SourceDoc],
    meta_source: Source, file_id: str, *, match_fn: MatchFn | None,
) -> None:
    """Build ``label``'s PMI network over ``network_sources`` and hand it to the
    writer as one ``NetworkData`` file (``meta_source`` is the source recorded in
    the file — a chapter, an article, or a whole-corpus stand-in)."""
    net = get_cooccurrence(p, label, network_sources, match_fn=match_fn)
    if net is None:
        print(f"  no co-occurrence for {label} in {meta_source.title}")
    occ = sum(count_occurrences(s.doc, label, match_fn)
              for s in network_sources)
    w.add_cooccurrence(label, meta_source, file_id, net, occ)


# ---------------------------------------------------------------------------
# Embeddings + similarity (share one cosine graph over the whole corpus)
# ---------------------------------------------------------------------------

def save_similarity(
    p: Pipeline, w: CorpusWriter, targets: set[str] | frozenset[str],
    corpus_source: Source, G: Graph, *, occ_counts: Counter,
) -> None:
    """Write one ``NetworkData`` file per term — its pruned cosine neighborhood —
    and record each term's whole-corpus occurrence count on the manifest."""
    for target in targets:
        pruned = prune_to_neighborhood(G, target, p.max_network_nodes)
        if pruned is None:
            print(f"  {target} absent from similarity graph")
        occ = int(occ_counts.get(target, 0))
        w.add_similarity(target, corpus_source, pruned, occ)
        w.set_total(target, occ)


# ---------------------------------------------------------------------------
# Corpus runners
# ---------------------------------------------------------------------------

def run_mengzi(p: Pipeline, *, artifacts: bool = False) -> None:
    sw = Stopwatch()
    targets = frozenset(t.hanzi for t in TERMS)

    mengzi = build_chinese_corpus()
    chapters = [SourceDoc(c, parse_mengzi_chapter(c)) for c in mengzi.chapters]
    full = SourceDoc(mengzi, Doc.from_docs([c.doc for c in chapters]))
    print(f"parsed     : {len(chapters)} chapters + full corpus")
    sw.lap("parse")

    writer = CorpusWriter(p, "mengzi", mengzi)

    # --- co-occurrence: full corpus + each chapter, per target ---
    # (the whole-corpus file first, so the manifest lists it first)
    for target in targets:
        save_cooccurrence(p, writer, target, [full],
                          mengzi, "mengzi", match_fn=None)
        for ch in chapters:
            save_cooccurrence(p, writer, target, [ch],
                              ch.source, ch.source.id, match_fn=None)
    sw.lap("co-occurrence")

    # --- embeddings over the whole corpus (chapters; full would double-count) ---
    embedder = Embedder(p.model)
    print(
        f"device     : {embedder.device_label}  hidden: {embedder.hidden_size}")
    pooler, doc_freq, n_docs = embed(p, embedder, chapters, targets,
                                     keep=targets if artifacts else frozenset())
    labels, matrix = pool(pooler, targets)
    matrix, mean, project = transform_matrix(p, matrix)
    sw.lap("embedding")

    G, _, community_map = analyze.build_networks(
        labels, matrix, method=p.sim_network, quantile=p.quantile, knn_k=p.knn_k,
        sim_transform=p.sim_transform, resolution=p.resolution)
    sw.lap("networks")

    occ_counts = Counter(
        t.lemma_ for t in full.doc if t.text in targets or t.lemma_ in targets)
    save_similarity(p, writer, targets, mengzi, G, occ_counts=occ_counts)

    norms = node_norms(labels, matrix)
    reduced = vectors.reduce_vectors(matrix, p.reduce_to_dims)
    path = writer.add_embeddings(
        Embeddings.from_matrix(mengzi, labels, reduced, targets, community_map,
                               doc_freq=doc_freq, documents=n_docs, graph=G,
                               norms=norms))
    print(f"embeddings : {len(labels)} nodes -> {path}")
    writer.save_index()
    sw.lap("similarity+export")

    if artifacts:
        run_artifacts(labels, matrix, targets, pooler, mean, project,
                      ANALYSIS / "mengzi", p.sim_network, p.quantile, p.knn_k,
                      p.sim_transform, p.resolution, p.max_network_nodes)
        sw.lap("artifacts")

    sw.summary("mengzi")


def run_sep(p: Pipeline, *, per_term: int = 12, artifacts: bool = False) -> None:
    sw = Stopwatch()
    renderings = [r for term in TERMS for r in term.renderings]
    labels_by_target = frozenset(r.label for r in renderings)

    def match_fn(lemma: str, pos: str | None = None) -> str | None:
        for r in renderings:
            if r.matches(lemma, pos):
                return r.label
        return None

    # Parse each unique article once; reuse across co-occurrence + embeddings.
    doc_cache: dict[str, SourceDoc] = {}

    def source_doc(a) -> SourceDoc:
        if a.url not in doc_cache:
            doc_cache[a.url] = SourceDoc(a, parse_sep_article(a))
        return doc_cache[a.url]

    searches = build_english_corpus(per_term)
    sw.lap("fetch+search")

    writer = CorpusWriter(p, "sep", SEP_CORPUS)

    # --- co-occurrence: each rendering over its combined search + each article ---
    # (also parses + caches every article, reused for the embedding space below)
    # The combined file goes first, so the manifest lists it first.
    for ts in searches:
        label, articles = ts.term.label, ts.search.articles
        adocs = [source_doc(a) for a in articles]
        save_cooccurrence(p, writer, label, adocs, ts.search,
                          "combined", match_fn=match_fn)
        for sd in adocs:
            save_cooccurrence(p, writer, label, [sd],
                              sd.source, sd.source.id, match_fn=match_fn)
    sw.lap("parse+co-occurrence")

    # --- occurrence counts within the excluded Chinese-philosophy articles ---
    # (parsed separately from `doc_cache` — must NOT feed the embedding corpus)
    chinese_phil_doc_cache: dict[str, SourceDoc] = {}

    def chinese_phil_source_doc(a) -> SourceDoc:
        if a.url not in chinese_phil_doc_cache:
            chinese_phil_doc_cache[a.url] = SourceDoc(a, parse_sep_article(a))
        return chinese_phil_doc_cache[a.url]

    n_cn_occ = 0
    for ts in searches:
        label = ts.term.label
        docs = [chinese_phil_source_doc(a) for a in ts.search.excluded]
        occ = sum(count_occurrences(sd.doc, label, match_fn) for sd in docs)
        writer.set_chinese_philosophy(label, occ)
        n_cn_occ += occ
    print(f"chinese-phil: {len(chinese_phil_doc_cache)} articles, "
          f"{n_cn_occ} occurrences")
    sw.lap("chinese-philosophy")

    # --- one combined embedding space over every (deduped) article ---
    combined = list(doc_cache.values())
    print(f"parsed     : {len(combined)} SEP articles")
    embedder = Embedder(p.model)
    print(
        f"device     : {embedder.device_label}  hidden: {embedder.hidden_size}")
    pooler, doc_freq, n_docs = embed(
        p, embedder, combined, labels_by_target, match_fn=match_fn,
        keep=labels_by_target if artifacts else frozenset())
    labels, matrix = pool(pooler, labels_by_target)
    matrix, mean, project = transform_matrix(p, matrix)
    sw.lap("embedding")

    G, _, community_map = analyze.build_networks(
        labels, matrix, method=p.sim_network, quantile=p.quantile, knn_k=p.knn_k,
        sim_transform=p.sim_transform, resolution=p.resolution)
    sw.lap("networks")

    occ_counts = Counter(
        lbl for sd in combined for t in sd.doc
        if (lbl := match_fn(t.lemma_.lower(), t.pos_)) is not None)
    save_similarity(p, writer, labels_by_target,
                    SEP_CORPUS, G, occ_counts=occ_counts)

    norms = node_norms(labels, matrix)
    reduced = vectors.reduce_vectors(matrix, p.reduce_to_dims)
    path = writer.add_embeddings(
        Embeddings.from_matrix(SEP_CORPUS, labels, reduced, labels_by_target,
                               community_map, doc_freq=doc_freq,
                               documents=n_docs, graph=G, norms=norms))
    print(f"embeddings : {len(labels)} nodes -> {path}")
    writer.save_index()
    sw.lap("similarity+export")

    if artifacts:
        run_artifacts(labels, matrix, labels_by_target, pooler, mean, project,
                      ANALYSIS / "sep", p.sim_network, p.quantile, p.knn_k,
                      p.sim_transform, p.resolution, p.max_network_nodes)
        sw.lap("artifacts")

    sw.summary("sep")


# ---------------------------------------------------------------------------
# Shared embedding helpers (corpus-agnostic)
# ---------------------------------------------------------------------------

def embed(
    p: Pipeline,
    emb: Embedder,
    sources: list[SourceDoc],
    target_labels: frozenset[str],
    *,
    match_fn: MatchFn | None = None,
    keep: set[str] | frozenset[str] = frozenset(),
) -> tuple[vectors.Pooler, Counter, int]:
    """Vocab -> segments -> streaming max-pool of per-occurrence span vectors.

    Returns ``(pooler, doc_freq, n_documents)`` — the pooled vectors plus the
    per-word document frequency and the corpus document count (for the scatter's
    doc-freq fields and the ``min_doc_freq``/``max_doc_freq`` bounds). The embedder
    yields occurrences one batch at a time; we fold each into a running max-pool
    here, so the whole corpus's occurrence vectors are never all resident at once.
    ``keep`` words additionally retain their full occurrence stacks (cohesion under
    ``--artifacts``)."""
    pos, stop = _to_set(p.content_pos), p.stopwords or set()
    vocab = build_vocab(sources, p.min_freq, match_fn, content_pos=pos, stopwords=stop)
    doc_freq = document_frequencies(sources, match_fn, content_pos=pos, stopwords=stop)
    n_docs = len(sources)

    if p.min_doc_freq is not None:
        floor = p.min_doc_freq if p.min_doc_freq > 1 else p.min_doc_freq * n_docs
        before = len(vocab)
        vocab = {w for w in vocab if doc_freq.get(w, 0) >= floor}
        print(f"doc-freq   : floor {floor:.0f}/{n_docs} docs -> dropped "
              f"{before - len(vocab)} of {before} vocab words")

    if p.max_doc_freq is not None:
        cap = p.max_doc_freq if p.max_doc_freq > 1 else p.max_doc_freq * n_docs
        before = len(vocab)
        vocab = {w for w in vocab if doc_freq.get(w, 0) <= cap}
        print(f"doc-freq   : cap {cap:.0f}/{n_docs} docs -> dropped "
              f"{before - len(vocab)} of {before} vocab words")

    vocab |= set(target_labels)     # targets always kept, regardless of the bounds
    unk_check(emb, target_labels)
    segments = segment(p, emb, sources, vocab, match_fn)
    pooler = vectors.Pooler(mode=p.occurrence_pooling, keep=set(keep))
    for word, vec in emb.embed(segments, p.batch_size,
                               subword_pooling=p.subword_pooling):
        pooler.add(word, vec)
    return pooler, doc_freq, n_docs


def segment(
    p: Pipeline,
    emb: Embedder,
    sources: list[SourceDoc],
    woi: set[str],
    match_fn: MatchFn | None,
) -> list[Segment]:
    """Split each source into sentences and greedily pack under the token cap."""
    segments = build_segments(
        sources, woi, match_fn=match_fn,
        sent_len_fn=emb.token_lengths, max_tokens=emb.max_length,
        content_pos=_to_set(p.content_pos), stopwords=p.stopwords)
    n_occ = sum(len(s.spans) for s in segments)
    n_tok = sum(len(s.doc) for s in sources)
    n_src = len({s.source.id for s in sources})
    print(f"segmented  : {n_tok} tokens ({n_src} sources) -> "
          f"{len(segments)} segments (<= {emb.max_length} tok); {n_occ} occurrences")
    return segments


def pool(pooler: vectors.Pooler, target_labels) -> tuple[list[str], ndarray]:
    labels, matrix = pooler.matrix()
    n_tgt = sum(1 for l in labels if l in target_labels)
    print(f"pooled     : {len(labels)} words ({n_tgt} targets)")
    return labels, matrix


def transform_matrix(
    p: Pipeline, matrix: ndarray
) -> tuple[ndarray, ndarray | None, vectors.Callable[[ndarray], ndarray] | None]:
    """Center (anisotropy) then optionally debias (remove frequency-dominated
    directions). Returns ``(matrix, mean, project)`` — ``mean`` centers other vectors
    into the same space (occurrence stacks), ``project`` applies the same debias map
    to them (both ``None`` when their step is off), so every downstream measurement
    (sim graph, PCA export, cohesion) shares one space."""
    mean = None
    if p.center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")
    project = None
    if p.debias != "none":
        matrix, project = vectors.debias_matrix(matrix, p.debias, p.debias_k)
        k = p.debias_k if p.debias_k is not None else "auto"
        print(f"debiased   : {p.debias} (k={k}) — removed dominant directions")
    return matrix, mean, project


def unk_check(emb: Embedder, words: set[str] | frozenset[str]) -> None:
    """Warn if any word we embed loses a character to [UNK]."""
    unk = emb.unk_words(sorted(words))
    if unk:
        print(f"WARNING: {len(unk)} word(s) tokenize to [UNK]:")
        for w, toks in list(unk.items())[:20]:
            print(f"    {w!r} -> {toks}")
    else:
        print(f"UNK check  : OK — all {len(words)} words in-vocab")


def node_norms(labels: list[str], matrix: ndarray) -> dict[str, float]:
    """Per-word L2 norm in the centered/debiased analysis space (≈ distance from the
    corpus centroid). The exported vectors are L2-normalized (direction only), so this
    is the only surviving radial signal — needed by the debias diagnostics."""
    return dict(zip(labels, np.linalg.norm(matrix, axis=1).tolist()))


def run_artifacts(labels, matrix, targets, pooler: vectors.Pooler, mean, project,
                  out_dir, network: SimMethod, quantile: float, knn_k: int,
                  sim_transform: analyze.SimTransform, resolution: float,
                  max_nodes: int) -> None:
    """Opt-in: the heavy PNG/CSV analysis dump, decoupled from the JSON outputs."""
    is_target = np.array([l in targets for l in labels])

    def to_space(s: ndarray) -> ndarray:
        """Project an occurrence stack into the same centered/debiased space as
        ``matrix``, so cohesion is measured where the scatter lives."""
        s = s if mean is None else s - mean
        return s if project is None else project(s)
    target_occ = {w: to_space(s) for w, s in pooler.stacks().items()}
    summary = analyze.run_analysis(
        labels, matrix, is_target, target_occ, out_dir, quantile, kmeans_k=4,
        method=network, knn_k=knn_k, sim_transform=sim_transform,
        resolution=resolution, max_nodes=max_nodes)
    print(f"artifacts  : {summary['louvain_communities']} communities -> "
          f"{out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Master index (composed from the per-corpus manifests each run wrote)
# ---------------------------------------------------------------------------

CORPUS_DIRS = {"mengzi": CTEXT, "sep": SEP}


def build_master(out_path=None) -> dict:
    """Assemble the master index from the per-corpus ``index.json`` manifests: per
    term, the Chinese (Mengzi) side and one side per English rendering, each listing
    its co-occurrence / similarity / embedding sources with their `data` web paths.

    The manifests are the only input — paths were recorded when the files were
    written, so nothing here re-derives one. A corpus that has never been run has no
    manifest; its sides come out empty (with a warning)."""
    out_path = out_path or (DATA / "terms.json")
    indexes: dict[str, CorpusIndex | None] = {
        corpus: CorpusIndex.load(out_dir / CorpusWriter.INDEX_NAME)
        for corpus, out_dir in CORPUS_DIRS.items()}
    for corpus, index in indexes.items():
        if index is None:
            print(f"  WARNING: no {CorpusWriter.INDEX_NAME} for {corpus} — its "
                  f"sides will be empty (run --corpus {corpus})")

    def side(label: str, corpus: str) -> dict:
        index = indexes[corpus]
        entry = (index.terms.get(label) if index else None) or TermIndex(
            TermData(label))
        total = entry.total_occurrences + entry.chinese_philosophy_occurrences
        embeddings = index.embeddings if index else None
        return {
            "corpus": corpus,
            "term": asdict(entry.term),
            "totalOccurrences": total,
            "chinesePhilosophyOccurrences": entry.chinese_philosophy_occurrences,
            "embeddings": ([asdict(replace(embeddings, occurrences=total))]
                           if embeddings else []),
            "similarity": [asdict(s) for s in entry.similarity],
            "cooccurrence": [asdict(s) for s in entry.cooccurrence],
        }

    terms = []
    for term in TERMS:
        terms.append({
            "hanzi": term.hanzi,
            "renderings": list(term.english),
            "chinese": side(term.hanzi, "mengzi"),
            "english": [side(r.label, "sep") for r in term.renderings],
        })

    master = {"terms": terms}
    out_path.write_text(json.dumps(master, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    n_src = sum(len(t["chinese"]["cooccurrence"])
                + sum(len(e["cooccurrence"]) for e in t["english"]) for t in terms)
    print(
        f"master     : {len(terms)} terms, {n_src} co-occurrence files -> {out_path}")
    return master


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["mengzi", "sep", "all"], default="all")
    p.add_argument("--artifacts", action="store_true",
                   help="Also write analysis/ PNG+CSV artifacts (decoupled).")
    p.add_argument("--per-term", type=int, default=12, dest="per_term",
                   help="SEP articles fetched per English rendering (caps corpus "
                        "size / embedding memory).")
    p.add_argument("--master-only", action="store_true",
                   help="Skip the corpora; just rebuild the master index from the "
                        "files already on disk.")
    return p.parse_args()


def main() -> None:
    from config import MENGZI_PIPELINE, SEP_PIPELINE
    args = parse_args()
    sw = Stopwatch()
    if not args.master_only:
        if args.corpus in ("mengzi", "all"):
            print("\n=== Mengzi ===")
            run_mengzi(MENGZI_PIPELINE, artifacts=args.artifacts)
            sw.lap("mengzi")
        if args.corpus in ("sep", "all"):
            print("\n=== SEP ===")
            run_sep(SEP_PIPELINE, per_term=args.per_term,
                    artifacts=args.artifacts)
            sw.lap("sep")
    print("\n=== Master index ===")
    build_master()
    sw.lap("master")
    print()
    sw.summary("total")


if __name__ == "__main__":
    main()
