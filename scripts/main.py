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
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from numpy import ndarray
from networkx import Graph
from spacy.tokens import Doc

from config import TERMS, CTEXT, SEP, DATA, ANALYSIS
from models import (
    CorpusIndex, Embeddings, Pipeline, Source, TermData, TermIndex)
from output import CorpusWriter
from renderings import check_coverage
from corpus.sep import SEP_CORPUS
from corpus.build import build_chinese_corpus, build_english_corpus
from corpus.parse import (
    parse_sep_article, parse_mengzi_chapter, mengzi_merge_config,
    mengzi_merge_report, verb_lemma)
from corpus.inpho import is_chinese_philosophy
from embeddings import analyze, families, layouts, vectors
from embeddings.analyze import Method as SimMethod
from embeddings.model import Embedder
from embeddings.occurrences import (
    MatchFn,
    Segment,
    SourceDoc,
    build_segments,
    content_frequencies,
    document_frequencies,
    dominant_pos,
    matched_lemmas,
    variant_list,
)
from cooccurrence import pmi_spacy
from graph.annotate import attach_variants
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
    alias: Mapping[str, str] | None = None,
) -> Graph | None:
    """The term's PMI co-occurrence neighborhood over ``sources`` (spaCy Docs).

    ``alias`` is the embedding lens's variant merge (``Pipeline.merge_cooccurrence``),
    so the two lenses agree on what one node is."""
    return pmi_spacy.build_cooccurrence_network(
        sources, term, p.cooccurrence_min_freq,
        max_nodes=p.max_network_nodes,
        match_fn=match_fn,
        content_pos=_to_set(p.content_pos),
        stopwords=p.stopwords or set(),
        alias=alias,
    )


def count_occurrences(doc, label: str, match_fn: MatchFn | None) -> int:
    """How many tokens in ``doc`` belong to ``label`` (via ``match_fn`` if given,
    else exact-lemma). Used for the ``TermData.occurrences`` display count.

    The count is the total of :func:`matched_lemmas`, so it can never disagree
    with the variant list built from the same pass."""
    return sum(matched_lemmas(doc, label, match_fn).values())


def target_matches(sources: Iterable[SourceDoc], label: str,
                   match_fn: MatchFn | None) -> Counter[str]:
    """:func:`matched_lemmas` folded over several sources — one scope's answer to
    "which words did ``label`` absorb here, and how often"."""
    counts: Counter[str] = Counter()
    for source in sources:
        counts += matched_lemmas(source.doc, label, match_fn)
    return counts


def save_cooccurrence(
    p: Pipeline, w: CorpusWriter, label: str, network_sources: list[SourceDoc],
    meta_source: Source, file_id: str, *, match_fn: MatchFn | None,
    alias: Mapping[str, str] | None = None,
    variants: Mapping[str, list[str]] | None = None,
) -> int:
    """Build ``label``'s PMI network over ``network_sources`` and hand it to the
    writer as one ``NetworkData`` file (``meta_source`` is the source recorded in
    the file — a chapter, an article, or a whole-corpus stand-in). Returns the
    occurrence count recorded on the manifest.

    ``variants`` is the other half of ``alias``: the merge decides what a node
    *is*, this records which words it stands for, on the node itself. The target
    adds its own entry, scoped to ``network_sources`` — which is one article or
    chapter here and a whole search there, exactly like the network itself."""
    counts = target_matches(network_sources, label, match_fn)
    matched = variant_list(counts, label)
    net = get_cooccurrence(p, label, network_sources, match_fn=match_fn,
                           alias=alias)
    if net is None:
        print(f"  no co-occurrence for {label} in {meta_source.title}")
    else:
        attach_variants(net, {**(variants or {}), label: matched},
                        always=(label,))
    occ = sum(counts.values())
    w.add_cooccurrence(label, meta_source, file_id, net, occ, matched)
    return occ


# ---------------------------------------------------------------------------
# Embeddings + similarity (share one cosine graph over the whole corpus)
# ---------------------------------------------------------------------------

def save_similarity(
    p: Pipeline, w: CorpusWriter, targets: set[str] | frozenset[str],
    corpus_source: Source, G: Graph, *, occ_counts: Counter,
    variants: Mapping[str, list[str]] | None = None,
) -> None:
    """Write one ``NetworkData`` file per term — its pruned cosine neighborhood —
    and record each term's whole-corpus occurrence count on the manifest.

    ``variants`` labels each node with the words it stands for — for an ordinary
    node the family the merge folded into it, for a target the lemmas it matched
    (the same map the embedding export carries, and whole-corpus like this graph);
    the pruned subgraphs inherit it."""
    attach_variants(G, variants, always=targets)
    for target in targets:
        pruned = prune_to_neighborhood(G, target, p.max_network_nodes)
        if pruned is None:
            print(f"  {target} absent from similarity graph")
        occ = int(occ_counts.get(target, 0))
        matched = list((variants or {}).get(target, ()))
        w.add_similarity(target, corpus_source, pruned, occ, matched)
        w.set_total(target, occ)
        w.set_variants(target, matched)


# ---------------------------------------------------------------------------
# Corpus runners
# ---------------------------------------------------------------------------

def run_mengzi(p: Pipeline, *, artifacts: bool = False,
               prune: bool = False) -> None:
    sw = Stopwatch()
    targets = frozenset(t.hanzi for t in TERMS)

    merge = mengzi_merge_config(p, targets)
    mengzi = build_chinese_corpus()
    chapters = [SourceDoc(c, parse_mengzi_chapter(c, merge))
                for c in mengzi.chapters]
    full = SourceDoc(mengzi, Doc.from_docs([c.doc for c in chapters]))
    print(f"parsed     : {len(chapters)} chapters + full corpus")
    report = mengzi_merge_report(merge)
    if report is not None:
        print(f"merged     : {report.summary()}")
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
    pooler, doc_freq, n_docs, freq = embed(
        p, chapters, targets,
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
    # Whole-corpus, like the graph. There is no `match_fn` here — a hanzi is a
    # target because its lemma *is* the label — so every list comes out empty;
    # it is carried anyway so the Chinese files say `[]` rather than nothing at
    # all, and so a term whose lemma differs from its glyph would show up here.
    target_variants = {t: variant_list(target_matches(chapters, t, None), t)
                       for t in targets}
    save_similarity(p, writer, targets, mengzi, G, occ_counts=occ_counts,
                    variants=target_variants)

    norms = node_norms(labels, matrix)
    reduced = vectors.reduce_vectors(matrix, p.reduce_to_dims)
    vectors.cache_analysis_matrix("mengzi", labels, matrix)
    path = writer.add_embeddings(
        Embeddings.from_matrix(mengzi, labels, reduced, targets, community_map,
                               doc_freq=doc_freq, documents=n_docs, graph=G,
                               norms=norms, variants=target_variants, freq=freq,
                               layouts=layouts.tsne_layouts(
                                   labels, {"reduced": reduced, "full": matrix}, p)))
    print(f"embeddings : {len(labels)} nodes -> {path}")
    writer.save_index()
    if prune:
        writer.prune()
    sw.lap("similarity+export")

    if artifacts:
        run_artifacts(labels, matrix, targets, pooler, mean, project,
                      ANALYSIS / "mengzi", p.sim_network, p.quantile, p.knn_k,
                      p.sim_transform, p.resolution, p.max_network_nodes)
        sw.lap("artifacts")

    sw.summary("mengzi")


def run_sep(p: Pipeline, *, per_term: int = 12, max_chinese_topic: float | None = None, artifacts: bool = False,
            prune: bool = False, allow_empty: bool = False) -> None:
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

    searches = build_english_corpus(
        per_term,
        max_chinese_topic=max_chinese_topic,
        min_freq=p.cooccurrence_min_freq
    )
    sw.lap("fetch+search")

    writer = CorpusWriter(p, "sep", SEP_CORPUS)

    # --- parse every article once, per rendering (deduped across renderings) ---
    # Co-occurrence used to run here and parse as a side effect, but it now needs
    # the variant merge, which only exists once there are vectors — so parsing is
    # its own phase and the PMI networks are built after the embedding below.
    per_label: list[tuple[str, Source, list[SourceDoc]]] = [
        (ts.term.label, ts.search, [source_doc(a) for a in ts.search.articles])
        for ts in searches]
    corpus_occ = {
        label: sum(count_occurrences(sd.doc, label, match_fn) for sd in adocs)
        for label, _, adocs in per_label}
    sw.lap("parse")

    # A rendering that matched nothing is a config/lemmatizer bug far more often
    # than a real absence, and it fails silently — every file it owns is written
    # with a null network. Check as soon as the counts exist, so the run still
    # fails before the expensive embedding phase rather than after it.
    check_coverage(renderings, corpus_occ,
                   [sd.doc for sd in doc_cache.values()],
                   {ts.term.label: ts.articles for ts in searches},
                   corpus="SEP", allow_empty=allow_empty)

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
        docs = [chinese_phil_source_doc(
            a) for a in ts.search.excluded if is_chinese_philosophy(a.url, max_chinese_topic)]
        occ = sum(count_occurrences(sd.doc, label, match_fn) for sd in docs)
        writer.set_chinese_philosophy(label, occ)
        n_cn_occ += occ
    print(f"chinese-phil: {len(chinese_phil_doc_cache)} articles, "
          f"{n_cn_occ} occurrences")
    sw.lap("chinese-philosophy")

    # --- one combined embedding space over every (deduped) article ---
    combined = list(doc_cache.values())
    print(f"parsed     : {len(combined)} SEP articles")
    pooler, doc_freq, n_docs, freq = embed(
        p, combined, labels_by_target, match_fn=match_fn,
        keep=labels_by_target if artifacts else frozenset())
    labels, matrix = pool(pooler, labels_by_target)
    matrix, mean, project = transform_matrix(p, matrix)

    # Collapse derivational variants, then rebuild the space from the merged
    # accumulators. The decision needs vectors, so this cannot move earlier; it
    # stays ahead of the graph, communities, norms and the PCA export so every
    # derived field is computed over the merged vocabulary.
    variants: dict[str, list[str]] = {}
    alias: dict[str, str] = {}
    if p.merge_variants:
        pos, stop = _to_set(p.content_pos), p.stopwords or set()
        extra = families.participial_pairs(labels, verb_lemma)
        # Dump the candidates and their cosines *in this space* before merging.
        # The exported vectors are PCA-reduced, so a threshold swept against the
        # artifact reads high; this is the only record of what the gate actually saw.
        dump_family_candidates(ANALYSIS / "sep", labels, matrix,
                               labels_by_target, extra)
        # Type-level POS, so a merged family can be named after its noun rather
        # than whichever member the corpus happens to say most often.
        word_pos = dominant_pos(
            combined, match_fn, content_pos=pos, stopwords=stop)
        alias, variants = families.merge_map(
            labels, matrix, threshold=p.merge_threshold,
            exclude=labels_by_target, counts=freq, pos=word_pos,
            extra_pairs=extra)
        if alias and p.merge_similarity:
            pooler.merge(alias)
            labels, matrix = pool(pooler, labels_by_target)
            matrix, mean, project = transform_matrix(p, matrix)
            doc_freq = document_frequencies(
                combined, match_fn, content_pos=pos, stopwords=stop, alias=alias)
            # Occurrences sum exactly under a merge (an article containing both
            # `inspire` and `inspiration` contributes both counts), so unlike
            # doc_freq this needs no second pass over the corpus.
            merged_freq: Counter[str] = Counter()
            for word, n in freq.items():
                merged_freq[alias.get(word, word)] += n
            freq = merged_freq
        lenses = " + ".join(
            [n for n, on in (("similarity", p.merge_similarity),
                             ("co-occurrence", p.merge_cooccurrence)) if on]
        ) or "nothing (both lenses opted out)"
        print(f"merged     : {len(alias)} variants into "
              f"{len(set(alias.values()))} words "
              f"(cosine >= {p.merge_threshold}) -> {len(labels)} nodes; "
              f"applied to {lenses}")

    # Only a lens the merge was actually *applied* to may claim the families on
    # its nodes: with a lens opted out its nodes are still one per lemma, so
    # listing variants there would be a lie about what the node covers.
    #
    # Target labels are the one class of node whose label abstracts over several
    # words *without* the merge — the rendering's glob does it — so they carry
    # their matched lemmas in the same field. `merge_map(exclude=labels_by_target)`
    # never merges a target, so these keys cannot collide. Whole-corpus over the
    # deduped articles, matching the scope of the graph and the export they
    # annotate (the per-source co-occurrence files scope their own).
    target_variants = {
        label: variant_list(target_matches(combined, label, match_fn), label)
        for label in labels_by_target}
    sim_variants = {**(variants if p.merge_similarity else {}),
                    **target_variants}
    cooc_variants = variants if p.merge_cooccurrence else {}
    sw.lap("embedding")

    # --- co-occurrence: each rendering over its combined search + each article ---
    # The combined file goes first, so the manifest lists it first. `cooc_alias`
    # is what makes a node mean the same word in both lenses; it is deliberately
    # separable, because the merge is *gated* on embedding cosine and this is the
    # point where a paradigmatic criterion reaches into the syntagmatic graphs.
    # See notes/derivational-variant-merging.md.
    cooc_alias = alias if p.merge_cooccurrence else {}
    for label, search, adocs in per_label:
        save_cooccurrence(p, writer, label, adocs, search, "combined",
                          match_fn=match_fn, alias=cooc_alias,
                          variants=cooc_variants)
        for sd in adocs:
            save_cooccurrence(p, writer, label, [sd], sd.source, sd.source.id,
                              match_fn=match_fn, alias=cooc_alias,
                              variants=cooc_variants)
    sw.lap("co-occurrence")

    G, _, community_map = analyze.build_networks(
        labels, matrix, method=p.sim_network, quantile=p.quantile, knn_k=p.knn_k,
        sim_transform=p.sim_transform, resolution=p.resolution)
    sw.lap("networks")

    occ_counts = Counter(
        lbl for sd in combined for t in sd.doc
        if (lbl := match_fn(t.lemma_.lower(), t.pos_)) is not None)
    save_similarity(p, writer, labels_by_target,
                    SEP_CORPUS, G, occ_counts=occ_counts,
                    variants=sim_variants)

    norms = node_norms(labels, matrix)
    reduced = vectors.reduce_vectors(matrix, p.reduce_to_dims)
    vectors.cache_analysis_matrix("sep", labels, matrix)
    path = writer.add_embeddings(
        Embeddings.from_matrix(SEP_CORPUS, labels, reduced, labels_by_target,
                               community_map, doc_freq=doc_freq,
                               documents=n_docs, graph=G, norms=norms,
                               variants=sim_variants, freq=freq,
                               layouts=layouts.tsne_layouts(
                                   labels, {"reduced": reduced, "full": matrix}, p)))
    print(f"embeddings : {len(labels)} nodes -> {path}")
    writer.save_index()
    if prune:
        writer.prune()
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
    sources: list[SourceDoc],
    target_labels: frozenset[str],
    *,
    match_fn: MatchFn | None = None,
    keep: set[str] | frozenset[str] = frozenset(),
) -> tuple[vectors.Pooler, Counter[str], int, Counter[str]]:
    """Vocab -> segments -> streaming max-pool of per-occurrence span vectors.

    Returns ``(pooler, doc_freq, n_documents, freq)`` — the pooled vectors, the
    per-word document frequency, the corpus document count (for the scatter's
    doc-freq fields and the ``min_doc_freq``/``max_doc_freq`` bounds), and the raw
    per-word occurrence counts (which label survives a variant merge). The embedder
    yields occurrences one batch at a time; we fold each into a running max-pool
    here, so the whole corpus's occurrence vectors are never all resident at once.
    ``keep`` words additionally retain their full occurrence stacks (cohesion under
    ``--artifacts``)."""

    emb = Embedder(p.model)
    print(
        f"device     : {emb.device_label}  hidden: {emb.hidden_size}")

    pos, stop = _to_set(p.content_pos), p.stopwords or set()
    freq = content_frequencies(
        sources, match_fn, content_pos=pos, stopwords=stop)
    vocab = {k for k, c in freq.items() if c >= p.min_freq}
    doc_freq = document_frequencies(
        sources, match_fn, content_pos=pos, stopwords=stop)
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

    # targets always kept, regardless of the bounds
    vocab |= set(target_labels)
    unk_check(emb, target_labels)
    segments = segment(p, emb, sources, vocab, match_fn)
    pooler = vectors.Pooler(mode=p.occurrence_pooling, keep=set(keep))
    for word, vec in emb.embed(segments, p.batch_size,
                               subword_pooling=p.subword_pooling):
        pooler.add(word, vec)
    return pooler, doc_freq, n_docs, freq


def dump_family_candidates(
    out_dir: Path,
    labels: list[str],
    matrix: ndarray,
    exclude: frozenset[str],
    extra_pairs,
) -> None:
    """Write every candidate family's pairwise cosines in the analysis space.

    One row per within-family pair, which is exactly what complete linkage
    consumes — so ``tools/family_diagnostics.py`` can replay any threshold
    against the real 768-d geometry instead of the PCA-reduced export, where
    truncation inflates cosine and makes a swept threshold read far too high.
    """
    import csv
    vocab = [lbl for lbl in labels if lbl not in exclude]
    fams = families.candidate_families(vocab, extra_pairs)
    index = {lbl: i for i, lbl in enumerate(labels)}
    unit = matrix / np.clip(np.linalg.norm(matrix,
                            axis=1, keepdims=True), 1e-12, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "family_candidates.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "a", "b", "cosine"])
        for n, fam in enumerate(fams):
            for i, a in enumerate(fam):
                for b in fam[i + 1:]:
                    w.writerow([n, a, b,
                                f"{float(unit[index[a]] @ unit[index[b]]):.4f}"])
    print(f"candidates : {len(fams)} families -> {path}")


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
        embeddings = index.embeddings if index else None
        # Every `Source.occurrences` is the term's count *in that source*, so the
        # embedding dataset gets the analyzed-corpus count — it never saw the
        # excluded Chinese-philosophy articles. Only the side's grand total adds
        # them back in.
        return {
            "corpus": corpus,
            "term": asdict(entry.term),
            "totalOccurrences": (entry.total_occurrences
                                 + entry.chinese_philosophy_occurrences),
            "chinesePhilosophyOccurrences": entry.chinese_philosophy_occurrences,
            "embeddings": ([asdict(replace(
                embeddings, occurrences=entry.total_occurrences))]
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
    p.add_argument("--max-chinese-topic", type=lambda n: float(n) if n is not None else None, default=None, dest="max_chinese_topic",
                   help="The max percentage an SEP article can be identified"
                   "with InPhO's Chinese philosophy topic before it is"
                   "discarded. This should be a float between 0 and 1. If"
                   "unset, it filters articles whose majority topic is Chinese"
                   "philosophy.")
    p.add_argument("--master-only", action="store_true",
                   help="Skip the corpora; just rebuild the master index from the "
                        "per-corpus index.json manifests.")
    p.add_argument("--prune", action="store_true",
                   help="After each corpus run, delete files in its output dir "
                        "that the run did not write (output left behind by terms "
                        "since removed from TERMS). Destructive — review the diff.")
    p.add_argument("--allow-empty-renderings", action="store_true",
                   dest="allow_empty",
                   help="Downgrade the empty-rendering check to a warning. A "
                        "rendering that matches no token writes null networks "
                        "for every file it owns; by default that aborts the run.")
    args = p.parse_args()
    if args.prune and args.master_only:
        p.error("--prune needs a corpus run to know what to keep; "
                "it cannot be combined with --master-only")
    return args


def main() -> None:
    from config import MENGZI_PIPELINE, SEP_PIPELINE
    args = parse_args()
    sw = Stopwatch()
    if not args.master_only:
        if args.corpus in ("mengzi", "all"):
            print("\n=== Mengzi ===")
            run_mengzi(MENGZI_PIPELINE, artifacts=args.artifacts,
                       prune=args.prune)
            sw.lap("mengzi")
        if args.corpus in ("sep", "all"):
            print("\n=== SEP ===")
            run_sep(SEP_PIPELINE, per_term=args.per_term,
                    max_chinese_topic=args.max_chinese_topic,
                    artifacts=args.artifacts, prune=args.prune,
                    allow_empty=args.allow_empty)
            sw.lap("sep")
    print("\n=== Master index ===")
    build_master()
    sw.lap("master")
    print()
    sw.summary("total")


if __name__ == "__main__":
    main()
