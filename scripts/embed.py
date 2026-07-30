"""WIP single pipeline: emit every JSON file the website needs, from one place.

Temporary scratch module (badly named, badly organized on purpose) — the plan is
to get one corpus working end-to-end here for a clean diff, then split into
``cli/main.py`` + shared modules. Outputs are shaped by the ``lib.py`` dataclasses
(``NetworkData`` / ``Embeddings``), NOT the current frontend Zod schemas; the site
is rewired to match in a later step.

Per corpus it writes three kinds of JSON:

- co-occurrence: one ``NetworkData`` file per (term, source) — the full corpus and
  each chapter/article — under ``public/ctext`` (or ``public/sep``).
- similarity:    one ``NetworkData`` file per term, the term's pruned cosine
  neighborhood over the whole corpus.
- embeddings:    one ``Embeddings`` file per corpus (PCA-reduced vectors).

Optional ``analysis/`` artifacts (PNGs/CSVs) are decoupled behind ``--artifacts``.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import networkx as nx
from numpy import ndarray
from networkx import Graph
from spacy.tokens import Doc

import config
import lib
from config import TERMS, CTEXT, SEP, EMBEDDINGS
from nlp.chinese import STOPWORDS as CHINESE_STOPWORDS
from corpus.build import build_chinese_corpus, build_english_corpus
from parse import parse_sep_article, parse_mengzi_chapter
from embeddings import analyze, vectors
from embeddings.model import Embedder
from embeddings.occurrences_unified import (
    CONTENT_POS,
    MatchFn,
    Segment,
    SourceDoc,
    build_segments,
    build_vocab,
)
from cooccurrence import pmi_spacy
from graph.prune import prune_to_neighborhood

MENGZI_MODEL = "hsc748NLP/GujiRoBERTa_fan"
SEP_MODEL = "roberta-base"

type WordVectors = dict[str, list[ndarray]]


# ---------------------------------------------------------------------------
# Co-occurrence (one NetworkData per (term, source))
# ---------------------------------------------------------------------------

def get_cooccurrence(
    term: str,
    sources: list[SourceDoc],
    min_freq: int,
    *,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = None,
    stopwords: set[str] | frozenset[str] = frozenset(),
    max_nodes: int = 15,
) -> Graph | None:
    """The term's PMI co-occurrence neighborhood over ``sources`` (spaCy Docs)."""
    return pmi_spacy.build_cooccurrence_network(
        sources, term, min_freq,
        max_nodes=max_nodes, match_fn=match_fn,
        content_pos=content_pos, stopwords=stopwords,
    )


def save_cooccurrences(
    targets, sources: list[SourceDoc], out_dir, *,
    min_freq: int, match_fn: MatchFn | None,
    content_pos: set[str] | None, stopwords, max_nodes: int,
) -> None:
    """Write one ``NetworkData`` file per (target, source)."""
    for src in sources:
        source, doc = src.source, src.doc
        counts = Counter(t.lemma_ for t in doc)
        for target in targets:
            term = lib.TermData(target, int(counts.get(target, 0)))
            net = get_cooccurrence(
                target, [src], min_freq, match_fn=match_fn,
                content_pos=content_pos, stopwords=stopwords, max_nodes=max_nodes)
            if net is None:
                print(f"  no co-occurrence for {target} in {source.title}")
            lib.NetworkData(term, source, net).save_json(
                out_dir / f"{target}_{source.id}.json")


# ---------------------------------------------------------------------------
# Embeddings + similarity (share one cosine graph over the whole corpus)
# ---------------------------------------------------------------------------

def cosine_graph(
    labels: list[str], matrix: ndarray, threshold: float,
) -> tuple[Graph, dict[str, int]]:
    """Full cosine-similarity graph over the vocab, with Louvain communities.

    Returns ``(G, community_map)``; isolated nodes are dropped (absent from the
    map, treated as community ``-1`` downstream)."""
    sim = analyze.cosine_matrix(matrix)
    G = analyze.build_cosine_graph(labels, sim, threshold)
    G.remove_nodes_from(list(nx.isolates(G)))
    analyze.annotate_communities(G)
    community_map = {n: int(G.nodes[n]["community"]) for n in G}
    print(f"cosine     : {G.number_of_nodes()} nodes, {G.number_of_edges()} "
          f"edges (>= {threshold})")
    return G, community_map


def save_similarity(
    targets, corpus_source, G: Graph, out_dir, *,
    max_nodes: int, occ_counts: Counter,
) -> None:
    """Write one ``NetworkData`` file per term: its pruned cosine neighborhood."""
    for target in targets:
        pruned = prune_to_neighborhood(G, target, max_nodes)
        if pruned is None:
            print(f"  {target} absent from similarity graph")
        term = lib.TermData(target, int(occ_counts.get(target, 0)))
        lib.NetworkData(term, corpus_source, pruned).save_json(
            out_dir / f"{target}_embeds.json")


# ---------------------------------------------------------------------------
# Corpus runners
# ---------------------------------------------------------------------------

def run_mengzi(
    *,
    min_freq: int = 5,
    center: bool = True,
    threshold: float = 0.3,
    reduce_to_dims: int = 50,
    max_nodes: int = 15,
    batch_size: int = 32,
    artifacts: bool = False,
) -> None:
    targets = frozenset(t.hanzi for t in TERMS)
    content_pos = None                 # classical Chinese: keep all POS
    stopwords = CHINESE_STOPWORDS

    mengzi = build_chinese_corpus()
    chapters = [SourceDoc(c, parse_mengzi_chapter(c)) for c in mengzi.chapters]
    full = SourceDoc(mengzi, Doc.from_docs([c.doc for c in chapters]))
    print(f"parsed     : {len(chapters)} chapters + full corpus")

    # --- co-occurrence: full corpus + each chapter, per target ---
    save_cooccurrences(
        targets, [full] + chapters, CTEXT,
        min_freq=min_freq, match_fn=None,
        content_pos=content_pos, stopwords=stopwords, max_nodes=max_nodes)

    # --- embeddings over the whole corpus (chapters; full would double-count) ---
    embedder = Embedder(MENGZI_MODEL)
    print(f"device     : {embedder.device_label}  hidden: {embedder.hidden_size}")
    word_vectors = embed(
        embedder, chapters, targets, min_freq=min_freq,
        content_pos=content_pos, stopwords=stopwords, batch_size=batch_size)
    labels, matrix = pool(word_vectors, targets)
    mean = None
    if center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")

    G, community_map = cosine_graph(labels, matrix, threshold)

    occ_counts = Counter(t.lemma_ for t in full.doc)
    save_similarity(targets, mengzi, G, CTEXT,
                    max_nodes=max_nodes, occ_counts=occ_counts)

    reduced = lib.reduce_vectors(matrix, reduce_to_dims)
    lib.Embeddings.from_matrix(mengzi, labels, reduced, targets, community_map) \
        .save_json(EMBEDDINGS / "mengzi.json")
    print(f"embeddings : {len(labels)} nodes -> {EMBEDDINGS / 'mengzi.json'}")

    if artifacts:
        run_artifacts(labels, matrix, targets, word_vectors, mean,
                      config.ANALYSIS / "mengzi", threshold, max_nodes)


def run_sep(
    *,
    per_term: int = 12,
    min_freq: int = 10,
    center: bool = True,
    threshold: float = 0.3,
    reduce_to_dims: int = 50,
    max_nodes: int = 15,
    batch_size: int = 16,
    artifacts: bool = False,
) -> None:
    renderings = [r for term in TERMS for r in term.renderings]
    labels_by_target = {r.label for r in renderings}

    def match_fn(lemma: str, pos: str | None = None) -> str | None:
        for r in renderings:
            if r.matches(lemma, pos):
                return r.label
        return None

    searches = build_english_corpus(per_term)
    # Every article, deduped by url, forms the combined embedding corpus.
    articles = {a.url: a for _, s in searches for a in s.search.articles}
    combined = [SourceDoc(a, parse_sep_article(a)) for a in articles.values()]
    print(f"parsed     : {len(combined)} SEP articles")

    # --- co-occurrence: each term over its own combined search + each article ---
    for term_search in searches:
        rendering, search = term_search.term, term_search.search
        srcs = [SourceDoc(search, parse_sep_article_of(search))] + [
            SourceDoc(a, parse_sep_article(a)) for a in search.articles]
        save_cooccurrences(
            [rendering.label], srcs, SEP,
            min_freq=min_freq, match_fn=match_fn,
            content_pos=CONTENT_POS, stopwords=frozenset(),
            max_nodes=max_nodes)

    # --- one combined embedding space ---
    embedder = Embedder(SEP_MODEL)
    print(f"device     : {embedder.device_label}  hidden: {embedder.hidden_size}")
    word_vectors = embed(
        embedder, combined, frozenset(labels_by_target), min_freq=min_freq,
        match_fn=match_fn, content_pos=CONTENT_POS, batch_size=batch_size)
    labels, matrix = pool(word_vectors, labels_by_target)
    mean = None
    if center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")

    G, community_map = cosine_graph(labels, matrix, threshold)
    save_similarity(labels_by_target, None, G, SEP,
                    max_nodes=max_nodes, occ_counts=Counter(labels))

    reduced = lib.reduce_vectors(matrix, reduce_to_dims)
    lib.Embeddings.from_matrix(None, labels, reduced, labels_by_target,
                               community_map).save_json(EMBEDDINGS / "sep.json")
    print(f"embeddings : {len(labels)} nodes -> {EMBEDDINGS / 'sep.json'}")

    if artifacts:
        run_artifacts(labels, matrix, labels_by_target, word_vectors, mean,
                      config.ANALYSIS / "sep", threshold, max_nodes)


def parse_sep_article_of(search) -> Doc:
    """A single combined Doc for a whole SEP search (all its articles' text)."""
    from parse import _get_en_nlp
    return _get_en_nlp()(search.text)


# ---------------------------------------------------------------------------
# Shared embedding helpers (corpus-agnostic)
# ---------------------------------------------------------------------------

def embed(
    emb: Embedder,
    sources: list[SourceDoc],
    target_labels: frozenset[str],
    *,
    match_fn: MatchFn | None = None,
    min_freq: int = 10,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: set[str] | frozenset[str] = frozenset(),
    batch_size: int = 32,
) -> WordVectors:
    """Vocab -> segments -> per-occurrence span vectors, for the whole corpus."""
    vocab = build_vocab(sources, min_freq, match_fn,
                        content_pos=content_pos, stopwords=stopwords)
    vocab |= set(target_labels)
    unk_check(emb, target_labels)
    segments = segment(emb, sources, vocab, match_fn,
                       content_pos=content_pos, stopwords=stopwords)
    return emb.embed(segments, batch_size)


def segment(
    emb: Embedder, sources: list[SourceDoc], woi: set[str],
    match_fn: MatchFn | None, *,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: set[str] | frozenset[str] = frozenset(),
) -> list[Segment]:
    """Split each source into sentences and greedily pack under the token cap."""
    segments = build_segments(
        sources, woi, match_fn=match_fn,
        sent_len_fn=emb.token_lengths, max_tokens=emb.max_length,
        content_pos=content_pos, stopwords=stopwords)
    n_occ = sum(len(s.spans) for s in segments)
    n_tok = sum(len(s.doc) for s in sources)
    n_src = len({s.source.id for s in sources})
    print(f"segmented  : {n_tok} tokens ({n_src} sources) -> "
          f"{len(segments)} segments (<= {emb.max_length} tok); {n_occ} occurrences")
    return segments


def pool(word_vectors: WordVectors, target_labels) -> tuple[list[str], ndarray]:
    labels, matrix = vectors.max_pool(word_vectors)
    n_tgt = sum(1 for l in labels if l in target_labels)
    print(f"pooled     : {len(labels)} words ({n_tgt} targets)")
    return labels, matrix


def occurrences(word_vectors: WordVectors, targets, mean: ndarray | None = None):
    """Per-target stacked occurrence vectors (centered with the pooled ``mean``)."""
    return {
        w: (s if mean is None else s - mean)
        for w in targets
        if (s := (np.stack(word_vectors[w]) if word_vectors.get(w) else None)) is not None
    }


def unk_check(emb: Embedder, words: set[str] | frozenset[str]) -> None:
    """Warn if any word we embed loses a character to [UNK]."""
    unk = emb.unk_words(sorted(words))
    if unk:
        print(f"WARNING: {len(unk)} word(s) tokenize to [UNK]:")
        for w, toks in list(unk.items())[:20]:
            print(f"    {w!r} -> {toks}")
    else:
        print(f"UNK check  : OK — all {len(words)} words in-vocab")


def run_artifacts(labels, matrix, targets, word_vectors, mean, out_dir,
                  threshold: float, max_nodes: int) -> None:
    """Opt-in: the heavy PNG/CSV analysis dump, decoupled from the JSON outputs."""
    is_target = np.array([l in targets for l in labels])
    target_occ = occurrences(word_vectors, targets, mean)
    summary = analyze.run_analysis(
        labels, matrix, is_target, target_occ, out_dir,
        threshold, kmeans_k=4, max_nodes=max_nodes)
    print(f"artifacts  : {summary['louvain_communities']} communities -> "
          f"{out_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["mengzi", "sep", "all"], default="all")
    p.add_argument("--artifacts", action="store_true",
                   help="Also write analysis/ PNG+CSV artifacts (decoupled).")
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Cosine edge threshold for the similarity graph.")
    p.add_argument("--min-freq", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    center = not args.no_center
    common = dict(center=center, threshold=args.threshold,
                  max_nodes=args.max_nodes, artifacts=args.artifacts)
    if args.corpus in ("mengzi", "all"):
        print("\n=== Mengzi ===")
        run_mengzi(min_freq=args.min_freq or 5, **common)
    if args.corpus in ("sep", "all"):
        print("\n=== SEP ===")
        run_sep(min_freq=args.min_freq or 10, **common)


if __name__ == "__main__":
    main()
