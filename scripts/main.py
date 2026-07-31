"""WIP single pipeline: emit every JSON file the website needs, from one place.

Temporary scratch module (badly named, badly organized on purpose) — the plan is
to get one corpus working end-to-end here for a clean diff, then split into
``cli/main.py`` + shared modules. Outputs are shaped by the ``models.py`` dataclasses
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
import json
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
from numpy import ndarray
from networkx import Graph
from spacy.tokens import Doc

import config
import models
from config import TERMS, CTEXT, SEP, EMBEDDINGS
from text.chinese import STOPWORDS as CHINESE_STOPWORDS
from corpus.build import build_chinese_corpus, build_english_corpus
from corpus.parse import parse_sep_article, parse_mengzi_chapter
from embeddings import analyze, vectors
from embeddings.model import Embedder
from embeddings.occurrences import (
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


def count_occurrences(doc, label: str, match_fn: MatchFn | None) -> int:
    """How many tokens in ``doc`` belong to ``label`` (via ``match_fn`` if given,
    else exact-lemma). Used for the ``TermData.occurrences`` display count."""
    if match_fn is not None:
        return sum(1 for t in doc if match_fn(t.lemma_.lower(), t.pos_) == label)
    return sum(1 for t in doc if t.lemma_ == label)


def save_cooccurrence(
    label: str, network_sources: list[SourceDoc], meta_source, file_id: str,
    out_dir, *, min_freq: int, match_fn: MatchFn | None,
    content_pos: set[str] | None, stopwords, max_nodes: int,
) -> None:
    """Build ``label``'s PMI network over ``network_sources`` and write it as one
    ``NetworkData`` file ``{label}_{file_id}.json`` (``meta_source`` is the source
    recorded in the file — a chapter, an article, or a whole-corpus stand-in)."""
    net = get_cooccurrence(
        label, network_sources, min_freq, match_fn=match_fn,
        content_pos=content_pos, stopwords=stopwords, max_nodes=max_nodes)
    if net is None:
        print(f"  no co-occurrence for {label} in {meta_source.title}")
    occ = sum(count_occurrences(s.doc, label, match_fn) for s in network_sources)
    models.NetworkData(models.TermData(label, occ), meta_source, net).save_json(
        out_dir / f"{label}_{file_id}.json")


# ---------------------------------------------------------------------------
# Embeddings + similarity (share one cosine graph over the whole corpus)
# ---------------------------------------------------------------------------

def save_similarity(
    targets, corpus_source, G: Graph, out_dir, *,
    max_nodes: int, occ_counts: Counter,
) -> None:
    """Write one ``NetworkData`` file per term: its pruned cosine neighborhood."""
    for target in targets:
        pruned = prune_to_neighborhood(G, target, max_nodes)
        if pruned is None:
            print(f"  {target} absent from similarity graph")
        term = models.TermData(target, int(occ_counts.get(target, 0)))
        models.NetworkData(term, corpus_source, pruned).save_json(
            out_dir / f"{target}_embeds.json")


# ---------------------------------------------------------------------------
# Corpus runners
# ---------------------------------------------------------------------------

def run_mengzi(
    *,
    min_freq: int = 5,
    center: bool = True,
    network: str = "knn",
    threshold: float = 0.3,
    knn_k: int = 8,
    sim_transform: analyze.SimTransform = "neglog",
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
    for target in targets:
        save_cooccurrence(target, [full], mengzi, "mengzi", CTEXT,
                          min_freq=min_freq, match_fn=None,
                          content_pos=content_pos, stopwords=stopwords,
                          max_nodes=max_nodes)
        for ch in chapters:
            save_cooccurrence(target, [ch], ch.source, ch.source.id, CTEXT,
                              min_freq=min_freq, match_fn=None,
                              content_pos=content_pos, stopwords=stopwords,
                              max_nodes=max_nodes)

    # --- embeddings over the whole corpus (chapters; full would double-count) ---
    embedder = Embedder(MENGZI_MODEL)
    print(f"device     : {embedder.device_label}  hidden: {embedder.hidden_size}")
    pooler = embed(
        embedder, chapters, targets, min_freq=min_freq,
        content_pos=content_pos, stopwords=stopwords, batch_size=batch_size,
        keep=targets if artifacts else frozenset())
    labels, matrix = pool(pooler, targets)
    mean = None
    if center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")

    G, _, community_map = analyze.build_networks(
        labels, matrix, method=network, threshold=threshold, knn_k=knn_k,
        sim_transform=sim_transform)

    occ_counts = Counter(t.lemma_ for t in full.doc)
    save_similarity(targets, mengzi, G, CTEXT,
                    max_nodes=max_nodes, occ_counts=occ_counts)

    reduced = models.reduce_vectors(matrix, reduce_to_dims)
    models.Embeddings.from_matrix(mengzi, labels, reduced, targets, community_map) \
        .save_json(EMBEDDINGS / "mengzi.json")
    print(f"embeddings : {len(labels)} nodes -> {EMBEDDINGS / 'mengzi.json'}")

    if artifacts:
        run_artifacts(labels, matrix, targets, pooler, mean,
                      config.ANALYSIS / "mengzi", network, threshold, knn_k,
                      sim_transform, max_nodes)


SEP_CORPUS = SimpleNamespace(
    id="sep", url="https://plato.stanford.edu/",
    title="SEP (combined)",
    description="Combined SEP corpus for the English renderings of 仁 / 義")


def _sep_slug(source) -> str:
    """Filesystem-safe id for a SEP article (its ``.../entries/<slug>/`` name)."""
    parts = [p for p in str(getattr(source, "url", "") or "").split("/") if p]
    slug = parts[-1] if parts else str(getattr(source, "id", "article"))
    return re.sub(r"[^\w-]+", "-", slug).strip("-") or "article"


def run_sep(
    *,
    per_term: int = 12,
    min_freq: int = 10,
    center: bool = True,
    network: str = "knn",
    threshold: float = 0.3,
    knn_k: int = 8,
    sim_transform: analyze.SimTransform = "neglog",
    reduce_to_dims: int = 50,
    max_nodes: int = 15,
    batch_size: int = 16,
    artifacts: bool = False,
) -> None:
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

    # --- co-occurrence: each rendering over its combined search + each article ---
    for ts in searches:
        label, articles = ts.term.label, ts.search.articles
        adocs = [source_doc(a) for a in articles]
        save_cooccurrence(label, adocs, ts.search, "combined", SEP,
                          min_freq=min_freq, match_fn=match_fn,
                          content_pos=CONTENT_POS, stopwords=frozenset(),
                          max_nodes=max_nodes)
        for sd in adocs:
            save_cooccurrence(label, [sd], sd.source, _sep_slug(sd.source), SEP,
                              min_freq=min_freq, match_fn=match_fn,
                              content_pos=CONTENT_POS, stopwords=frozenset(),
                              max_nodes=max_nodes)

    # --- one combined embedding space over every (deduped) article ---
    combined = list(doc_cache.values())
    print(f"parsed     : {len(combined)} SEP articles")
    embedder = Embedder(SEP_MODEL)
    print(f"device     : {embedder.device_label}  hidden: {embedder.hidden_size}")
    pooler = embed(
        embedder, combined, labels_by_target, min_freq=min_freq,
        match_fn=match_fn, content_pos=CONTENT_POS, batch_size=batch_size,
        keep=labels_by_target if artifacts else frozenset())
    labels, matrix = pool(pooler, labels_by_target)
    mean = None
    if center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")

    G, _, community_map = analyze.build_networks(
        labels, matrix, method=network, threshold=threshold, knn_k=knn_k,
        sim_transform=sim_transform)
    occ_counts = Counter(
        lbl for sd in combined for t in sd.doc
        if (lbl := match_fn(t.lemma_.lower(), t.pos_)) is not None)
    save_similarity(labels_by_target, SEP_CORPUS, G, SEP,
                    max_nodes=max_nodes, occ_counts=occ_counts)

    reduced = models.reduce_vectors(matrix, reduce_to_dims)
    models.Embeddings.from_matrix(SEP_CORPUS, labels, reduced, labels_by_target,
                               community_map).save_json(EMBEDDINGS / "sep.json")
    print(f"embeddings : {len(labels)} nodes -> {EMBEDDINGS / 'sep.json'}")

    if artifacts:
        run_artifacts(labels, matrix, labels_by_target, pooler, mean,
                      config.ANALYSIS / "sep", network, threshold, knn_k,
                      sim_transform, max_nodes)


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
    keep: set[str] | frozenset[str] = frozenset(),
) -> vectors.Pooler:
    """Vocab -> segments -> streaming max-pool of per-occurrence span vectors.

    The embedder yields occurrences one batch at a time; we fold each into a
    running max-pool here, so the whole corpus's occurrence vectors are never all
    resident at once (the memory that OOM'd on the large SEP corpus). ``keep``
    words additionally retain their full occurrence stacks (for cohesion under
    ``--artifacts``); pass none to keep only the running maxes."""
    vocab = build_vocab(sources, min_freq, match_fn,
                        content_pos=content_pos, stopwords=stopwords)
    vocab |= set(target_labels)
    unk_check(emb, target_labels)
    segments = segment(emb, sources, vocab, match_fn,
                       content_pos=content_pos, stopwords=stopwords)
    pooler = vectors.Pooler(keep=set(keep))
    for word, vec in emb.embed(segments, batch_size):
        pooler.add(word, vec)
    return pooler


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


def pool(pooler: vectors.Pooler, target_labels) -> tuple[list[str], ndarray]:
    labels, matrix = pooler.matrix()
    n_tgt = sum(1 for l in labels if l in target_labels)
    print(f"pooled     : {len(labels)} words ({n_tgt} targets)")
    return labels, matrix


def unk_check(emb: Embedder, words: set[str] | frozenset[str]) -> None:
    """Warn if any word we embed loses a character to [UNK]."""
    unk = emb.unk_words(sorted(words))
    if unk:
        print(f"WARNING: {len(unk)} word(s) tokenize to [UNK]:")
        for w, toks in list(unk.items())[:20]:
            print(f"    {w!r} -> {toks}")
    else:
        print(f"UNK check  : OK — all {len(words)} words in-vocab")


def run_artifacts(labels, matrix, targets, pooler: vectors.Pooler, mean, out_dir,
                  network: str, threshold: float, knn_k: int,
                  sim_transform: analyze.SimTransform, max_nodes: int) -> None:
    """Opt-in: the heavy PNG/CSV analysis dump, decoupled from the JSON outputs."""
    is_target = np.array([l in targets for l in labels])
    target_occ = {w: (s if mean is None else s - mean)
                  for w, s in pooler.stacks().items()}
    summary = analyze.run_analysis(
        labels, matrix, is_target, target_occ, out_dir, threshold, kmeans_k=4,
        method=network, knn_k=knn_k, sim_transform=sim_transform, max_nodes=max_nodes)
    print(f"artifacts  : {summary['louvain_communities']} communities -> "
          f"{out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Master index (scans the written files so paths reflect what's on disk)
# ---------------------------------------------------------------------------

def _scan_networks(out_dir, web_prefix: str) -> dict[str, dict]:
    """Group a directory's NetworkData files by term label into
    ``{label: {"sources": [{id,title,cooccurrence}], "similarity": path|None}}``.
    Co-occurrence files are ``{label}_{source.id}.json``; similarity is
    ``{label}_embeds.json``. Legacy/unrelated JSON (no ``term``/``source`` dict)
    is skipped."""
    by_term: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        term = obj.get("term") if isinstance(obj, dict) else None
        if not (isinstance(term, dict) and isinstance(obj.get("source"), dict)):
            continue
        label = term["label"]
        entry = by_term.setdefault(label, {"sources": [], "similarity": None})
        web = f"{web_prefix}/{path.name}"
        if path.name == f"{label}_embeds.json":
            entry["similarity"] = web
        else:
            src = obj["source"]
            entry["sources"].append(
                {"id": src.get("id"), "title": src.get("title"),
                 "cooccurrence": web})
    return by_term


def _full_first(sources: list[dict]) -> list[dict]:
    """Order sources with the whole-corpus one first (identified by its file name,
    ``*_mengzi.json`` / ``*_combined.json`` — its source.id may be a URL)."""
    def is_full(s: dict) -> bool:
        p = s.get("cooccurrence", "")
        return p.endswith("_mengzi.json") or p.endswith("_combined.json")
    return sorted(sources, key=lambda s: (not is_full(s), str(s.get("id"))))


def build_master(out_path=None) -> dict:
    """Assemble the master index over whatever the pipeline has written: per term,
    the Chinese (Mengzi) side and its English renderings, each with its source
    file paths, similarity network, and corpus embedding. Scans the output dirs so
    paths mirror what is actually on disk."""
    out_path = out_path or (config.DATA / "terms.json")
    ctext = _scan_networks(CTEXT, "/ctext")
    sep = _scan_networks(SEP, "/sep")

    def side(entry, corpus, embeddings):
        return {
            "corpus": corpus,
            "embeddings": embeddings,
            "similarity": entry["similarity"],
            "sources": _full_first(entry["sources"]),
        }

    empty = {"sources": [], "similarity": None}
    terms = []
    for term in TERMS:
        english = [
            {"label": r.label,
             **side(sep.get(r.label, empty), "sep", "/embeddings/sep.json")}
            for r in term.renderings
        ]
        terms.append({
            "hanzi": term.hanzi,
            "renderings": list(term.english),
            "chinese": side(ctext.get(term.hanzi, empty),
                            "mengzi", "/embeddings/mengzi.json"),
            "english": english,
        })

    master = {"terms": terms}
    out_path.write_text(json.dumps(master, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    n_src = sum(len(t["chinese"]["sources"])
                + sum(len(e["sources"]) for e in t["english"]) for t in terms)
    print(f"master     : {len(terms)} terms, {n_src} source files -> {out_path}")
    return master


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["mengzi", "sep", "all"], default="all")
    p.add_argument("--artifacts", action="store_true",
                   help="Also write analysis/ PNG+CSV artifacts (decoupled).")
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--network", choices=["knn", "threshold"], default="knn",
                   help="Similarity graph: relative kNN neighborhoods (default) "
                        "or an absolute cosine threshold.")
    p.add_argument("--knn-k", type=int, default=8,
                   help="Neighbors per node when --network knn.")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Cosine edge threshold when --network threshold.")
    p.add_argument("--sim-transform", choices=["none", "neglog", "poslog"],
                   default="neglog", dest="sim_transform",
                   help="Reweight cosine before the similarity graph: neglog "
                        "expands high similarities, poslog compresses them.")
    p.add_argument("--per-term", type=int, default=12, dest="per_term",
                   help="SEP articles fetched per English rendering (caps corpus "
                        "size / embedding memory).")
    p.add_argument("--min-freq", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=15)
    p.add_argument("--master-only", action="store_true",
                   help="Skip the corpora; just rebuild the master index from the "
                        "files already on disk.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.master_only:
        center = not args.no_center
        common = dict(center=center, network=args.network,
                      threshold=args.threshold, knn_k=args.knn_k,
                      sim_transform=args.sim_transform,
                      max_nodes=args.max_nodes, artifacts=args.artifacts)
        if args.corpus in ("mengzi", "all"):
            print("\n=== Mengzi ===")
            run_mengzi(min_freq=args.min_freq or 5, **common)
        if args.corpus in ("sep", "all"):
            print("\n=== SEP ===")
            run_sep(min_freq=args.min_freq or 10, per_term=args.per_term, **common)
    print("\n=== Master index ===")
    build_master()


if __name__ == "__main__":
    main()
