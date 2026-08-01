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

Optional ``analysis/`` artifacts (PNGs/CSVs) are decoupled behind ``--artifacts``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
from numpy import ndarray
from networkx import Graph
from spacy.tokens import Doc

from config import TERMS, CTEXT, SEP, EMBEDDINGS, DATA, ANALYSIS
from models import Pipeline, Source, TermData, NetworkData, Embeddings
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

MENGZI_MODEL = "hsc748NLP/GujiRoBERTa_fan"
SEP_MODEL = "roberta-base"


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
    p: Pipeline, label: str, network_sources: list[SourceDoc], meta_source: Source, file_id: str,
    *, match_fn: MatchFn | None,
) -> None:
    """Build ``label``'s PMI network over ``network_sources`` and write it as one
    ``NetworkData`` file ``{label}_{file_id}.json`` (``meta_source`` is the source
    recorded in the file — a chapter, an article, or a whole-corpus stand-in)."""
    net = get_cooccurrence(p, label, network_sources, match_fn=match_fn)
    if net is None:
        print(f"  no co-occurrence for {label} in {meta_source.title}")
    occ = sum(count_occurrences(s.doc, label, match_fn)
              for s in network_sources)
    NetworkData(TermData(label), meta_source, net, occurrences=occ).save_json(
        p.out_dir / f"{label}_{file_id}.json")


# ---------------------------------------------------------------------------
# Embeddings + similarity (share one cosine graph over the whole corpus)
# ---------------------------------------------------------------------------

def save_similarity(
    p: Pipeline, targets: set[str] | frozenset[str], corpus_source: Source, G: Graph,
    *, occ_counts: Counter,
) -> None:
    """Write one ``NetworkData`` file per term: its pruned cosine neighborhood."""
    for target in targets:
        pruned = prune_to_neighborhood(G, target, p.max_network_nodes)
        if pruned is None:
            print(f"  {target} absent from similarity graph")
        NetworkData(TermData(target), corpus_source, pruned,
                    occurrences=int(occ_counts.get(target, 0))).save_json(
            p.out_dir / f"{target}_embeds.json")


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

    # --- co-occurrence: full corpus + each chapter, per target ---
    for target in targets:
        save_cooccurrence(p, target, [full], mengzi, "mengzi", match_fn=None)
        for ch in chapters:
            save_cooccurrence(
                p, target, [ch], ch.source, ch.source.id, match_fn=None)
    sw.lap("co-occurrence")

    # --- embeddings over the whole corpus (chapters; full would double-count) ---
    embedder = Embedder(p.model)
    print(
        f"device     : {embedder.device_label}  hidden: {embedder.hidden_size}")
    pooler, doc_freq, n_docs = embed(p, embedder, chapters, targets,
                                     keep=targets if artifacts else frozenset())
    labels, matrix = pool(pooler, targets)
    mean = None
    if p.center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")
    sw.lap("embedding")

    G, _, community_map = analyze.build_networks(
        labels, matrix, method=p.sim_network, quantile=p.quantile, knn_k=p.knn_k,
        sim_transform=p.sim_transform, resolution=p.resolution)
    sw.lap("networks")

    occ_counts = Counter(
        t.lemma_ for t in full.doc if t.text in targets or t.lemma_ in targets)
    save_similarity(p, targets, mengzi, G, occ_counts=occ_counts)

    reduced = vectors.reduce_vectors(matrix, p.reduce_to_dims)
    Embeddings.from_matrix(mengzi, labels, reduced, targets, community_map,
                           doc_freq=doc_freq, documents=n_docs, graph=G) \
        .save_json(EMBEDDINGS / "mengzi.json")
    print(f"embeddings : {len(labels)} nodes -> {EMBEDDINGS / 'mengzi.json'}")
    sw.lap("similarity+export")

    if artifacts:
        run_artifacts(labels, matrix, targets, pooler, mean,
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

    # --- co-occurrence: each rendering over its combined search + each article ---
    # (also parses + caches every article, reused for the embedding space below)
    for ts in searches:
        label, articles = ts.term.label, ts.search.articles
        adocs = [source_doc(a) for a in articles]
        save_cooccurrence(p, label, adocs, ts.search,
                          "combined", match_fn=match_fn)
        for sd in adocs:
            save_cooccurrence(
                p, label, [sd], sd.source, sd.source.id, match_fn=match_fn)
    sw.lap("parse+co-occurrence")

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
    mean = None
    if p.center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")
    sw.lap("embedding")

    G, _, community_map = analyze.build_networks(
        labels, matrix, method=p.sim_network, quantile=p.quantile, knn_k=p.knn_k,
        sim_transform=p.sim_transform, resolution=p.resolution)
    sw.lap("networks")

    occ_counts = Counter(
        lbl for sd in combined for t in sd.doc
        if (lbl := match_fn(t.lemma_.lower(), t.pos_)) is not None)
    save_similarity(p, labels_by_target, SEP_CORPUS, G, occ_counts=occ_counts)

    reduced = vectors.reduce_vectors(matrix, p.reduce_to_dims)
    Embeddings.from_matrix(SEP_CORPUS, labels, reduced, labels_by_target,
                           community_map, doc_freq=doc_freq, documents=n_docs,
                           graph=G).save_json(EMBEDDINGS / "sep.json")
    print(f"embeddings : {len(labels)} nodes -> {EMBEDDINGS / 'sep.json'}")
    sw.lap("similarity+export")

    if artifacts:
        run_artifacts(labels, matrix, labels_by_target, pooler, mean,
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
    doc-freq fields and the ``max_doc_freq`` cap). The embedder yields occurrences
    one batch at a time; we fold each into a running max-pool here, so the whole
    corpus's occurrence vectors are never all resident at once. ``keep`` words
    additionally retain their full occurrence stacks (cohesion under ``--artifacts``)."""
    pos, stop = _to_set(p.content_pos), p.stopwords or set()
    vocab = build_vocab(sources, p.min_freq, match_fn, content_pos=pos, stopwords=stop)
    doc_freq = document_frequencies(sources, match_fn, content_pos=pos, stopwords=stop)
    n_docs = len(sources)

    if p.max_doc_freq is not None:
        cap = p.max_doc_freq if p.max_doc_freq > 1 else p.max_doc_freq * n_docs
        before = len(vocab)
        vocab = {w for w in vocab if doc_freq.get(w, 0) <= cap}
        print(f"doc-freq   : cap {cap:.0f}/{n_docs} docs -> dropped "
              f"{before - len(vocab)} of {before} vocab words")

    vocab |= set(target_labels)     # targets always kept, regardless of the cap
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
                  network: SimMethod, quantile: float, knn_k: int,
                  sim_transform: analyze.SimTransform, resolution: float,
                  max_nodes: int) -> None:
    """Opt-in: the heavy PNG/CSV analysis dump, decoupled from the JSON outputs."""
    is_target = np.array([l in targets for l in labels])
    target_occ = {w: (s if mean is None else s - mean)
                  for w, s in pooler.stacks().items()}
    summary = analyze.run_analysis(
        labels, matrix, is_target, target_occ, out_dir, quantile, kmeans_k=4,
        method=network, knn_k=knn_k, sim_transform=sim_transform,
        resolution=resolution, max_nodes=max_nodes)
    print(f"artifacts  : {summary['louvain_communities']} communities -> "
          f"{out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Master index (scans the written files so paths reflect what's on disk)
# ---------------------------------------------------------------------------

def _source_entry(src: dict, web: str) -> dict:
    """One master-index `Source`: a file's recorded provenance + its web `data` path."""
    return {
        "id": src.get("id"), "url": src.get("url") or "",
        "title": src.get("title") or "", "description": src.get("description"),
        "occurrences": src.get("occurrences"), "data": web,
    }


def _scan_networks(out_dir, web_prefix: str) -> dict[str, dict]:
    """Group a directory's NetworkData files by term label into
    ``{label: {"variants", "total", "cooccurrence": [Source], "similarity": [Source]}}``.
    Co-occurrence files are ``{label}_{source.id}.json``; the similarity file is
    ``{label}_embeds.json`` (whole corpus). Each entry is a master-index `Source`
    (provenance + `data` path + per-source `occurrences`). Non-NetworkData JSON is
    skipped."""
    by_term: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        term = obj.get("term") if isinstance(obj, dict) else None
        if not (isinstance(term, dict) and isinstance(obj.get("source"), dict)):
            continue
        label = term["label"]
        entry = by_term.setdefault(
            label, {"variants": term.get("variants", []), "total": 0,
                    "cooccurrence": [], "similarity": []})
        src = _source_entry(obj["source"], f"{web_prefix}/{path.name}")
        occ = src["occurrences"] or 0
        if path.name == f"{label}_embeds.json":
            entry["similarity"].append(src)
            entry["total"] = occ or entry["total"]        # similarity = whole corpus
        else:
            entry["cooccurrence"].append(src)
            if path.name.endswith(("_mengzi.json", "_combined.json")):
                entry["total"] = occ or entry["total"]    # whole-corpus co-occurrence
    return by_term


def _full_first(sources: list[dict]) -> list[dict]:
    """Order co-occurrence sources with the whole-corpus one first (by ``data`` name,
    ``*_mengzi.json`` / ``*_combined.json``)."""
    def is_full(s: dict) -> bool:
        d = s.get("data") or ""
        return d.endswith("_mengzi.json") or d.endswith("_combined.json")
    return sorted(sources, key=lambda s: (not is_full(s), str(s.get("id"))))


def _embedding_source(corpus: str, total: int) -> dict:
    """The corpus's embedding dataset as a master-index `Source` (`data` → its JSON,
    `occurrences` = the term's whole-corpus total). Reuses the file's own provenance."""
    path = EMBEDDINGS / f"{corpus}.json"
    src = (json.loads(path.read_text(encoding="utf-8")).get("source") or {}) \
        if path.exists() else {}
    entry = _source_entry(src, f"/embeddings/{corpus}.json")
    entry["id"] = entry["id"] or corpus
    entry["title"] = entry["title"] or corpus
    entry["occurrences"] = total
    return entry


def build_master(out_path=None) -> dict:
    """Assemble the master index over whatever the pipeline has written: per term,
    the Chinese (Mengzi) side and one side per English rendering. Each side lists its
    co-occurrence / similarity / embedding sources (with `data` web paths), scanned
    from the output dirs so paths mirror what is on disk."""
    out_path = out_path or (DATA / "terms.json")
    ctext = _scan_networks(CTEXT, "/ctext")
    sep = _scan_networks(SEP, "/sep")

    empty = {"variants": [], "total": 0, "cooccurrence": [], "similarity": []}

    def side(label: str, corpus: str, scan: dict) -> dict:
        e = scan.get(label, empty)
        return {
            "corpus": corpus,
            "term": {"label": label, "variants": e["variants"]},
            "occurrences": e["total"],
            "embeddings": [_embedding_source(corpus, e["total"])],
            "similarity": e["similarity"],
            "cooccurrence": _full_first(e["cooccurrence"]),
        }

    terms = []
    for term in TERMS:
        terms.append({
            "hanzi": term.hanzi,
            "renderings": list(term.english),
            "chinese": side(term.hanzi, "mengzi", ctext),
            "english": [side(r.label, "sep", sep) for r in term.renderings],
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
