"""Runner for the SEP English semantic-space pipeline (Task 2, part 1).

Mirrors ``embed/run.py`` but on the English side: a combined Stanford
Encyclopedia of Philosophy corpus (roberta-base embeddings) with the four English
renderings of 仁 / 義 as targets. The vector pooling, centering, and analysis
(``vectors`` / ``analyze``) are shared with the Chinese pipeline unchanged.

Run from ``scripts/``:

    .venv/bin/python -m embed.run_sep --dry-run          # small fetch, cosines only
    .venv/bin/python -m embed.run_sep --center --network knn --knn-k 8 --max-nodes 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import TERMS, Rendering, ANALYSIS
from embeddings import analyze, vectors
from embeddings.model import Embedder
from embeddings.sep_occurrences import (
    build_segments,
    build_vocab,
    fetch_corpus,
    parse_docs,
    MatchFn
)

DEFAULT_MODEL = "roberta-base"
DEFAULT_OUT = ANALYSIS / "sep"


def _load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")


def target_config(args: argparse.Namespace) -> tuple[list[Rendering], MatchFn]:
    renderings = [r for term in TERMS for r in term.renderings]
    if args.terms:
        labels = set(args.terms)
        renderings = [r for r in renderings if r.label in labels]

    def match_fn(lemma: str, pos: str | None = None) -> str | None:
        """Return the canonical rendering label a (lemma, POS) belongs to, or None."""
        for r in renderings:
            if r.matches(lemma, pos):
                return r.label

    return renderings, match_fn


def segment(emb, sentences, woi):
    segments = build_segments(
        sentences, woi, emb.token_lengths, emb.max_length)
    n_occ = sum(len(s.spans) for s in segments)
    n_docs = len({s.chapter for s in segments})
    print(f"segmented  : {len(sentences)} sentences ({n_docs} articles) -> "
          f"{len(segments)} segments (<= {emb.max_length} tok); {n_occ} occurrences")
    return segments


def unk_check(emb: Embedder, words: set[str]):
    unk = emb.unk_words(sorted(words))
    if unk:
        print(f"WARNING: {len(unk)} word(s) tokenize to [UNK]:")
        for w, toks in list(unk.items())[:20]:
            print(f"    {w!r} -> {toks}")
    else:
        print(f"UNK check  : OK — all {len(words)} words in-vocab")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--per-term", type=int, default=12,
                   help="SEP articles fetched per English term (deduped).")
    p.add_argument("--limit-docs", type=int, default=6,
                   help="Dry-run cap on articles parsed.")
    p.add_argument("--terms", nargs="*", default=None,
                   help="Override English targets (default: config / DEFAULT_TERMS).")
    p.add_argument("--min-freq", type=int, default=10,
                   help="Min lemma frequency for the content-word vocabulary.")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--center", action="store_true",
                   help="Mean-center vectors (anisotropy fix; retune --threshold).")
    p.add_argument(
        "--network", choices=["threshold", "knn"], default="threshold")
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--max-nodes", type=int, default=15)
    p.add_argument("--sim-transform", choices=["none", "neglog", "poslog"],
                   default="none")
    p.add_argument("--kmeans-k", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def dry_run(args: argparse.Namespace, targets: list[Rendering], match_fn: MatchFn) -> None:
    docs = fetch_corpus(targets, args.per_term)[: args.limit_docs]
    print(f"[dry-run] articles          : {len(docs)}")
    print(
        f"[dry-run] targets           : {', '.join([t.label for t in targets])}")
    nlp = _load_spacy()
    sentences = parse_docs(docs, nlp, match_fn)
    print(f"[dry-run] sentences         : {len(sentences)}")

    emb = Embedder(args.model)
    print(f"[dry-run] device            : {emb.device_label}")
    unk_check(emb, set([t.label for t in targets]))
    segments = segment(emb, sentences, set(targets))
    if sum(len(s.spans) for s in segments) == 0:
        print("[dry-run] no target occurrences found.")
        return
    by_word = emb.embed(segments, batch_size=args.batch_size)
    labels, matrix = vectors.max_pool(by_word)
    for w in labels:
        print(f"[dry-run]   {w}: {len(by_word[w])} occurrences")
    if len(labels) >= 2:
        sim = cosine_similarity(matrix)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                print(
                    f"[dry-run] cos({labels[i]},{labels[j]}) = {sim[i, j]:.4f}")
    print("[dry-run] OK — SEP fetch + roberta span mapping validated.")


def full_run(args: argparse.Namespace, targets: list[Rendering], match_fn: MatchFn) -> None:
    # ensure that target_labels matches the results from match_fn
    target_labels = frozenset(
        [l for l in [match_fn(t.label) for t in targets] if l is not None])

    docs = fetch_corpus(targets, args.per_term)
    print(f"articles   : {len(docs)} (>= {args.per_term}/term, deduped)")
    nlp = _load_spacy()
    sentences = parse_docs(docs, nlp, match_fn)
    vocab = build_vocab(sentences, target_labels, args.min_freq)
    print(
        f"sentences  : {len(sentences)}  vocab: {len(vocab)} (min_freq={args.min_freq})")

    emb = Embedder(args.model)
    print(f"device     : {emb.device_label}  hidden: {emb.hidden_size}")
    unk_check(emb, vocab)
    segments = segment(emb, sentences, vocab)
    by_word = emb.embed(segments, batch_size=args.batch_size)
    labels, matrix = vectors.max_pool(by_word)
    is_target = np.array([lbl in target_labels for lbl in labels])
    print(f"pooled     : {len(labels)} words ({int(is_target.sum())} targets)")

    mean = None
    if args.center:
        matrix, mean = vectors.center_matrix(matrix)
        print("centered   : subtracted vocab centroid (anisotropy fix)")

    vectors.save_vectors(args.out, labels, matrix,
                         target_labels, by_word, mean=mean)
    target_occ = vectors.load_target_occurrences(args.out)
    if mean is not None:
        target_occ = {w: s - mean for w, s in target_occ.items()}

    summary = analyze.run_analysis(
        labels, matrix, is_target, target_occ,
        args.out, args.threshold, args.kmeans_k,
        method=args.network, knn_k=args.knn_k, sim_transform=args.sim_transform,
        max_nodes=args.max_nodes,
    )
    print("\n=== summary ===")
    for row in summary["cohesion_variance"]:
        print(f"  {row['term']}: n={row['n']} cohesion={row['cohesion']} "
              f"variance={row['variance']}")
    net = (f"knn (k={args.knn_k})" if args.network == "knn"
           else f"threshold {args.threshold}")
    print(f"  network: {net}  centered={args.center}  "
          f"sim_transform={args.sim_transform}")
    print(f"  Louvain communities: {summary['louvain_communities']}")
    print(f"  artifacts written to: {args.out.resolve()}")


def main() -> None:
    args = parse_args()
    targets, match_fn = target_config(args)

    if args.dry_run:
        dry_run(args, targets, match_fn)
    else:
        full_run(args, targets, match_fn)


if __name__ == "__main__":
    main()
