"""Runner for the GujiRoBERTa semantic-space pipeline (Task 1).

Consumes per-chapter passage segmentation (``segpos/chapters/*.jsonl`` from
``xunzi/run.py --unit line``), splits sentences within each passage, and packs
them into <=512-token segments that never cross passage / chapter boundaries.

Run from the ``scripts/`` directory:

    .venv/bin/python -m embed.run --dry-run      # targets-only smoke test
    .venv/bin/python -m embed.run                # full virtue + vocab pass
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import TERMS
from embed import analyze, vectors
from embed.model import DEFAULT_MODEL, Embedder
from embed.occurrences import (
    DEFAULT_CORPUS_DIR,
    Passage,
    Segment,
    build_segments,
    build_vocab,
    load_passages,
)

DEFAULT_OUT = Path("../analysis/task1")


def segment(emb: Embedder, passages: list[Passage], woi: set[str]) -> list[Segment]:
    """Split each passage into sentences and greedily pack under the token cap."""
    segments = build_segments(passages, woi, emb.token_lengths, emb.max_length)
    n_occ = sum(len(s.spans) for s in segments)
    n_chapters = len({p.chapter for p in passages})
    print(f"segmented  : {len(passages)} passages ({n_chapters} chapters) -> "
          f"{len(segments)} segments (<= {emb.max_length} tok); {n_occ} occurrences")
    return segments


def unk_check(emb: Embedder, words: set[str]) -> None:
    """Warn if any word we embed loses a character to [UNK]."""
    unk = emb.unk_words(sorted(words))
    if unk:
        print(f"WARNING: {len(unk)} word(s) tokenize to [UNK]:")
        for w, toks in list(unk.items())[:20]:
            print(f"    {w!r} -> {toks}")
    else:
        print(f"UNK check  : OK — all {len(words)} words in-vocab")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Smoke test: targets only, first --limit passages, no full analysis.")
    p.add_argument("--limit", type=int, default=20,
                   help="Dry-run passage cap (default 20).")
    p.add_argument("--terms", nargs="*", default=None,
                   help="Override target hanzi (default: config.TERMS).")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Cosine edge threshold for the network (tune 0.65-0.85).")
    p.add_argument("--min-freq", type=int, default=5,
                   help="Min token frequency for the content-word vocabulary.")
    p.add_argument("--kmeans-k", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR,
                   help="Directory of per-chapter passage JSONL files.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def target_set(args: argparse.Namespace) -> set[str]:
    return set(args.terms) if args.terms else {t.hanzi for t in TERMS}


def dry_run(args: argparse.Namespace, targets: set[str]) -> None:
    passages = load_passages(args.corpus)[: args.limit]
    print(f"[dry-run] passages scanned  : {len(passages)}")
    print(f"[dry-run] targets           : {sorted(targets)}")

    emb = Embedder(args.model)
    print(f"[dry-run] device            : {emb.device_label}")
    print(f"[dry-run] hidden size       : {emb.hidden_size}")
    unk_check(emb, targets)
    segments = segment(emb, passages, targets)
    if sum(len(s.spans) for s in segments) == 0:
        print("[dry-run] no occurrences found — check corpus / target chars.")
        return
    by_word = emb.embed(segments, batch_size=args.batch_size)
    labels, matrix = vectors.max_pool(by_word)
    print(f"[dry-run] pooled words      : {labels}")
    print(f"[dry-run] vector shape      : {matrix[0].shape}")
    for w in labels:
        print(f"[dry-run]   {w}: {len(by_word[w])} occurrences")
    if len(labels) >= 2:
        sim = cosine_similarity(matrix)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                print(
                    f"[dry-run] cos({labels[i]},{labels[j]}) = {sim[i, j]:.4f}")
    print("[dry-run] OK — span mapping + GPU path validated.")


def full_run(args: argparse.Namespace, targets: set[str]) -> None:
    passages = load_passages(args.corpus)
    vocab = build_vocab(passages, targets, args.min_freq)
    print(f"targets    : {sorted(targets)}")
    print(f"vocab size : {len(vocab)} (min_freq={args.min_freq})")

    emb = Embedder(args.model)
    print(f"device     : {emb.device_label}  hidden: {emb.hidden_size}")
    unk_check(emb, vocab)
    segments = segment(emb, passages, vocab)
    by_word = emb.embed(segments, batch_size=args.batch_size)
    labels, matrix = vectors.max_pool(by_word)
    is_target = np.array([lbl in targets for lbl in labels])
    print(f"pooled     : {len(labels)} words ({int(is_target.sum())} targets)")

    vectors.save_vectors(args.out, labels, matrix, targets, by_word)

    summary = analyze.run_analysis(
        labels, matrix, is_target,
        vectors.load_target_occurrences(args.out),
        args.out, args.threshold, args.kmeans_k,
    )
    print("\n=== summary ===")
    for row in summary["cohesion_variance"]:
        print(f"  {row['term']}: n={row['n']} cohesion={row['cohesion']} "
              f"variance={row['variance']}")
    print(f"  Louvain communities: {summary['louvain_communities']}")
    print(f"  artifacts written to: {args.out.resolve()}")


def main() -> None:
    args = parse_args()
    targets = target_set(args)
    if args.dry_run:
        dry_run(args, targets)
    else:
        full_run(args, targets)


if __name__ == "__main__":
    main()
