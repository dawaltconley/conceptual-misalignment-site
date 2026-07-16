"""Runner for the GujiRoBERTa semantic-space pipeline (Task 1).

Run from the ``scripts/`` directory:

    .venv/bin/python -m embed.run --dry-run      # 仁-only smoke test, first 50 sents
    .venv/bin/python -m embed.run                # full four-virtue + vocab pass
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
    DEFAULT_CORPUS,
    Passage,
    build_passages,
    build_vocab,
    load_sentences,
)

DEFAULT_OUT = Path("../analysis/task1")


def pack(emb: Embedder, sentences: list[dict], woi: set[str]) -> list[Passage]:
    """Token-length each sentence, then greedily pack under the model cap."""
    lens = emb.token_lengths([s["sentence"] for s in sentences])
    passages = build_passages(sentences, woi, lens, emb.max_length)
    n_occ = sum(len(p.spans) for p in passages)
    print(f"packed     : {len(sentences)} sentences -> {len(passages)} passages "
          f"(<= {emb.max_length} tok); {n_occ} occurrences")
    return passages


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
                   help="Smoke test: targets only, first --limit sentences, no full analysis.")
    p.add_argument("--limit", type=int, default=50,
                   help="Dry-run sentence cap (default 50).")
    p.add_argument("--terms", nargs="*", default=None,
                   help="Override target hanzi (default: config.TERMS).")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Cosine edge threshold for the network (tune 0.65-0.85).")
    p.add_argument("--min-freq", type=int, default=5,
                   help="Min token frequency for the content-word vocabulary.")
    p.add_argument("--kmeans-k", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def target_set(args: argparse.Namespace) -> set[str]:
    return set(args.terms) if args.terms else {t.hanzi for t in TERMS}


def dry_run(args: argparse.Namespace, targets: set[str]) -> None:
    sentences = load_sentences(args.corpus)[: args.limit]
    print(f"[dry-run] sentences scanned : {len(sentences)}")
    print(f"[dry-run] targets           : {sorted(targets)}")

    emb = Embedder(args.model)
    print(f"[dry-run] device            : {emb.device_label}")
    print(f"[dry-run] hidden size       : {emb.hidden_size}")
    unk_check(emb, targets)
    passages = pack(emb, sentences, targets)
    if sum(len(p.spans) for p in passages) == 0:
        print("[dry-run] no occurrences found — check corpus / target chars.")
        return
    by_word = emb.embed(passages, batch_size=args.batch_size)
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
    sentences = load_sentences(args.corpus)
    vocab = build_vocab(sentences, targets, args.min_freq)
    print(f"targets    : {sorted(targets)}")
    print(f"vocab size : {len(vocab)} (min_freq={args.min_freq})")

    emb = Embedder(args.model)
    print(f"device     : {emb.device_label}  hidden: {emb.hidden_size}")
    unk_check(emb, vocab)
    passages = pack(emb, sentences, vocab)
    by_word = emb.embed(passages, batch_size=args.batch_size)
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
