"""What word recombination does to the Mengzi — the review surface for it.

Recombining subword tokens (``corpus.recombine``) is the one place in the Chinese
pipeline where a machine decision silently changes what counts as a word, so it
needs to be readable rather than trusted. This prints, for the corpus as a whole:

  A. every merged word type with its token count and the POS it inherits, so the
     list can be read end to end as a philological check;
  B. every group a guard rejected, by reason — most importantly the ones skipped
     for containing a target term (仁 義 禮 智 信), which are logged rather than
     swallowed;
  C. the characters that lose the most occurrences to a merge, since that is what
     shifts the vocabulary out from under ``Pipeline.min_freq``.

Anything the report shows as wrong is fixed in ``scripts/data/merge_overrides.json``
(``merge`` to add a word, ``never_merge`` to drop one) rather than by widening the
relation set — see ``notes/multi-character-tokenization.md``.

Run:  scripts/.venv/bin/python -m tools.merge_report [--pos NOUN,PROPN] [--min-count 1]
"""

from __future__ import annotations

import argparse
from collections import Counter

from config import MENGZI_CONLLU, MENGZI_PIPELINE, TERMS
from corpus.conllu import load_conllu
from corpus.parse import mengzi_merge_config
from corpus.recombine import MergeConfig, MergeReport


def build_report(merge: MergeConfig) -> tuple[MergeReport, Counter[str]]:
    """Merge every chapter and return the report plus per-character loss counts
    (how many occurrences of each character were absorbed into some word)."""
    report = MergeReport()
    absorbed: Counter[str] = Counter()
    for _ in load_conllu(MENGZI_CONLLU, merge=merge, report=report):
        pass
    for word, count in report.merged.items():
        for char in word:
            absorbed[char] += count
    return report, absorbed


def word_pos(report: MergeReport, word: str) -> str:
    """The POS a word inherits, or ``NOUN/PROPN`` when occurrences disagree."""
    counts = report.pos_by_word.get(word)
    return "/".join(pos for pos, _ in counts.most_common()) if counts else "_"


def print_merged(report: MergeReport, pos_filter: set[str] | None,
                 min_count: int) -> None:
    # The POS a merged token inherits decides whether the pipeline's content_pos
    # filter keeps it at all, so it is reported alongside the count.
    print(f"\n=== merged words: {report.types} types / {report.tokens} tokens ===")
    print("by inherited POS: " + ", ".join(
        f"{pos} {n}" for pos, n in report.root_pos.most_common()))
    if pos_filter:
        print(f"(listing only {'/'.join(sorted(pos_filter))})")
    print()
    for word, count in report.merged.most_common():
        if count < min_count:
            continue
        pos = word_pos(report, word)
        if pos_filter and not (set(pos.split("/")) & pos_filter):
            continue
        forced = " [override]" if report.forced.get(word) else ""
        print(f"  {word:<8} {count:>4}  {pos}{forced}")


def print_sources(report: MergeReport) -> None:
    """Which source proposed each merge. The reason for a second boundary source
    is the words the UD relations cannot reach, so `lexicon` alone is the number
    that says whether it earned its place."""
    print("\n=== merges by source ===")
    for src, words in sorted(report.by_source.items(),
                             key=lambda kv: -sum(kv[1].values())):
        top = " ".join(w for w, _ in words.most_common(10))
        print(f"  {src:<16} {sum(words.values()):>5} tokens / "
              f"{len(words):>3} types   {top}")


def print_skipped(report: MergeReport) -> None:
    print("\n=== groups a guard rejected ===")
    if not report.skipped:
        print("  (none)")
        return
    for reason, words in sorted(report.skipped.items()):
        total = sum(words.values())
        print(f"\n  {reason} — {total} group(s)")
        for word, count in words.most_common():
            print(f"      {word:<8} {count:>4}")


def print_absorbed(absorbed: Counter[str], report: MergeReport,
                   limit: int) -> None:
    print(f"\n=== characters losing the most occurrences to a merge "
          f"(top {limit}) ===")
    print("  these no longer count as standalone nodes — re-check "
          "Pipeline.min_freq\n")
    for char, count in absorbed.most_common(limit):
        into = [w for w in report.merged if char in w]
        into.sort(key=lambda w: -report.merged[w])
        shown = " ".join(into[:6]) + (" …" if len(into) > 6 else "")
        print(f"  {char}  {count:>4}  -> {shown}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pos", default=None,
                   help="Comma-separated UPOS filter for the merged-word listing "
                        "(e.g. NOUN,PROPN). Counts are always over everything.")
    p.add_argument("--min-count", type=int, default=1, dest="min_count",
                   help="Hide merged words occurring fewer than N times.")
    p.add_argument("--absorbed", type=int, default=25,
                   help="How many characters to list in the absorption table.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    targets = frozenset(t.hanzi for t in TERMS)
    merge = mengzi_merge_config(MENGZI_PIPELINE, targets)
    if merge is None:
        print("MENGZI_PIPELINE.merge_deps is None — recombination is off, "
              "nothing to report. Set it in config.py to enable.")
        return

    overrides = merge.overrides
    print(f"relations  : {', '.join(sorted(merge.deps)) or '(none)'}")
    print(f"targets    : {' '.join(sorted(targets))} (never merged into a word)")
    print(f"overrides  : {len(overrides.merge)} merge, "
          f"{len(overrides.never_merge)} never-merge")
    lex = merge.lexicon_path
    print(f"lexicon    : {lex if lex and lex.is_file() else 'none'}")

    report, absorbed = build_report(merge)
    print_merged(report, set(args.pos.split(",")) if args.pos else None,
                 args.min_count)
    print_sources(report)
    print_skipped(report)
    print_absorbed(absorbed, report, args.absorbed)


if __name__ == "__main__":
    main()
