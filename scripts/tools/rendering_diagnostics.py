"""Audit what every ``Rendering`` in ``config.TERMS`` actually matched — and,
more usefully, what it *nearly* matched and lost.

The pipeline guard (``renderings.check_coverage``) only fires when a rendering
matches nothing at all. That is the loud case. This is the quiet one: a
rendering can be matching fine overall while the lemmatizer quietly diverts a
slice of its occurrences elsewhere. The signature of both is the same — a token
whose **surface** form matches the rendering's patterns but whose **lemma** does
not — so this reports it per rendering with the losing lemma named, which is
what you need to write the ``scripts/lemmas/english.conf`` line.

Read the output as candidates, not as a fix list. The conf table is deliberately
"measured, not guessed" (see ``notes/spacy-lemma-exceptions.md``), and the loss
is sometimes *correct*: 55 of the shadowed hits on 義's ``meaning`` are the verb
("meaning that p is true") going to lemma ``mean``, which is exactly where it
belongs. Check the POS column before adding anything.

Run:  scripts/.venv/bin/python scripts/tools/rendering_diagnostics.py
      scripts/.venv/bin/python scripts/tools/rendering_diagnostics.py --per-term 4
      scripts/.venv/bin/python scripts/tools/rendering_diagnostics.py --out audit.md
"""

from __future__ import annotations
from corpus.parse import parse_sep_article
from corpus.build import build_english_corpus
from renderings import NOTE, RenderingAudit
import config

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def audit_corpus(per_term: int, min_freq: int = 1) -> tuple[list[RenderingAudit], int]:
    """Full audit over the SEP corpus: ``(audits, n_articles)``.

    One pass over every parsed article, testing each token's lemma and (only
    when they differ) its surface form against the whole rendering list, with
    the same first-wins semantics ``run_sep`` uses.
    """
    renderings = [r for term in config.TERMS for r in term.renderings]

    def label_for(word: str, pos: str | None) -> str | None:
        for r in renderings:
            if r.matches(word, pos):
                return r.label
        return None

    searches = build_english_corpus(per_term, min_freq=min_freq)
    # Dedupe: one article can be a search result for several renderings.
    articles = {a.url: a for ts in searches for a in ts.search.articles}
    print(f"\nparsing {len(articles)} articles...")

    matched: Counter[str] = Counter()
    shadowed: dict[str, Counter[tuple[str, str, str]]] = {
        r.label: Counter() for r in renderings}
    for i, article in enumerate(articles.values(), 1):
        if i % 25 == 0:
            print(f"  {i}/{len(articles)}")
        for token in parse_sep_article(article):
            lemma, surface = token.lemma_.lower(), token.text.lower()
            by_lemma = label_for(lemma, token.pos_)
            if by_lemma is not None:
                matched[by_lemma] += 1
            if surface == lemma:
                continue  # nothing for the lemmatizer to have moved
            by_surface = label_for(surface, token.pos_)
            if by_surface is not None and by_lemma is None:
                shadowed[by_surface][(surface, lemma, token.pos_)] += 1

    audits = [RenderingAudit(label=r.label, patterns=r.patterns,
                             matched=matched[r.label],
                             shadowed=shadowed[r.label])
              for r in renderings]
    return audits, len(articles)


def report(audits: list[RenderingAudit], n_articles: int) -> list[str]:
    lines = ["# Rendering coverage audit", "",
             f"{len(audits)} renderings over {n_articles} SEP articles. "
             f"*Shadowed* = tokens whose surface form matches the rendering but "
             f"whose lemma does not; see [[spacy-lemma-exceptions]].", ""]

    empty = [a for a in audits if a.empty]
    print(f"\n=== {len(empty)} rendering(s) matched NOTHING ===")
    lines += [f"## Matched nothing ({len(empty)})", ""]
    if empty:
        for a in empty:
            print(a.diagnosis())
            lines.append(f"- **{a.label}** — patterns "
                         f"{', '.join(f'`{p}`' for p in a.patterns)}"
                         + ("; multi-word, cannot match"
                            if a.multiword else
                            f"; {sum(a.shadowed.values())} shadowed"))
    else:
        print("  (none)")
        lines.append("_None — every rendering matched at least one token._")

    partial = sorted((a for a in audits if not a.empty and a.shadowed),
                     key=lambda a: -sum(a.shadowed.values()))
    print(f"\n=== {len(partial)} rendering(s) losing SOME occurrences ===")
    lines += ["", f"## Partial loss ({len(partial)})", "",
              "| rendering | matched | shadowed | surface -> lemma (POS) |",
              "|---|---|---|---|"]
    for a in partial:
        detail = "; ".join(f"`{s}` -> `{lem}` ({pos}) x{n}"
                           for (s, lem, pos), n in a.shadowed.most_common(4))
        total = sum(a.shadowed.values())
        print(f"  {a.label:18} matched {a.matched:6}  shadowed {total:5}  "
              f"{detail}")
        lines.append(f"| {a.label} | {a.matched} | {total} | {detail} |")
    if not partial:
        print("  (none)")

    healthy = [a for a in audits if not a.empty and not a.shadowed]
    lines += ["", f"## Clean ({len(healthy)})", "",
              ", ".join(f"{a.label} ({a.matched})" for a in healthy) or "_none_"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-term", type=int, default=12, dest="per_term",
                    help="SEP articles per rendering — match the pipeline run "
                         "you are auditing (default 12).")
    ap.add_argument("--min-freq", type=int, default=3, dest="min_freq",
                    help="SEP articles per rendering — match the pipeline run "
                         "you are auditing (default 12).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Markdown report path "
                         "(default analysis/sep/rendering_diagnostics.md).")
    args = ap.parse_args()

    audits, n_articles = audit_corpus(args.per_term, args.min_freq)
    lines = report(audits, n_articles)
    out = args.out or (config.ANALYSIS / "sep" / "rendering_diagnostics.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nfull report -> {out}\nbackground  -> {NOTE}")


if __name__ == "__main__":
    main()
