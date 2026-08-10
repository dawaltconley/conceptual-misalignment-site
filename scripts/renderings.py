"""Does every configured ``Rendering`` actually match anything?

A ``Rendering`` matches a token's **lemma** (``models.Rendering.matches``), so
anything that moves the lemma out from under the pattern deletes the rendering
from the run — silently, because a term with no occurrences is indistinguishable
from a term the corpus happens not to discuss. 禮's ``mores`` was dead this way
for an unknown number of runs: ``en_core_web_sm`` lemmatizes ``mores`` to
``more`` (it is a Latin pluralium tantum, not a plural of *more*), so all 11
occurrences missed the pattern, every ``mores_*.json`` was written with
``"network": null``, and the only signal was a ``no co-occurrence`` line that
reads like a sparse-data result. See ``notes/spacy-lemma-exceptions.md``.

Two distinct failure modes, and they want different fixes:

- **lemma drift** — the surface form is in the corpus, but its lemma is not what
  the pattern expects. Fixed by one line in ``scripts/lemmas/english.conf``.
- **multi-word pattern** — a pattern containing a space can never match, because
  the thing it is matched against is a single token's lemma. Fixed by merging the
  phrase into one token at parse time, not by an exception table.

:func:`check_coverage` is the pipeline guard (cheap: it is handed the occurrence
counts the run already computed, and only pays for a corpus scan on the
renderings that came out empty). ``scripts/tools/rendering_diagnostics.py`` is
the full audit, which also finds *partial* shadowing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from models import Rendering

# One place to point a confused reader, referenced from the error message.
NOTE = "notes/spacy-lemma-exceptions.md"
TOOL = "scripts/tools/rendering_diagnostics.py"
LEMMA_CONF = "scripts/lemmas/english.conf"


class EmptyRenderingError(RuntimeError):
    """A configured rendering matched nothing, so its networks would all be null."""


@dataclass(frozen=True)
class RenderingAudit:
    """What a rendering matched, and — if nothing — what got in the way."""

    label: str
    patterns: tuple[str, ...]
    matched: int = 0
    """Tokens whose lemma matched (i.e. real occurrences)."""

    shadowed: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    """``(surface, lemma, pos) -> count`` for tokens whose **surface** matches a
    pattern but whose lemma does not: occurrences the lemmatizer moved out of
    reach. Non-empty alongside ``matched > 0`` means a *partial* loss."""

    @property
    def multiword(self) -> tuple[str, ...]:
        """Patterns that can never match, because a lemma is a single token."""
        return tuple(p for p in self.patterns if " " in p)

    @property
    def empty(self) -> bool:
        return self.matched == 0

    def diagnosis(self) -> str:
        """Why this rendering is empty, and what to do about it — indented for
        inclusion in the guard's error message."""
        lines = [f"  {self.label}  (patterns: "
                 f"{', '.join(repr(p) for p in self.patterns)})"]
        if self.multiword:
            lines += [
                f"      {len(self.multiword)} multi-word pattern(s): "
                f"{', '.join(repr(p) for p in self.multiword)}",
                "      A rendering is matched against one token's lemma, so a "
                "pattern containing",
                "      a space can never match. Needs the phrase merged into a "
                "single token at",
                "      parse time — an exception table cannot reach this.",
            ]
        if self.shadowed:
            total = sum(self.shadowed.values())
            lines.append(
                f"      {total} token(s) whose SURFACE matches were lemmatized "
                f"out of reach:")
            for (surface, lemma, pos), n in self.shadowed.most_common(5):
                lines.append(
                    f"        {surface!r} -> lemma {lemma!r} ({pos}) x{n}")
            fixes = sorted({s for s, _, _ in self.shadowed})[:3]
            lines.append(
                f"      Fix: add {' / '.join(f'`{s} -> {s}`' for s in fixes)} "
                f"to {LEMMA_CONF}")
        if not self.multiword and not self.shadowed:
            lines.append(
                "      No token in the corpus matches on surface form either — "
                "the corpus")
            lines.append(
                "      genuinely does not use this word. Drop it from TERMS, or "
                "widen the pattern.")
        return "\n".join(lines)


def shadowed_surfaces(
    rendering: Rendering, docs: Iterable, *, others: Sequence[Rendering] = (),
) -> Counter[tuple[str, str, str]]:
    """Tokens whose surface matches ``rendering`` but whose lemma does not.

    ``others`` are the renderings that would win the token first (matching is
    first-wins across the whole ``TERMS`` list, so a token already claimed by
    another rendering is not evidence of loss here).
    """
    hits: Counter[tuple[str, str, str]] = Counter()
    for doc in docs:
        for token in doc:
            surface, lemma = token.text.lower(), token.lemma_.lower()
            if surface == lemma or not rendering.matches(surface, token.pos_):
                continue
            if rendering.matches(lemma, token.pos_):
                continue
            if any(o.matches(lemma, token.pos_) for o in others):
                continue
            hits[(surface, lemma, token.pos_)] += 1
    return hits


def audit_empty(
    renderings: Sequence[Rendering],
    occurrences: Mapping[str, int],
    docs: Sequence,
) -> list[RenderingAudit]:
    """Audit only the renderings that ``occurrences`` reports as empty.

    Splitting it this way keeps the guard nearly free on a healthy run: the
    counts come from work the pipeline already did, and the corpus is only
    re-scanned for the renderings that actually failed.
    """
    audits: list[RenderingAudit] = []
    for r in renderings:
        if occurrences.get(r.label, 0) > 0:
            continue
        others = [o for o in renderings if o is not r]
        audits.append(RenderingAudit(
            label=r.label,
            patterns=r.patterns,
            matched=0,
            shadowed=shadowed_surfaces(r, docs, others=others),
        ))
    return audits


def check_coverage(
    renderings: Sequence[Rendering],
    occurrences: Mapping[str, int],
    docs: Sequence,
    *,
    corpus: str = "SEP",
    allow_empty: bool = False,
) -> list[RenderingAudit]:
    """Raise :class:`EmptyRenderingError` if any rendering matched nothing.

    Returns the audits (empty list on a clean run) so a caller that passes
    ``allow_empty`` can still report them.
    """
    audits = audit_empty(renderings, occurrences, docs)
    if not audits:
        return []
    report = "\n\n".join(a.diagnosis() for a in audits)
    message = (
        f"{len(audits)} of {len(renderings)} renderings matched no token in the "
        f"{corpus} corpus.\nEvery network they own would be written null:\n\n"
        f"{report}\n\n"
        f"A rendering is matched against a token's LEMMA, so a lemmatizer error "
        f"erases it\nsilently — an empty term looks exactly like a term the "
        f"corpus does not discuss.\nBackground and the measured error classes: "
        f"{NOTE}\nFull audit (including partial losses): {TOOL}\n"
        f"Pass --allow-empty-renderings to write the null networks anyway.")
    if allow_empty:
        print(f"\nWARNING: {message}\n")
        return audits
    raise EmptyRenderingError(message)
