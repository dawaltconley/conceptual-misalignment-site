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

Three distinct failure modes, and they want different fixes:

- **lemma drift** — the surface form is in the corpus, but its lemma is not what
  the pattern expects. Fixed by one line in ``scripts/lemmas/english.conf``.
- **multi-token pattern** — the thing a pattern is matched against is a single
  token's lemma, so a phrase can only match if it is *made* one token first;
  ``corpus.parse.merge_phrases`` does that, and no exception table could.
- **no article cleared the frequency floor** — ``corpus.build`` admits an article
  only if the rendering occurs in it at least ``cooccurrence_min_freq`` times, so
  a word the SEP uses only once or twice per entry gets an empty corpus while the
  word itself is everywhere (愛's ``cherish``: 26 candidate entries, best had 4
  against a floor of 5). Nothing downstream can see this — the rejected articles
  are never parsed — so ``corpus.build`` records it in :class:`ArticleAudit` as it
  filters, and the audit reports it rather than blaming the corpus.

:func:`check_coverage` is the pipeline guard (cheap: it is handed the occurrence
counts the run already computed, and only pays for a corpus scan on the
renderings that came out empty). ``scripts/tools/rendering_diagnostics.py`` is
the full audit, which also finds *partial* shadowing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from models import Rendering

# One place to point a confused reader, referenced from the error message.
NOTE = "notes/spacy-lemma-exceptions.md"
TOOL = "scripts/tools/rendering_diagnostics.py"
LEMMA_CONF = "scripts/lemmas/english.conf"


class EmptyRenderingError(RuntimeError):
    """A configured rendering matched nothing, so its networks would all be null."""


@dataclass(frozen=True)
class ArticleAudit:
    """What ``corpus.build``'s min-frequency filter did to one rendering.

    Recorded during the filtering itself because it is unrecoverable afterwards:
    a rejected article is unparsed and never enters ``docs``, so an empty
    rendering whose corpus was *filtered* away looks identical to one the SEP
    never discusses.
    """

    min_freq: int
    """The floor an article had to clear (``Pipeline.cooccurrence_min_freq``)."""

    admitted: int = 0
    rejected: int = 0
    best: int = 0
    """Highest occurrence count among the REJECTED articles — how close the best
    near-miss came to ``min_freq``. (Admitted articles stop counting at the
    floor, so they are not represented here.)"""

    best_url: str | None = None

    @property
    def filtered_out(self) -> bool:
        """Every candidate was rejected — the rendering has no corpus at all."""
        return self.admitted == 0 and self.rejected > 0

    @property
    def no_candidates(self) -> bool:
        """The SEP search itself returned nothing to filter."""
        return self.admitted == 0 and self.rejected == 0


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

    present: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    """``(surface, lemma, pos) -> count`` for tokens the rendering DOES match,
    found anywhere in the parsed corpus. On an empty rendering these are hits in
    articles admitted for some *other* rendering: proof the corpus contains the
    word, and that the loss is upstream of matching."""

    articles: ArticleAudit | None = None
    """Article-selection outcome, when the caller has it (the pipeline does; the
    standalone diagnostics tool builds audits straight from parsed docs)."""

    @property
    def multiword(self) -> tuple[str, ...]:
        """Patterns needing more than one token — handled by the phrase merger.

        Asks ``corpus.parse`` rather than looking for a space, so the answer is
        the merger's own (``heart-mind*`` is multi-token too). Imported lazily:
        it loads the spaCy model, and a healthy run never gets here.
        """
        from corpus.parse import spans_multiple_tokens
        return tuple(p for p in self.patterns if spans_multiple_tokens(p))

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
                f"      {len(self.multiword)} multi-token pattern(s): "
                f"{', '.join(repr(p) for p in self.multiword)}",
                "      These are reachable — corpus.parse.merge_phrases merges "
                "the phrase into one",
                "      token before matching — so the corpus simply never "
                "contained it as tokenized.",
                "      Check the pattern against how the tokenizer actually "
                "splits the phrase, and",
                "      that its head inflects the way the pattern expects.",
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
        a = self.articles
        explained = bool(self.multiword or self.shadowed)
        if a is not None and a.filtered_out:
            explained = True
            lines += [
                f"      No corpus to match against: all {a.rejected} candidate "
                f"article(s) were",
                f"      rejected by the min-frequency filter, which admits an "
                f"article only if the",
                f"      rendering occurs >= {a.min_freq} times in it "
                f"(Pipeline.cooccurrence_min_freq).",
            ]
            if a.best:
                lines.append(
                    f"      The best candidate had {a.best}: {a.best_url}")
            lines += [
                "      So this is a THRESHOLD result, not an absent word: the "
                "SEP uses it too",
                "      thinly per entry to sustain a network. Lower "
                "cooccurrence_min_freq, widen",
                "      the patterns so the whole family clears the floor, or "
                "drop the rendering.",
            ]
        elif a is not None and a.no_candidates:
            explained = True
            lines.append(
                "      The SEP search returned no candidate articles at all — "
                "check the patterns,")
            lines.append(
                "      which are also what the search query is built from.")
        if self.present:
            total = sum(self.present.values())
            forms = ", ".join(
                f"{s!r} ({pos}) x{n}"
                for (s, _, pos), n in self.present.most_common(4))
            lines += [
                f"      For the record, the word IS in the parsed corpus — "
                f"{total} matching token(s)",
                f"      in articles admitted for other renderings: {forms}",
            ]
        if not explained and not self.present:
            lines.append(
                "      No token in the parsed corpus matches on surface form "
                "either, and no article")
            lines.append(
                "      was filtered out — the corpus genuinely does not use "
                "this word. Drop it from")
            lines.append(
                "      TERMS, or widen the pattern.")
        elif not explained:
            lines.append(
                "      Nothing here explains it: the corpus contains the word, "
                "so the loss is in")
            lines.append(
                "      matching, or in article selection (not recorded by this "
                f"caller). Run {TOOL}.")
        return "\n".join(lines)


class DocScan(NamedTuple):
    """One pass over the parsed corpus, from an empty rendering's point of view."""

    shadowed: Counter[tuple[str, str, str]]
    present: Counter[tuple[str, str, str]]


def scan_docs(
    rendering: Rendering, docs: Iterable, *, others: Sequence[Rendering] = (),
) -> DocScan:
    """Where the rendering's word does and does not appear in ``docs``.

    ``shadowed``: tokens whose surface matches ``rendering`` but whose lemma does
    not — occurrences the lemmatizer moved out of reach. ``others`` are the
    renderings that would win the token first (matching is first-wins across the
    whole ``TERMS`` list, so a token already claimed by another rendering is not
    evidence of loss here).

    ``present``: tokens the rendering matches outright. For a rendering the run
    counted as empty these can only come from articles admitted for some *other*
    rendering — which is the evidence that "the corpus does not use this word" is
    the wrong conclusion.
    """
    shadowed: Counter[tuple[str, str, str]] = Counter()
    present: Counter[tuple[str, str, str]] = Counter()
    for doc in docs:
        for token in doc:
            surface, lemma = token.text.lower(), token.lemma_.lower()
            if rendering.matches(lemma, token.pos_):
                present[(surface, lemma, token.pos_)] += 1
                continue
            if surface == lemma or not rendering.matches(surface, token.pos_):
                continue
            if any(o.matches(lemma, token.pos_) for o in others):
                continue
            shadowed[(surface, lemma, token.pos_)] += 1
    return DocScan(shadowed, present)


def shadowed_surfaces(
    rendering: Rendering, docs: Iterable, *, others: Sequence[Rendering] = (),
) -> Counter[tuple[str, str, str]]:
    """Tokens whose surface matches ``rendering`` but whose lemma does not."""
    return scan_docs(rendering, docs, others=others).shadowed


def audit_empty(
    renderings: Sequence[Rendering],
    occurrences: Mapping[str, int],
    docs: Sequence,
    articles: Mapping[str, ArticleAudit | None] | None = None,
) -> list[RenderingAudit]:
    """Audit only the renderings that ``occurrences`` reports as empty.

    Splitting it this way keeps the guard nearly free on a healthy run: the
    counts come from work the pipeline already did, and the corpus is only
    re-scanned for the renderings that actually failed. ``articles`` is
    ``corpus.build``'s record of article selection, keyed by rendering label —
    optional, because a caller working from parsed docs alone has no such record.
    """
    audits: list[RenderingAudit] = []
    for r in renderings:
        if occurrences.get(r.label, 0) > 0:
            continue
        others = [o for o in renderings if o is not r]
        scan = scan_docs(r, docs, others=others)
        audits.append(RenderingAudit(
            label=r.label,
            patterns=r.patterns,
            matched=0,
            shadowed=scan.shadowed,
            present=scan.present,
            articles=articles.get(r.label) if articles else None,
        ))
    return audits


def check_coverage(
    renderings: Sequence[Rendering],
    occurrences: Mapping[str, int],
    docs: Sequence,
    articles: Mapping[str, ArticleAudit | None] | None = None,
    *,
    corpus: str = "SEP",
    allow_empty: bool = False,
) -> list[RenderingAudit]:
    """Raise :class:`EmptyRenderingError` if any rendering matched nothing.

    Returns the audits (empty list on a clean run) so a caller that passes
    ``allow_empty`` can still report them.
    """
    audits = audit_empty(renderings, occurrences, docs, articles)
    if not audits:
        return []
    report = "\n\n".join(a.diagnosis() for a in audits)
    message = (
        f"{len(audits)} of {len(renderings)} renderings matched no token in the "
        f"{corpus} corpus.\nEvery network they own would be written null:\n\n"
        f"{report}\n\n"
        f"A rendering is matched against a token's LEMMA, in a corpus its own "
        f"occurrence count\nselected, so either a lemmatizer error or the "
        f"min-frequency filter erases it silently —\nan empty term looks exactly "
        f"like a term the corpus does not discuss.\nBackground and the measured error classes: "
        f"{NOTE}\nFull audit (including partial losses): {TOOL}\n"
        f"Pass --allow-empty-renderings to write the null networks anyway.")
    if allow_empty:
        print(f"\nWARNING: {message}\n")
        return audits
    raise EmptyRenderingError(message)
