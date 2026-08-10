"""Derivational word families: which vocabulary entries are variants of one word.

The SEP vocabulary keys on a spaCy **lemma**, which is inflectional only. So
``inspires``/``inspired`` already collapse to ``inspire``, but ``inspire`` and
``inspiration`` stay two nodes that land on top of each other in the scatter.
About a quarter of the SEP vocabulary is entangled this way.

This module proposes candidate families and then **gates them on embedding
cosine**, so a family only merges when its members are already near-identical in
the analysis space. That ordering matters in both directions:

- Candidates alone are not enough. ``know``/``knowledge``/``knowable`` share a
  root but not a meaning, and merging them would destroy a real distinction.
- Cosine alone is not enough either, and this is the load-bearing half. The
  highest-cosine pairs in the SEP space that are *not* morphologically related
  are ``monotonic``/``nonmonotonic`` (0.935) and ``posteriori``/``priori``
  (0.928) — negation and antonym pairs, whose vectors are near-identical
  precisely because they are contraries. Merging on similarity alone would fuse
  them. **Corollary: strip suffixes only, never prefixes.**

Because the merge cannot pull together anything that was not already within
``threshold``, it declutters the view without manufacturing proximity — see
``notes/derivational-variant-merging.md`` for the writeup and the caveat.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Mapping, Sequence

import numpy as np

# Open English Wordnet — the maintained successor to Princeton WordNet 3.0.
# Pinned to a dated release so the family set is reproducible. Measured against
# the SEP vocabulary, 2025 finds 1083 in-vocab derivational pairs vs nltk
# WordNet 3.0's 1047, with 1044 shared (39 OEWN-only, 3 nltk-only).
LEXICON = "oewn:2025"

_wordnet = None


def _lexicon():
    global _wordnet
    if _wordnet is None:
        import wn
        _wordnet = wn.Wordnet(LEXICON)
    return _wordnet


Pair = tuple[str, str]


def _pair(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def derivational_pairs(vocab: Iterable[str]) -> set[Pair]:
    """In-vocabulary pairs linked by an OEWN ``derivation`` sense relation.

    This is the active candidate source. It catches irregular derivations that
    no suffix rule can reach (``belief``/``believe``, ``democracy``/
    ``democratic``, ``conceive``/``conception``, ``fail``/``failure``), and it
    declines to link ``knowledge`` to ``know`` at all — OEWN records no
    derivational relation for it, so the motivating counter-example is excluded
    before the cosine gate even runs.
    """
    words = set(vocab)
    en = _lexicon()
    pairs: set[Pair] = set()
    for word in words:
        for sense in en.senses(word.replace(" ", "_")):
            for related in sense.get_related("derivation"):
                other = related.word().lemma().lower().replace("_", " ")
                if other != word and other in words:
                    pairs.add(_pair(word, other))
    return pairs


def participial_pairs(
    vocab: Iterable[str], verb_lemma: Callable[[str], str | None]
) -> set[Pair]:
    """Pairs linking an ``-ed``/``-ing`` entry to its base verb, when both are in vocab.

    spaCy tags participial adjectives ``JJ``, not ``VBN``/``VBG``, so their lemma
    stays inflected and ``devoted``/``alleged``/``inclined`` enter the vocabulary
    alongside ``devote``/``allege``/``incline``. There is no tag-level signal
    separating those lemmatizer artifacts from genuinely lexicalized adjectives
    (``determined``, ``concerned``) or from deverbal nouns that must not collapse
    (``building``, ``beginning``, ``accounting``), which is exactly why these are
    proposed as *candidates* and left for the cosine gate to judge.

    ``verb_lemma`` maps a surface form to its verb lemma (see
    ``corpus.parse.verb_lemma``).
    """
    words = set(vocab)
    pairs: set[Pair] = set()
    for word in words:
        if not word.endswith(("ed", "ing")):
            continue
        base = verb_lemma(word)
        if base is not None and base != word and base in words:
            pairs.add(_pair(word, base))
    return pairs


# ---------------------------------------------------------------------------
# Suffix stripping — NOT used by the pipeline
# ---------------------------------------------------------------------------

_MIN_STEM = 4

# Ordered longest-first; (suffix, replacement). Suffixes only — stripping a
# prefix would fuse negation pairs (see the module docstring).
_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ization", "ize"), ("isation", "ize"), ("ational", "ate"),
    ("fulness", "ful"), ("ousness", "ous"), ("iveness", "ive"),
    ("ability", "able"), ("ibility", "ible"),
    ("ledge", ""),
    ("ation", ""), ("ition", ""), ("ution", ""),
    ("ment", ""), ("ness", ""), ("ance", ""), ("ence", ""),
    ("ancy", ""), ("ency", ""), ("able", ""), ("ible", ""),
    ("ical", "ic"), ("ship", ""), ("hood", ""),
    ("ism", ""), ("ist", ""), ("ity", ""), ("ety", ""), ("ive", ""),
    ("ory", ""), ("ary", ""), ("ous", ""), ("ful", ""), ("ess", ""),
    ("ize", ""), ("ise", ""), ("ify", ""), ("ate", ""), ("ion", ""),
    ("ing", ""), ("er", ""), ("or", ""), ("ly", ""), ("al", ""),
    ("ic", ""), ("ed", ""), ("es", ""), ("s", ""),
)
_VOWELS = frozenset("aeiou")


def stem_key(word: str) -> str:
    """An aggressive derivational stem, e.g. ``inspiration``/``inspire`` -> ``inspir``.

    **Not called by the pipeline** — kept here so it can be swapped in as a
    second candidate source alongside :func:`derivational_pairs`. Measured on
    the SEP vocabulary the two are complementary rather than redundant: at
    cosine 0.70 the union yields 231 merges, of which WordNet alone misses 31%
    and this stemmer alone misses 26%. Its unique contribution is philosophy
    jargon that no lexicon contains — 24 ``-ism``/``-ist`` pairs such as
    ``compatibilism``/``compatibilist`` and ``fideism``/``fideist``. Its cost is
    over-merging (``virtual``/``virtue``/``virtuous`` share a stem), which the
    cosine gate then has to clean up.
    """
    stem = word.lower()
    for _ in range(3):
        for suffix, replacement in _SUFFIXES:
            if stem.endswith(suffix) and (
                len(stem) - len(suffix) + len(replacement) >= _MIN_STEM
            ):
                stem = stem[: -len(suffix)] + replacement
                break
        else:
            break
    if stem.endswith("e") and len(stem) > _MIN_STEM:
        stem = stem[:-1]
    if len(stem) > _MIN_STEM and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
        stem = stem[:-1]
    if stem.endswith("y"):
        stem = stem[:-1] + "i"
    return stem


def stem_pairs(vocab: Iterable[str]) -> set[Pair]:
    """Pairs sharing a :func:`stem_key`. **Not called by the pipeline.**"""
    groups: dict[str, list[str]] = {}
    for word in vocab:
        groups.setdefault(stem_key(word), []).append(word)
    return {
        _pair(a, b)
        for members in groups.values() if len(members) > 1
        for i, a in enumerate(sorted(members)) for b in sorted(members)[i + 1:]
    }


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def candidate_families(
    vocab: Iterable[str], extra_pairs: Iterable[Pair] = ()
) -> list[list[str]]:
    """Connected groups of morphologically related words, before any gating.

    Exposed separately from :func:`merge_map` so the threshold can be swept
    offline: these groups plus their pairwise cosines are everything the gate
    consumes, and dumping them lets ``tools/family_diagnostics.py`` reproduce any
    threshold's outcome exactly, in the real analysis space, without re-running
    the embedder.
    """
    words = set(vocab)
    pairs = derivational_pairs(words)
    pairs |= {_pair(a, b) for a, b in extra_pairs if a in words and b in words}
    return _components(pairs)


def _components(pairs: Iterable[Pair]) -> list[list[str]]:
    """Connected components of the candidate graph (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for word in parent:
        groups.setdefault(find(word), []).append(word)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def _complete_linkage(
    family: Sequence[str], sim: Callable[[str, str], float], threshold: float
) -> list[list[str]]:
    """Split a family so that **every** pair inside a cluster clears ``threshold``.

    Complete (not single) linkage on purpose: single linkage would let a chain
    of adjacent pairs drag a distant member in, which is precisely the
    ``know``-``knowable``-``knowledge`` failure this is meant to avoid.
    """
    clusters = [[w] for w in family]
    while True:
        best, bi, bj = threshold, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                worst = min(sim(a, b) for a in clusters[i] for b in clusters[j])
                if worst >= best:
                    best, bi, bj = worst, i, j
        if bi < 0:
            return clusters
        clusters[bi] += clusters.pop(bj)


def merge_map(
    labels: Sequence[str],
    matrix: np.ndarray,
    *,
    threshold: float,
    exclude: Container[str] = frozenset(),
    counts: Mapping[str, int] | None = None,
    extra_pairs: Iterable[Pair] = (),
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Decide which vocabulary entries collapse together.

    ``matrix`` must be in the **analysis space** (centred, debiased) — raw pooled
    vectors sit in an anisotropic band where every cosine is ~0.84 and a
    threshold means nothing.

    ``exclude`` holds words that must never merge (the target renderings:
    ``Rendering`` globs already family-merge those by hand, and stemming would
    collapse ``humaneness`` and ``humanity`` into one). ``counts`` picks each
    family's surviving label — the most frequent member, falling back to the
    shortest then alphabetical order.

    Returns ``(alias, variants)``: ``alias`` maps every absorbed word to its
    surviving label, and ``variants`` maps each surviving label to the sorted
    list of words folded into it.
    """
    index = {label: i for i, label in enumerate(labels)}
    unit = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    def sim(a: str, b: str) -> float:
        return float(unit[index[a]] @ unit[index[b]])

    vocab = {label for label in labels if label not in exclude}

    def rank(word: str) -> tuple[int, int, str]:
        return (-(counts or {}).get(word, 0), len(word), word)

    alias: dict[str, str] = {}
    variants: dict[str, list[str]] = {}
    for family in candidate_families(vocab, extra_pairs):
        for cluster in _complete_linkage(family, sim, threshold):
            if len(cluster) < 2:
                continue
            primary = min(cluster, key=rank)
            absorbed = sorted(w for w in cluster if w != primary)
            for word in absorbed:
                alias[word] = primary
            variants[primary] = absorbed
    return alias, variants
