"""Recombine subword tokens into whole words before anything else sees the Doc.

The Kyoto treebank annotates classical Chinese one character per token, so the
corpus arrives as characters rather than words: 天下 is 天 + 下, 父母 is 父 + 母,
文王 is 文 + 王. Keying nodes on those characters inflates the frequency of generic
glyphs and splits one word's occurrences across two nodes, in both the PMI
networks and the embedding scatter.

The treebank already says which adjacent tokens form one word — the UD
*word-formation* relations ``compound``, ``flat`` and ``fixed``. This module turns
those relations (plus a hand-curated override list) into token-index groups, and
applies them with spaCy's retokenizer so every consumer downstream sees words.

Two things are deliberately *not* merged:

- **Coordination and modification.** ``conj`` and ``nmod`` are excluded, so 仁義
  (義 --conj--> 仁) stays two tokens. This matters: 仁 and 義 are target terms.
- **Bisyllabic words the treebank labels ``nmod``** — 諸侯, 天子, 大夫, 聖人. Adding
  ``nmod`` to the relation set would over-merge genuine modifiers, so these are
  left to the curated override list (and, later, to a segmentation-derived
  lexicon). See ``notes/multi-character-tokenization.md``.

The grouping functions are pure — index sequences in, index groups out — so they
can be exercised without spaCy, and so a second boundary source can join in by
contributing pairs rather than by touching the merge logic.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spacy.tokens import Doc

# UD relations that join a token to its head *within a single word*. Kept narrow
# on purpose — see the module docstring for what is excluded and why.
MERGE_DEPS = frozenset({"compound", "flat", "fixed"})


# ---------------------------------------------------------------------------
# Grouping (pure: indices in, indices out)
# ---------------------------------------------------------------------------

def _components(n: int, pairs: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    """Union-find over ``0..n-1``; returns ``{root: sorted indices}``."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    out: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        out[find(i)].append(i)
    return out


def groups_from_components(
    n: int, components: dict[int, list[int]]
) -> list[list[int]]:
    """Turn components into words: a component only becomes one if its members
    form a **contiguous run with no foreign token interleaved**.

    That drops the cases where a relation reaches across an intervening token
    (欣然, 子男, 民夫婦), which a merge would silently reorder the text around.

    Returns an ordered list of groups covering every index; unmerged tokens come
    back as singletons, so a caller can rebuild the token sequence group by group.
    """
    merge_root: dict[int, int] = {}
    for root, idxs in components.items():
        lo, hi = idxs[0], idxs[-1]
        if len(idxs) > 1 and len(idxs) == hi - lo + 1:
            for i in idxs:
                merge_root[i] = root

    groups: list[list[int]] = []
    i = 0
    while i < n:
        root = merge_root.get(i)
        if root is None:
            groups.append([i])
            i += 1
            continue
        group = [i]
        i += 1
        while i < n and merge_root.get(i) == root:
            group.append(i)
            i += 1
        groups.append(group)
    return groups


def contiguous_groups(
    n: int, pairs: Iterable[tuple[int, int]]
) -> list[list[int]]:
    """The words ``pairs`` describe: union the connected tokens, keep the
    contiguous components. See :func:`groups_from_components`."""
    return groups_from_components(n, _components(n, pairs))


def word_formation_pairs(
    deps: Sequence[str | None],
    heads: Sequence[int],
    merge_deps: Iterable[str] = MERGE_DEPS,
) -> list[tuple[int, int]]:
    """``(token, head)`` pairs for every word-formation relation.

    ``deps[i]`` is token *i*'s dependency label and ``heads[i]`` is the list index
    of its head (its own index, or any out-of-range value, for a root).
    """
    merge_deps = frozenset(merge_deps)
    n = len(deps)
    return [
        (i, head)
        for i, (dep, head) in enumerate(zip(deps, heads))
        if dep in merge_deps and 0 <= head < n and head != i
    ]


def word_formation_groups(
    deps: Sequence[str | None],
    heads: Sequence[int],
    merge_deps: Iterable[str] = MERGE_DEPS,
) -> list[list[int]]:
    """The words ``deps``/``heads`` describe, as contiguous index groups.

    Tokenizer-agnostic: a caller only has to map its own token objects onto the
    ``(dep, head)`` inputs. (Originally written for the suparkanbun parse on the
    ``suparkanbun-cltk-comparison`` branch; the CoNLL-U loader feeds it the
    treebank's gold columns instead.)
    """
    return contiguous_groups(len(deps), word_formation_pairs(deps, heads, merge_deps))


# ---------------------------------------------------------------------------
# Overrides — the hand-curated escape hatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Overrides:
    """Curated word list applied on top of the UD relations.

    ``merge`` force-merges a run of whole tokens spelling one of its words,
    wherever that run appears inside a sentence — the way to add the bisyllabic
    words the treebank labels ``nmod`` (諸侯, 天子, 大夫, 聖人) without loosening the
    relation set for everything else. Because listing a word is an explicit
    judgement, a forced group bypasses the stopword and target guards below; it
    still has to be contiguous and inside one sentence.

    ``never_merge`` drops a word the relations produce but you don't want as a
    node. It is checked first, so listing a word in both means it is not merged —
    an explicit refusal beats an explicit request.

    Both match on either the joined surface forms or the joined lemmas, so a word
    can be listed with whichever glyphs you have to hand (荅 or 答).
    """

    merge: frozenset[str] = frozenset()
    never_merge: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.merge or self.never_merge)


EMPTY_OVERRIDES = Overrides()


def load_overrides(path: str | Path | None) -> Overrides:
    """Read an overrides JSON file. A missing file is not an error — it just
    means no overrides, so the pipeline never depends on one existing."""
    if path is None:
        return EMPTY_OVERRIDES
    path = Path(path)
    if not path.is_file():
        return EMPTY_OVERRIDES
    data = json.loads(path.read_text(encoding="utf-8"))
    return Overrides(
        merge=frozenset(data.get("merge", ())),
        never_merge=frozenset(data.get("never_merge", ())),
    )


def override_pairs(
    forms: Sequence[str],
    lemmas: Sequence[str],
    sent_ids: Sequence[int],
    words: Iterable[str],
) -> list[tuple[int, int]]:
    """``(i, i+1)`` pairs for every run of tokens spelling an override word.

    Scans each start position for the longest listed word it begins, matching the
    joined forms or the joined lemmas, and never crossing a sentence boundary.
    """
    words = {w for w in words if w}
    if not words:
        return []
    max_len = max(len(w) for w in words)
    n = len(forms)

    pairs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        hit = 0
        form = lemma = ""
        for j in range(i, n):
            if sent_ids[j] != sent_ids[i]:
                break
            form += forms[j]
            lemma += lemmas[j]
            if len(form) > max_len and len(lemma) > max_len:
                break
            if j > i and (form in words or lemma in words):
                hit = j
        if hit:
            pairs.extend((k, k + 1) for k in range(i, hit))
            i = hit + 1
        else:
            i += 1
    return pairs


# ---------------------------------------------------------------------------
# Configuration + reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeConfig:
    """What to merge, and what to refuse to merge. Hashable, so it can key the
    parsed-chapter cache in :mod:`corpus.parse`."""

    deps: frozenset[str] = MERGE_DEPS
    """UD relations treated as word-formation. Empty disables relation merging
    (overrides still apply)."""

    stopwords: frozenset[str] = frozenset()
    """A group whose constituents are *all* stopwords is left alone. Today those
    characters are each dropped downstream, so skipping the merge reproduces
    current behaviour rather than inventing a 可以 / 得而 / 有以 node."""

    targets: frozenset[str] = frozenset()
    """A group containing one of these is left alone, so the term keeps every
    occurrence. Skipped groups are recorded in the report, not swallowed."""

    overrides: Overrides = EMPTY_OVERRIDES


@dataclass
class MergeReport:
    """What a merge pass did, for ``tools/merge_report.py`` and the run summary."""

    merged: Counter[str] = field(default_factory=Counter)
    """Merged word (joined surface forms) -> token count."""

    root_pos: Counter[str] = field(default_factory=Counter)
    """Merged tokens by the POS they inherit from their syntactic root."""

    pos_by_word: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter))
    """Merged word -> the POS it inherited, per occurrence. The inherited POS is
    what decides whether ``Pipeline.content_pos`` keeps the word at all, so it
    belongs next to the word in any review."""

    forced: Counter[str] = field(default_factory=Counter)
    """Merges that came from ``Overrides.merge``."""

    skipped: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter))
    """Reason -> word -> count, for every group a guard rejected."""

    @property
    def tokens(self) -> int:
        return sum(self.merged.values())

    @property
    def types(self) -> int:
        return len(self.merged)

    def summary(self) -> str:
        skips = ", ".join(f"{sum(c.values())} {reason}"
                          for reason, c in sorted(self.skipped.items()))
        line = f"{self.types} types / {self.tokens} tokens"
        return f"{line} (skipped {skips})" if skips else line


# ---------------------------------------------------------------------------
# Applying it to a Doc
# ---------------------------------------------------------------------------

def _sentence_ids(doc: "Doc") -> list[int]:
    """Per-token sentence index, so a group can be tested for crossing one."""
    ids = [0] * len(doc)
    for s, sent in enumerate(doc.sents):
        for token in sent:
            ids[token.i] = s
    return ids


def merged_pos(doc: "Doc", group: Sequence[int], root: int) -> str:
    """The POS a merged token should carry — its root's, except for names.

    Inheriting the syntactic head's POS is right for ordinary compounds (天下 is
    the NOUN 下 was) but wrong for a name: in a Chinese name+title compound the
    *modifier* carries the referential identity and the head is an ordinary noun,
    so 文 (PROPN) --``compound``--> 王 (NOUN) would come out NOUN and 文王 would
    read as a kind of king rather than as King Wen. A group containing a proper
    noun **is** a proper noun, so it inherits PROPN instead.

    This matters because it is what keeps names out of the vocabulary:
    ``content_pos`` is ``{"NOUN","VERB","ADJ"}`` precisely to exclude them, and
    the English side excludes proper nouns the same way. Without this, merging
    would quietly re-admit 文王, 周公, 宣王, 武王 …

    The treebank's gold annotation is the whole source here — no NER pass needed.
    To restrict this to *persons* only, additionally require the root's XPOS to
    start with ``n,名詞,人`` (王 公 伯 人 夫 子 弟 徒); that leaves the 7 place/state
    compounds (齊國, 岐山, 幽州 …) as NOUN, all of which fall below both frequency
    floors anyway.
    """
    if doc[root].pos_ != "PROPN" and any(doc[i].pos_ == "PROPN" for i in group):
        return "PROPN"
    return doc[root].pos_


def _root(doc: "Doc", group: Sequence[int]) -> int:
    """The group's syntactic root: the member whose head lies outside it. The
    merged token inherits this token's POS/tag/morph/dep, so 天下 comes out as the
    NOUN 下 was, and 足以 as the AUX — which the pipeline's ``content_pos`` filter
    then discards on its own."""
    members = set(group)
    for i in group:
        head = doc[i].head.i
        if head == i or head not in members:
            return i
    return group[0]


def merge_doc(
    doc: "Doc",
    config: MergeConfig,
    report: MergeReport | None = None,
) -> "Doc":
    """Recombine ``doc``'s subword tokens in place, and return it.

    The UD relations and the override list each contribute pairs; they are
    unioned so a curated word can extend a relation-derived group rather than
    fight it. The surviving groups are applied in one retokenizer pass, which
    remaps heads for us. The merged token's lemma is the joined lemmas (node ids
    key on the lemma) and its text is the joined forms (the display glyph).

    ``doc.text`` and every ``token.idx`` are unchanged by retokenization, so the
    character-offset arithmetic in ``embeddings.occurrences`` stays exact.
    """
    report = report if report is not None else MergeReport()
    n = len(doc)
    if n == 0:
        return doc

    forms = [t.text for t in doc]
    lemmas = [t.lemma_ or t.text for t in doc]
    sent_ids = _sentence_ids(doc)

    deps = [t.dep_ for t in doc]
    heads = [t.head.i for t in doc]
    pairs = word_formation_pairs(deps, heads, config.deps) if config.deps else []

    forced_pairs = override_pairs(
        forms, lemmas, sent_ids, config.overrides.merge)
    forced_tokens = {i for pair in forced_pairs for i in pair}
    pairs = [*pairs, *forced_pairs]
    components = _components(n, pairs)

    keep: list[tuple[list[int], str]] = []
    for group in groups_from_components(n, components):
        if len(group) == 1:
            continue
        word = "".join(forms[i] for i in group)
        lemma = "".join(lemmas[i] for i in group)

        if word in config.overrides.never_merge or lemma in config.overrides.never_merge:
            report.skipped["override"][word] += 1
            continue
        if len({sent_ids[i] for i in group}) > 1:
            report.skipped["cross-sentence"][word] += 1
            continue
        forced = bool(forced_tokens & set(group))
        if not forced:
            if all(lemmas[i] in config.stopwords for i in group):
                report.skipped["all-stopword"][word] += 1
                continue
            if any(lemmas[i] in config.targets for i in group):
                report.skipped["target"][word] += 1
                continue
        keep.append((group, word))
        report.merged[word] += 1
        if forced:
            report.forced[word] += 1

    # Groups the relations produced but that no longer form a contiguous run are
    # reported too — they are a real signal that the parse disagrees with itself.
    for idxs in components.values():
        if len(idxs) > 1 and len(idxs) != idxs[-1] - idxs[0] + 1:
            report.skipped["non-contiguous"][
                "".join(forms[i] for i in idxs)] += 1

    if not keep:
        return doc

    with doc.retokenize() as retokenizer:
        for group, word in keep:
            root_i = _root(doc, group)
            root = doc[root_i]
            pos = merged_pos(doc, group, root_i)
            span = doc[group[0]:group[-1] + 1]
            attrs: dict[str | int, Any] = {
                "LEMMA": "".join(lemmas[i] for i in group),
                "POS": pos,
                "TAG": root.tag_,
                "DEP": root.dep_,
                "MORPH": str(root.morph),
            }
            retokenizer.merge(span, attrs=attrs)
            report.root_pos[pos or "_"] += 1
            # Keyed by the joined forms, exactly as ``report.merged`` is — a span
            # that happened to contain whitespace would key differently.
            report.pos_by_word[word][pos or "_"] += 1
    return doc
