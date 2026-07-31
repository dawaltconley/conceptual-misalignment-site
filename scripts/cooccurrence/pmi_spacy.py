"""PMI / co-occurrence networks over spaCy Docs (the ``occurrences_unified`` path).

A spaCy-native re-expression of :mod:`cooccurrence.pmi`. The original works on
pre-tokenized ``list[list[str]]`` sentences; here the input is the same
``SourceDoc``s the embedding pipeline already builds, so node selection reuses
:func:`~embeddings.occurrences_unified.content_key` and
:func:`~embeddings.occurrences_unified.build_vocab` — one definition of "what is a
node, and under what key" for both pipelines.

Two things are carried over unchanged from the string pipeline by *reusing* the
pure functions in :mod:`cooccurrence.pmi` (``pmi``, ``count_pair_cooccurrences``,
``build_pmi_graph``, ``build_cosine_similarity_graph`` and the two network
builders): once a corpus is reduced to node-key sentences the PMI math is
corpus-agnostic, so the resulting graph is identical to the old one for the same
node lists — nothing is re-derived here.

What is new:

- **Node id = the content key (lemma / target label).** Orthographic variants of a
  word (敎/教) pool onto one node because the treebank lemma is the key.
- **Node attribute ``form`` = display glyph.** Each node also carries the most
  common *surface* form seen for that key, so the reader sees the familiar glyph
  (教) even though the graph keys on the orthodox lemma (敎). Purely a label; it
  does not affect edges. See the ``display-glyph-normalization`` note.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import networkx as nx
from spacy.tokens import Span

# Reused unchanged — the PMI math is the same once we have node-key sentences.
from cooccurrence.pmi import (
    count_pair_cooccurrences,
    pmi,
    build_cooccurrence_network as _build_cooccurrence_network,
    build_cosine_similarity_graph as _build_cosine_similarity_graph,
    build_pmi_graph as _build_pmi_graph,
    build_similarity_network as _build_similarity_network,
)
from embeddings.occurrences_unified import (
    CONTENT_POS,
    MatchFn,
    SourceDoc,
    build_vocab,
    content_key,
)

Targets = set[str] | frozenset[str]


# ---------------------------------------------------------------------------
# spaCy input → node-key sentences (replaces filter_to_sent_node_lists)
# ---------------------------------------------------------------------------

def sentence_node_keys(
    sent: Span,
    nodes: set[str] | frozenset[str],
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """The node keys occurring in one spaCy sentence, restricted to ``nodes``.

    Each kept token contributes its :func:`content_key` (a target label or the
    token lemma); non-content tokens and keys outside ``nodes`` are dropped.
    Duplicates within the sentence are preserved (they are de-duplicated per
    sentence by the co-occurrence counter downstream).
    """
    keys: list[str] = []
    for token in sent:
        key = content_key(token, match_fn, content_pos, stopwords)
        if key is not None and key in nodes:
            keys.append(key)
    return keys


def collect_node_sentences(
    sources: Sequence[SourceDoc],
    min_freq: int,
    targets: Targets = frozenset(),
    *,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
    min_sent_nodes: int = 2,
) -> tuple[list[list[str]], set[str], dict[str, str]]:
    """Reduce spaCy Docs to frequency-filtered node-key sentences for PMI.

    Returns ``(sent_node_lists, nodes, forms)``:

    - ``nodes`` — the frequency-filtered content-key vocabulary (:func:`build_vocab`)
      unioned with ``targets`` (always kept, regardless of frequency).
    - ``sent_node_lists`` — each source sentence reduced to its in-vocab node keys,
      dropping sentences with fewer than ``min_sent_nodes`` (matches the original's
      ``len >= 2``). These feed the reused PMI/similarity builders unchanged.
    - ``forms`` — ``node id -> display glyph``: the most common surface form seen
      for that key, for use as a node label. Collected over every occurrence, not
      only the kept sentences.
    """
    nodes = build_vocab(
        sources, min_freq,
        match_fn=match_fn, content_pos=content_pos, stopwords=stopwords,
    )
    nodes |= set(targets)

    surface: dict[str, Counter[str]] = defaultdict(Counter)
    sent_node_lists: list[list[str]] = []
    for source in sources:
        for sent in source.doc.sents:
            keys: list[str] = []
            for token in sent:
                key = content_key(token, match_fn, content_pos, stopwords)
                if key is not None and key in nodes:
                    keys.append(key)
                    surface[key][token.text] += 1
            if len(keys) >= min_sent_nodes:
                sent_node_lists.append(keys)

    forms = {key: counts.most_common(1)[0][0] for key, counts in surface.items()}
    return sent_node_lists, nodes, forms


def attach_forms(G: nx.Graph, forms: dict[str, str]) -> nx.Graph:
    """Tag each node with its display glyph (``form``); falls back to the id."""
    for node in G.nodes:
        G.nodes[node]["form"] = forms.get(node, node)
    return G


# ---------------------------------------------------------------------------
# Graph builders — spaCy in, lemma-keyed graph (+ display forms) out
# ---------------------------------------------------------------------------

def build_pmi_graph(
    sources: Sequence[SourceDoc],
    min_freq: int,
    targets: Targets = frozenset(),
    *,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
    min_sent_nodes: int = 2,
) -> tuple[nx.Graph, Counter[tuple[str, str]], Counter[str], int]:
    """PMI-weighted graph over spaCy Docs; returns the same tuple as the base
    :func:`cooccurrence.pmi.build_pmi_graph` (``G, pair_cooc, sent_freq, n_sents``)
    so callers can compute further per-term PMI without re-scanning. ``G`` nodes
    carry the ``form`` display attribute."""
    sent_node_lists, nodes, forms = collect_node_sentences(
        sources, min_freq, targets, match_fn=match_fn,
        content_pos=content_pos, stopwords=stopwords, min_sent_nodes=min_sent_nodes,
    )
    G, pair_cooc, sent_freq, n_sents = _build_pmi_graph(sent_node_lists, nodes)
    attach_forms(G, forms)
    return G, pair_cooc, sent_freq, n_sents


def build_cosine_similarity_graph(
    sources: Sequence[SourceDoc],
    threshold: float,
    min_freq: int,
    targets: Targets = frozenset(),
    *,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
    min_sent_nodes: int = 2,
) -> nx.Graph:
    """Co-occurrence-vector cosine graph over spaCy Docs; nodes carry ``form``."""
    sent_node_lists, nodes, forms = collect_node_sentences(
        sources, min_freq, targets, match_fn=match_fn,
        content_pos=content_pos, stopwords=stopwords, min_sent_nodes=min_sent_nodes,
    )
    S = _build_cosine_similarity_graph(sent_node_lists, sorted(nodes), threshold)
    return attach_forms(S, forms)


def build_cooccurrence_network(
    sources: Sequence[SourceDoc],
    term: str,
    min_freq: int,
    *,
    max_nodes: int = 15,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> nx.Graph | None:
    """PMI network pruned to ``term``'s neighborhood; nodes carry ``form``.

    ``term`` is a node key (a target hanzi or a family label); it is always kept
    in the vocabulary. Returns ``None`` if ``term`` never survives into the graph.
    """
    sent_node_lists, nodes, forms = collect_node_sentences(
        sources, min_freq, {term}, match_fn=match_fn,
        content_pos=content_pos, stopwords=stopwords,
    )
    G = _build_cooccurrence_network(sent_node_lists, nodes, term, max_nodes)
    return attach_forms(G, forms) if G is not None else None


def build_similarity_network(
    sources: Sequence[SourceDoc],
    term: str,
    min_freq: int,
    *,
    max_nodes: int = 15,
    sim_threshold: float = 0.7,
    match_fn: MatchFn | None = None,
    content_pos: set[str] | None = CONTENT_POS,
    stopwords: frozenset[str] | set[str] = frozenset(),
) -> nx.Graph | None:
    """Cosine-similarity network pruned to ``term``'s neighborhood; nodes carry
    ``form``. Returns ``None`` if ``term`` is absent from the graph."""
    sent_node_lists, nodes, forms = collect_node_sentences(
        sources, min_freq, {term}, match_fn=match_fn,
        content_pos=content_pos, stopwords=stopwords,
    )
    S = _build_similarity_network(
        sent_node_lists, nodes, term, max_nodes, sim_threshold)
    return attach_forms(S, forms) if S is not None else None
