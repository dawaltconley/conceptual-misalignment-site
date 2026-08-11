"""Per-node display annotations shared by both lenses.

The graphs the site renders are keyed on one canonical label per word; anything a
reader needs in order to know *what that label stands for* rides along as a node
attribute (``cooccurrence.pmi_spacy.attach_forms`` does the same job for the
display glyph). Nothing here affects edges or any measurement — these are labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def attach_variants(
    G: "nx.Graph", variants: Mapping[str, Sequence[str]] | None
) -> "nx.Graph":
    """Tag each node with the vocabulary entries folded into it by the variant
    merge (``inspiration`` -> ``['inspirational', 'inspire']``), so the site can
    say which words a node actually covers.

    Corpus-level, matching the ``Vector.variants`` field on the embedding export:
    a family is a property of the merged vocabulary, not of the one article a
    given co-occurrence file was built from, so a node lists its whole family even
    where only one member occurs in that source.

    Unmerged nodes get no attribute at all (the merge touches a small minority of
    the vocabulary, and most graphs have none) — read it as optional downstream.
    """
    if not variants:
        return G
    for node in G.nodes:
        merged = variants.get(node)
        if merged:
            G.nodes[node]["variants"] = list(merged)
    return G
