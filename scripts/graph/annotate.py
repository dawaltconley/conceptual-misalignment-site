"""Per-node display annotations shared by both lenses.

The graphs the site renders are keyed on one canonical label per word; anything a
reader needs in order to know *what that label stands for* rides along as a node
attribute (``cooccurrence.pmi_spacy.attach_forms`` does the same job for the
display glyph). Nothing here affects edges or any measurement — these are labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def attach_variants(
    G: "nx.Graph", variants: Mapping[str, Sequence[str]] | None,
    *, always: "Iterable[str]" = (),
) -> "nx.Graph":
    """Tag each node with the other words it stands for, so the site can say what
    a label actually covers. Two things feed it, and they are the same claim:

    - the **derivational merge** (``inspiration`` -> ``['inspirational',
      'inspire']``) — corpus-level, matching the ``Vector.variants`` field on the
      embedding export: a family is a property of the merged vocabulary, not of
      the one article a given co-occurrence file was built from, so a node lists
      its whole family even where only one member occurs in that source.
    - the **target match** (``wisdom`` -> ``['wise', 'wisely']``) — scoped to the
      file, since a rendering's glob absorbs whatever that source happens to say.

    Nodes named in ``always`` (the targets) get the attribute even when the list
    is empty, because there ``[]`` is an answer — this source used only the label
    form — rather than the absence of one. Every other unmerged node gets no
    attribute at all (the merge touches a small minority of the vocabulary, and
    most graphs have none), so read it as optional downstream.
    """
    for node in always:
        if node in G:
            G.nodes[node]["variants"] = list((variants or {}).get(node, ()))
    if not variants:
        return G
    for node in G.nodes:
        merged = variants.get(node)
        if merged:
            G.nodes[node]["variants"] = list(merged)
    return G
