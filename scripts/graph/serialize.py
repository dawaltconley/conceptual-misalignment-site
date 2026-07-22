"""Serialize a NetworkX graph to the site's node-link JSON schema."""

import json
from pathlib import Path

import networkx as nx


def save_graph_json(G: nx.Graph, path: Path) -> None:
    """Write a NetworkX graph to a node-link JSON file."""
    path.write_text(json.dumps(nx.node_link_data(G),
                    indent=2), encoding="utf-8")
