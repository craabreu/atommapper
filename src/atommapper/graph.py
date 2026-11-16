"""Attributed and alignment graphs as plain NetworkX data. No RDKit here."""

from collections.abc import Mapping
from typing import Any

import networkx as nx

from .coloring import Coloring

Qualifier = tuple[int, int, int, int]

VERTEX_ATTRIBUTES: tuple[str, ...] = (
    "element",
    "charge",
    "isotope",
    "hcount",
    "radicals",
    "aromatic",
)
EDGE_ATTRIBUTES: tuple[str, ...] = ("order", "aromatic")


def qualifier(attrs: Mapping[str, Any]) -> Qualifier:
    """Element, attached ordinary hydrogens, cycle membership and charge of a vertex.

    Pairing is decided by the color, not by this tuple. It is read only for single-atom
    components: equal tuples pair first, and decide whether the pair is tagged seed or
    relaxed.
    """
    return (
        int(attrs["element"]),
        int(attrs["hcount"]),
        int(attrs["cycles"]),
        int(attrs["charge"]),
    )


def alignment_graph(attributed: nx.Graph, coloring: Coloring) -> nx.Graph:
    """Adjacency with a qualifier, a color and a total degree per vertex and an order per edge.

    The color is what decides whether two atoms can be paired (``coloring.Coloring``); the
    caller names the coloring.

    Total degree counts alignment neighbours plus attached ordinary hydrogens (the
    isolated-vertex pairing reads it). The order is what the objective charges when a conserved
    bond changes it.
    """
    g = nx.Graph()
    for v, attrs in attributed.nodes(data=True):
        g.add_node(
            v,
            q=qualifier(attrs),
            color=coloring.color(attrs),
            degree=int(attrs["hcount"]) + attributed.degree(v),
        )
    g.add_edges_from((u, v, {"order": d["order"]}) for u, v, d in attributed.edges(data=True))
    return g


OrderKey = tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]


def tie_order(
    mapping: Mapping[int, int],
    source_classes: Mapping[int, int],
    target_classes: Mapping[int, int],
    source_ranks: Mapping[int, int],
    target_ranks: Mapping[int, int],
) -> OrderKey:
    """The canonical order of the last tie-break (``preference.preference_key``):
    symmetry classes first, then ranks to separate correspondences that pair the same
    classes, each read from the source and from the target, the smaller kept, so a tie
    resolves to the same correspondence whichever molecule is the source."""
    inverse = {t: s for s, t in mapping.items()}
    forward = (
        canonical_key(mapping, source_classes, target_classes),
        canonical_key(mapping, source_ranks, target_ranks),
    )
    backward = (
        canonical_key(inverse, target_classes, source_classes),
        canonical_key(inverse, target_ranks, source_ranks),
    )
    return min(forward, backward)


def canonical_key(
    mapping: Mapping[int, int], source_rank: Mapping[int, int], target_rank: Mapping[int, int]
) -> tuple[tuple[int, int], ...]:
    """Order-independent key of a correspondence under canonical ranks; every tie-break uses it."""
    return tuple(sorted((source_rank[s], target_rank[t]) for s, t in mapping.items()))
