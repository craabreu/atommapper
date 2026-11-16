"""The size of a change, the default objective. No RDKit here.

Everything that differs counts one: an atom or bond present on one side only, a
conserved bond whose order changes, or a conserved atom whose element changes (possible
only when the coloring leaves element out). Nothing else is charged, so the cost of a complete
correspondence is an integer, the size of the change.
"""

import time
from collections.abc import Mapping
from typing import Any

import networkx as nx

FORBIDDEN = 1e6
"""Stands for an impossible pairing in the assignment bounds."""


def compatible(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two alignment vertices can be the same atom: their hard colors agree."""
    return a["color"] == b["color"]


def order_changed(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    """1 when a conserved bond changes order: it counts like a bond broken or formed, which
    is what the output reports and what a reaction model sees."""
    return int(a["order"] != b["order"])


def element_changed(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    """1 when a conserved atom changes element; always 0 when element is a hard color."""
    return int(a["q"][0] != b["q"][0])


def change_size(
    source: nx.Graph,
    target: nx.Graph,
    mapping: Mapping[int, int],
    weights: tuple[int, int] = (1, 1),
) -> int:
    """The size of the change: atoms and bonds present on one side only, plus conserved
    bonds whose order changes and conserved atoms whose element changes. What the
    ``"change"`` objective minimises.

    ``weights`` (atom, bond) charge an atom and a bond present on one side only; a change
    of element or order always counts one. ``lomap_weights`` makes the sum lexicographic.

    Raises ``ValueError`` for a mapping that pairs atoms of different colors.
    """
    for s, t in mapping.items():
        if not compatible(source.nodes[s], target.nodes[t]):
            raise ValueError(f"color mismatch in mapping: {s} -> {t}")
    atom, bond = weights
    inverse = {t: s for s, t in mapping.items()}
    size = atom * (source.number_of_nodes() + target.number_of_nodes() - 2 * len(mapping))
    size += sum(element_changed(source.nodes[s], target.nodes[t]) for s, t in mapping.items())
    for u, v, attrs in source.edges(data=True):
        if u in mapping and v in mapping and target.has_edge(mapping[u], mapping[v]):
            size += order_changed(attrs, target.edges[mapping[u], mapping[v]])
        else:
            size += bond
    for u, v in target.edges:
        if not (u in inverse and v in inverse and source.has_edge(inverse[u], inverse[v])):
            size += bond
    return size


def lomap_weights(source: nx.Graph, target: nx.Graph, stereo: int = 0) -> tuple[int, int]:
    """Weights (atom, bond) under which ``change_size`` ranks correspondences by the bonds
    they lose, then by the atoms, then by the changes of element and order: LOMAP's
    objective (the most bonds kept), ties to the most atoms, then to the smallest change.

    An atom outweighs every change a correspondence can hold, and a bond outweighs every
    atom and change together. ``stereo`` is the most inverted stereocentres a correspondence
    can keep (when ``map_atoms`` allows them), each one more change.
    """
    n, k = source.number_of_nodes(), target.number_of_nodes()
    changes = min(n, k) + min(source.number_of_edges(), target.number_of_edges()) + stereo
    atom = changes + 1
    return atom, atom * (n + k + 1)


def kept_counts(source: nx.Graph, target: nx.Graph, mapping: Mapping[int, int]) -> dict[str, int]:
    """The atoms and bonds a correspondence keeps; a bond is kept when both sides hold it."""
    bonds = sum(
        1
        for u, v in source.edges
        if u in mapping and v in mapping and target.has_edge(mapping[u], mapping[v])
    )
    return {"atoms": len(mapping), "bonds": bonds}


def global_bound(
    source: nx.Graph, target: nx.Graph, budget: float, weights: tuple[int, int] = (1, 1)
) -> tuple[float | None, bool]:
    """Unconstrained edit distance with the same costs (``weights`` as in
    ``cost.change_size``); a lower bound when it completes."""
    atom, bond = float(weights[0]), float(weights[1])

    def node_cost(a: Mapping, b: Mapping) -> float:
        return float(element_changed(a, b)) if compatible(a, b) else FORBIDDEN

    best: float | None = None
    start = time.perf_counter()
    for _vertices, _edges, cost in nx.optimize_edit_paths(
        source,
        target,
        node_subst_cost=node_cost,
        node_del_cost=lambda _a: atom,
        node_ins_cost=lambda _a: atom,
        edge_subst_cost=order_changed,
        edge_del_cost=lambda _a: bond,
        edge_ins_cost=lambda _a: bond,
        timeout=budget,
    ):
        best = cost
    return best, time.perf_counter() - start < budget
