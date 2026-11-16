"""The tie-break among correspondences of equal cost, and deduplication of seeds by
symmetry class."""

from collections.abc import Collection, Mapping
from typing import Protocol

import networkx as nx

from .graph import OrderKey, canonical_key, tie_order


class SearchSide(Protocol):
    @property
    def graph(self) -> nx.Graph: ...
    @property
    def alignment(self) -> nx.Graph: ...
    @property
    def ranks(self) -> dict[int, int]: ...
    @property
    def classes(self) -> dict[int, int]: ...


def order_key(source: SearchSide, target: SearchSide, mapping: Mapping[int, int]) -> OrderKey:
    """The tie-break's canonical order (``graph.tie_order``) for two molecule views."""
    return tie_order(mapping, source.classes, target.classes, source.ranks, target.ranks)


def _moved_size(graph: nx.Graph, u: int, v: int) -> int:
    """Atoms on the smaller side of bond ``(u, v)`` once it is cut; 0 for a ring bond.

    Both sides are measured within the bond's own component, so other components of the
    molecule (a counter-ion, a solvent) do not change the answer."""
    reached = {u}
    frontier = [u]
    while frontier:
        x = frontier.pop()
        for y in graph.neighbors(x):
            if (x, y) == (u, v) or y in reached:
                continue
            if y == v:
                return 0
            reached.add(y)
            frontier.append(y)
    return min(len(reached), len(nx.node_connected_component(graph, u)) - len(reached))


def moved_atoms(source: nx.Graph, target: nx.Graph, mapping: Mapping[int, int]) -> int:
    """Size of what moves when a bond between conserved atoms breaks or forms: the smaller
    side of each such bond, summed over both sides, so that among equal-cost descriptions
    the one that moves the smaller group wins."""
    image = set(mapping.values())
    total = 0
    for u, v in source.edges:
        if u in mapping and v in mapping and not target.has_edge(mapping[u], mapping[v]):
            total += _moved_size(source, u, v)
    inverse = {t: s for s, t in mapping.items()}
    for u, v in target.edges:
        if u in image and v in image and not source.has_edge(inverse[u], inverse[v]):
            total += _moved_size(target, u, v)
    return total


def preference_key(
    source: SearchSide, target: SearchSide, mapping: Mapping[int, int]
) -> tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Tie-break among equal-cost correspondences: fewest attachment
    points changed (conserved atoms whose heavy-atom neighbour count differs), the smaller
    group moving, then the canonical order read the same from either side. Keys for ring
    changes, attribute edits and one-sided hydrogens were removed on 2026-11-08, after
    decisive counts of 6, 39 and 24 in 1,500 pilot pairs (in moldelta, the project this
    library was extracted from)."""
    gs, gt = source.graph, target.graph
    attachment_changes = sum(gs.degree(s) != gt.degree(t) for s, t in mapping.items())
    return (
        attachment_changes,
        moved_atoms(gs, gt, mapping),
        *order_key(source, target, mapping),
    )


def dedupe(
    solutions: Collection[dict[int, int]], source: SearchSide, target: SearchSide
) -> list[dict[int, int]]:
    """Drop solutions that pair the same symmetry classes as an earlier one."""
    seen: set[tuple[tuple[int, int], ...]] = set()
    kept = []
    for solution in sorted(solutions, key=lambda m: order_key(source, target, m)):
        key = canonical_key(solution, source.classes, target.classes)
        if key not in seen:
            seen.add(key)
            kept.append(solution)
    return kept
