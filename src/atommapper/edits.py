"""The edit script between two attributed graphs. No RDKit here."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .graph import EDGE_ATTRIBUTES, VERTEX_ATTRIBUTES

Attrs = tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class EditScript:
    mapping: tuple[tuple[int, int], ...]
    deleted_vertices: tuple[tuple[int, Attrs], ...]
    inserted_vertices: tuple[tuple[int, Attrs], ...]
    vertex_changes: tuple[tuple[int, int, str, Any, Any], ...]
    deleted_edges: tuple[tuple[int, int, Attrs], ...]
    inserted_edges: tuple[tuple[int, int, Attrs], ...]
    edge_changes: tuple[tuple[int, int, int, int, str, Any, Any], ...]

    @property
    def is_empty(self) -> bool:
        return not (
            self.deleted_vertices
            or self.inserted_vertices
            or self.vertex_changes
            or self.deleted_edges
            or self.inserted_edges
            or self.edge_changes
        )


def _attrs(data: Mapping[str, Any], keys: Iterable[str]) -> Attrs:
    return tuple((key, data[key]) for key in keys)


def _edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def compute_edits(source: nx.Graph, target: nx.Graph, mapping: Mapping[int, int]) -> EditScript:
    """Every difference between the attributed graphs under a fixed correspondence."""
    m = dict(sorted(mapping.items()))
    inverse = {t: s for s, t in m.items()}
    deleted_vertices = tuple(
        (s, _attrs(source.nodes[s], VERTEX_ATTRIBUTES)) for s in sorted(source) if s not in m
    )
    inserted_vertices = tuple(
        (t, _attrs(target.nodes[t], VERTEX_ATTRIBUTES)) for t in sorted(target) if t not in inverse
    )
    vertex_changes = tuple(
        sorted(
            (s, t, key, source.nodes[s][key], target.nodes[t][key])
            for s, t in m.items()
            for key in VERTEX_ATTRIBUTES
            if source.nodes[s][key] != target.nodes[t][key]
        )
    )
    deleted_edges: list[tuple[int, int, Attrs]] = []
    edge_changes: list[tuple[int, int, int, int, str, Any, Any]] = []
    for u, v in sorted(_edge(*e) for e in source.edges):
        if u in m and v in m and target.has_edge(m[u], m[v]):
            for key in EDGE_ATTRIBUTES:
                before, after = source.edges[u, v][key], target.edges[m[u], m[v]][key]
                if before != after:
                    edge_changes.append((u, v, m[u], m[v], key, before, after))
        else:
            deleted_edges.append((u, v, _attrs(source.edges[u, v], EDGE_ATTRIBUTES)))
    inserted_edges = tuple(
        (u, v, _attrs(target.edges[u, v], EDGE_ATTRIBUTES))
        for u, v in sorted(_edge(*e) for e in target.edges)
        if not (u in inverse and v in inverse and source.has_edge(inverse[u], inverse[v]))
    )
    return EditScript(
        tuple(m.items()),
        deleted_vertices,
        inserted_vertices,
        vertex_changes,
        tuple(deleted_edges),
        inserted_edges,
        tuple(sorted(edge_changes)),
    )


def apply_edits(script: EditScript, source: nx.Graph) -> nx.Graph:
    """The graph obtained by applying the script to the source, keyed by target ids."""
    m = dict(script.mapping)
    g = nx.Graph()
    for s, t in m.items():
        g.add_node(t, **{key: source.nodes[s][key] for key in VERTEX_ATTRIBUTES})
    for t, attrs in script.inserted_vertices:
        g.add_node(t, **dict(attrs))
    for _s, t, key, _before, after in script.vertex_changes:
        g.nodes[t][key] = after
    deleted = {_edge(u, v) for u, v, _ in script.deleted_edges}
    for u, v in source.edges:
        if _edge(u, v) not in deleted:
            g.add_edge(m[u], m[v], **{key: source.edges[u, v][key] for key in EDGE_ATTRIBUTES})
    for u, v, attrs in script.inserted_edges:
        g.add_edge(u, v, **dict(attrs))
    for _su, _sv, tu, tv, key, _before, after in script.edge_changes:
        g.edges[tu, tv][key] = after
    return g


def reverse_edits(script: EditScript) -> EditScript:
    """The script that turns the target back into the source."""
    return EditScript(
        mapping=tuple(sorted((t, s) for s, t in script.mapping)),
        deleted_vertices=script.inserted_vertices,
        inserted_vertices=script.deleted_vertices,
        vertex_changes=tuple(
            sorted((t, s, key, after, before) for s, t, key, before, after in script.vertex_changes)
        ),
        deleted_edges=script.inserted_edges,
        inserted_edges=script.deleted_edges,
        edge_changes=tuple(
            sorted(
                (tu, tv, su, sv, key, after, before)
                for su, sv, tu, tv, key, before, after in script.edge_changes
            )
        ),
    )


def same_graph(a: nx.Graph, b: nx.Graph) -> bool:
    """Vertex-by-vertex and edge-by-edge equality on the compared attributes."""
    if set(a) != set(b):
        return False
    if any(a.nodes[v][key] != b.nodes[v][key] for v in a for key in VERTEX_ATTRIBUTES):
        return False
    edges_a = {_edge(*e) for e in a.edges}
    if edges_a != {_edge(*e) for e in b.edges}:
        return False
    return all(
        a.edges[u, v][key] == b.edges[u, v][key] for u, v in edges_a for key in EDGE_ATTRIBUTES
    )
