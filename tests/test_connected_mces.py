"""The maximum connected common edge subgraph (``_mces.fragments(best_only=True)``),
checked against brute force on small graphs."""

import itertools

import networkx as nx
import pytest
from networkx.generators.atlas import graph_atlas_g

from atommapper import _mces


def brute_connected(g1: nx.Graph, g2: nx.Graph) -> int:
    """Largest connected common edge subgraph, over every one-to-one atom map."""
    best = 0
    for image in itertools.permutations(g2, len(g1)):
        m = dict(zip(g1, image, strict=True))
        kept = nx.Graph((u, v) for u, v in g1.edges if g2.has_edge(m[u], m[v]))
        for component in nx.connected_components(kept):
            best = max(best, kept.subgraph(component).number_of_edges())
    return best


def connected(g1: nx.Graph, g2: nx.Graph):
    return _mces.fragments(
        [0] * len(g1),
        list(g1.edges),
        [0] * len(g2),
        list(g2.edges),
        max_steps=10_000_000,
        max_fragments=4096,
        min_bonds=1,
        best_only=True,
    )


GRAPHS = [
    g
    for g in graph_atlas_g()[1:]
    if 4 <= g.number_of_nodes() <= 6 and nx.is_connected(g) and max(d for _, d in g.degree) <= 4
]
PAIRS = [
    (a, b)
    for k, (a, b) in enumerate(itertools.product(GRAPHS, GRAPHS))
    if k % 97 == 0 and a.number_of_nodes() <= b.number_of_nodes()
]


@pytest.mark.parametrize(("g1", "g2"), PAIRS)
def test_best_only_finds_the_largest_connected_fragment(g1, g2):
    fragments, _steps, stop, truncated, _split, _pairs, _score, _work = connected(g1, g2)
    assert stop is None and not truncated
    expected = brute_connected(g1, g2)
    assert {len(bonds) for _atoms, bonds in fragments} == {expected}
    for atoms, bonds in fragments:
        m = dict(atoms)
        edges1 = list(g1.edges)
        sub = nx.Graph(edges1[i] for i in bonds)
        assert nx.is_connected(sub)
        assert all(g2.has_edge(m[u], m[v]) for u, v in sub.edges)


def test_dimethylbutane_against_hexane():
    dmb = nx.Graph([(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)])
    hexane = nx.path_graph(6)
    fragments = connected(dmb, hexane)[0]
    assert fragments and all(len(bonds) == 3 for _atoms, bonds in fragments)
