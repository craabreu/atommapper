import networkx as nx
import pytest

from atommapper.coloring import COLORINGS
from atommapper.cost import change_size, compatible
from atommapper.graph import alignment_graph, canonical_key, qualifier
from atommapper.settings import Settings
from tests._toy import attributed, ring

RING = COLORINGS["element-ring"]
ETHANOL = attributed([(6, 3, 0), (6, 2, 0), (8, 1, 0)], [(0, 1, "SINGLE"), (1, 2, "SINGLE")])


def test_qualifier_and_total_degree():
    a = alignment_graph(ETHANOL, RING)
    assert qualifier(ETHANOL.nodes[1]) == (6, 2, 0, 0)
    assert a.nodes[1]["q"] == (6, 2, 0, 0)
    assert a.nodes[1]["degree"] == 4
    assert a.nodes[2]["degree"] == 2
    assert sorted(a.edges) == [(0, 1), (1, 2)]


def test_canonical_key_ignores_insertion_order():
    ranks_s, ranks_t = {0: 2, 1: 0}, {5: 1, 6: 3}
    assert canonical_key({0: 5, 1: 6}, ranks_s, ranks_t) == canonical_key(
        {1: 6, 0: 5}, ranks_s, ranks_t
    )
    assert canonical_key({0: 5, 1: 6}, ranks_s, ranks_t) == ((0, 3), (2, 1))


def test_only_the_color_decides_whether_two_atoms_can_pair():
    carbon = {"q": (6, 1, 1), "color": (6,), "degree": 3}
    assert not compatible(carbon, {"q": (7, 0, 1), "color": (7,), "degree": 2})
    assert compatible(carbon, {"q": (6, 1, 1), "color": (6,), "degree": 3})
    # a changed hydrogen count or ring count is not a change of color under "element"
    assert compatible(carbon, {"q": (6, 2, 0), "color": (6,), "degree": 4})
    # under "element-ring" a ring carbon and a chain carbon have different colors
    ring_carbon = {"q": (6, 1, 1), "color": (6, 1), "degree": 3}
    assert not compatible(ring_carbon, {"q": (6, 2, 0), "color": (6, 0), "degree": 4})


def test_alignment_graph_keeps_bond_order():
    a = alignment_graph(ETHANOL, RING)
    assert a.edges[0, 1]["order"] == "SINGLE"


def test_benzene_to_cyclohexane_identity_costs_six_order_changes():
    benzene = alignment_graph(ring(6, 1, 6, "AROMATIC"), RING)
    cyclohexane = alignment_graph(ring(6, 2, 6, "SINGLE"), RING)
    identity = {i: i for i in range(6)}
    assert change_size(benzene, cyclohexane, identity) == 6


def test_unmatched_vertices_and_edges_cost_one_each():
    ethane = alignment_graph(attributed([(6, 3, 0), (6, 3, 0)], [(0, 1, "SINGLE")]), RING)
    propane = alignment_graph(
        attributed([(6, 3, 0), (6, 2, 0), (6, 3, 0)], [(0, 1, "SINGLE"), (1, 2, "SINGLE")]),
        RING,
    )
    # both ethane carbons on the propane ends:
    # one inserted vertex, one deleted edge and two inserted edges
    assert change_size(ethane, propane, {0: 0, 1: 2}) == 4
    # growth at the end: CH3 -> CH2 keeps total degree 4, one inserted vertex and one inserted edge
    assert change_size(ethane, propane, {0: 0, 1: 1}) == 2


def test_change_size_rejects_a_cross_color_mapping():
    water = alignment_graph(attributed([(8, 2, 0)], []), RING)
    methane = alignment_graph(attributed([(6, 4, 0)], []), RING)
    with pytest.raises(ValueError, match="color"):
        change_size(water, methane, {0: 0})


def test_alignment_graph_keeps_isolated_vertices():
    g = nx.Graph()
    g.add_node(
        0, element=11, charge=1, isotope=0, hcount=0, radicals=0, aromatic=False, cycles=0, index=0
    )
    assert alignment_graph(g, RING).nodes[0]["degree"] == 0


def test_work_budget_by_default_with_a_long_clock_as_safety_net():
    settings = Settings()
    assert settings.node_budget > 0
    assert settings.time_limit_s >= 60.0
    assert not hasattr(settings, "refine_budget_s")
    assert not hasattr(settings, "fragment_timeout_s")


def test_alignment_graph_takes_its_coloring_from_the_caller():
    with pytest.raises(TypeError):
        alignment_graph(ETHANOL)  # ty: ignore[missing-argument]
