from atommapper.edits import apply_edits, compute_edits, reverse_edits, same_graph
from tests._toy import attributed

ETHANOL = attributed([(6, 3, 0), (6, 2, 0), (8, 1, 0)], [(0, 1, "SINGLE"), (1, 2, "SINGLE")])
ACETALDEHYDE = attributed([(6, 3, 0), (6, 1, 0), (8, 0, 0)], [(0, 1, "SINGLE"), (1, 2, "DOUBLE")])
ETHANE = attributed([(6, 3, 0), (6, 3, 0)], [(0, 1, "SINGLE")])
PROPANE = attributed([(6, 3, 0), (6, 2, 0), (6, 3, 0)], [(0, 1, "SINGLE"), (1, 2, "SINGLE")])


def test_attribute_edits_on_a_full_correspondence():
    script = compute_edits(ETHANOL, ACETALDEHYDE, {0: 0, 1: 1, 2: 2})
    assert script.vertex_changes == ((1, 1, "hcount", 2, 1), (2, 2, "hcount", 1, 0))
    assert script.edge_changes == ((1, 2, 1, 2, "order", "SINGLE", "DOUBLE"),)
    assert not script.deleted_vertices and not script.inserted_vertices
    assert not script.deleted_edges and not script.inserted_edges


def test_apply_reproduces_the_target_and_reverse_reproduces_the_source():
    for source, target, mapping in [
        (ETHANOL, ACETALDEHYDE, {0: 0, 1: 1, 2: 2}),
        (ETHANE, PROPANE, {0: 0, 1: 2}),
        (PROPANE, ETHANE, {0: 0, 2: 1}),
    ]:
        script = compute_edits(source, target, mapping)
        assert same_graph(apply_edits(script, source), target)
        assert same_graph(apply_edits(reverse_edits(script), target), source)


def test_structural_edits_when_a_vertex_is_inserted():
    script = compute_edits(ETHANE, PROPANE, {0: 0, 1: 2})
    assert [t for t, _ in script.inserted_vertices] == [1]
    assert [(u, v) for u, v, _ in script.deleted_edges] == [(0, 1)]
    assert sorted((u, v) for u, v, _ in script.inserted_edges) == [(0, 1), (1, 2)]
    assert script.vertex_changes == ()


def test_identical_graphs_give_an_empty_script():
    script = compute_edits(ETHANOL, ETHANOL, {0: 0, 1: 1, 2: 2})
    assert script.is_empty


def test_same_graph_detects_attribute_and_edge_differences():
    assert same_graph(ETHANOL, ETHANOL)
    assert not same_graph(ETHANOL, ACETALDEHYDE)
    assert not same_graph(ETHANE, PROPANE)


def test_reverse_twice_is_the_identity():
    script = compute_edits(ETHANE, PROPANE, {0: 0, 1: 2})
    assert reverse_edits(reverse_edits(script)) == script
