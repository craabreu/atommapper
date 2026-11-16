from atommapper.adapter import build_view, parse
from atommapper.coloring import coloring
from atommapper.stage1 import pair_isolated


def side(smiles: str):
    return build_view(parse(smiles, "test"), "test", coloring("element-ring"))


def element_vertex(v, element: int) -> int:
    return next(n for n, d in v.graph.nodes(data=True) if d["element"] == element)


def test_isolated_vertices_pair_by_qualifier_in_rank_order():
    s, t = side("[Na+].c1ccccc1"), side("[Na+].C1CCCCC1")
    pairs = pair_isolated(s.alignment, t.alignment, s.ranks, t.ranks)
    assert pairs == {element_vertex(s, 11): element_vertex(t, 11)}
