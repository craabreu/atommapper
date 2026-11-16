"""Colorings: which atoms can be the same atom."""

import pytest

from atommapper.adapter import build_view, parse
from atommapper.coloring import coloring


def test_presets_name_their_hard_attributes():
    assert coloring("element-ring").hard == ("element", "ring")
    assert coloring("ring").hard == ("ring",)


def test_an_unknown_preset_lists_the_valid_names():
    with pytest.raises(KeyError, match="element-ring"):
        coloring("any")


def test_ring_membership_splits_the_carbons_of_toluene():
    view = build_view(parse("Cc1ccccc1", "t"), "t", coloring("element-ring"))
    assert sorted({d["color"] for _, d in view.alignment.nodes(data=True)}) == [(6, 0), (6, 1)]


def test_ring_leaves_out_the_element():
    view = build_view(parse("COc1ccncc1", "t"), "t", coloring("ring"))
    colors = {view.graph.nodes[v]["index"]: view.alignment.nodes[v]["color"] for v in view.graph}
    # chain atoms share one color whatever their element, ring atoms another
    assert colors[0] == colors[1] == (0,)
    assert {colors[i] for i in range(2, 8)} == {(1,)}
    assert not coloring("ring").element_hard
