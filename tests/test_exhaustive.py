"""map_atoms against a brute force that shares none of its search: every one-to-one
pairing of atoms of one color, judged by the rules and priced by the cost, on molecules
of up to six heavy atoms. It checks the refinement's bound, pruning and weights, which the
exhaustive oracle (built on the same refinement) cannot."""

import itertools

import pytest

from atommapper import map_atoms
from atommapper.adapter import build_view, parse
from atommapper.coloring import coloring
from atommapper.cost import change_size, lomap_weights
from atommapper.mapper import _Rules

PAIRS = [
    ("C", "CC"),
    ("CCO", "CO"),
    ("CC=O", "CCO"),
    ("CB", "CN"),
    ("C[C@H](N)O", "C[C@@H](N)O"),
    ("C[C@H](N)O", "C[C@@H](N)S"),
    ("c1ccccc1", "C1CCCC1"),
    ("C1CCCCC1", "CC1CCCC1"),
    ("OC1CCC1", "NC1CC1"),
    ("C1CC2CC12", "C1CC2CCC12"),
    ("CC1CC1", "C1CC1C"),
    ("[K+].CC(=O)[O-]", "[Na+].[OH-]"),
    ("[Na+].CC(=O)[O-]", "[K+].CC(=O)[O-]"),
    ("CCO.N", "CCN.O"),
]


def brute_force(
    source: str,
    target: str,
    element_change: bool,
    objective: str,
    allow_stereo_inversion: bool = False,
) -> int:
    paint = coloring("ring" if element_change else "element-ring")
    source_view = build_view(parse(source, "s"), "s", paint)
    target_view = build_view(parse(target, "t"), "t", paint)
    rules = _Rules(source_view, target_view, allow_stereo_inversion)
    s, t = source_view.alignment, target_view.alignment
    centres = rules.max_inversions if allow_stereo_inversion else 0
    weights = lomap_weights(s, t, centres) if objective == "bonds" else (1, 1)
    best = change_size(s, t, {}, weights)
    for k in range(1, min(len(s), len(t)) + 1):
        for sources in itertools.combinations(list(s), k):
            for targets in itertools.permutations(list(t), k):
                mapping = dict(zip(sources, targets, strict=True))
                if any(s.nodes[a]["color"] != t.nodes[b]["color"] for a, b in mapping.items()):
                    continue
                if rules.accept(mapping):
                    cost = change_size(s, t, mapping, weights)
                    if allow_stereo_inversion:
                        cost += len(rules.inverted_elements(mapping))
                    best = min(best, cost)
    return best


@pytest.mark.parametrize("objective", ["change", "bonds"])
@pytest.mark.parametrize("element_change", [True, False])
@pytest.mark.parametrize(("source", "target"), PAIRS)
def test_map_atoms_matches_a_brute_force(source, target, element_change, objective):
    result = map_atoms(source, target, element_change=element_change, objective=objective)
    assert result.metadata["cost"] == brute_force(source, target, element_change, objective)


STEREO_PAIRS = [
    ("C[C@H](N)O", "C[C@@H](N)O"),
    ("C[C@H](N)O", "C[C@@H](N)S"),
    ("C[C@H](F)[C@H](C)O", "C[C@@H](F)[C@@H](C)O"),
    ("C[C@H](F)[C@H](C)O", "C[C@@H](F)[C@H](C)O"),
    ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)OC"),
    ("C[C@H]1CC1C", "C[C@@H]1CC1C"),
    ("C/C=C/C(=O)O", "C/C=C\\C(=O)O"),
    ("C/C=C/C", "C/C=C\\C"),
    ("C/C=C/[C@H](C)O", "C/C=C\\[C@@H](C)O"),
    ("C/C=C/[C@H](C)O", "C/C=C/[C@@H](C)O"),
]


@pytest.mark.parametrize("objective", ["change", "bonds"])
@pytest.mark.parametrize("element_change", [True, False])
@pytest.mark.parametrize(("source", "target"), PAIRS + STEREO_PAIRS)
def test_map_atoms_with_inversions_allowed_matches_a_brute_force(
    source, target, element_change, objective
):
    result = map_atoms(
        source,
        target,
        element_change=element_change,
        objective=objective,
        allow_stereo_inversion=True,
    )
    assert result.metadata["cost"] == brute_force(source, target, element_change, objective, True)


INVERTED_PAIRS = [
    # the best correspondence under the rule pairs the surroundings of the inverted element
    # otherwise than every largest fragment does: the chain shifted so that the double bond
    # maps onto single bonds, the arms of a centre swapped; the search is seeded from the
    # atoms the stereo trim drops to reach it
    ("C/C=C/C(=O)O", "C/C=C\\C(=O)O"),
    ("C/C=C/C", "C/C=C\\C"),
    ("CC/C=C/CC", "CC/C=C\\CC"),
    ("C/C=C/C=C/C", "C/C=C\\C=C/C"),
    ("F/C=C/F", "F/C=C\\F"),
    ("N/C=N/C", "N/C=N\\C"),
    ("[Na+].C/C=C/C", "[K+].C/C=C\\C"),
    ("C/C=C/[C@H](C)O", "C/C=C\\[C@H](C)O"),
    ("C[C@H](O)CC", "C[C@@H](O)CC"),
    ("C[C@H](O)C=C", "C[C@@H](O)C=C"),
]


@pytest.mark.parametrize("objective", ["change", "bonds"])
@pytest.mark.parametrize("element_change", [True, False])
@pytest.mark.parametrize(("source", "target"), INVERTED_PAIRS)
def test_map_atoms_matches_a_brute_force_across_an_inverted_element(
    source, target, element_change, objective
):
    result = map_atoms(source, target, element_change=element_change, objective=objective)
    assert result.metadata["cost"] == brute_force(source, target, element_change, objective)
