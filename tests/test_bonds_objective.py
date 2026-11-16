"""map_atoms(objective="bonds"): LOMAP's rules and LOMAP's objective, the most bonds kept,
ties to the most atoms, then to the smallest change."""

import pytest

from atommapper import map_atoms
from atommapper.adapter import build_view, parse
from atommapper.coloring import coloring
from atommapper.cost import change_size, kept_counts, lomap_weights
from atommapper.mapper import _Rules
from tests._exhaustive import lomap_optimum
from tests.test_mapper import SMALL_PILOT

SMALL = [
    ("C", "CC"),
    ("CC", "CC1CC1"),
    ("CB", "CN"),
    ("CCO", "CO"),
    ("CN", "C[C@H](N)O"),
    ("[Na+].CC(=O)[O-]", "[K+].CC(=O)[O-]"),
    ("O", "N"),
]


@pytest.mark.parametrize("element_change", [True, False])
@pytest.mark.parametrize(("source", "target"), SMALL_PILOT + SMALL)
def test_the_bonds_objective_reaches_the_exhaustive_optimum(source, target, element_change):
    result = map_atoms(source, target, element_change=element_change, objective="bonds")
    assert result.status != "timed_out"
    assert result.metadata["cost"] == lomap_optimum(source, target, element_change, "bonds")


def test_the_weights_rank_bonds_then_atoms_then_changes():
    paint = coloring("ring")
    s = build_view(parse("CCCC", "s"), "s", paint).alignment
    t = build_view(parse("CCCO", "t"), "t", paint).alignment
    atom, bond = lomap_weights(s, t)
    whole = {0: 0, 1: 1, 2: 2, 3: 3}  # one element change
    assert kept_counts(s, t, whole) == {"atoms": 4, "bonds": 3}
    assert change_size(s, t, whole, (atom, bond)) == 1
    part = {0: 0, 1: 1, 2: 2}  # no change, one atom and one bond lost on each side
    assert change_size(s, t, part, (atom, bond)) == 2 * atom + 2 * bond
    # one change never outweighs an atom, and atoms and changes never outweigh a bond
    assert atom > min(4, 4) + min(3, 3)
    assert bond > atom * (4 + 4) + atom - 1


def test_one_more_bond_is_kept_at_the_price_of_a_larger_change():
    source, target = "N[C@H](CO)Cc1ccccc1", "OCCCNC(=S)S"
    bonds = map_atoms(source, target, objective="bonds")
    assert (bonds.metadata["kept"], bonds.metadata["size"]) == ({"atoms": 5, "bonds": 4}, 24)
    assert bonds.metadata["cost"] == lomap_optimum(source, target, True, "bonds")
    change = map_atoms(source, target)
    assert (change.metadata["kept"], change.metadata["size"]) == ({"atoms": 4, "bonds": 3}, 23)
    assert change.metadata["cost"] == lomap_optimum(source, target, True, "change")


def test_metadata_names_the_objective_and_counts_what_is_kept():
    bonds = map_atoms("Cc1ccccc1", "Cc1ccccc1N", objective="bonds")
    assert bonds.metadata["objective_name"] == "bonds"
    assert bonds.metadata["kept"] == {"atoms": 7, "bonds": 7}
    assert bonds.metadata["size"] == 2
    change = map_atoms("Cc1ccccc1", "Cc1ccccc1N")
    assert change.metadata["objective_name"] == "change"
    assert change.metadata["cost"] == change.metadata["size"] == 2


def test_an_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="bonds"):
        map_atoms("CC", "CC", objective="atoms")


def test_lomaps_own_mapping_keeps_no_more_bonds():
    # pair 171 of the pilot: OpenFE LOMAP 3.3.0 keeps the macrocycle (27 atoms)
    source = "OCCN1CCN(c2ccc3cc2COCC=CCOCc2cccc(c2)-c2ccnc(n2)N3)CC1"
    target = "C1=CCOCc2cc(ccc2OCCN2CCCCC2)Nc2nccc(n2)-c2cccc(c2)COC1"
    lomap = [(7, 10), (8, 9), (9, 8), (10, 7), (11, 6), (12, 5), (13, 4), (14, 3), (15, 2)]
    lomap += [(16, 1), (17, 0), (18, 35), (19, 34), (20, 33), (21, 31), (22, 30), (23, 29)]
    lomap += [(24, 28), (25, 27), (26, 32), (27, 25), (28, 24), (29, 23), (30, 22), (31, 21)]
    lomap += [(32, 26), (33, 20)]
    s = build_view(parse(source, "s"), "s", coloring("ring"))
    t = build_view(parse(target, "t"), "t", coloring("ring"))
    mapping = {s.input_to_vertex[a]: t.input_to_vertex[b] for a, b in lomap}
    assert _Rules(s, t).accept(mapping)
    result = map_atoms(source, target, objective="bonds")
    assert result.status == "complete"
    theirs = kept_counts(s.alignment, t.alignment, mapping)
    assert result.metadata["kept"]["bonds"] >= theirs["bonds"]


def test_the_global_bound_proves_the_bonds_objective_too():
    from atommapper import Settings

    result = map_atoms(
        "Cc1ccccc1", "Cc1ccccc1N", objective="bonds", settings=Settings(global_budget_s=5)
    )
    assert result.metadata["optimality"] == "optimal"
    assert result.metadata["global_check"]["gap"] == 0


def test_the_cli_takes_the_objective(capsys):
    from atommapper.cli import main

    assert main(["N[C@H](CO)Cc1ccccc1", "OCCCNC(=S)S", "--objective", "bonds"]) == 0
    bonds = capsys.readouterr().out
    assert main(["N[C@H](CO)Cc1ccccc1", "OCCCNC(=S)S"]) == 0
    assert capsys.readouterr().out != bonds


@pytest.mark.xfail(strict=True, reason="the largest trimmed pieces miss the optimum (conventions)")
def test_the_largest_trimmed_pieces_can_miss_bonds_the_smaller_ones_reach():
    # pilot pair 50056 (one centre inverted): the trims leave pieces of 10, 3 and single
    # atoms; the optimum, 11 atoms and 12 bonds, grows from the smaller pieces only
    source = "CC1(C)S[C@@H]2[C@@H](S)C(=O)N2[C@H]1C(=O)O"
    target = "CC1(C)S[C@@H]2[C@H](S)C(=O)N2[C@H]1C(=O)O"
    result = map_atoms(source, target, objective="bonds")
    assert result.metadata["cost"] == lomap_optimum(source, target, True, "bonds")
