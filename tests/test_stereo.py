import pytest
from rdkit import Chem

from atommapper.stereo import stereo_verdicts


def same_molecule(source: str, target: str):
    """Molecules with the same skeleton; the mapping comes from a stereo-blind match."""
    ms, mt = Chem.MolFromSmiles(source), Chem.MolFromSmiles(target)
    match = mt.GetSubstructMatch(ms, useChirality=False)
    assert match
    return ms, mt, dict(enumerate(match))


def verdicts(source: str, target: str, mapping: dict[int, int] | None = None):
    if mapping is None:
        ms, mt, mapping = same_molecule(source, target)
    else:
        ms, mt = Chem.MolFromSmiles(source), Chem.MolFromSmiles(target)
    return [
        (v.kind, v.status, v.source_cip, v.target_cip) for v in stereo_verdicts(ms, mt, mapping)
    ]


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("F[C@H](Cl)Br", "Cl[C@H](Br)F", ("atom", "retained", "R", "R")),
        ("F[C@H](Cl)Br", "Br[C@@H](F)Cl", ("atom", "inverted", "R", "S")),
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br", ("atom", "inverted", "R", "S")),
        ("F[C@H](Cl)Br", "FC(Cl)Br", ("atom", "lost", "R", None)),
        ("FC(Cl)Br", "F[C@H](Cl)Br", ("atom", "gained", None, "R")),
        ("FC(Cl)Br", "FC(Cl)Br", ("atom", "unspecified", None, None)),
        ("C/C=C/F", "F/C=C/C", ("bond", "retained", "E", "E")),
        ("C/C=C/F", "C/C=C\\F", ("bond", "inverted", "E", "Z")),
    ],
)
def test_same_skeleton(source, target, expected):
    assert verdicts(source, target) == [expected]


def test_lost_by_symmetry_and_its_reverse():
    assert verdicts("F[C@H](Cl)Br", "FC(Cl)Cl", {0: 0, 1: 1, 2: 2}) == [
        ("atom", "lost_by_symmetry", "R", None)
    ]
    assert verdicts("FC(Cl)Cl", "F[C@H](Cl)Br", {0: 0, 1: 1, 2: 2}) == [
        ("atom", "gained_by_desymmetrization", None, "R")
    ]


def test_two_unconserved_neighbours_make_the_verdict_undetermined():
    assert verdicts("F[C@H](Cl)Br", "I[C@H](C)O", {1: 1}) == [("atom", "undetermined", "R", "R")]


def test_one_replaced_substituent_keeps_its_slot():
    assert verdicts("F[C@H](Cl)Br", "F[C@H](Cl)I", {0: 0, 1: 1, 2: 2}) == [
        ("atom", "retained", "R", "R")
    ]
    assert verdicts("F[C@H](Cl)Br", "F[C@@H](Cl)I", {0: 0, 1: 1, 2: 2}) == [
        ("atom", "inverted", "R", "S")
    ]


def test_a_replaced_implicit_hydrogen_takes_the_last_slot_and_cip_may_flip():
    assert verdicts("F[C@H](Cl)Br", "F[C@](Cl)(Br)I", {0: 0, 1: 1, 2: 2, 3: 3}) == [
        ("atom", "retained", "R", "S")
    ]


def test_cip_relabel_without_geometric_change_is_retained():
    assert verdicts("C/C=C/Cl", "C/C=C(/Cl)Br", {0: 0, 1: 1, 2: 2, 3: 3}) == [
        ("bond", "retained", "E", "Z")
    ]


def test_a_hydrogenated_double_bond_loses_its_stereo_and_the_reverse_gains_it():
    assert verdicts("C/C=C/F", "CCCF", {0: 0, 1: 1, 2: 2, 3: 3}) == [("bond", "lost", "E", None)]
    assert verdicts("CCCF", "C/C=C/F", {0: 0, 1: 1, 2: 2, 3: 3}) == [("bond", "gained", None, "E")]


def test_ring_cis_trans_is_reported_per_atom():
    result = verdicts("C[C@H]1CC[C@@H](C)CC1", "C[C@H]1CC[C@H](C)CC1", {i: i for i in range(8)})
    assert sorted(status for _, status, _, _ in result) == ["inverted", "retained"]


def test_square_planar_centers_are_unsupported():
    mol = "F[Pt@SP1](Cl)(Br)I"
    assert [s for _, s, _, _ in verdicts(mol, mol, {i: i for i in range(5)})] == ["unsupported"]


def test_deleted_centers_get_no_verdict():
    assert verdicts("F[C@H](Cl)Br", "FCl", {0: 0, 2: 1}) == []


def test_a_centre_that_becomes_trigonal_is_lost_not_lost_by_symmetry():
    identity = {i: i for i in range(5)}
    assert [s for _, s, _, _ in verdicts("C[C@H](O)CC", "CC(=O)CC", identity)] == ["lost"]
    assert [s for _, s, _, _ in verdicts("CC(=O)CC", "C[C@H](O)CC", identity)] == ["gained"]


def test_specification_decides_before_the_neighbour_order():
    """Precedence of conventions.md: lost and gained come before undetermined, and a centre
    unspecified on both sides is unspecified whatever happened to its neighbours."""
    first_four = {i: i for i in range(4)}
    assert [s for _, s, _, _ in verdicts("CCC[C@H](F)Cl", "CCCC(Br)I", first_four)] == ["lost"]
    assert [s for _, s, _, _ in verdicts("CCCC(Br)I", "CCC[C@H](F)Cl", first_four)] == ["gained"]
    assert [s for _, s, _, _ in verdicts("CCCC(F)Cl", "CCCC(Br)I", first_four)] == ["unspecified"]
