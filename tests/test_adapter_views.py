import pytest
from rdkit import Chem

from atommapper.adapter import build_view, parse
from atommapper.coloring import coloring
from atommapper.errors import InvalidInput, UnsupportedInput


def view(smiles: str):
    return build_view(parse(smiles, "test"), "test", coloring("element-ring"))


def elements(v) -> list[int]:
    return sorted(d["element"] for _, d in v.graph.nodes(data=True))


def test_hydrogen_spellings_give_the_same_graph():
    implicit = view("CCO")
    explicit = view("[H]OC([H])([H])C([H])([H])[H]")
    assert implicit.smiles == explicit.smiles
    assert explicit.graph.number_of_nodes() == 3
    assert sorted(d["hcount"] for _, d in explicit.graph.nodes(data=True)) == [1, 2, 3]


def test_input_indices_survive_hydrogen_merging():
    v = view("[H]OCC")
    assert set(v.input_to_vertex) == {1, 2, 3}
    assert v.graph.nodes[v.input_to_vertex[1]]["element"] == 8


def test_isotopic_and_molecular_hydrogens_stay_vertices():
    assert elements(view("C[2H]")) == [1, 6]
    h2 = view("[H][H]")
    assert h2.graph.number_of_nodes() == 2 and h2.graph.number_of_edges() == 1


def test_charged_hydrogen_on_a_heavy_atom_is_unsupported():
    with pytest.raises(UnsupportedInput, match="hydrogen"):
        view("C[H+]")


@pytest.mark.parametrize("smiles", ["C1CC", "not a molecule", ""])
def test_invalid_or_empty_input_raises(smiles):
    with pytest.raises(InvalidInput):
        parse(smiles, "source")


def test_map_numbers_are_cleared():
    v = view("[CH3:7][OH:3]")
    assert all(a.GetAtomMapNum() == 0 for a in v.mol.GetAtoms())


def test_cycle_counts_follow_the_symmetrized_ring_set():
    naphthalene = view("c1ccc2ccccc2c1")
    assert sorted(d["cycles"] for _, d in naphthalene.graph.nodes(data=True)) == [1] * 8 + [2] * 2
    cubane = view("C12C3C4C1C5C2C3C45")
    assert {d["cycles"] for _, d in cubane.graph.nodes(data=True)} == {3}


def test_kekule_and_aromatic_spellings_agree():
    assert view("C1=CC=CC=C1").smiles == view("c1ccccc1").smiles
    assert all(d["aromatic"] for _, d in view("C1=CC=CC=C1").graph.nodes(data=True))


def test_components_are_kept():
    salt = view("[Na+].[Cl-]")
    assert salt.graph.number_of_nodes() == 2 and salt.graph.number_of_edges() == 0


def test_rdkit_mol_input_is_copied_and_cleared():
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("[CH3:1][OH:2]", params)
    v = build_view(parse(mol, "source"), "source", coloring("element-ring"))
    assert mol.GetAtomWithIdx(0).GetAtomMapNum() == 1
    assert v.smiles == view("CO").smiles


def test_ranks_and_classes_cover_every_vertex():
    v = view("Cc1ccccc1")
    assert set(v.ranks) == set(v.graph) == set(v.classes)
    assert len(set(v.ranks.values())) == 7
    assert len(set(v.classes.values())) == 5  # toluene has two pairs of equivalent ring carbons
