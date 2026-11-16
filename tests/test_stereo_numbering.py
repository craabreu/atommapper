"""A molecule against itself in another atom order: no change, no inverted centre."""

import random

import pytest
from rdkit import Chem

from atommapper import map_atoms

CHIRAL = [
    # RDKit's possible-stereo perception misread this centre in some atom orders
    "C[C@@]1(Cl)CCC(C)(C)CN1",
    "C[C@H](N)C(=O)O",
    "C[C@H]1C[C@@H](O)[C@H](N)C1",
    "CC(=O)N[C@H](C(=O)NC1(C(=O)N[C@@H](CC(N)=O)C(N)=O)CCC1)c1ccc(O)cc1",
    "C/C=C/[C@@H](O)c1ccccc1",
]


@pytest.mark.parametrize("smiles", CHIRAL)
def test_renumbering_changes_nothing(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(0)
    for _ in range(12):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        other = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        result = map_atoms(other, smiles)
        assert result.metadata["size"] == 0
        assert {v["status"] for v in result.metadata["stereo"]} <= {"retained"}
