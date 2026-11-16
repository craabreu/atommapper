from rdkit import Chem

from atommapper import map_atoms
from atommapper.adapter import (
    build_view,
    parse,
    parse_reaction,
    raw_tag_agreement,
    round_trip,
    write_reaction,
)
from atommapper.coloring import coloring
from atommapper.numbering import assign_identifiers
from atommapper.stereo import stereo_verdicts


def prepared(source: str, target: str):
    colors = coloring("ring")
    s = build_view(parse(source, "source"), "source", colors)
    t = build_view(parse(target, "target"), "target", colors)
    result = map_atoms(source, target)
    mapping = {
        s.input_to_vertex[a.source_index]: t.input_to_vertex[a.target_index] for a in result.mapping
    }
    numbering = assign_identifiers(mapping, s.graph, t.graph, s.ranks, t.ranks)
    return s, t, mapping, numbering, write_reaction(s, t, numbering)


def map_numbers(mol: Chem.Mol) -> list[int]:
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]


def test_every_atom_is_numbered_and_conserved_atoms_share_numbers():
    _, _, mapping, numbering, reaction = prepared("CCO", "CC=O")
    left, right = parse_reaction(reaction)
    assert all(n > 0 for n in map_numbers(left) + map_numbers(right))
    shared = set(map_numbers(left)) & set(map_numbers(right))
    assert shared == {numbering.source[v] for v in mapping}


def test_hydrogens_stay_implicit():
    _, _, _, _, reaction = prepared("CCO", "CC=O")
    left, right = parse_reaction(reaction)
    assert not any(a.GetAtomicNum() == 1 for a in [*left.GetAtoms(), *right.GetAtoms()])


def test_round_trip_recovers_both_inputs_and_detects_a_swap():
    s, t, _, _, reaction = prepared("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)OC")
    assert round_trip(reaction, s, t)
    left, right = reaction.split(">>")
    assert not round_trip(f"{right}>>{left}", s, t)


def test_raw_tags_of_a_retained_center_written_alike_agree():
    s, t, mapping, numbering, reaction = prepared("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)OC")
    verdicts = stereo_verdicts(s.mol, t.mol, mapping)
    checks = raw_tag_agreement(reaction, verdicts, numbering)
    assert [c.kind for c in checks] == ["atom"]
    assert checks[0].agree
