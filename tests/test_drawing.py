import pytest
from rdkit import Chem

from atommapper import draw_png, draw_svg, map_atoms, save_image
from atommapper.cli import main
from atommapper.drawing import (
    ADDED,
    CHANGED,
    LEGEND_HEIGHT,
    REMOVED,
    _framed,
    _legend,
    _misdrawn_double_bonds,
    _panels,
)

WALK = ("Cc1ccc(F)cc1", "Cc1cccc(F)c1")
E_TO_Z = ("C/C=C/C(=O)O", "C/C=C\\C(=O)O")


def hex_color(color: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(255 * c):02X}" for c in color)


def notes(mol: Chem.Mol) -> dict[int, str]:
    return {a.GetIdx(): a.GetProp("atomNote") for a in mol.GetAtoms() if a.HasProp("atomNote")}


def test_what_leaves_is_marked_on_the_source_and_what_arrives_on_the_target():
    source, target, counts = _panels(map_atoms(*WALK), numbers=False)
    # the para fluorine leaves with its bond, the meta fluorine arrives with its own
    assert set(source.atoms.values()) == {REMOVED}
    assert set(target.atoms.values()) == {ADDED}
    assert [source.mol.GetAtomWithIdx(i).GetSymbol() for i in source.atoms] == ["F"]
    assert [target.mol.GetAtomWithIdx(i).GetSymbol() for i in target.atoms] == ["F"]
    assert set(source.bonds.values()) == {REMOVED}
    assert set(target.bonds.values()) == {ADDED}
    one_each = {"kept": 7, "removed": 1, "added": 1, "changed": 0}
    assert counts == {"atoms": one_each, "bonds": one_each}


def test_kept_changes_are_marked_on_both_sides_and_hydrogen_counts_are_not():
    # the carbonyl becomes an alcohol: one order change; two hydrogen counts change
    source, target, counts = _panels(map_atoms("CC=O", "CCO"), numbers=False)
    assert source.atoms == {} and target.atoms == {}
    assert set(source.bonds.values()) == set(target.bonds.values()) == {CHANGED}
    # the change is a bond's, not an atom's
    assert counts["atoms"] == {"kept": 3, "removed": 0, "added": 0, "changed": 0}
    assert counts["bonds"] == {"kept": 2, "removed": 0, "added": 0, "changed": 1}
    # an element change marks the atom on both sides
    source, target, counts = _panels(map_atoms("c1ccncc1", "c1ccccc1"), numbers=False)
    assert len(source.atoms) == len(target.atoms) == 1
    assert set(source.atoms.values()) == {CHANGED}
    assert (counts["atoms"]["changed"], counts["bonds"]["changed"]) == (1, 0)


def test_an_inverted_centre_carries_its_label_on_each_side():
    result = map_atoms("C[C@H](N)O", "C[C@@H](N)O", allow_stereo_inversion=True)
    source, target, _ = _panels(result, numbers=False)
    assert set(source.atoms.values()) == set(target.atoms.values()) == {CHANGED}
    assert sorted(notes(source.mol).values()) + sorted(notes(target.mol).values()) in (
        ["(R)", "(S)"],
        ["(S)", "(R)"],
    )


def test_an_inverted_double_bond_carries_its_label_and_is_drawn_the_right_way():
    result = map_atoms(*E_TO_Z, allow_stereo_inversion=True)
    source, target, _ = _panels(result, numbers=False)
    labels = [
        [b.GetProp("bondNote") for b in panel.mol.GetBonds() if b.HasProp("bondNote")]
        for panel in (source, target)
    ]
    assert labels == [["(E)"], ["(Z)"]]
    # pinned to the source's positions, RDKit would draw the Z target as E
    assert not _misdrawn_double_bonds(source.mol)
    assert not _misdrawn_double_bonds(target.mol)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("C/C=C\\c1ccccc1", "CC/C(F)=C\\c1ccccc1"),
        ("OC(=O)/C=C\\c1ccc(Cl)cc1", "COC(=O)/C=C\\c1ccc(Br)cc1"),
        ("CC/C=C/CC", "CC/C=C\\CC"),
    ],
)
def test_every_double_bond_is_drawn_as_the_molecule_has_it(source, target):
    for panel in _panels(map_atoms(source, target), numbers=False)[:2]:
        assert not _misdrawn_double_bonds(panel.mol)


def test_kept_atoms_sit_where_the_source_has_them():
    source, target, _ = _panels(map_atoms(*WALK), numbers=True)
    numbered = notes(source.mol)
    assert len(numbered) == 7  # the seven kept atoms; the fluorines carry no number
    assert all(a.GetAtomMapNum() == 0 for a in source.mol.GetAtoms())
    here, there = source.mol.GetConformer(), target.mol.GetConformer()
    for index, note in notes(target.mol).items():
        mate = next(i for i, other in numbered.items() if other == note)
        a, b = here.GetAtomPosition(mate), there.GetAtomPosition(index)
        assert (a.x, a.y) == pytest.approx((b.x, b.y))


def test_both_panels_share_one_frame():
    # a benzene against a much larger molecule: the same two corners on both sides, so
    # RDKit gives both panels the same scale and origin
    result = map_atoms("c1ccccc1", "CC(C)(C)c1ccc(C(=O)N2CCOCC2)cc1")
    source, target, _ = _panels(result, numbers=False)
    small, large = _framed((source.mol, target.mol), 400, 300)
    corners = []
    for mol in (small, large):
        conformer = mol.GetConformer()
        ends = [
            conformer.GetAtomPosition(i) for i in (mol.GetNumAtoms() - 2, mol.GetNumAtoms() - 1)
        ]
        corners.append([(round(p.x, 6), round(p.y, 6)) for p in ends])
    assert corners[0] == corners[1]
    assert small.GetNumAtoms() == source.mol.GetNumAtoms() + 2


def test_numbers_are_left_out_unless_asked_for():
    source, _, _ = _panels(map_atoms(*WALK), numbers=False)
    assert notes(source.mol) == {}


COUNTED = [
    ("Cc1ccc(F)cc1", "Cc1cccc(F)c1", {}),
    ("CC=O", "CCO", {}),
    ("c1ccncc1", "c1ccccc1", {}),
    ("CC1CCCCC1", "CC1CCCC1", {}),
    ("C1CCCCC1", "CCCCCC", {"element_change": False}),
    ("CC(=O)N1CCC(c2nc3ccccc3[nH]2)CC1", "CS(=O)(=O)N1CCC(c2nc3cc(Cl)ccc3[nH]2)CC1", {}),
]


@pytest.mark.parametrize(("source", "target", "options"), COUNTED)
def test_the_counts_agree_with_the_result(source, target, options):
    result = map_atoms(source, target, **options)
    panels = _panels(result, numbers=False)
    counts = panels[2]
    assert {unit: counts[unit]["kept"] for unit in counts} == result.metadata["kept"]
    kinds = [c.kind for c in result.atom_changes]
    assert counts["atoms"]["removed"] == kinds.count("deleted")
    assert counts["atoms"]["added"] == kinds.count("inserted")
    kinds = [c.kind for c in result.bond_changes]
    assert counts["bonds"]["removed"] == kinds.count("deleted")
    assert counts["bonds"]["added"] == kinds.count("inserted")
    # every bond of each side is kept, or removed, or added
    for panel, gone in zip(panels[:2], ("removed", "added"), strict=True):
        assert panel.mol.GetNumBonds() == counts["bonds"]["kept"] + counts["bonds"][gone]


def test_an_atom_or_bond_that_changes_in_several_ways_counts_once():
    # the nitrogen changes element and charge: one changed atom
    result = map_atoms("CC[NH3+]", "CCO")
    assert len([c for c in result.atom_changes if c.attribute in ("element", "charge")]) == 2
    assert _panels(result, numbers=False)[2]["atoms"]["changed"] == 1


def test_the_legend_has_a_row_for_atoms_and_a_row_for_bonds():
    # a bond changes and nothing else: no column for removed or added, and an empty cell
    # where the atoms have no change
    counts = _panels(map_atoms("CC=O", "CCO"), numbers=False)[2]
    assert _legend(counts) == [
        [("atoms", None), ("3 kept", None), ("", None)],
        [("bonds", None), ("2 kept", None), ("1 changed", CHANGED)],
    ]
    counts = _panels(map_atoms(*WALK), numbers=False)[2]
    assert _legend(counts) == [
        [("atoms", None), ("7 kept", None), ("1 removed", REMOVED), ("1 added", ADDED)],
        [("bonds", None), ("7 kept", None), ("1 removed", REMOVED), ("1 added", ADDED)],
    ]


def test_svg_and_png_drawings():
    result = map_atoms(*WALK)
    svg = draw_svg(result, width=300, height=200)
    assert "<svg" in svg and "width='600px'" in svg and f"height='{200 + LEGEND_HEIGHT}px'" in svg
    assert hex_color(REMOVED) in svg and hex_color(ADDED) in svg
    assert draw_png(result).startswith(b"\x89PNG")


def test_a_notebook_shows_the_drawing():
    result = map_atoms(*WALK)
    assert result._repr_svg_() == draw_svg(result)
    assert map_atoms("C[H+]", "C")._repr_svg_() is None


def test_a_result_with_nothing_to_draw_is_an_error():
    with pytest.raises(ValueError, match="unsupported_input"):
        draw_svg(map_atoms("C[H+]", "C"))


def test_save_image_follows_the_suffix(tmp_path):
    result = map_atoms(*WALK)
    save_image(result, tmp_path / "pair.svg")
    save_image(result, tmp_path / "pair.PNG")
    assert (tmp_path / "pair.svg").read_text().lstrip().startswith("<?xml")
    assert (tmp_path / "pair.PNG").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match=r"\.svg or \.png"):
        save_image(result, tmp_path / "pair.jpg")


def test_the_cli_draws_to_a_file(tmp_path, capsys):
    path = tmp_path / "pair.png"
    assert main([*WALK, "--draw", str(path), "--size", "300x200"]) == 0
    assert ">>" in capsys.readouterr().out
    plain = path.read_bytes()
    assert plain.startswith(b"\x89PNG")
    assert main([*WALK, "--draw", str(path), "--size", "300x200", "--numbers"]) == 0
    assert path.read_bytes() != plain


@pytest.mark.parametrize("args", [["--draw", "pair.jpg"], ["--draw", "pair.svg", "--size", "3x"]])
def test_the_cli_rejects_a_bad_file_or_size(args):
    with pytest.raises(SystemExit) as exit_info:
        main([*WALK, *args])
    assert exit_info.value.code == 2
