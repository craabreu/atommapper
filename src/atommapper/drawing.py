"""Drawings of a mapped pair: the source and the target side by side.

Both panels share one scale and one origin, and the target is laid out with its kept atoms
where the source has them, so a kept atom sits at the same place in each panel. What
leaves is marked on the source and what arrives on the target; kept atoms and bonds that
change (element, charge, order, stereo) are marked on both, a changed stereo element with
its label on each side. A changed hydrogen count is not marked: a bond that forms or
breaks at the atom already shows it. The marks are light tints under the structure, of
three colors of the Okabe-Ito palette chosen to stay apart under deuteranopia, protanopia
and tritanopia: their OKLab distance (times 100) is at least 14 between any two under each,
simulated as Machado, Oliveira and Fernandes (2009) do at full severity, and at least 18
with full color vision. A legend under the panels counts, for atoms and for bonds apart,
what is kept, removed, added and changed.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem, Geometry
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from .adapter import parse_reaction

if TYPE_CHECKING:
    from .result import MappingResult

Color = tuple[float, float, float]


def _rgb(code: str) -> Color:
    """An ``RRGGBB`` hex code as RDKit's fractions."""
    red, green, blue = (int(code[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (red, green, blue)


def _tint(color: Color, white: float) -> Color:
    """``color`` mixed with a fraction ``white`` of white, light enough to sit under bonds
    and labels: the legend's dark gray text (``_INK``) keeps a WCAG contrast of at least 5:1
    on each tint below, and black bonds more. RDKit's colored atom labels (red oxygen, blue
    nitrogen) keep less, down to about 2:1 on the vermillion tint."""
    red, green, blue = (c + (1 - c) * white for c in color)
    return (red, green, blue)


REMOVED: Color = _tint(_rgb("D55E00"), 0.45)
"""Atoms and bonds only in the source (a tint of vermillion)."""
ADDED: Color = _tint(_rgb("56B4E9"), 0.45)
"""Atoms and bonds only in the target (a tint of sky blue)."""
CHANGED: Color = _tint(_rgb("F0E442"), 0.25)
"""Kept atoms and bonds that change (a tint of yellow, less white since yellow is light
already)."""

UNMARKED_ATTRIBUTES = frozenset({"hcount"})
"""Atom changes left unmarked: implied by the bonds that form or break."""

LEGEND_HEIGHT = 58
"""Pixels under the panels for the legend: one row for atoms, one for bonds."""
Counts = dict[str, dict[str, int]]
"""Per unit (``"atoms"``, ``"bonds"``), how many are kept, removed, added and changed."""
_MAX_BOND_PIXELS = 42
"""A bond is never drawn longer than this, so small molecules are not blown up."""
_BOND_LENGTH = 1.5
"""The bond length of RDKit's depictions."""
_INK: Color = (0.25, 0.25, 0.25)
_E_Z = (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)


class _Panel:
    """One side of the drawing: its molecule and what to mark on it."""

    def __init__(self, mol: Chem.Mol) -> None:
        self.mol = mol
        self.index = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        self.atoms: dict[int, Color] = {}
        self.bonds: dict[int, Color] = {}

    def mark_atom(self, identifier: int, color: Color) -> Chem.Atom | None:
        if identifier not in self.index:
            return None
        self.atoms[self.index[identifier]] = color
        return self.mol.GetAtomWithIdx(self.index[identifier])

    def mark_bond(self, identifiers: tuple[int, int], color: Color) -> Chem.Bond | None:
        a, b = identifiers
        if a not in self.index or b not in self.index:
            return None
        bond = self.mol.GetBondBetweenAtoms(self.index[a], self.index[b])
        if bond is not None:
            self.bonds[bond.GetIdx()] = color
        return bond

    def side(self, start: int, blocked: int) -> set[int]:
        """Atoms reached from ``start`` without passing through ``blocked``, ``start``
        left out."""
        seen, stack = {start}, [start]
        while stack:
            for neighbor in self.mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                if neighbor.GetIdx() not in seen and neighbor.GetIdx() != blocked:
                    seen.add(neighbor.GetIdx())
                    stack.append(neighbor.GetIdx())
        return seen - {start}


def _panels(result: "MappingResult", numbers: bool) -> tuple[_Panel, _Panel, Counts]:
    """Both sides, marked and laid out, and the counts of what was marked: an atom or a
    bond that changes in several ways counts once."""
    if not result.reaction_smiles:
        raise ValueError(f"nothing to draw: the result is {result.status!r}")
    source, target = (_Panel(mol) for mol in parse_reaction(result.reaction_smiles))
    changed_atoms: set[int] = set()
    changed_bonds: set[tuple[int, int]] = set()
    notes: dict[tuple[int, int], str] = {}
    for change in result.atom_changes:
        if change.kind == "deleted":
            source.mark_atom(change.identifier, REMOVED)
        elif change.kind == "inserted":
            target.mark_atom(change.identifier, ADDED)
        elif change.attribute not in UNMARKED_ATTRIBUTES:
            changed_atoms.add(change.identifier)
            sides = ((source, change.before), (target, change.after))
            for side, (panel, label) in enumerate(sides):
                atom = panel.mark_atom(change.identifier, CHANGED)
                if change.kind == "stereo" and atom is not None and label:
                    notes[side, atom.GetIdx()] = f"({label})"
    for change in result.bond_changes:
        if change.kind == "deleted":
            source.mark_bond(change.identifiers, REMOVED)
        elif change.kind == "inserted":
            target.mark_bond(change.identifiers, ADDED)
        else:
            changed_bonds.add(change.identifiers)
            for panel, label in ((source, change.before), (target, change.after)):
                bond = panel.mark_bond(change.identifiers, CHANGED)
                if change.kind == "stereo" and bond is not None and label:
                    bond.SetProp("bondNote", f"({label})")
    kept = source.index.keys() & target.index.keys()
    for side, panel in enumerate((source, target)):
        for atom in panel.mol.GetAtoms():
            note = [str(atom.GetAtomMapNum())] if numbers and atom.GetAtomMapNum() in kept else []
            note += [notes[side, atom.GetIdx()]] if (side, atom.GetIdx()) in notes else []
            if note:
                atom.SetProp("atomNote", " ".join(note))
            atom.SetAtomMapNum(0)
    _lay_out(source, target, kept)
    kept_bonds = sum(
        1
        for bond in source.mol.GetBonds()
        if bond.GetIdx() not in source.bonds or source.bonds[bond.GetIdx()] != REMOVED
    )
    counts = {
        "atoms": {
            "kept": len(kept),
            "removed": sum(color == REMOVED for color in source.atoms.values()),
            "added": sum(color == ADDED for color in target.atoms.values()),
            "changed": len(changed_atoms),
        },
        "bonds": {
            "kept": kept_bonds,
            "removed": sum(color == REMOVED for color in source.bonds.values()),
            "added": sum(color == ADDED for color in target.bonds.values()),
            "changed": len(changed_bonds),
        },
    }
    return source, target, counts


def _lay_out(source: _Panel, target: _Panel, kept: set[int]) -> None:
    """Depict the source, then the target with its kept atoms where the source has them."""
    rdDepictor.Compute2DCoords(source.mol, useRingTemplates=True)
    positions = source.mol.GetConformer()
    fixed = {
        target.index[i]: Geometry.Point2D(
            positions.GetAtomPosition(source.index[i]).x,
            positions.GetAtomPosition(source.index[i]).y,
        )
        for i in kept
    }
    try:
        rdDepictor.Compute2DCoords(target.mol, coordMap=fixed, useRingTemplates=True)
    except (RuntimeError, ValueError):
        rdDepictor.Compute2DCoords(target.mol, useRingTemplates=True)
    if not _fix_double_bonds(target):
        # A right drawing comes before an aligned one.
        rdDepictor.Compute2DCoords(target.mol, useRingTemplates=True)


def _misdrawn_double_bonds(mol: Chem.Mol) -> list[Chem.Bond]:
    """Double bonds whose 2D coordinates show the other of E and Z."""
    drawn = Chem.Mol(mol)
    for bond in drawn.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.DetectBondStereochemistry(drawn)
    Chem.AssignStereochemistry(drawn, cleanIt=True, force=True)
    return [
        bond
        for bond, shown in zip(mol.GetBonds(), drawn.GetBonds(), strict=True)
        if bond.GetStereo() in _E_Z and shown.GetStereo() in _E_Z
        if bond.GetStereo() != shown.GetStereo()
    ]


def _fix_double_bonds(panel: _Panel) -> bool:
    """Mirror the smaller side of each double bond drawn the wrong way; False when one
    cannot be mirrored (a ring bond).

    With pinned coordinates RDKit lays out a molecule without reading its double-bond
    stereo, even for the atoms left free, so a Z bond can come out drawn as E."""
    conformer = panel.mol.GetConformer()
    for _ in range(panel.mol.GetNumBonds()):
        misdrawn = _misdrawn_double_bonds(panel.mol)
        if not misdrawn:
            return True
        bond = misdrawn[0]
        if bond.IsInRing():
            return False
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        side_a, side_b = panel.side(a, b), panel.side(b, a)
        side, origin, other = (side_a, a, b) if len(side_a) <= len(side_b) else (side_b, b, a)
        start, end = conformer.GetAtomPosition(origin), conformer.GetAtomPosition(other)
        ux, uy = end.x - start.x, end.y - start.y
        length = (ux * ux + uy * uy) ** 0.5
        ux, uy = ux / length, uy / length
        for index in side:
            point = conformer.GetAtomPosition(index)
            dx, dy = point.x - start.x, point.y - start.y
            along = dx * ux + dy * uy
            mirrored = Geometry.Point3D(
                start.x + 2 * along * ux - dx, start.y + 2 * along * uy - dy, 0.0
            )
            conformer.SetAtomPosition(index, mirrored)
    return not _misdrawn_double_bonds(panel.mol)


def _framed(mols: tuple[Chem.Mol, Chem.Mol], width: int, height: int) -> list[Chem.Mol]:
    """Copies of the molecules with two unseen atoms at the corners of one frame, so that
    both panels get the same scale and origin (RDKit fits each molecule to its panel,
    whatever scale is set beforehand). The frame is at least as large as a panel of bonds
    ``_MAX_BOND_PIXELS`` long."""
    points = [
        mol.GetConformer().GetAtomPosition(i) for mol in mols for i in range(mol.GetNumAtoms())
    ]
    low_x, high_x = min(p.x for p in points), max(p.x for p in points)
    low_y, high_y = min(p.y for p in points), max(p.y for p in points)
    centre_x, centre_y = (low_x + high_x) / 2, (low_y + high_y) / 2
    margin = 0.6 * _BOND_LENGTH
    half_width = max((high_x - low_x) / 2 + margin, width / _MAX_BOND_PIXELS * _BOND_LENGTH / 2)
    half_height = max((high_y - low_y) / 2 + margin, height / _MAX_BOND_PIXELS * _BOND_LENGTH / 2)
    corners = (
        (centre_x - half_width, centre_y - half_height),
        (centre_x + half_width, centre_y + half_height),
    )
    framed = []
    for mol in mols:
        editable = Chem.RWMol(mol)
        conformer = editable.GetConformer()
        for x, y in corners:
            corner = Chem.Atom(0)
            corner.SetProp("_displayLabel", " ")
            corner.SetNoImplicit(True)
            conformer.SetAtomPosition(editable.AddAtom(corner), Geometry.Point3D(x, y, 0.0))
        framed.append(editable.GetMol())
    return framed


_TINTS: dict[str, Color | None] = {
    "kept": None,
    "removed": REMOVED,
    "added": ADDED,
    "changed": CHANGED,
}


def _legend(counts: Counts) -> list[list[tuple[str, Color | None]]]:
    """The legend as a table: a row for atoms and a row for bonds, each starting with its
    unit, then one cell per kind of count that is not zero in both rows. A cell is its
    text and the tint of its chip; a zero is an empty cell, so the columns stay aligned."""
    kinds = [
        kind for kind in _TINTS if kind == "kept" or any(counts[unit][kind] for unit in counts)
    ]
    return [
        [(unit, None)]
        + [
            (f"{counts[unit][kind]} {kind}", _TINTS[kind])
            if counts[unit][kind] or kind == "kept"
            else ("", None)
            for kind in kinds
        ]
        for unit in ("atoms", "bonds")
    ]


def _draw_legend(
    drawer: rdMolDraw2D.MolDraw2D,
    rows: list[list[tuple[str, Color | None]]],
    width: int,
    height: int,
) -> None:
    """The table centred under the two panels. Each text is centred in a cell sized from
    the longest text of its column, so the table is centred without measuring text."""
    point = Geometry.Point2D
    gap, row_height = 10, 24
    widths = [8 * max(len(row[k][0]) for row in rows) + 18 for k in range(len(rows[0]))]
    start = width - (sum(widths) + gap * (len(widths) - 1)) / 2
    drawer.SetFontSize(13)
    drawer.SetFillPolys(True)
    for index, row in enumerate(rows):
        middle = height + (LEGEND_HEIGHT - row_height * len(rows)) / 2 + row_height * (index + 0.5)
        left = start
        for (text, tint), cell in zip(row, widths, strict=True):
            if tint is not None:
                drawer.SetColour(tint)
                drawer.DrawRect(
                    point(left, middle - 9), point(left + cell, middle + 9), rawCoords=True
                )
            if text:
                drawer.SetColour(_INK)
                drawer.DrawString(text, point(left + cell / 2, middle), 0, rawCoords=True)
            left += cell + gap


def _draw(
    drawer: rdMolDraw2D.MolDraw2D,
    result: "MappingResult",
    width: int,
    height: int,
    numbers: bool,
) -> rdMolDraw2D.MolDraw2D:
    source, target, counts = _panels(result, numbers)
    options = drawer.drawOptions()
    options.padding = 0.02
    options.highlightRadius = 0.32
    options.highlightBondWidthMultiplier = 16
    options.annotationFontScale = 0.75
    drawer.DrawMolecules(
        _framed((source.mol, target.mol), width, height),
        highlightAtoms=[list(source.atoms), list(target.atoms)],
        highlightAtomColors=[source.atoms, target.atoms],
        highlightBonds=[list(source.bonds), list(target.bonds)],
        highlightBondColors=[source.bonds, target.bonds],
    )
    drawer.SetOffset(0, 0)
    point = Geometry.Point2D
    drawer.DrawArrow(
        point(width - 16, height / 2),
        point(width + 16, height / 2),
        color=_INK,
        rawCoords=True,
        frac=0.4,
    )
    _draw_legend(drawer, _legend(counts), width, height)
    drawer.FinishDrawing()
    return drawer


def draw_svg(
    result: "MappingResult", *, width: int = 400, height: int = 300, numbers: bool = False
) -> str:
    """The source and target side by side as SVG text, each panel ``width`` by ``height``
    pixels, over a legend ``LEGEND_HEIGHT`` pixels high; ``numbers`` labels the kept atoms
    with their identifiers.

    Raises ``ValueError`` for a result with nothing to draw (``unsupported_input``)."""
    drawer = rdMolDraw2D.MolDraw2DSVG(2 * width, height + LEGEND_HEIGHT, width, height)
    return _draw(drawer, result, width, height, numbers).GetDrawingText()


def draw_png(
    result: "MappingResult", *, width: int = 400, height: int = 300, numbers: bool = False
) -> bytes:
    """As ``draw_svg``, as PNG bytes."""
    drawer = rdMolDraw2D.MolDraw2DCairo(2 * width, height + LEGEND_HEIGHT, width, height)
    return _draw(drawer, result, width, height, numbers).GetDrawingText()


def save_image(
    result: "MappingResult",
    path: str | Path,
    *,
    width: int = 400,
    height: int = 300,
    numbers: bool = False,
) -> None:
    """Write the drawing to ``path``, as SVG or PNG by its suffix (``.svg`` or ``.png``).

    Raises ``ValueError`` for another suffix or a result with nothing to draw."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".svg":
        path.write_text(draw_svg(result, width=width, height=height, numbers=numbers))
    elif suffix == ".png":
        path.write_bytes(draw_png(result, width=width, height=height, numbers=numbers))
    else:
        raise ValueError(f"the image format follows the suffix, .svg or .png; got {path.name!r}")
