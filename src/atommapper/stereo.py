"""Stereo verdicts relative to the correspondence.

CIP labels are reported but never decide a verdict: the comparison maps the source's
controlling atoms through the correspondence and combines the permutation parity with the
two parity descriptors.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

Token = tuple[str, int]
Key = tuple[str, tuple[int, ...]]

_H: Token = ("H", -1)
# The controlling-atom placeholder for an implicit neighbour. RDKit 2026.09 no longer
# exposes StereoInfo.NOATOM but still uses the same value (unsigned -1).
_NOATOM = getattr(Chem.StereoInfo, "NOATOM", 2**32 - 1)
_TETRAHEDRAL = Chem.StereoType.Atom_Tetrahedral
_DOUBLE = Chem.StereoType.Bond_Double
_SPECIFIED = Chem.StereoSpecified.Specified


@dataclass(frozen=True)
class StereoVerdict:
    kind: str
    source: tuple[int, ...]
    target: tuple[int, ...] | None
    status: str
    source_cip: str | None
    target_cip: str | None


def _perceive(mol: Chem.Mol) -> tuple[Chem.Mol, dict[Key, Chem.StereoInfo]]:
    """Every stereo element, possible ones included, by kind and atoms.

    A specified element is taken from a perception that lists specified elements only:
    with ``flagPossible``, RDKit (2026.03) sometimes returns a descriptor that does not
    match the order of the controlling atoms it returns, so the same centre read in another
    atom order compared as inverted. The possible but unspecified elements come from the
    other perception; their descriptors are never compared.
    """
    mol = Chem.Mol(mol)
    possible = Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True, flagPossible=True)
    specified = Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True, flagPossible=False)
    rdCIPLabeler.AssignCIPLabels(mol)
    elements = _by_key(mol, possible)
    elements.update(
        (key, info) for key, info in _by_key(mol, specified).items() if info.specified == _SPECIFIED
    )
    return mol, elements


def _by_key(mol: Chem.Mol, infos: Sequence[Chem.StereoInfo]) -> dict[Key, Chem.StereoInfo]:
    elements: dict[Key, Chem.StereoInfo] = {}
    for info in infos:
        if info.type == _TETRAHEDRAL:
            elements[("atom", (info.centeredOn,))] = info
        elif info.type == _DOUBLE:
            bond = mol.GetBondWithIdx(info.centeredOn)
            elements[("bond", (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))] = info
        elif str(info.type).startswith("Bond"):
            bond = mol.GetBondWithIdx(info.centeredOn)
            elements[("other", (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))] = info
        else:
            elements[("other", (info.centeredOn,))] = info
    return elements


def _cip(mol: Chem.Mol, atoms: tuple[int, ...]) -> str | None:
    if len(atoms) == 1:
        obj = mol.GetAtomWithIdx(atoms[0])
    else:
        obj = mol.GetBondBetweenAtoms(atoms[0], atoms[1])
    if obj is None or not obj.HasProp("_CIPCode"):
        return None
    return obj.GetProp("_CIPCode")


def _parity(source: Sequence[Token], target: Sequence[Token]) -> int | None:
    """Parity of the permutation from source slots to target slots, pairing at most one
    unknown on each side; None when the order cannot be established."""
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        return None
    source_unknown = [i for i, token in enumerate(source) if token not in target]
    target_unknown = [i for i, token in enumerate(target) if token not in source]
    if len(source_unknown) != len(target_unknown) or len(source_unknown) > 1:
        return None
    permutation = [
        target_unknown[0] if i in source_unknown else target.index(token)
        for i, token in enumerate(source)
    ]
    inversions = sum(
        1
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
        if permutation[i] > permutation[j]
    )
    return inversions % 2


def _source_tokens(atoms: Sequence[int], mapping: Mapping[int, int]) -> list[Token]:
    return [
        _H if a == _NOATOM else (("t", mapping[a]) if a in mapping else ("x", a)) for a in atoms
    ]


def _target_tokens(atoms: Sequence[int]) -> list[Token]:
    return [_H if a == _NOATOM else ("t", a) for a in atoms]


def _compare(
    si: Chem.StereoInfo,
    ti: Chem.StereoInfo,
    source_atoms: tuple[int, ...],
    target_atoms: tuple[int, ...],
    mapping: Mapping[int, int],
) -> str:
    """The status of an element present on both sides, in the precedence of the docs:
    whether each side specifies it decides first, the neighbour order only after."""
    source_specified = si.specified == _SPECIFIED
    target_specified = ti.specified == _SPECIFIED
    if source_specified and not target_specified:
        return "lost"
    if target_specified and not source_specified:
        return "gained"
    if not source_specified:
        return "unspecified"
    if si.type == _TETRAHEDRAL:
        src = _source_tokens(list(si.controllingAtoms), mapping)
        tgt = _target_tokens(list(ti.controllingAtoms))
        if len(src) == 3:
            src.append(_H)
        if len(tgt) == 3:
            tgt.append(_H)
        parity = _parity(src, tgt)
    else:
        controlling = list(si.controllingAtoms)
        halves = [controlling[:2], controlling[2:]]
        if mapping[source_atoms[0]] != target_atoms[0]:
            halves.reverse()
        target_controlling = list(ti.controllingAtoms)
        first = _parity(_source_tokens(halves[0], mapping), _target_tokens(target_controlling[:2]))
        second = _parity(_source_tokens(halves[1], mapping), _target_tokens(target_controlling[2:]))
        parity = None if first is None or second is None else (first + second) % 2
    if parity is None:
        return "undetermined"
    same = si.descriptor == ti.descriptor
    return "retained" if same == (parity == 0) else "inverted"


def _same_geometry(a: Chem.Atom, b: Chem.Atom) -> bool:
    """Whether a centre keeps its hybridization and number of neighbours, hydrogens
    included: then a centre that stops being stereogenic does so because two of its
    substituents became alike, not because its geometry changed."""
    return a.GetHybridization() == b.GetHybridization() and a.GetTotalDegree() == b.GetTotalDegree()


def stereo_verdicts(
    source: Chem.Mol, target: Chem.Mol, mapping: Mapping[int, int]
) -> tuple[StereoVerdict, ...]:
    ms, source_elements = _perceive(source)
    mt, target_elements = _perceive(target)
    inverse = {t: s for s, t in mapping.items()}
    verdicts: list[StereoVerdict] = []
    seen: set[Key] = set()

    for (kind, atoms), si in source_elements.items():
        if any(a not in mapping for a in atoms):
            continue
        mapped = tuple(mapping[a] for a in atoms)
        specified = si.specified == _SPECIFIED
        if kind == "other":
            seen.add(("other", mapped))
            verdicts.append(StereoVerdict("other", atoms, mapped, "unsupported", None, None))
            continue
        target_atoms: tuple[int, ...] = mapped
        if kind == "bond":
            bond = mt.GetBondBetweenAtoms(*mapped)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                if specified:
                    verdicts.append(
                        StereoVerdict("bond", atoms, None, "lost", _cip(ms, atoms), None)
                    )
                continue
            target_atoms = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        ti = target_elements.get((kind, target_atoms))
        if ti is None:
            if specified:
                by_symmetry = kind == "bond" or _same_geometry(
                    ms.GetAtomWithIdx(atoms[0]), mt.GetAtomWithIdx(target_atoms[0])
                )
                verdicts.append(
                    StereoVerdict(
                        kind,
                        atoms,
                        target_atoms,
                        "lost_by_symmetry" if by_symmetry else "lost",
                        _cip(ms, atoms),
                        _cip(mt, target_atoms),
                    )
                )
            continue
        seen.add((kind, target_atoms))
        status = _compare(si, ti, atoms, target_atoms, mapping)
        verdicts.append(
            StereoVerdict(
                kind, atoms, target_atoms, status, _cip(ms, atoms), _cip(mt, target_atoms)
            )
        )

    for (kind, atoms), ti in target_elements.items():
        if (kind, atoms) in seen or ti.specified != _SPECIFIED:
            continue
        if any(a not in inverse for a in atoms):
            continue
        source_atoms = tuple(inverse[a] for a in atoms)
        if kind == "other":
            verdicts.append(StereoVerdict("other", source_atoms, atoms, "unsupported", None, None))
            continue
        if kind == "bond":
            bond = ms.GetBondBetweenAtoms(*source_atoms)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                verdicts.append(
                    StereoVerdict("bond", source_atoms, atoms, "gained", None, _cip(mt, atoms))
                )
                continue
        by_symmetry = kind == "bond" or _same_geometry(
            ms.GetAtomWithIdx(source_atoms[0]), mt.GetAtomWithIdx(atoms[0])
        )
        status = "gained_by_desymmetrization" if by_symmetry else "gained"
        verdicts.append(StereoVerdict(kind, source_atoms, atoms, status, None, _cip(mt, atoms)))
    return tuple(verdicts)
