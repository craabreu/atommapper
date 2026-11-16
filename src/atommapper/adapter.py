"""The RDKit side: parsing, views and serialization."""

from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
from rdkit import Chem, rdBase

from .coloring import Coloring
from .errors import InvalidInput, UnsupportedInput
from .graph import alignment_graph
from .numbering import Numbering
from .stereo import StereoVerdict

INPUT_INDEX = "atommapper_input_index"


@dataclass(frozen=True)
class MolView:
    """One input molecule as the search and the edit extraction see it."""

    mol: Chem.Mol
    graph: nx.Graph
    alignment: nx.Graph
    ranks: dict[int, int]
    classes: dict[int, int]
    input_to_vertex: dict[int, int]
    smiles: str


def parse(molecule: str | Chem.Mol, label: str) -> Chem.Mol:
    """Parse with explicit hydrogens kept, sanitize, and clear map numbers."""
    if isinstance(molecule, Chem.Mol):
        mol = Chem.Mol(molecule)
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
        except Exception as error:  # RDKit raises several sanitization exception types
            raise InvalidInput(f"{label}: sanitization failed: {error}") from error
    elif not isinstance(molecule, str):
        raise InvalidInput(
            f"{label}: expected a SMILES string or an RDKit Mol, got {type(molecule).__name__}"
        )
    else:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        with rdBase.BlockLogs():
            parsed = Chem.MolFromSmiles(molecule, params)
        if parsed is None:
            raise InvalidInput(f"{label}: invalid SMILES {molecule!r}")
        mol = parsed
    if mol.GetNumAtoms() == 0:
        raise InvalidInput(f"{label}: empty molecule")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def _reject_charged_hydrogen(atom: Chem.Atom, label: str) -> None:
    """RDKit would merge a charged or radical hydrogen into its neighbour and lose the charge."""
    if atom.GetAtomicNum() != 1 or atom.GetIsotope() != 0:
        return
    if atom.GetFormalCharge() == 0 and atom.GetNumRadicalElectrons() == 0:
        return
    heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
    if atom.GetDegree() == 1 and len(heavy) == 1:
        raise UnsupportedInput(
            f"{label}: charged or radical hydrogen (input atom {atom.GetIdx()}) bonded to "
            f"atom {heavy[0].GetIdx()}"
        )


def build_view(mol: Chem.Mol, label: str, coloring: Coloring) -> MolView:
    """Merge ordinary hydrogens and build the attributed and alignment graphs, colored by
    ``coloring``."""
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp(INPUT_INDEX, atom.GetIdx())
        _reject_charged_hydrogen(atom, label)
    with rdBase.BlockLogs():
        heavy = Chem.RemoveHs(mol)
    rings = heavy.GetRingInfo()
    graph = nx.Graph()
    for atom in heavy.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            element=atom.GetAtomicNum(),
            charge=atom.GetFormalCharge(),
            isotope=atom.GetIsotope(),
            hcount=atom.GetTotalNumHs(),
            radicals=atom.GetNumRadicalElectrons(),
            aromatic=atom.GetIsAromatic(),
            cycles=rings.NumAtomRings(atom.GetIdx()),
            index=atom.GetIntProp(INPUT_INDEX),
        )
    for bond in heavy.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(
            u,
            v,
            order=str(bond.GetBondType()),
            aromatic=bond.GetIsAromatic(),
        )
    ranks = {i: int(r) for i, r in enumerate(Chem.CanonicalRankAtoms(heavy, breakTies=True))}
    classes = {i: int(r) for i, r in enumerate(Chem.CanonicalRankAtoms(heavy, breakTies=False))}
    return MolView(
        mol=heavy,
        graph=graph,
        alignment=alignment_graph(graph, coloring),
        ranks=ranks,
        classes=classes,
        input_to_vertex={graph.nodes[v]["index"]: v for v in graph},
        smiles=Chem.MolToSmiles(heavy),
    )


@dataclass(frozen=True)
class RawTagCheck:
    kind: str
    identifiers: tuple[int, ...]
    agree: bool


def _numbered(mol: Chem.Mol, identifiers: Mapping[int, int]) -> Chem.Mol:
    """A copy with map numbers set."""
    rw = Chem.RWMol(mol)
    for vertex, identifier in identifiers.items():
        rw.GetAtomWithIdx(vertex).SetAtomMapNum(identifier)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def _in_identifier_order(mol: Chem.Mol) -> str:
    order = sorted(range(mol.GetNumAtoms()), key=lambda i: mol.GetAtomWithIdx(i).GetAtomMapNum())
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


def write_reaction(source: MolView, target: MolView, numbering: Numbering) -> str:
    """``source>>target`` with every heavy atom numbered; hydrogens stay implicit."""
    left = _numbered(source.mol, numbering.source)
    right = _numbered(target.mol, numbering.target)
    return f"{_in_identifier_order(left)}>>{_in_identifier_order(right)}"


def parse_reaction(reaction: str) -> tuple[Chem.Mol, Chem.Mol]:
    left, right = reaction.split(">>")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    with rdBase.BlockLogs():
        source = Chem.MolFromSmiles(left, params)
        target = Chem.MolFromSmiles(right, params)
    if source is None or target is None:
        raise ValueError(f"unparsable reaction SMILES {reaction!r}")
    return source, target


def round_trip(reaction: str, source: MolView, target: MolView) -> bool:
    """Stripping the numbers recovers both input identities."""
    try:
        sides = parse_reaction(reaction)
    except ValueError:
        return False
    for mol, view in zip(sides, (source, target), strict=True):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        with rdBase.BlockLogs():
            stripped = Chem.RemoveHs(mol)
        if Chem.MolToSmiles(stripped) != view.smiles:
            return False
    return True


def raw_tag_agreement(
    reaction: str, verdicts: tuple[StereoVerdict, ...], numbering: Numbering
) -> tuple[RawTagCheck, ...]:
    """For each retained stereo element, whether the written raw tags of both sides agree."""
    left, right = parse_reaction(reaction)
    left_index = {a.GetAtomMapNum(): a.GetIdx() for a in left.GetAtoms()}
    right_index = {a.GetAtomMapNum(): a.GetIdx() for a in right.GetAtoms()}
    checks = []
    for verdict in verdicts:
        if verdict.status != "retained":
            continue
        ids = tuple(numbering.source[a] for a in verdict.source)
        if verdict.kind == "atom":
            agree = (
                left.GetAtomWithIdx(left_index[ids[0]]).GetChiralTag()
                == right.GetAtomWithIdx(right_index[ids[0]]).GetChiralTag()
            )
        else:
            a = left.GetBondBetweenAtoms(left_index[ids[0]], left_index[ids[1]])
            b = right.GetBondBetweenAtoms(right_index[ids[0]], right_index[ids[1]])
            agree = a is not None and b is not None and a.GetStereo() == b.GetStereo()
        checks.append(RawTagCheck(verdict.kind, ids, agree))
    return tuple(checks)
