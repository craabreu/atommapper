"""The result of a mapping and its assembly from a correspondence."""

from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import version
from typing import Any

import networkx
import rdkit
import scipy

from .adapter import MolView, raw_tag_agreement, write_reaction
from .checks import run_checks
from .clock import reference_id
from .edits import EditScript, compute_edits
from .numbering import Numbering, assign_identifiers
from .settings import Settings
from .stereo import StereoVerdict, stereo_verdicts


@dataclass(frozen=True)
class MappedAtom:
    """One atom correspondence.

    ``identifier`` is its one-based atom-map number in ``reaction_smiles``.
    ``source_index`` and ``target_index`` are zero-based indices in the inputs.
    ``tag`` is "seed" or "refined", describing search provenance, not confidence.
    """

    identifier: int
    source_index: int
    target_index: int
    tag: str


@dataclass(frozen=True)
class AtomChange:
    """An atom edit keyed by its atom-map ``identifier``, not its input index.

    ``kind`` is "deleted", "inserted", "changed", or "stereo". For "changed",
    ``attribute`` names the changed property and ``before``/``after`` hold its values.
    For "stereo", attribute is the stereo verdict and the values are the CIP labels
    when available. Insertions and deletions leave these optional fields as None.
    """

    kind: str
    identifier: int
    attribute: str | None = None
    before: Any = None
    after: Any = None


@dataclass(frozen=True)
class BondChange:
    """A bond edit keyed by the atom-map ``identifiers`` of its two endpoints.

    ``kind`` is "deleted", "inserted", "changed", or "stereo". Attribute changes
    include their ``attribute``, ``before``, and ``after`` values. For stereo changes,
    attribute is the verdict and the values are the E/Z labels when available.
    These identifiers are not input atom indices or RDKit bond indices.
    """

    kind: str
    identifiers: tuple[int, int]
    attribute: str | None = None
    before: Any = None
    after: Any = None


@dataclass(frozen=True)
class _Context:
    source: MolView
    target: MolView
    mapping: dict[int, int]
    tags: dict[int, str]
    element_hard: bool


@dataclass(frozen=True)
class MappingResult:
    """Atom mapping, edits, and diagnostics returned by :func:`map_atoms`.

    ``reaction_smiles`` uses shared atom-map numbers for mapped atoms and distinct
    numbers for source-only and target-only atoms. ``mapping`` connects these numbers
    to input atom indices; ``atom_changes`` and ``bond_changes`` use the map numbers.

    Check ``status`` before consuming the result: "complete", "timed_out" (a partial
    result after a work or time limit), "no_common_vertices" (empty mapping), or
    "unsupported_input" (empty mapped SMILES). ``reason`` explains non-complete outcomes.
    "complete" does not imply global optimality; consult ``metadata["optimality"]``.
    Unsupported results have only version and settings metadata.

    Display the result in a notebook to see an SVG drawing, or use ``save_image``.
    Use ``to_dict()`` to serialize its public fields without internal molecule objects.
    """

    reaction_smiles: str
    mapping: tuple[MappedAtom, ...]
    atom_changes: tuple[AtomChange, ...]
    bond_changes: tuple[BondChange, ...]
    status: str
    reason: str | None
    metadata: dict[str, Any]
    _context: _Context | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return public fields as plain data, excluding the internal molecule context.

        Records become dictionaries and lists. The metadata dictionary is shared with
        the result, not deep-copied. The output can be passed to ``json.dumps``.
        """
        return {
            "reaction_smiles": self.reaction_smiles,
            "mapping": [asdict(m) for m in self.mapping],
            "atom_changes": [asdict(c) for c in self.atom_changes],
            "bond_changes": [asdict(c) for c in self.bond_changes],
            "status": self.status,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    def _repr_svg_(self) -> str | None:
        """The drawing a notebook shows (``drawing.draw_svg``); none for a result with
        nothing to draw."""
        from .drawing import draw_svg

        return draw_svg(self) if self.reaction_smiles else None

    def reverse(self) -> "MappingResult":
        """Return the same correspondence in the opposite direction without a new search.

        Swap source and target indices, invert the edits, and regenerate map numbers.
        Search diagnostics describe the original search; ``metadata["reversed"]``
        tracks the direction. The original result is not modified.
        """
        metadata = {**self.metadata, "reversed": not self.metadata.get("reversed", False)}
        context = self._context
        if context is None:
            return replace(self, metadata=metadata)
        inverse = {t: s for s, t in context.mapping.items()}
        tags = {context.mapping[s]: tag for s, tag in context.tags.items()}
        return assemble(
            context.target,
            context.source,
            inverse,
            tags,
            element_hard=context.element_hard,
            status=self.status,
            reason=self.reason,
            metadata=metadata,
        )


def versions() -> dict[str, str]:
    return {
        "atommapper": version("atommapper"),
        "clock": reference_id(),
        "rdkit": rdkit.__version__,
        "networkx": networkx.__version__,
        "scipy": scipy.__version__,
    }


def unsupported(reason: str, settings: Settings) -> MappingResult:
    metadata = {"versions": versions(), "settings": asdict(settings)}
    return MappingResult("", (), (), (), "unsupported_input", reason, metadata)


UNCHANGED_STEREO = frozenset({"retained", "unspecified"})
"""Verdicts that are not changes: they appear in ``metadata["stereo"]`` only."""


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _changes(
    script: EditScript, verdicts: tuple[StereoVerdict, ...], numbering: Numbering
) -> tuple[tuple[AtomChange, ...], tuple[BondChange, ...]]:
    src, tgt = numbering.source, numbering.target
    atoms = [AtomChange("deleted", src[s]) for s, _ in script.deleted_vertices]
    atoms += [AtomChange("inserted", tgt[t]) for t, _ in script.inserted_vertices]
    atoms += [
        AtomChange("changed", src[s], key, before, after)
        for s, _t, key, before, after in script.vertex_changes
    ]
    bonds = [BondChange("deleted", _pair(src[u], src[v])) for u, v, _ in script.deleted_edges]
    bonds += [BondChange("inserted", _pair(tgt[u], tgt[v])) for u, v, _ in script.inserted_edges]
    bonds += [
        BondChange("changed", _pair(src[su], src[sv]), key, before, after)
        for su, sv, _tu, _tv, key, before, after in script.edge_changes
    ]
    for verdict in verdicts:
        if verdict.status in UNCHANGED_STEREO:
            continue
        ids = tuple(src[a] for a in verdict.source)
        if len(ids) == 2:
            bonds.append(
                BondChange(
                    "stereo", _pair(*ids), verdict.status, verdict.source_cip, verdict.target_cip
                )
            )
        else:
            atoms.append(
                AtomChange("stereo", ids[0], verdict.status, verdict.source_cip, verdict.target_cip)
            )
    atoms.sort(key=lambda c: (c.identifier, c.kind, c.attribute or ""))
    bonds.sort(key=lambda c: (c.identifiers, c.kind, c.attribute or ""))
    return tuple(atoms), tuple(bonds)


def assemble(
    source: MolView,
    target: MolView,
    mapping: dict[int, int],
    tags: dict[int, str],
    *,
    element_hard: bool,
    status: str,
    reason: str | None,
    metadata: dict[str, Any],
) -> MappingResult:
    """Edits, numbering, reaction SMILES, stereo and checks for a fixed correspondence that
    keeps one connected piece (``run_checks``); ``element_hard`` when every pair keeps its
    element. Hydrogens stay implicit: a changed count is an attribute change of its atom."""
    script = compute_edits(source.graph, target.graph, mapping)
    numbering = assign_identifiers(mapping, source.graph, target.graph, source.ranks, target.ranks)
    reaction = write_reaction(source, target, numbering)
    verdicts = stereo_verdicts(source.mol, target.mol, mapping)
    raw_tags = raw_tag_agreement(reaction, verdicts, numbering)
    checks = run_checks(
        source,
        target,
        script,
        numbering,
        reaction,
        element_hard=element_hard,
        one_piece=True,
    )
    atom_changes, bond_changes = _changes(script, verdicts, numbering)
    mapped = tuple(
        MappedAtom(
            numbering.source[s],
            source.graph.nodes[s]["index"],
            target.graph.nodes[t]["index"],
            tags.get(s, "refined"),
        )
        for s, t in sorted(mapping.items(), key=lambda st: numbering.source[st[0]])
    )
    metadata = {
        **metadata,
        "coverage": {
            "source": len(mapping) / source.graph.number_of_nodes(),
            "target": len(mapping) / target.graph.number_of_nodes(),
        },
        "checks": checks.as_dict(),
        "stereo": [
            {
                "kind": v.kind,
                "identifiers": [numbering.source[a] for a in v.source],
                "status": v.status,
                "source_cip": v.source_cip,
                "target_cip": v.target_cip,
            }
            for v in verdicts
        ],
        "stereo_raw_tags": [
            {"kind": c.kind, "identifiers": list(c.identifiers), "agree": c.agree} for c in raw_tags
        ],
        "stereo_raw_tags_disagreeing": sum(1 for c in raw_tags if not c.agree),
    }
    return MappingResult(
        reaction,
        mapped,
        atom_changes,
        bond_changes,
        status,
        reason,
        metadata,
        _Context(source, target, dict(mapping), dict(tags), element_hard),
    )
