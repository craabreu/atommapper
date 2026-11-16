"""Checks of each result against its own output, recorded in its metadata."""

from dataclasses import asdict, dataclass

from rdkit import Chem

from .adapter import MolView, round_trip
from .edits import EditScript, apply_edits, reverse_edits, same_graph
from .numbering import Numbering


@dataclass(frozen=True)
class CheckReport:
    identifiers_valid: bool
    elements_preserved: bool | None
    """None when the coloring lets a kept atom change element."""
    apply_reproduces_target: bool
    reverse_reproduces_source: bool
    round_trip: bool
    identical_inputs_give_no_change: bool | None
    """None when the inputs differ, or under a one-piece rule (``map_atoms``) when an
    input has two components with bonds: only one of them can be kept."""

    def as_dict(self) -> dict[str, bool | None]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return all(value is not False for value in self.as_dict().values())


def run_checks(
    source: MolView,
    target: MolView,
    script: EditScript,
    numbering: Numbering,
    reaction: str,
    element_hard: bool = True,
    one_piece: bool = False,
) -> CheckReport:
    source_ids = list(numbering.source.values())
    target_ids = list(numbering.target.values())
    conserved = {numbering.source[s] for s, _ in script.mapping}
    identifiers_valid = (
        all(i > 0 for i in [*source_ids, *target_ids])
        and len(set(source_ids)) == len(source_ids)
        and len(set(target_ids)) == len(target_ids)
        and set(source_ids) & set(target_ids) == conserved
    )
    elements_preserved = (
        all(
            source.graph.nodes[s]["element"] == target.graph.nodes[t]["element"]
            for s, t in script.mapping
        )
        if element_hard
        else None
    )
    return CheckReport(
        identifiers_valid=identifiers_valid,
        elements_preserved=elements_preserved,
        apply_reproduces_target=same_graph(apply_edits(script, source.graph), target.graph),
        reverse_reproduces_source=same_graph(
            apply_edits(reverse_edits(script), target.graph), source.graph
        ),
        round_trip=round_trip(reaction, source, target),
        identical_inputs_give_no_change=(
            script.is_empty
            if source.smiles == target.smiles and not (one_piece and _pieces(source) > 1)
            else None
        ),
    )


def _pieces(view: MolView) -> int:
    """Connected components with at least one bond."""
    return sum(1 for atoms in Chem.GetMolFrags(view.mol) if len(atoms) > 1)
