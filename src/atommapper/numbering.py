"""Identifier numbering of the reaction SMILES. No RDKit here."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Numbering:
    source: dict[int, int]
    target: dict[int, int]


def assign_identifiers(
    mapping: Mapping[int, int],
    source_vertices: Iterable[int],
    target_vertices: Iterable[int],
    source_rank: Mapping[int, int],
    target_rank: Mapping[int, int],
) -> Numbering:
    """Conserved atoms 1..n in source canonical order, then source-only atoms, then
    target-only atoms, each in canonical order."""
    conserved = sorted(mapping, key=source_rank.__getitem__)
    image = set(mapping.values())
    source: dict[int, int] = {}
    target: dict[int, int] = {}
    identifier = 0
    for s in conserved:
        identifier += 1
        source[s] = identifier
        target[mapping[s]] = identifier
    for s in sorted((v for v in source_vertices if v not in mapping), key=source_rank.__getitem__):
        identifier += 1
        source[s] = identifier
    for t in sorted((v for v in target_vertices if v not in image), key=target_rank.__getitem__):
        identifier += 1
        target[t] = identifier
    return Numbering(source, target)
