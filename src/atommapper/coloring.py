"""Colorings: which atoms can be the same atom.

A coloring names the hard node attributes: two atoms whose colors differ are never paired,
by the search or by the fragment engine. Bond order is soft: a kept bond whose order
changes counts one. Element is soft when the coloring leaves it out (``ring``, used with
``element_change``): a kept atom whose element changes counts one.
Everything else about an atom or bond is reported in the output, not compared.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Coloring:
    name: str
    hard: tuple[str, ...]
    """Node attributes compared for identity: ``"element"`` (the atomic number) and
    ``"ring"`` (whether the atom lies in a ring of the heavy-atom graph)."""

    def color(self, attrs: Mapping[str, Any]) -> tuple[int, ...]:
        """The color of an attributed-graph node."""
        values = {"element": int(attrs["element"]), "ring": int(int(attrs["cycles"]) > 0)}
        return tuple(values[key] for key in self.hard)

    @property
    def element_hard(self) -> bool:
        """Whether every pair keeps its element."""
        return "element" in self.hard


COLORINGS: dict[str, Coloring] = {
    "element-ring": Coloring("element-ring", ("element", "ring")),
    "ring": Coloring("ring", ("ring",)),
}


def coloring(name: str) -> Coloring:
    """The preset called ``name``."""
    try:
        return COLORINGS[name]
    except KeyError:
        raise KeyError(f"unknown coloring {name!r}; valid: {', '.join(COLORINGS)}") from None
