"""Atom mapping of two molecules under LOMAP's rules."""

from .errors import InvalidInput, UnsupportedInput
from .mapper import map_atoms
from .result import AtomChange, BondChange, MappedAtom, MappingResult
from .settings import Settings

__version__ = "0.1.0"

__all__ = [
    "AtomChange",
    "BondChange",
    "InvalidInput",
    "MappedAtom",
    "MappingResult",
    "Settings",
    "UnsupportedInput",
    "__version__",
    "map_atoms",
]
