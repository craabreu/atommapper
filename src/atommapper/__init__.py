"""Fast atom mapping for related molecules with connected cores and complete rings.

Use ``map_atoms`` with SMILES strings or RDKit molecules to obtain a ``MappingResult``.
Configure search limits with ``Settings`` and save pair drawings with ``save_image``.
``calibrate`` compares the virtual seconds of the time limit with seconds of this machine.
"""

from .calibration import calibrate
from .drawing import draw_png, draw_svg, save_image
from .errors import InvalidInput, UnsupportedInput
from .mapper import map_atoms
from .result import AtomChange, BondChange, MappedAtom, MappingResult
from .settings import Settings

__version__ = "0.2.0"

__all__ = [
    "AtomChange",
    "BondChange",
    "InvalidInput",
    "MappedAtom",
    "MappingResult",
    "Settings",
    "UnsupportedInput",
    "__version__",
    "calibrate",
    "draw_png",
    "draw_svg",
    "map_atoms",
    "save_image",
]
