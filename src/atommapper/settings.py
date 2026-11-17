"""Budgets and limits of a mapping call."""

import math
from dataclasses import dataclass, fields

_COUNTS = {
    # field: smallest value allowed
    "node_budget": 0,
    "fragment_steps": 0,
    "coarse_fragment_steps": 0,
    "fragment_bond_pairs": 0,
    "max_fragments": 1,
    "max_alternatives": 1,
}
_OPTIONAL_COUNTS = {"k_seeds": 1}
_SECONDS = ("time_limit_s", "global_budget_s")


@dataclass(frozen=True)
class Settings:
    node_budget: int = 1_000_000
    """Branch-and-bound nodes for the whole call, shared across seeds. A work budget, so
    results do not depend on machine speed or load."""
    time_limit_s: float = 60.0
    """Wall-clock limit for the whole call, from its start: the fragment search (every
    1,024 steps of the engine, about a millisecond; more often on large search graphs, and
    while those are built), seeding and refinement all read it, so a call ends within tens
    of milliseconds of it. Parsing and stereo perception run to the end. A call stopped
    while the fragment search still builds its search graph has no fragment to start from.
    A result the clock stopped reports ``"time"`` in ``metadata["budget"]["stopped_by"]``
    and depends on machine speed and load; a result within the limit is the one the work
    budgets alone give."""
    fragment_steps: int = 20_000_000
    """Search steps for the fragment search over the whole call, the residual searches
    included, shared across seeds like the node budget. A work budget: a search it stops
    returns the same solutions on every machine. Most pairs need a few thousand steps; a
    step costs about half a microsecond on drug-sized pairs and more on the largest search
    graphs, so the default can take tens of seconds on the hardest pairs (31 s on the
    slowest of 1,117 hard pilot pairs, under load). ``time_limit_s`` cuts it short."""
    coarse_fragment_steps: int = 100_000
    """Search steps, within ``fragment_steps``, for the coarse pass of the fragment search.
    With ``element_change`` the search runs twice: on labels that hold the element, then on
    ring and chain labels alone (the coarse pass), where every ring bond can pair with every
    other. On polycyclic pairs that pass takes most of the time and seldom changes the
    result: its fragments rarely beat those on element labels by more than a bond or two,
    which refinement recovers. Unlike the other budgets, reaching this cap does not stop
    the call: the fragments found so far seed the search, the status stays ``complete`` and
    ``metadata["fragment_search"]["coarse_capped"]`` says so. Zero skips the pass. The
    default changed no mapping of the LOMAP benchmark (3,529 pairs) and the size of no
    change on 2,961 neighbouring pairs of the MoleculeACE sets; of 118 pairs there that
    needed more than 10 s without it, 116 kept the size of theirs."""
    fragment_bond_pairs: int = 20_000
    """Ceiling on the compatible bond pairs the fragment search takes. The search graph grows with
    its square, so a pair above it is not searched and reports ``stopped_by == "pairs"``.
    Drug-sized pairs stay far below it; long same-element chains reach it (a 150-carbon
    chain against itself needs about 22,000 and then takes a few seconds)."""
    global_budget_s: float = 0.0
    """Seconds for an optional exact check after the search: the unconstrained graph edit
    distance under the same counts, which bounds every correspondence from below. When it
    finishes and the result meets it, ``metadata["optimality"]`` says ``optimal``; the
    bound and the gap are in ``metadata["global_check"]``. Exponential, so off by default
    and meant for small molecules and benchmarks."""
    max_fragments: int = 4096
    """Cap on the common fragments the fragment search lists; reaching it is reported as truncation.
    Atoms reach the engine in canonical order, so the cut does not depend on the input atom
    order, but it can depend on which molecule is the source; the cap is set high."""
    k_seeds: int | None = None
    """Seeds refined, after deduplication by symmetry class; None refines all of
    them, which keeps the result independent of input atom order and of which molecule is
    the source. A number trades that independence for speed."""
    max_alternatives: int = 16
    """Tied mappings kept per seed; the tie-break picks among them. Reaching the cap keeps the
    smallest under the canonical order, and ``metadata["ambiguity"]`` says so."""

    def __post_init__(self) -> None:
        """Reject a bad value when the settings are built, not deep inside a search."""
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name in _COUNTS or (name in _OPTIONAL_COUNTS and value is not None):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer, got {value!r}")
                least = _COUNTS.get(name, _OPTIONAL_COUNTS.get(name, 1))
                if value < least:
                    raise ValueError(f"{name} must be at least {least}, got {value}")
            elif name in _SECONDS:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise TypeError(f"{name} must be a number of seconds, got {value!r}")
                if math.isnan(value) or value < 0:
                    raise ValueError(f"{name} must not be negative or NaN, got {value}")
