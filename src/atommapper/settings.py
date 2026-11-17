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
    """Configure the work budgets, time limit, and candidate caps for ``map_atoms``.

    Pass an instance as ``map_atoms(source, target, settings=Settings(...))``. Defaults
    suit drug-like molecular pairs. Work budgets give reproducible stopping points
    provided the wall-clock limit does not intervene. A limit can return a partial
    result with status "timed_out"; the coarse-pass cap is part of the normal search.

    Counts must be integers, and time budgets must be nonnegative numbers. Invalid
    values raise TypeError or ValueError at construction. Restricting ``k_seeds`` or
    truncating fragments can affect mapping quality and consistency under reversal.
    """

    node_budget: int = 1_000_000
    """Total branch-and-bound nodes, shared across seeds. Exhaustion sets status to
    "timed_out" and includes "nodes" in metadata["budget"]["stopped_by"]."""
    time_limit_s: float = 60.0
    """Wall-clock search limit, measured from the start of the call. Fragment search,
    seeding, and refinement check it periodically. Parsing, stereo perception, and result
    assembly are not interrupted, so this is not a hard process deadline. The optional
    global check has its own additional budget. Expiry sets status to "timed_out" and
    includes "time" in metadata["budget"]["stopped_by"]."""
    fragment_steps: int = 20_000_000
    """Total fragment-search steps across all passes. Exhaustion sets status to
    "timed_out" and includes "steps" in metadata["budget"]["stopped_by"]."""
    coarse_fragment_steps: int = 100_000
    """Work cap for the element-independent fragment pass, within fragment_steps.
    Used when element changes are allowed. Zero skips the pass. Reaching this cap is
    normal: fragments found so far seed refinement, status can still be "complete", and
    metadata["fragment_search"]["coarse_capped"] records the cap. Increasing it may improve
    candidate coverage but can be expensive for polycyclic pairs."""
    fragment_bond_pairs: int = 20_000
    """Ceiling on compatible bond pairs used to build a fragment-search graph.
    Pairs above the ceiling are not searched; "pairs" appears in the stop reasons.
    Raising it permits larger searches but can increase memory and time substantially."""
    global_budget_s: float = 0.0
    """Additional seconds for an exact lower-bound check after the mapping search.
    Zero disables it. The check uses unconstrained graph edit distance under the same
    objective. If it finishes and matches a completed mapping's cost, optimality becomes
    "optimal"; otherwise no global proof is established. See metadata["global_check"]
    for the bound and gap. Intended for small molecules and validation."""
    max_fragments: int = 4096
    """Maximum number of common fragments retained. Truncation is recorded in
    metadata["fragment_search"]["truncated"] and can affect consistency under reversal."""
    k_seeds: int | None = None
    """Maximum seeds refined after deduplication by symmetry class. None refines all.
    A smaller limit trades candidate coverage and input-order consistency for speed."""
    max_alternatives: int = 16
    """Maximum tied mappings retained per seed for tie-breaking. When capped,
    metadata["ambiguity"]["lower_bound"] indicates that its count may be incomplete."""

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
