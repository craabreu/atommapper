"""The virtual clock: a work limit that reads no clock.

``Settings.work_limit`` bounds a call in virtual seconds. Each stage of the search counts
its work in units whose cost does not depend on the pair, and the virtual clock advances by
a fixed number of picoseconds per unit (``REFERENCE_RATES``). A call stops when its virtual
clock passes the limit, so the pairs that stop, and what they return, are the same on every
machine and under any load, as with the step and node budgets. Solvers offer the same kind
of limit for the same reason: CPLEX's deterministic time limit, in ticks, and Gurobi's
``WorkLimit``.

The rates are a constant of the package, not a setting: they define the reference clock, on
which one virtual second is one second of the reference machine (``REFERENCE_RATES``).
Scaling every rate by a factor is the same as dividing the limit by it, so the limit is the
only knob a user needs; ``atommapper.calibrate`` (``calibration``) measures how a virtual
second compares with a second of the machine at hand. ``CLOCK_ID`` names the rates in the
settings and in every result, so that results made on different rates are not taken for the
same.

The wall clock is still read, as a guard: it stops a call after
``Settings.wall_guard_factor`` times the limit in real seconds, and a call it stops says
so (``"time"`` among the stop reasons, against ``"work"`` for the virtual clock). On a
machine like the reference one, the guard does not fire.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, fields

PICOSECONDS = 10**12


@dataclass(frozen=True)
class ClockRates:
    """Virtual picoseconds per unit of work, by stage.

    The first four price the fragment search engine's units (``engine::Work``). The others
    price a unit of the Python stages, each counted once per atom and bond of the pair (the
    atoms and bonds of both molecules, added up): those stages read both molecules whole at
    every step. The package's own rates are ``REFERENCE_RATES``.
    """

    relation_pair: int
    """A pair of compatible bond pairs whose relation the engine's set-up reads."""
    step: int
    """A step of the fragment enumeration, whatever it reads."""
    word: int
    """A 64-bit word of a bitset the fragment enumeration reads."""
    leaf_bond: int
    """A bond of a clique the engine splits, scores or turns into a fragment."""
    call: int
    """Once per call: parsing, the views of both molecules, the rules, the result."""
    trim: int
    """A trim of a fragment to the rules (two per fragment, four with inversions allowed)."""
    piece: int
    """A piece or seed candidate read: turned, tested against the rules, ranked."""
    seed: int
    """The set-up of one refinement."""
    node: int
    """A branch-and-bound node of the refinement."""

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field.name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{field.name} must not be negative, got {value}")

    @classmethod
    def off(cls) -> "ClockRates":
        """Rates of zero: the virtual clock never advances, and only the budgets and the
        wall-clock guard stop a call."""
        return cls(**{field.name: 0 for field in fields(cls)})


REFERENCE_RATES = ClockRates(
    relation_pair=10_000,
    step=1_400_000,
    word=410,
    leaf_bond=10_000,
    call=36_000_000,
    trim=590_000,
    piece=17_000,
    seed=1_400_000,
    node=400_000,
)
"""The reference clock: the rates every call is timed with. One virtual second is one second
of the reference machine, an AMD Ryzen Threadripper PRO 5975WX (32 cores) running eight
calls at a time.

They were fitted on 2026-11-18 on 2,400 pairs of public medicinal-chemistry data, which
are not shipped with the package (2,948 calls, with and without stereo inversions), 400
from each of six bands of the pair's heavy atoms (up to 50, 70, 90, 120, 200 and above),
under a 10 s wall clock with the virtual clock off:
for each stage, the least-squares rates, without intercept and not below zero, of the
stage's wall time on its units, with each call weighted by the inverse of its time so that
the slowest pairs do not set the rates alone, then rounded to two digits. On the same pairs
mapped again with these rates, a call's wall time was 0.76 to 1.63 times its virtual time
for 98% of the calls (0.5 to 3.7 for all of them, 0.75 to 2.0 for those over a second).

The number of calls running at a time changes what a second buys, not which calls stop: the
same calls took about 1.8 times as long (the median; 1.1 to 2.2 for nine calls in ten) in a
run that kept all 32 cores busy as in the run of eight the rates come from. A virtual
second is then about 1.8 s of that machine.

The rates are frozen: changing one changes which calls stop, and with it ``CLOCK_ID``.
"""

STAGES = {
    "fragment_search": ("relation_pair", "step", "word", "leaf_bond"),
    "seeding": ("trim", "piece"),
    "refinement": ("seed", "node"),
}
"""The rates of each stage whose wall time a call reports (``metadata["timings"]``)."""


def clock_id(rates: ClockRates) -> str:
    """A name for a set of rates: the first twelve hexadecimal digits of the SHA-256 of
    their JSON. Two sets of rates have the same name only if every rate is the same."""
    text = json.dumps(asdict(rates), sort_keys=True)
    return "clock-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def reference_id() -> str:
    """The name of the rates in use, read when called: ``CLOCK_ID``, unless a test or a new
    fit has put other rates in ``REFERENCE_RATES``. It is a field of ``Settings``
    (``reference_clock``) and an entry of every result's ``metadata["versions"]``, so a
    cache keyed by either tells results made on other rates apart."""
    return clock_id(REFERENCE_RATES)


CLOCK_ID = reference_id()
"""The name of the reference clock."""


def stage_seconds(work: dict[str, int], rates: ClockRates = REFERENCE_RATES) -> dict[str, float]:
    """The virtual seconds of each stage of ``STAGES``, from a call's units of work
    (``metadata["budget"]["work"]``)."""
    return {
        stage: sum(work[name] * getattr(rates, name) for name in names) / PICOSECONDS
        for stage, names in STAGES.items()
    }


class VirtualClock:
    """The virtual clock of one call, and its wall-clock guard."""

    def __init__(
        self, limit_s: float, rates: ClockRates, guard_factor: float, start: float
    ) -> None:
        self.rates = rates
        self.limit: int | None = None if math.isinf(limit_s) else round(limit_s * PICOSECONDS)
        """The limit in virtual picoseconds; None without one."""
        self.spent = 0
        """Virtual picoseconds so far."""
        self.work: dict[str, int] = {field.name: 0 for field in fields(rates)}
        """Units of work so far, by the name of their rate."""
        guarded = math.isfinite(guard_factor) and math.isfinite(limit_s)
        self.deadline = start + guard_factor * limit_s if guarded else math.inf
        """The wall-clock guard, a ``time.perf_counter`` reading; infinite without one."""
        self.stops: set[str] = set()
        """Which clock stopped a stage: ``"work"``, ``"time"`` (the guard) or both."""

    def charge(self, name: str, units: int) -> None:
        """Advance the clock by ``units`` of the work priced by the rate ``name``."""
        self.work[name] += units
        self.spent += units * getattr(self.rates, name)

    @property
    def expired(self) -> bool:
        """Whether the virtual clock is past the limit. Reads no clock."""
        return self.limit is not None and self.spent > self.limit

    @property
    def left(self) -> int | None:
        """Virtual picoseconds left; None without a limit."""
        return None if self.limit is None else max(self.limit - self.spent, 0)

    def late(self) -> bool:
        """Whether a stage should stop, and why (``stops``): the virtual clock is past the
        limit or, failing that, the wall-clock guard has passed."""
        if self.expired:
            self.stops.add("work")
            return True
        if time.perf_counter() > self.deadline:
            self.stops.add("time")
            return True
        return False

    @property
    def seconds(self) -> float:
        """Virtual seconds so far."""
        return self.spent / PICOSECONDS
