"""Calibration: how the virtual seconds of the time limit compare with seconds of this machine.

``Settings.work_limit`` is counted in work, on a reference clock (``clock``): the same
calls stop on every machine, and a virtual second is a second of the reference machine.
``calibrate`` maps a benchmark shipped with the package, the same work on every machine, and
compares each call's wall time with its virtual time. From their ratio it suggests the
limit that takes a wanted wall time here.

The benchmark (``calibration_pairs.tsv``) is 300 calls on pairs of synthetic molecules, 50
in each of six bands of the pair's heavy atoms, 64 of them slow (0.5 to 4 virtual seconds):
about two minutes on one core of the reference machine. The molecules are put together
from building blocks by ``benchmarks/make_calibration_pairs.py`` in the repository (chains
of rings, ring systems, macrocycles, peptides); none comes from a database. The rates of
the clock were fitted on other pairs, which are not shipped (``clock.REFERENCE_RATES``).
"""

import argparse
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import resources

from .clock import CLOCK_ID, STAGES, stage_seconds
from .mapper import map_atoms
from .settings import Settings

Call = tuple[str, str, bool]
"""A call of the benchmark: source SMILES, target SMILES, ``allow_stereo_inversion``."""

VIRTUAL_LIMIT_S = 5.0
"""The time limit of the benchmark's calls, in virtual seconds: above what its slowest call
takes, so every call runs to its end, and deterministic, so a change of the package that
made one much slower would still not make the benchmark endless."""

SLOW_S = 0.5
"""Calls of at least this many virtual seconds are the slow ones, whose ratio suggests the
limit: a limit matters for slow calls, and the wall time of a fast one is mostly noise."""

STAGE_S = 0.05
"""A stage enters its own ratio when it took at least this many virtual seconds."""

REFERENCE = (
    "On the reference machine: 1.02 for the slow calls with eight calls at a time, 0.97 with one."
)
"""What this benchmark measures where the rates were fitted (2026-11-18). Not 1.00 to the
last digit: the rates were fitted on other pairs, and a second buys a little less when
eight calls run at a time than when one does."""

GUIDANCE = (
    "Results are comparable across machines only at the same work_limit: a calibrated "
    "limit gives up exact comparability with a preparation made at the reference limit."
)


def benchmark() -> list[Call]:
    """The calls of the benchmark shipped with the package, in the file's order."""
    text = resources.files("atommapper").joinpath("calibration_pairs.tsv").read_text()
    calls = []
    for line in text.splitlines():
        if line and not line.startswith("#"):
            source, target, allow = line.split("\t")
            calls.append((source, target, allow == "1"))
    return calls


@dataclass(frozen=True)
class Ratio:
    """Wall seconds per virtual second over some calls: the median and two percentiles."""

    calls: int
    p5: float
    median: float
    p95: float


@dataclass(frozen=True)
class Calibration:
    """What ``calibrate`` measured. ``str()`` gives the report the command prints."""

    calls: int
    workers: int
    wall_s: float
    """Wall seconds of all the calls, added up."""
    virtual_s: float
    """Virtual seconds of all the calls, added up: the same on every machine."""
    ratios: dict[str, Ratio]
    """Wall seconds per virtual second: over ``"all calls"``, over the ``"slow calls"``
    and over each stage of ``clock.STAGES``; a group with no call is left out."""
    factor: float
    """Wall seconds of this machine per virtual second: the median over the slow calls, or
    over all calls when fewer than five are slow."""
    wall_seconds: float
    """The wall time asked for."""
    work_limit: float
    """The suggested ``Settings.work_limit``: ``wall_seconds / factor``."""
    clock: str = CLOCK_ID

    def __str__(self) -> str:
        lines = [
            f"AtomMapper calibration on the reference clock {self.clock}: {self.calls} calls, "
            f"{self.workers} at a time, {self.wall_s:.1f} wall s for {self.virtual_s:.1f} "
            "virtual s.",
            f"Wall seconds per virtual second. {REFERENCE}",
            f"  {'':18}{'calls':>6}{'5%':>8}{'median':>8}{'95%':>8}",
        ]
        lines += [
            f"  {name.replace('_', ' '):18}{r.calls:>6}{r.p5:>8.2f}{r.median:>8.2f}{r.p95:>8.2f}"
            for name, r in self.ratios.items()
        ]
        lines += [
            f"For about {self.wall_seconds:g} s of this machine per call at the limit, use "
            f"work_limit = {self.work_limit:.2g} (a virtual second takes {self.factor:.2f} s "
            "here).",
            GUIDANCE,
        ]
        return "\n".join(lines)


def _measure(call: Call) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """A call's wall and virtual seconds, whole and by stage. The wall-clock guard is off:
    a slow machine must do the same work."""
    source, target, allow = call
    settings = Settings(work_limit=VIRTUAL_LIMIT_S, wall_guard_factor=math.inf)
    result = map_atoms(source, target, allow_stereo_inversion=allow, settings=settings)
    budget, timings = result.metadata["budget"], result.metadata["timings"]
    return timings["total"], budget["virtual_s"], timings, stage_seconds(budget["work"])


def _warm_up() -> None:
    """A first call, not measured: it pays what a process pays once."""
    map_atoms("CCO", "CC=O")


def _quantile(ordered: Sequence[float], q: float) -> float:
    """The ``q`` quantile of sorted values, between the two nearest ones."""
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def _ratio(pairs: Sequence[tuple[float, float]]) -> Ratio | None:
    ordered = sorted(wall / virtual for wall, virtual in pairs)
    if not ordered:
        return None
    return Ratio(
        len(ordered), _quantile(ordered, 0.05), _quantile(ordered, 0.5), _quantile(ordered, 0.95)
    )


def calibrate(
    wall_seconds: float = 10.0, *, workers: int = 1, calls: Sequence[Call] | None = None
) -> Calibration:
    """Compare virtual seconds with seconds of this machine, and suggest a time limit.

    Parameters
    ----------
    wall_seconds : float, default 10.0
        The wall time a call should take here when the time limit stops it.
    workers : int, default 1
        Calls mapped at a time, each in its own process. Use the number the real work will
        run with: a second buys less when more calls run at a time.
    calls : sequence of (source, target, allow_stereo_inversion), optional
        The calls to map; the benchmark shipped with the package (``benchmark()``) when
        None. Ratios measured on other calls are not comparable with those of the benchmark.

    Returns
    -------
    Calibration
        The ratios of wall to virtual time and the suggested ``Settings.work_limit``.
        Printing it gives the report of ``atommapper calibrate``.

    Notes
    -----
    A time limit other than the one a set of results was made with gives other results
    for the pairs between the two limits, on whatever machine: results are comparable
    across machines only at the same ``work_limit``.

    Examples
    --------
    >>> report = calibrate(10.0, calls=[("CCO", "CC=O", False)])
    >>> report.calls, report.clock == CLOCK_ID
    (1, True)
    """
    if not wall_seconds > 0:
        raise ValueError(f"wall_seconds must be positive, got {wall_seconds}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    todo = benchmark() if calls is None else list(calls)
    if not todo:
        raise ValueError("no call to map")
    if workers == 1:
        _warm_up()
        measured = [_measure(call) for call in todo]
    else:
        with ProcessPoolExecutor(workers, initializer=_warm_up) as pool:
            measured = list(pool.map(_measure, todo))
    whole = [(wall, virtual) for wall, virtual, _, _ in measured]
    slow = [(wall, virtual) for wall, virtual in whole if virtual >= SLOW_S]
    groups = {"all calls": whole, "slow calls": slow}
    for stage in STAGES:
        groups[stage] = [
            (timings[stage], virtual[stage])
            for _, _, timings, virtual in measured
            if virtual[stage] >= STAGE_S
        ]
    ratios = {name: ratio for name, pairs in groups.items() if (ratio := _ratio(pairs))}
    factor = ratios["slow calls" if len(slow) >= 5 else "all calls"].median
    return Calibration(
        calls=len(todo),
        workers=workers,
        wall_s=sum(wall for wall, _ in whole),
        virtual_s=sum(virtual for _, virtual in whole),
        ratios=ratios,
        factor=factor,
        wall_seconds=wall_seconds,
        work_limit=wall_seconds / factor,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The ``atommapper calibrate`` command."""
    parser = argparse.ArgumentParser(
        prog="atommapper calibrate",
        description="Map the benchmark shipped with the package and compare each call's "
        "wall time with its virtual time, the unit of --work-limit and Settings.work_limit. "
        "Print their ratio and the time limit that takes --wall-seconds on this machine. "
        "About two minutes on one core of the reference machine.",
        epilog=GUIDANCE,
    )
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=10.0,
        help="wall time a call should take here when the limit stops it (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="calls mapped at a time; use the number the real work will run with (default: 1)",
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=None,
        metavar="N",
        help="map only N calls of the benchmark, spread over it (a quick check, not a calibration)",
    )
    args = parser.parse_args(argv)
    calls = benchmark()
    if args.calls is not None:
        if args.calls < 1:
            parser.error(f"--calls must be at least 1, got {args.calls}")
        step = max(len(calls) // args.calls, 1)
        calls = calls[::step][: args.calls]
    try:
        print(calibrate(args.wall_seconds, workers=args.workers, calls=calls))
    except ValueError as error:
        parser.error(str(error))
    return 0
