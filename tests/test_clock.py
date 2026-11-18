"""The virtual clock: the time limit counted in work, and the wall-clock guard behind it."""

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, fields, replace

import pytest

from atommapper import Settings, clock, map_atoms
from atommapper.clock import CLOCK_ID, REFERENCE_RATES, ClockRates, VirtualClock, clock_id

# Two branched alkanes whose fragment search alone takes seconds.
DENDRIMERS = (
    "CC(C)(C)CC(CC(C)(C)C)(CC(C)(C)C)CC(CC(C)(C)C)(CC(C)(C)C)CC(C)(C)C",
    "CC(C)(C)CC(CC(C)(C)C)(CC(C)(C)C)CC(CC(C)(C)C)(CC(C)(C)CC)CC(C)(C)C",
)
# Two alkane chains: 19,460 compatible bond pairs, a search graph that takes 1.7 s to build.
CHAINS = ("C" * 140, "C" * 141)
# The clock stops the refinement of this pair: its fragment search is short.
NODE_HUNGRY = (
    "CO[C@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
    "CO[C@@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
)
# No wall-clock guard: these tests are about the virtual clock alone.
NO_GUARD = math.inf


def outcome(result) -> dict:
    """What a call returned, without its wall times."""
    return {
        "status": result.status,
        "reaction_smiles": result.reaction_smiles,
        "mapping": result.mapping,
        "budget": result.metadata["budget"],
        "fragment_search": result.metadata["fragment_search"],
        "seeds": result.metadata["seeds"],
        "size": result.metadata["size"],
    }


def test_a_call_the_virtual_clock_stops_returns_the_same_every_time():
    settings = Settings(work_limit=0.2, wall_guard_factor=NO_GUARD)
    first = map_atoms(*DENDRIMERS, settings=settings)
    assert first.status == "timed_out"
    assert first.metadata["budget"]["stopped_by"] == "work"
    assert first.metadata["budget"]["virtual_s"] > 0.2
    assert len(first.mapping) > 1
    for _ in range(2):
        assert outcome(map_atoms(*DENDRIMERS, settings=settings)) == outcome(first)


def test_a_call_the_virtual_clock_stops_returns_the_same_under_load():
    settings = Settings(work_limit=0.2, wall_guard_factor=NO_GUARD)
    idle = map_atoms(*DENDRIMERS, settings=settings)
    done = threading.Event()

    def spin() -> None:
        while not done.is_set():
            sum(range(1000))

    threads = [threading.Thread(target=spin) for _ in range(3)]
    for thread in threads:
        thread.start()
    try:
        loaded = map_atoms(*DENDRIMERS, settings=settings)
    finally:
        done.set()
        for thread in threads:
            thread.join()
    assert loaded.metadata["budget"]["stopped_by"] == "work"
    assert outcome(loaded) == outcome(idle)


def test_a_refinement_the_virtual_clock_stops_returns_the_same_every_time():
    pair = NODE_HUNGRY
    settings = Settings(work_limit=0.05, wall_guard_factor=NO_GUARD)
    first = map_atoms(*pair, settings=settings)
    assert first.metadata["budget"]["stopped_by"] == "work"
    assert first.metadata["fragment_search"]["stopped_by"] is None
    assert first.metadata["budget"]["nodes_used"] > 0
    assert outcome(map_atoms(*pair, settings=settings)) == outcome(first)


def test_the_wall_clock_guard_does_not_fire_on_an_ordinary_pair():
    result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1", settings=Settings(work_limit=10.0))
    budget = result.metadata["budget"]
    assert result.status == "complete" and budget["stopped_by"] is None
    # Far from both clocks: the virtual one, and the guard at ten times the limit.
    assert 0 < budget["virtual_s"] < 1.0
    assert result.metadata["timings"]["total"] < 1.0
    assert budget["work"]["call"] > 0 and budget["work"]["node"] > 0


def test_the_wall_clock_guard_does_not_fire_when_the_virtual_clock_stops_a_call():
    start = time.perf_counter()
    result = map_atoms(*DENDRIMERS, settings=Settings(work_limit=0.2))
    elapsed = time.perf_counter() - start
    # The guard is at 2 s: a virtual second is about a real one, so it is not reached.
    assert result.metadata["budget"]["stopped_by"] == "work"
    assert elapsed < 1.2


def test_a_set_up_that_costs_more_than_the_limit_stops_before_the_search():
    start = time.perf_counter()
    result = map_atoms(*CHAINS, settings=Settings(work_limit=0.2))
    elapsed = time.perf_counter() - start
    search, budget = result.metadata["fragment_search"], result.metadata["budget"]
    assert result.status == "timed_out"
    assert search["stopped_by"] == "work" and search["steps"] == 0
    # The set-up was charged, not done: every relation of the 19,460 bond pairs.
    assert budget["work"]["relation_pair"] == 19_460 * 19_459 // 2
    assert budget["work"]["word"] == 0
    assert budget["virtual_s"] > 0.2
    assert elapsed < 0.5


def test_with_the_virtual_clock_off_the_wall_clock_guard_stops_the_call(monkeypatch):
    # The rates are no setting: only a test turns the virtual clock off.
    monkeypatch.setattr(clock, "REFERENCE_RATES", ClockRates.off())
    settings = Settings(work_limit=0.2, wall_guard_factor=1.0)
    start = time.perf_counter()
    result = map_atoms(*DENDRIMERS, settings=settings)
    elapsed = time.perf_counter() - start
    assert result.status == "timed_out"
    assert result.metadata["fragment_search"]["stopped_by"] == "time"
    assert "time" in result.metadata["budget"]["stopped_by"].split(",")
    assert result.metadata["budget"]["virtual_s"] == 0
    assert len(result.mapping) > 1
    assert elapsed < 0.2 + 1.0
    # The set-up of a large search graph reads the guard too.
    start = time.perf_counter()
    result = map_atoms(*CHAINS, settings=settings)
    assert result.metadata["fragment_search"]["stopped_by"] == "time"
    assert time.perf_counter() - start < 0.2 + 1.0


def cache_key(settings: Settings) -> str:
    """A cache key as a user of the package makes one: a hash of the settings."""
    return hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()


def test_the_reference_clock_is_frozen_and_named_in_the_settings_and_the_results():
    # The rates are a constant of the package. Changing one changes which calls stop: this
    # name must then change with it, here and wherever results are cached by it.
    assert CLOCK_ID == clock_id(REFERENCE_RATES) == "clock-60ae29dd4850"
    settings = asdict(Settings())
    assert settings["reference_clock"] == CLOCK_ID
    assert "clock_rates" not in settings
    result = map_atoms("CCO", "CC=O")
    assert result.metadata["settings"]["reference_clock"] == CLOCK_ID
    assert result.metadata["versions"]["clock"] == CLOCK_ID
    # Not a setting to choose.
    with pytest.raises(TypeError, match="reference_clock"):
        Settings(reference_clock="mine")  # ty: ignore[unknown-argument]
    assert replace(Settings(), work_limit=2.0).reference_clock == CLOCK_ID


def test_a_change_of_any_rate_changes_the_name_of_the_clock_and_a_cache_key(monkeypatch):
    names = {CLOCK_ID}
    for field in fields(ClockRates):
        changed = replace(REFERENCE_RATES, **{field.name: getattr(REFERENCE_RATES, field.name) + 1})
        names.add(clock_id(changed))
    assert len(names) == 1 + len(fields(ClockRates))
    # The settings carry the name, so a key made of them follows the rates; the guard is
    # in it too.
    key = cache_key(Settings())
    slower = replace(REFERENCE_RATES, node=2 * REFERENCE_RATES.node)
    monkeypatch.setattr(clock, "REFERENCE_RATES", slower)
    assert Settings().reference_clock == clock_id(slower) != CLOCK_ID
    assert cache_key(Settings()) != key
    assert map_atoms("CCO", "CC=O").metadata["versions"]["clock"] == clock_id(slower)
    monkeypatch.undo()
    assert cache_key(Settings()) == key
    assert cache_key(Settings(wall_guard_factor=5.0)) != key


@pytest.mark.parametrize(
    ("pair", "limit"),
    [
        (DENDRIMERS, 0.2),
        (NODE_HUNGRY, 0.05),
        (CHAINS, 0.2),
        (("Cc1ccc(F)cc1", "Cc1cccc(F)c1"), 1.0),
    ],
)
@pytest.mark.parametrize("factor", [2, 3])
def test_scaling_the_limit_is_scaling_every_rate_by_the_inverse(monkeypatch, pair, limit, factor):
    # Why the limit is the only knob: a clock `factor` times as fast with a limit `factor`
    # times as long stops the same calls at the same point.
    reference = map_atoms(*pair, settings=Settings(work_limit=limit, wall_guard_factor=NO_GUARD))
    fast = ClockRates(**{name: factor * rate for name, rate in asdict(REFERENCE_RATES).items()})
    monkeypatch.setattr(clock, "REFERENCE_RATES", fast)
    scaled = map_atoms(
        *pair, settings=Settings(work_limit=factor * limit, wall_guard_factor=NO_GUARD)
    )
    expected, found = outcome(reference), outcome(scaled)
    seconds = expected["budget"].pop("virtual_s")
    assert found["budget"].pop("virtual_s") == pytest.approx(factor * seconds, rel=1e-12)
    assert found == expected
    monkeypatch.undo()
    # The limit alone, on the same rates, is another limit: a call it stopped goes further.
    longer = map_atoms(
        *pair, settings=Settings(work_limit=factor * limit, wall_guard_factor=NO_GUARD)
    )
    if reference.status == "timed_out" and reference.metadata["fragment_search"]["steps"]:
        assert longer.metadata["budget"]["work"] != reference.metadata["budget"]["work"]


def test_bad_clock_settings_are_rejected():
    with pytest.raises(ValueError, match=r"wall_guard_factor.*NaN"):
        Settings(wall_guard_factor=float("nan"))
    with pytest.raises(ValueError, match="wall_guard_factor"):
        Settings(wall_guard_factor=-1.0)
    with pytest.raises(TypeError, match="clock_rates"):
        Settings(clock_rates=REFERENCE_RATES)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match="node"):
        replace(REFERENCE_RATES, node=1.5)
    with pytest.raises(ValueError, match="word"):
        replace(REFERENCE_RATES, word=-1)


def test_the_virtual_clock_counts_work_and_reads_no_clock_to_expire():
    rates = replace(ClockRates.off(), node=250, seed=100)
    virtual = VirtualClock(1e-9, rates, 10.0, 0.0)
    assert virtual.limit == 1000 and virtual.left == 1000 and not virtual.expired
    virtual.charge("node", 4)
    assert (virtual.spent, virtual.left, virtual.expired) == (1000, 0, False)
    virtual.charge("seed", 1)
    assert virtual.expired and virtual.late() and virtual.stops == {"work"}
    assert virtual.work["node"] == 4 and virtual.work["seed"] == 1
    # Without a limit the clock never expires, and without a guard it is never late.
    endless = VirtualClock(math.inf, REFERENCE_RATES, 10.0, time.perf_counter())
    endless.charge("node", 10**12)
    assert endless.left is None and not endless.expired and not endless.late()
    unguarded = VirtualClock(0.0, REFERENCE_RATES, math.inf, time.perf_counter() - 3600)
    assert unguarded.deadline == math.inf and not unguarded.late()
