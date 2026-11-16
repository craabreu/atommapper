"""Budgets, clocks and other regressions, through the public API."""

import time

import pytest

from atommapper import Settings, map_atoms


def test_a_forced_timeout_is_visible():
    result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1", settings=Settings(node_budget=1))
    assert result.status == "timed_out"
    assert result.metadata["optimality"] == "not_established"
    assert result.reason is not None


def cyclic_peptide(side_chains: list[str]) -> str:
    """cyclo-peptide SMILES from side chains, closing the ring between the first N and last C."""
    residues = [f"N[C@@H]({chain})C(=O)" for chain in side_chains]
    residues[0] = residues[0].replace("N", "N1", 1)
    residues[-1] = residues[-1][: -len("C(=O)")] + "C1=O"
    return "".join(residues)


def test_a_large_macrocycle_returns_within_budget():
    chains = [
        "C",
        "CC(C)C",
        "CC(C)C",
        "Cc1ccccc1",
        "CO",
        "C(C)O",
        "Cc1ccc(O)cc1",
        "CCCCN",
        "CC(=O)O",
        "CCSC",
    ]
    source = cyclic_peptide(chains)
    target = cyclic_peptide([*chains[:4], "C", *chains[5:]])
    start = time.perf_counter()
    result = map_atoms(source, target, settings=Settings(time_limit_s=4.0))
    elapsed = time.perf_counter() - start
    assert result.status in {"complete", "timed_out"}
    assert elapsed < 4.0 + 1.0
    assert all(value is not False for value in result.metadata["checks"].values())


# A pilot pair that needs about 3,800 nodes with elements free.
NODE_HUNGRY = (
    "CO[C@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
    "CO[C@@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
)


def test_results_do_not_depend_on_the_clock_when_the_node_budget_decides():
    short = map_atoms(*NODE_HUNGRY, settings=Settings(node_budget=1000, time_limit_s=60.0))
    long = map_atoms(*NODE_HUNGRY, settings=Settings(node_budget=1000, time_limit_s=600.0))
    assert short.metadata["budget"]["stopped_by"] == "nodes"
    assert short.reaction_smiles == long.reaction_smiles
    assert short.status == long.status
    assert short.metadata["budget"] == long.metadata["budget"]


def test_the_node_budget_is_a_total_for_the_call():
    result = map_atoms(*NODE_HUNGRY, settings=Settings(node_budget=1000))
    assert result.metadata["budget"]["nodes_used"] <= 1000
    assert result.status == "timed_out" and result.metadata["budget"]["stopped_by"] == "nodes"


# Branched alkanes whose Stage 1 needs millions of steps: over 30 s (the whole step budget)
# without a time limit.
DENDRIMERS = (
    "CC(C)(C)CC(CC(C)(C)C)(CC(C)(C)C)CC(CC(C)(C)C)(CC(C)(C)C)CC(C)(C)C",
    "CC(C)(C)CC(CC(C)(C)C)(CC(C)(C)C)CC(CC(C)(C)C)(CC(C)(C)CC)CC(C)(C)C",
)


def test_the_time_limit_bounds_the_whole_call_stage_1_included():
    start = time.perf_counter()
    result = map_atoms(*DENDRIMERS, settings=Settings(time_limit_s=0.2))
    elapsed = time.perf_counter() - start
    assert result.status == "timed_out"
    assert result.metadata["stage1"]["stopped_by"] == "time"
    assert "time" in result.metadata["budget"]["stopped_by"].split(",")
    # The fragments Stage 1 found by the deadline still seed the search: map_atoms once
    # dropped them all and fell back to a single atom pair.
    assert len(result.mapping) > 1
    assert elapsed < 0.2 + 1.0


def test_a_zero_time_limit_searches_nothing_and_says_so():
    result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1", settings=Settings(time_limit_s=0.0))
    assert result.status == "timed_out"
    assert result.metadata["stage1"]["stopped_by"] == "time"
    assert result.metadata["stage1"]["steps"] == 0


def test_a_nan_time_limit_is_rejected_by_settings():
    # NaN passes "< 0" and every deadline comparison; it used to reach the engine.
    with pytest.raises(ValueError, match=r"time_limit_s.*NaN"):
        Settings(time_limit_s=float("nan"))
    with pytest.raises(ValueError, match=r"global_budget_s.*NaN"):
        Settings(global_budget_s=float("nan"))


def test_the_time_limit_bounds_the_set_up_of_a_large_search_graph():
    # 19,460 bond pairs: building Stage 1's search graph took 1.7 s, so a 0.2 s limit
    # took 2.2 s before the engine read the clock there.
    start = time.perf_counter()
    result = map_atoms("C" * 140, "C" * 141, settings=Settings(time_limit_s=0.2))
    elapsed = time.perf_counter() - start
    assert result.status == "timed_out"
    assert result.metadata["stage1"]["stopped_by"] == "time"
    assert elapsed < 0.2 + 1.0
