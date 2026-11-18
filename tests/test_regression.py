"""Budgets, clocks and other regressions, through the public API."""

import math

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
    result = map_atoms(source, target, settings=Settings(work_limit=4.0))
    assert result.status in {"complete", "timed_out"}
    assert result.metadata["budget"]["virtual_s"] < 4.0 + 1.0
    assert all(value is not False for value in result.metadata["checks"].values())


# A pilot pair that needs about 3,800 nodes with elements free.
NODE_HUNGRY = (
    "CO[C@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
    "CO[C@@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
)


def test_results_do_not_depend_on_the_clock_when_the_node_budget_decides():
    short = map_atoms(*NODE_HUNGRY, settings=Settings(node_budget=1000, work_limit=60.0))
    long = map_atoms(*NODE_HUNGRY, settings=Settings(node_budget=1000, work_limit=600.0))
    assert short.metadata["budget"]["stopped_by"] == "nodes"
    assert short.reaction_smiles == long.reaction_smiles
    assert short.status == long.status
    assert short.metadata["budget"] == long.metadata["budget"]


def test_the_node_budget_is_a_total_for_the_call():
    result = map_atoms(*NODE_HUNGRY, settings=Settings(node_budget=1000))
    assert result.metadata["budget"]["nodes_used"] <= 1000
    assert result.status == "timed_out" and result.metadata["budget"]["stopped_by"] == "nodes"


# Branched alkanes whose fragment search needs millions of steps: over 30 s (the whole step budget)
# without a time limit.
DENDRIMERS = (
    "CC(C)(C)CC(CC(C)(C)C)(CC(C)(C)C)CC(CC(C)(C)C)(CC(C)(C)C)CC(C)(C)C",
    "CC(C)(C)CC(CC(C)(C)C)(CC(C)(C)C)CC(CC(C)(C)C)(CC(C)(C)CC)CC(C)(C)C",
)


def test_the_work_limit_bounds_the_whole_call_the_fragment_search_included():
    # Without the wall-clock guard, which a loaded machine could reach first.
    result = map_atoms(*DENDRIMERS, settings=Settings(work_limit=0.2, wall_guard_factor=math.inf))
    assert result.status == "timed_out"
    # The limit is counted in work (tests/test_clock.py): "work", not "time".
    assert result.metadata["fragment_search"]["stopped_by"] == "work"
    assert result.metadata["budget"]["stopped_by"] == "work"
    # The fragments found by the limit still seed the search: map_atoms once
    # dropped them all and fell back to a single atom pair.
    assert len(result.mapping) > 1
    assert result.metadata["budget"]["virtual_s"] < 0.2 + 1.0


def test_a_zero_work_limit_searches_nothing_and_says_so():
    result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1", settings=Settings(work_limit=0.0))
    assert result.status == "timed_out"
    assert result.metadata["fragment_search"]["stopped_by"] == "work"
    assert result.metadata["fragment_search"]["steps"] == 0


def test_a_nan_work_limit_is_rejected_by_settings():
    # NaN passes "< 0" and every deadline comparison; it used to reach the engine.
    with pytest.raises(ValueError, match=r"work_limit.*NaN"):
        Settings(work_limit=float("nan"))
    with pytest.raises(ValueError, match=r"global_budget_s.*NaN"):
        Settings(global_budget_s=float("nan"))


def test_the_work_limit_bounds_the_set_up_of_a_large_search_graph():
    # 19,460 bond pairs: building the search graph took 1.7 s, so a 0.2 s limit
    # took 2.2 s before the engine read the clock there. The set-up is now charged to
    # the virtual clock before it starts, and is not done.
    result = map_atoms("C" * 140, "C" * 141, settings=Settings(work_limit=0.2))
    assert result.status == "timed_out"
    assert result.metadata["fragment_search"]["stopped_by"] == "work"
    assert result.metadata["fragment_search"]["steps"] == 0


# Two synthetic dimers: twice a ring system on a benzene ring, joined through a chain of six
# carbons. The first is a tricycle with a spiro oxolane, on an ether chain; the second a
# chromane fused to an N-methylpiperidine, on a chain of two amines. With elements free to
# change, the fragment search runs on element labels (39,873 steps here) and then on ring
# and chain labels alone, the coarse pass: 232,683 steps that find no better fragment.
RING_SYSTEM = "c1ccc2c(c1)CC1CCCC3(CCOC3)C21"
OTHER_RING_SYSTEM = "c1ccc2c(c1)OC1CCN(C)CC1C2"
DIMERS = (
    f"O(CCCCCCO{RING_SYSTEM}){RING_SYSTEM}",
    f"N(CCCCCCNC{OTHER_RING_SYSTEM})C{OTHER_RING_SYSTEM}",
)


def test_the_coarse_pass_stops_at_its_cap_and_the_call_completes():
    result = map_atoms(*DIMERS, allow_stereo_inversion=True)
    search = result.metadata["fragment_search"]
    assert result.status == "complete"
    assert search["coarse_capped"] is True
    assert search["stopped_by"] is None
    assert search["steps"] <= 39_873 + Settings().coarse_fragment_steps
    # The change the uncapped search finds (coarse_fragment_steps=20_000_000), and what the
    # molecules give: 42 + 49 and 40 + 45 atoms and bonds; the ring systems differ, so one
    # benzene ring and the chain between the two halves are kept (14 atoms, 14 bonds):
    # 176 - 56 = 120; and three atoms of the chain change element, two oxygens onto carbons
    # and a carbon onto a nitrogen.
    assert result.metadata["size"] == 123
    assert result.metadata["kept"] == {"atoms": 14, "bonds": 14}


def test_the_step_budget_running_out_in_the_coarse_pass_is_a_stop_not_the_cap():
    # 39,873 steps on element labels leave the coarse pass fewer than its cap.
    result = map_atoms(
        *DIMERS, allow_stereo_inversion=True, settings=Settings(fragment_steps=50_000)
    )
    assert result.status == "timed_out"
    assert result.metadata["fragment_search"]["stopped_by"] == "steps"
    assert result.metadata["fragment_search"]["coarse_capped"] is False


def test_a_cap_the_coarse_pass_does_not_reach_is_not_reported():
    result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1")
    assert result.status == "complete"
    assert result.metadata["fragment_search"]["coarse_capped"] is False


def test_with_elements_fixed_there_is_no_coarse_pass_to_cap():
    result = map_atoms(
        "Cc1ccc(F)cc1",
        "Cc1cccc(F)c1",
        element_change=False,
        settings=Settings(coarse_fragment_steps=0),
    )
    assert result.status == "complete"
    assert result.metadata["fragment_search"]["coarse_capped"] is False
    assert len(result.mapping) == 7


def test_a_zero_cap_skips_the_coarse_pass():
    result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1", settings=Settings(coarse_fragment_steps=0))
    assert result.status == "complete"
    assert result.metadata["fragment_search"]["coarse_capped"] is True


def test_a_negative_cap_is_rejected_by_settings():
    with pytest.raises(ValueError, match="coarse_fragment_steps"):
        Settings(coarse_fragment_steps=-1)


def test_the_renamed_settings_bound_the_fragment_search():
    assert map_atoms("CCCCCC", "CCCCCC", settings=Settings(fragment_bond_pairs=10)).status == (
        "timed_out"
    )
    capped = map_atoms("CCCC", "CCCC", settings=Settings(max_fragments=1))
    assert len(capped.mapping) == 4
