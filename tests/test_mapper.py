"""map_atoms: LOMAP's mapping rules under the smallest-change objective.

The rules: one connected piece; ring atoms pair with ring atoms; every kept ring atom and
ring bond lies on a cycle of the kept part (complete rings, one ring at a time); no kept
stereocentre is inverted; hydrogens implicit. With element_change (the default), elements
and bond orders may change, each change counting one.
"""

import random
import time

import networkx as nx
import pytest
from rdkit import Chem

from atommapper import Settings, map_atoms


def shuffled(smiles: str, rng: random.Random) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    order = list(range(mol.GetNumAtoms()))
    rng.shuffle(order)
    return Chem.RenumberAtoms(mol, order)


def kept(result) -> set[int]:
    return {m.source_index for m in result.mapping}


def one_piece(result, source: str, target_smiles: str) -> bool:
    mol = Chem.MolFromSmiles(source)
    atoms = kept(result)
    g = nx.Graph()
    g.add_nodes_from(atoms)
    pairs = {m.source_index: m.target_index for m in result.mapping}
    target = Chem.MolFromSmiles(target_smiles)
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in atoms and v in atoms and target.GetBondBetweenAtoms(pairs[u], pairs[v]):
            g.add_edge(u, v)
    return nx.number_connected_components(g) <= 1


@pytest.mark.parametrize(
    ("source", "target", "size", "atoms_kept"),
    [
        # one connected piece: a 4-atom chain, not two 3-atom pieces
        ("CC(C)C(C)C", "CCCCCC", 8, 4),
        # a ring is complete on its own: benzene is kept as one ring of naphthalene
        ("c1ccccc1", "c1ccc2ccccc2c1", 9, 6),
        # elements may change: one change for the ring nitrogen
        ("Cc1ccncc1", "Cc1ccccc1", 1, 7),
        # two linker atoms change element; everything is kept
        ("CC(C)SSc1ccccc1", "CC(C)OOc1ccccc1", 2, 11),
    ],
)
def test_lomap_rules(source, target, size, atoms_kept):
    result = map_atoms(source, target)
    assert (result.metadata["size"], len(result.mapping)) == (size, atoms_kept)
    assert result.status == "complete"
    assert one_piece(result, source, target)
    checks = result.metadata["checks"]
    assert checks["apply_reproduces_target"] and checks["reverse_reproduces_source"]


def test_a_ring_never_maps_onto_a_ring_of_another_size():
    result = map_atoms("CC1CCCCC1", "CC1CCCC1")
    ring = set(range(1, 7))
    assert not kept(result) & ring


def test_without_element_change_a_ring_with_another_element_is_not_kept():
    assert map_atoms("c1ccncc1", "c1ccccc1").metadata["size"] == 1
    result = map_atoms("c1ccncc1", "c1ccccc1", element_change=False)
    assert (result.metadata["size"], len(result.mapping)) == (24, 0)


def test_an_inverted_stereocentre_is_not_kept():
    source, target = "C[C@H](N)O", "C[C@@H](N)O"
    # elements fixed: keeping N or O with the centre would invert it, so the centre stays
    # with the methyl alone, its configuration undetermined (LOMAP trims an inverted
    # centre's branches until at most two heavy neighbours remain)
    result = map_atoms(source, target, element_change=False)
    assert (result.metadata["size"], kept(result)) == (8, {0, 1})
    assert [v["status"] for v in result.metadata["stereo"]] == ["undetermined"]
    # elements free: swapping N and O keeps the centre's configuration, at two changes
    result = map_atoms(source, target)
    assert (result.metadata["size"], len(result.mapping)) == (2, 4)


MIRRORED = [
    # (source, target, inverted centres kept, size of the change with stereo ignored): the
    # target mirrors the centres of the source
    ("C[C@H](N)O", "C[C@@H](N)O", 1, 0),
    ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)OC", 1, 2),
    ("C[C@H](F)[C@H](C)O", "C[C@@H](F)[C@@H](C)O", 2, 0),
    ("C[C@H](F)[C@H](C)O", "C[C@@H](F)[C@H](C)O", 1, 0),
    ("C/C=C/C(=O)O", "C/C=C\\C(=O)O", 1, 0),
    ("C/C=C/[C@H](C)O", "C/C=C\\[C@@H](C)O", 2, 0),
]


def test_an_allowed_inverted_stereocentre_is_kept_and_counts_one_change():
    source, target = "C[C@H](N)O", "C[C@@H](N)O"
    # elements fixed: without the option the centre keeps only the methyl (size 8, above);
    # with it, every atom is kept and the inversion is the one change
    result = map_atoms(source, target, element_change=False, allow_stereo_inversion=True)
    assert (result.metadata["size"], result.metadata["cost"]) == (1, 1)
    assert kept(result) == {0, 1, 2, 3}
    assert [v["status"] for v in result.metadata["stereo"]] == ["inverted"]
    assert result.metadata["allow_stereo_inversion"] is True
    assert "count each inverted stereo element" in result.metadata["objective"]
    # elements free: the inversion (one change) is cheaper than swapping N and O (two)
    result = map_atoms(source, target, allow_stereo_inversion=True)
    assert result.metadata["size"] == 1
    assert [v["status"] for v in result.metadata["stereo"]] == ["inverted"]


def test_an_inverted_double_bond_is_not_kept_unless_allowed_and_then_counts_one_change():
    source, target = "C/C=C/C(=O)O", "C/C=C\\C(=O)O"
    # by default no kept stereo element is inverted, a double bond included: part of it goes
    plain = map_atoms(source, target, element_change=False)
    assert plain.metadata["size"] > 1
    assert len(plain.mapping) < 6
    assert "inverted" not in [v["status"] for v in plain.metadata["stereo"]]
    # allowed, all six atoms are kept and the inversion is the one change
    allowed = map_atoms(source, target, element_change=False, allow_stereo_inversion=True)
    assert (allowed.metadata["size"], len(allowed.mapping)) == (1, 6)
    assert [(v["kind"], v["status"]) for v in allowed.metadata["stereo"]] == [("bond", "inverted")]


def test_the_default_still_forbids_an_inverted_stereocentre():
    result = map_atoms("C[C@H](N)O", "C[C@@H](N)O", element_change=False)
    assert result.metadata["allow_stereo_inversion"] is False
    assert (result.metadata["size"], kept(result)) == (8, {0, 1})


@pytest.mark.parametrize("objective", ["change", "bonds"])
@pytest.mark.parametrize(("source", "target", "inverted", "unstereo"), MIRRORED)
def test_each_inverted_centre_adds_one_to_the_size_of_the_change(
    source, target, inverted, unstereo, objective
):
    result = map_atoms(
        source, target, element_change=False, objective=objective, allow_stereo_inversion=True
    )
    verdicts = [v["status"] for v in result.metadata["stereo"]]
    assert verdicts.count("inverted") == inverted
    # the same atoms are kept as when stereo is ignored, with each inversion charged
    assert result.metadata["size"] == unstereo + inverted


MIRRORED_CORE = (
    "CNC(=O)CN1C[C@]2(C(=O)N(c3cncc4ccccc34)C[C@H]2COC)c2cc(Cl)ccc2C1=O",
    "C[C@H]1CN(c2cncc3ccccc23)C(=O)[C@@]12CN(CCN1C[C@H]3C[C@@H]1CO3)C(=O)c1ccc(Cl)cc12",
)


def test_a_mirrored_core_is_kept_when_inversions_are_allowed():
    # both molecules hold one spirocyclic core, with both of its centres mirrored: under the
    # rule, the core goes (15 atoms stay); allowed, the core stays at one change per centre
    source, target = MIRRORED_CORE
    assert len(map_atoms(source, target).mapping) == 15
    result = map_atoms(source, target, allow_stereo_inversion=True)
    assert result.status == "complete"
    assert len(result.mapping) == 30
    assert [v["status"] for v in result.metadata["stereo"]].count("inverted") == 2
    # 26 is the size of the change with stereo ignored
    assert result.metadata["size"] == 26 + 2


# Pairs of CHEMBL2034 ligands (MoleculeACE) whose largest common fragments hold a mirrored
# core: the strict trim cuts one down to a single atom, the seed of the best mapping. With
# inversions allowed that trim drops nothing, and before the strict seeds were added the
# search ended at sizes 90, 76, 71 and 89.
CHEMBL2034 = [
    (
        "Cc1cccc([C@@](C)(O)[C@H]2CCCC3=Cc4c(-c5ccc(F)cc5)ncn4C[C@@]32C)c1",
        "Cc1ccc(S(=O)(=O)N2CCC3=Cc4c(cnn4-c4ccc(F)cc4)C[C@]3(C(=O)c3ccccn3)C2)cc1",
        60,
    ),
    (
        "COC[C@]12Cc3cnn(-c4ccc(F)cc4)c3C=C1CCN(S(=O)(=O)c1ccccc1)C2",
        "C[C@@](O)(c1ccccn1)[C@H]1CCCC2=Cc3c(-c4ccc(F)cc4)ncn3C[C@@]21C",
        45,
    ),
    (
        "C=C1C[C@H](C)C[C@@H](C)[C@@H]1Cc1cc(OC)c(Br)cc1O",
        "Cc1ccccc1S(=O)(=O)N1CCC2=Cc3c(cnn3-c3ccc(F)cc3)C[C@]2(C(=O)c2ccccn2)C1",
        70,
    ),
    (
        "C[C@@](O)(c1ccncc1)[C@H]1CCCC2=Cc3c(-c4ccc(F)cc4)ncn3C[C@@]21C",
        "O=C(c1nccs1)[C@]12Cc3cnn(-c4ccc(F)cc4)c3C=C1CCN(S(=O)(=O)c1ccc(Cl)c(Cl)c1)C2",
        58,
    ),
]


@pytest.mark.parametrize(
    ("source", "target"),
    [pair[:2] for pair in CHEMBL2034] + [MIRRORED_CORE, ("CC/C=C/CC", "CC/C=C\\CC")],
)
def test_allowing_inversion_never_makes_the_change_larger(source, target):
    # every mapping the rule accepts is accepted when inversions are allowed, and costs the
    # same; the strict search's seeds are searched too, so none of its results is missed
    strict = map_atoms(source, target)
    allowed = map_atoms(source, target, allow_stereo_inversion=True)
    assert allowed.metadata["size"] <= strict.metadata["size"]


@pytest.mark.parametrize(("source", "target", "size"), CHEMBL2034)
def test_an_allowed_mapping_without_inversion_is_the_strict_one(source, target, size):
    strict = map_atoms(source, target)
    allowed = map_atoms(source, target, allow_stereo_inversion=True)
    assert strict.metadata["size"] == allowed.metadata["size"] == size
    assert "inverted" not in [v["status"] for v in allowed.metadata["stereo"]]
    assert allowed.reaction_smiles == strict.reaction_smiles


@pytest.mark.parametrize(
    ("source", "target"),
    [("CCO", "CCN"), ("c1ccccc1", "C1CCCC1"), ("CC1CC1", "C1CC1C"), ("OC1CCC1", "NC1CC1")],
)
@pytest.mark.parametrize("objective", ["change", "bonds"])
def test_allowing_inversion_changes_nothing_without_stereocentres(source, target, objective):
    plain = map_atoms(source, target, objective=objective)
    allowed = map_atoms(source, target, objective=objective, allow_stereo_inversion=True)
    assert allowed.reaction_smiles == plain.reaction_smiles
    assert allowed.metadata["size"] == plain.metadata["size"]
    assert allowed.metadata["cost"] == plain.metadata["cost"]


def test_hydrogens_stay_implicit():
    result = map_atoms("CCO", "CCN")
    assert "[H" not in result.reaction_smiles


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("CC(C)C(C)C", "CCCCCC"),
        ("Cc1ccc(F)cc1", "Cc1cccc(F)c1"),
        ("CC(=O)Nc1ccc(O)cc1", "CC(=O)Nc1ccc(OC)cc1"),
        ("c1ccc(Cc2ccccn2)cc1", "c1ccc(Oc2ccccn2)cc1"),
    ],
)
@pytest.mark.parametrize("objective", ["change", "bonds"])
def test_renumbered_inputs_and_swapped_molecules_give_the_same_result(source, target, objective):
    def outcome(a, b):
        metadata = map_atoms(a, b, objective=objective).metadata
        return metadata["size"], metadata["kept"]

    reference = outcome(source, target)
    rng = random.Random(f"{source}>>{target}")
    for _ in range(3):
        assert outcome(shuffled(source, rng), shuffled(target, rng)) == reference
    assert outcome(target, source) == reference


def test_rule_3_chooses_among_every_tied_mapping_of_a_seed():
    # two mappings tie at size 31; the one a seed lists first in canonical order changes 3
    # attachment points, the other 2 (reduced from pilot pair 3340)
    source, target = "CC(=O)CC(C)=NNC(=O)c1ccccc1", "CC(=O)CSC(=S)N1CC1"
    result = map_atoms(source, target)
    assert result.metadata["size"] == 31
    assert result.metadata["ambiguity"]["count"] == 2
    s, t = Chem.MolFromSmiles(source), Chem.MolFromSmiles(target)
    attachment_changes = sum(
        s.GetAtomWithIdx(a.source_index).GetDegree() != t.GetAtomWithIdx(a.target_index).GetDegree()
        for a in result.mapping
    )
    assert attachment_changes == 2


def test_seeds_include_fragments_that_survive_the_ring_trim():
    # the largest ring/chain-label fragment threads through the macrocycle and the
    # piperazine and keeps 3 atoms once trimmed; LOMAP's complete-ring mapping (27 atoms)
    # costs 38 by rule 1
    result = map_atoms(
        "OCCN1CCN(c2ccc3cc2COCC=CCOCc2cccc(c2)-c2ccnc(n2)N3)CC1",
        "C1=CCOCc2cc(ccc2OCCN2CCCCC2)Nc2nccc(n2)-c2cccc(c2)COC1",
    )
    assert result.status == "complete"
    assert result.metadata["size"] == 38


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("CC(=O)Nc1ccc(O)cc1", "CC(=O)Nc1ccc(OC)cc1"),
        ("c1ccc(Cc2ccccn2)cc1", "c1ccc(Oc2ccccn2)cc1"),
        ("CC(=O)CC(C)=NNC(=O)c1ccccc1", "CC(=O)CSC(=S)N1CC1"),
        ("CC/C=C/CC", "CC/C=C\\CC"),
        (
            "O=C(O)c1cn2c3c(c(F)c(F)cc3c1=O)Oc1ccccc1-2",
            "N[C@H]1CCN(c2c(F)cc3c(=O)c(C(=O)O)cn4c3c2Sc2ccccc2-4)C1",
        ),
    ],
)
def test_the_connected_bound_cuts_no_branch_the_cost_so_far_keeps(source, target, monkeypatch):
    # A bound that cut a branch holding a best leaf would change the size, the mapping
    # or the tied mappings counted; the cost of the decisions so far, a far weaker bound,
    # cuts almost nothing.
    from atommapper.refine import _BranchAndBound

    def outcomes():
        return [
            (
                r.metadata["size"],
                sorted((a.source_index, a.target_index) for a in r.mapping),
                r.metadata["ambiguity"]["count"],
                r.status,
            )
            for r in (
                map_atoms(source, target, element_change=e, objective=o)
                for e in (True, False)
                for o in ("change", "bonds")
            )
        ]

    connected = outcomes()
    monkeypatch.setattr(
        _BranchAndBound, "_connected_bound", lambda search: search._decided_cost()[0]
    )
    assert outcomes() == connected


def test_a_seed_that_runs_out_of_its_share_retries_with_what_the_others_left():
    # pilot pair 1191: 8 seeds of unequal work; with half again the nodes they need in all,
    # the largest needs more than its share and finishes on a retry (a fifth more sufficed
    # before the connected bound of 2026-11-16 changed how the work splits among seeds)
    source = "O=C(O)c1cn2c3c(c(F)c(F)cc3c1=O)Oc1ccccc1-2"
    target = "N[C@H]1CCN(c2c(F)cc3c(=O)c(C(=O)O)cn4c3c2Sc2ccccc2-4)C1"
    ample = map_atoms(source, target, element_change=False)
    assert ample.status == "complete"
    assert ample.metadata["budget"]["retries"] == 0
    budget = ample.metadata["budget"]["nodes_used"] * 3 // 2
    tight = map_atoms(source, target, element_change=False, settings=Settings(node_budget=budget))
    assert tight.metadata["budget"]["retries"] >= 1
    assert tight.status == "complete"
    assert tight.metadata["size"] == ample.metadata["size"]


# --- review fixes (2026-11-14) -------------------------------------------------------

from atommapper.adapter import build_view, parse  # noqa: E402
from atommapper.coloring import coloring  # noqa: E402
from atommapper.cost import change_size  # noqa: E402
from atommapper.mapper import _bridges, _color, _connected_fragments, _Rules  # noqa: E402
from atommapper.refine import refine  # noqa: E402
from tests._exhaustive import lomap_optimum  # noqa: E402

SMALL_PILOT = [
    # the optimum held no seed: fragments on ring and chain labels ignore elements
    ("CC1=C2C(=O)OCC[C@H]2[C@H](O)C1", "CC1=C2COC(=O)[C@H](CO)[C@H]2[C@H](O)C1"),
    ("Clc1ccc2c(c1Cl)CNCCC2", "CCC1Cc2ccccc2CN1"),
    ("NC1=NC(CO)CSCC1", "CC1CC(N)=NC2CCCCC2S1"),
    ("CN1CCCC1COc1cncnc1", "CN1C2C=C(c3cccnc3)CC1CC2"),
    ("C[C@H]1[C@H](C(=O)O)[C@H]1[C@H](N)C(=O)O", "N[C@H](C(=O)O)[C@@H]1[C@@H](C(=O)O)C1(F)F"),
    # no common bond: a single kept atom
    ("OCCCNc1ccnc2cc(Cl)ccc12", "O=[N+]([O-])c1cn2c(n1)OC[C@@H](O)C2"),
    ("C#CCOC/C=C(\\C)CCC=C(C)C", "CC1CC(=O)O[C@@H]2C[C@H](C)CC[C@@H]12"),
]


@pytest.mark.parametrize("element_change", [True, False])
@pytest.mark.parametrize(("source", "target"), SMALL_PILOT)
def test_small_pairs_reach_the_exhaustive_optimum(source, target, element_change):
    result = map_atoms(source, target, element_change=element_change)
    assert result.status != "timed_out"
    assert result.metadata["size"] == lomap_optimum(source, target, element_change)


@pytest.mark.parametrize(
    ("source", "target", "size"),
    [
        ("C", "CC", 2),  # no common bond: one kept carbon
        ("CC", "CC1CC1", 9),  # a chain carbon may stay alone; ring atoms may not
        ("CB", "CN", 1),  # one bond, turned around: C-B against N-C
        ("CCO", "CO", 2),
        ("CN", "C[C@H](N)O", 4),
        ("[Na+].CC(=O)[O-]", "[K+].CC(=O)[O-]", 1),  # lone atoms pair by color
        ("O", "N", 1),
    ],
)
def test_small_inputs_with_elements_free(source, target, size):
    assert map_atoms(source, target).metadata["size"] == size
    assert map_atoms(source, target).metadata["size"] == lomap_optimum(source, target)


def test_with_elements_fixed_a_lone_ion_is_not_turned_into_another():
    assert (
        map_atoms("[Na+].CC(=O)[O-]", "[K+].CC(=O)[O-]", element_change=False).metadata["size"] == 2
    )


def test_of_two_components_with_bonds_only_one_can_stay():
    result = map_atoms("CCO.CCN", "CCO.CCN")
    assert result.metadata["size"] > 0
    assert result.metadata["checks"]["identical_inputs_give_no_change"] is None
    assert map_atoms("CCO", "CCO").metadata["checks"]["identical_inputs_give_no_change"]


def test_the_result_is_labelled_optimal_given_its_seeds_unless_a_bound_proves_more():
    plain = map_atoms("Cc1ccccc1", "Cc1ccccc1N")
    assert plain.metadata["optimality"] == "optimal_given_seed"
    checked = map_atoms("Cc1ccccc1", "Cc1ccccc1N", settings=Settings(global_budget_s=5))
    assert checked.metadata["optimality"] == "optimal"
    assert checked.metadata["global_check"]["gap"] == 0


@pytest.mark.parametrize(
    ("source", "target"),
    [("c1ccc2ccccc2c1", "c1ccc2cccc2cc1"), ("C1CCC2CCCCC2C1", "C1CCC2CCCC2CC1")],
)
def test_fused_rings_are_not_matched_along_another_fusion_bond(source, target):
    # the 10-atom perimeters match, every perimeter bond kept, only with each fusion bond
    # lost between kept atoms
    s, t, rules = _views(source, target)
    perimeter = dict(zip(_perimeter(s.alignment), _perimeter(t.alignment), strict=True))
    assert sum(len(n) for n in rules.kept_adjacency(perimeter).values()) == 2 * 10
    assert rules.lost_between_kept(perimeter)
    assert not rules.accept(perimeter)
    result = map_atoms(source, target)
    assert (result.metadata["size"], len(result.mapping)) == (42, 0)
    assert result.metadata["size"] == lomap_optimum(source, target)


def _perimeter(g):
    """The outer cycle of two fused rings, from one fusion atom."""
    fusion = [v for v in g if g.degree(v) == 3]
    outer = g.copy()
    outer.remove_edge(*fusion)
    return [u for u, _ in nx.find_cycle(outer, source=fusion[0])]


def test_a_bond_lost_between_paired_atoms_makes_a_node_hopeless():
    *_, rules = _views("CC1CCC1", "CC1CCCC1")
    # source 1-4 closes the four-ring; pairing 1 and 4 with unbonded target atoms loses it
    assert rules.lost_between_kept({1: 1, 4: 3}) == {1, 4}
    assert rules.hopeless({1: 1, 4: 3}, {1: 1, 3: 4}, {})


def test_seeds_from_the_atoms_a_stereo_trim_drops_reach_the_optimum():
    # pilot pair 31620: every largest fragment inverts the centre, and the pieces the
    # trim leaves reach 20; seeded from the centre's pairs too, the search finds 14
    source = "CC(=O)N[C@H](C)C(=O)CCC(=O)O"
    target = "C[C@H](NC(=O)C(F)(F)F)C(=O)CCC(=O)O"
    result = map_atoms(source, target)
    assert result.metadata["shift_seeds"] > 0
    assert result.metadata["size"] == lomap_optimum(source, target) == 14


STEREO_SHIFTS = [
    # (source, target, element_change): the optimum under the stereo rule pairs the
    # surroundings of the inverted element otherwise than the fragments do
    # a chain shifted along the double bond, so that it maps onto single bonds
    ("CC/C=C/CC", "CC/C=C\\CC", True),
    ("C/C=C/C=C/C", "C/C=C\\C=C/C", True),
    ("C/C=C/[C@H](C)O", "C/C=C\\[C@H](C)O", True),
    ("[Na+].C/C=C/C", "[K+].C/C=C\\C", True),
    # cinnamic acid, E to Z: the ring stays, the chain goes
    ("c1ccccc1/C=C/C(=O)O", "c1ccccc1/C=C\\C(=O)O", True),
    ("c1ccccc1/C=C/C(=O)O", "c1ccccc1/C=C\\C(=O)O", False),
    # the two carbon arms of the centre swapped
    ("C[C@H](O)CC", "C[C@@H](O)CC", False),
    ("C[C@H](O)C=C", "C[C@@H](O)C=C", False),
]


@pytest.mark.parametrize("objective", ["change", "bonds"])
@pytest.mark.parametrize(("source", "target", "element_change"), STEREO_SHIFTS)
def test_the_result_reaches_the_exhaustive_optimum_across_a_stereo_trim(
    source, target, element_change, objective
):
    result = map_atoms(source, target, element_change=element_change, objective=objective)
    assert result.status == "complete"
    assert result.metadata["shift_seeds"] > 0
    assert result.metadata["cost"] == lomap_optimum(source, target, element_change, objective)
    assert "inverted" not in [v["status"] for v in result.metadata["stereo"]]


def test_a_chain_shifts_along_an_inverted_double_bond():
    # 5 atoms stay, the double bond on single bonds (two order changes), not 3 with it
    result = map_atoms("CC/C=C/CC", "CC/C=C\\CC")
    assert (result.metadata["size"], len(result.mapping)) == (6, 5)
    assert sum(c.kind == "changed" for c in result.bond_changes) == 2


@pytest.mark.parametrize(("source", "target", "element_change"), STEREO_SHIFTS)
def test_shifted_seeds_keep_the_result_independent_of_atom_order(source, target, element_change):
    def outcome(a, b):
        metadata = map_atoms(a, b, element_change=element_change).metadata
        return metadata["size"], metadata["kept"]

    reference = outcome(source, target)
    rng = random.Random(f"{source}>>{target}")
    for _ in range(3):
        assert outcome(shuffled(source, rng), shuffled(target, rng)) == reference
    assert outcome(target, source) == reference


def test_no_seed_is_added_without_a_stereo_trim():
    # without a stereo element there is nothing to drop, inversions allowed or not
    assert map_atoms("CCO", "CCN").metadata["shift_seeds"] == 0
    assert map_atoms("CCO", "CCN", allow_stereo_inversion=True).metadata["shift_seeds"] == 0


def test_allowed_inversions_are_seeded_from_the_strict_trims_too():
    # the strict trim drops the double bond's ends: its shift seeds start the search with
    # inversions allowed as well
    source, target = "CC/C=C/CC", "CC/C=C\\CC"
    strict = map_atoms(source, target).metadata["shift_seeds"]
    assert strict > 0
    assert map_atoms(source, target, allow_stereo_inversion=True).metadata["shift_seeds"] == strict


@pytest.mark.parametrize("element_change", [True, False])
@pytest.mark.parametrize(
    ("source", "target"),
    [
        # pilot pair 31620: every largest fragment inverts the centre
        ("CC(=O)N[C@H](C)C(=O)CCC(=O)O", "C[C@H](NC(=O)C(F)(F)F)C(=O)CCC(=O)O"),
        ("C[C@H]1[C@H](C(=O)O)[C@H]1[C@H](N)C(=O)O", "N[C@H](C(=O)O)[C@@H]1[C@@H](C(=O)O)C1(F)F"),
        ("C[C@H](N)C(=O)N[C@@H](C)C(=O)O", "C[C@@H](N)C(=O)N[C@H](C)C(=O)OC"),
    ],
)
def test_with_inversions_allowed_the_result_reaches_the_exhaustive_optimum(
    source, target, element_change
):
    result = map_atoms(source, target, element_change=element_change, allow_stereo_inversion=True)
    assert result.status != "timed_out"
    optimum = lomap_optimum(source, target, element_change, allow_stereo_inversion=True)
    assert result.metadata["size"] == optimum


def test_lomaps_own_mapping_costs_no_less():
    # pair 171 of the pilot: OpenFE LOMAP 3.3.0 keeps the macrocycle (27 atoms)
    source = "OCCN1CCN(c2ccc3cc2COCC=CCOCc2cccc(c2)-c2ccnc(n2)N3)CC1"
    target = "C1=CCOCc2cc(ccc2OCCN2CCCCC2)Nc2nccc(n2)-c2cccc(c2)COC1"
    lomap = [(7, 10), (8, 9), (9, 8), (10, 7), (11, 6), (12, 5), (13, 4), (14, 3), (15, 2)]
    lomap += [(16, 1), (17, 0), (18, 35), (19, 34), (20, 33), (21, 31), (22, 30), (23, 29)]
    lomap += [(24, 28), (25, 27), (26, 32), (27, 25), (28, 24), (29, 23), (30, 22), (31, 21)]
    lomap += [(32, 26), (33, 20)]
    s = build_view(parse(source, "s"), "s", coloring("ring"))
    t = build_view(parse(target, "t"), "t", coloring("ring"))
    mapping = {s.input_to_vertex[a]: t.input_to_vertex[b] for a, b in lomap}
    assert _Rules(s, t).accept(mapping)
    result = map_atoms(source, target)
    assert result.status == "complete"
    assert result.metadata["size"] <= change_size(s.alignment, t.alignment, mapping)


def test_bridges_agree_with_networkx():
    rng = random.Random(0)
    for _ in range(300):
        g = nx.gnm_random_graph(rng.randint(1, 14), rng.randint(0, 20), seed=rng.randrange(10**9))
        adj = {v: list(g.neighbors(v)) for v in g}
        expected = {(u, v) for a, b in nx.bridges(g) for u, v in ((a, b), (b, a))}
        assert _bridges(adj, adj) == expected


PAIRS_FOR_RULES = [
    ("CC1=C2C(=O)OCC[C@H]2[C@H](O)C1", "CC1=C2COC(=O)[C@H](CO)[C@H]2[C@H](O)C1"),
    ("Clc1ccc2c(c1Cl)CNCCC2", "CCC1Cc2ccccc2CN1"),
    ("c1ccc2ccccc2c1", "c1ccccc1CCc1ccccc1"),
]


def _views(source, target, element_change=True):
    paint = coloring("ring" if element_change else "element-ring")
    s = build_view(parse(source, "s"), "s", paint)
    t = build_view(parse(target, "t"), "t", paint)
    return s, t, _Rules(s, t)


@pytest.mark.parametrize(("source", "target"), PAIRS_FOR_RULES)
def test_every_trimmed_piece_satisfies_the_rules(source, target):
    s, t, rules = _views(source, target)
    fragments, *_ = _connected_fragments(s, t, rules, Settings(), _color, 10**7)
    assert fragments
    for fragment in fragments:
        for piece in rules.trimmed_pieces(fragment):
            assert rules.accept(piece)


@pytest.mark.parametrize(("source", "target"), PAIRS_FOR_RULES)
def test_pruning_by_the_rules_never_changes_the_answer(source, target):
    s, t, rules = _views(source, target)
    costs = []
    for prune in (None, rules.hopeless):
        best = None
        for a in s.alignment:
            for b in t.alignment:
                if s.alignment.nodes[a]["color"] != t.alignment.nodes[b]["color"]:
                    continue
                anchors = {a: b}
                out = refine(
                    s.alignment,
                    t.alignment,
                    anchors,
                    deadline=time.perf_counter() + 600,
                    max_nodes=10**7,
                    max_alternatives=1,
                    source_rank=s.ranks,
                    target_rank=t.ranks,
                    source_classes=s.classes,
                    target_classes=t.classes,
                    accept=lambda m, anchors=anchors: m == anchors or rules.accept(m),
                    prune=prune,
                    bound=best,
                )
                if rules.accept(out.mapping) and (best is None or out.cost < best):
                    best = out.cost
        costs.append(best)
    assert costs[0] == costs[1]
