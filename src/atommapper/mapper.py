"""Atom mapping under LOMAP's rules, with the smallest change or LOMAP's objective.

LOMAP (Liu et al., J Comput Aided Mol Des 2013; the OpenFE code) maps two ligands through
one connected maximum common substructure in which ring atoms match only ring atoms and
every ring is complete, trimmed of inverted stereocentres. ``map_atoms`` keeps those rules:

- **One piece.** What is kept is one connected piece (kept atoms joined by kept bonds);
  atoms without bonds on either side, such as counter-ions, are exempt.
- **Ring to ring.** A ring atom is kept only as a ring atom (the ``ring`` coloring, or
  ``element-ring`` without ``element_change``).
- **Complete rings.** Every kept ring atom and every kept ring bond lies on a cycle of the
  kept part: a ring is kept whole or not at all, one ring at a time, so a benzene can be
  kept as one ring of a naphthalene (LOMAP's ``completeRingsOnly`` and
  ``delete_broken_ring``).
- **No bond lost between kept atoms.** Two kept atoms bonded on one side are bonded on
  the other, so fused rings cannot be matched along a different fusion bond (LOMAP's
  ``trim_mcs_fix_broken_rdkit_code``, which checks the first molecule only; here both).
- **No inverted stereo element**, unless ``allow_stereo_inversion`` is set (below). A kept
  tetrahedral centre keeps its configuration (LOMAP's ``trim_mcs_chiral_atoms``), and so does
  a kept double bond whose four neighbours are kept or hydrogen (E stays E; LOMAP names
  chiral atoms only, and this extends the rule so that no stereo element is free); which
  atoms go instead is the objective's choice. A centre's hydrogen
  counts as kept with it, so two kept heavy neighbours can already fix the configuration.
  LOMAP compares only the neighbours inside its heavy-atom match and misses such an
  inversion (it maps the hydrogens afterwards); this rule is stricter on purpose.
- **Hydrogens implicit.** Only heavy atoms are mapped; a changed hydrogen count is an
  attribute change of its atom.

With ``element_change`` (LOMAP's default), a kept atom may change element and a kept bond
may change order.

With ``allow_stereo_inversion`` (not LOMAP's rule), the last rule is lifted: a kept
tetrahedral centre or double bond may be inverted, and each inverted stereo element counts one
change, like a change of element or order, under either objective. A mapping that keeps a
mirrored core and inverts its centres then competes with one that drops the core; with the
option off, it cannot.

Two objectives:

- ``"change"`` (the default): the smallest change; each atom or bond
  left out and each change of element or order counts one.
- ``"bonds"``: LOMAP's. Its MCS (RDKit ``FindMCS``, any bond matching any bond) keeps the
  most bonds, whatever their orders; ties go to the most atoms, then to the smallest
  change. LOMAP trims one MCS to its rules after choosing it, which can lose bonds another
  MCS would have kept; here the rules are applied during the search, around seeds.

The search starts from connected common fragments (Koch's c-cliques, ``_mces.fragments``):
every largest one, and every one whose largest piece keeps the most bonds once trimmed to
complete rings. It trims each to the rules, keeps the largest pieces each trim leaves, and
refines each with the anchored branch-and-bound of ``refine``, recording only leaves that
satisfy the rules. With ``element_change`` the fragments are searched on element labels and
then on ring and chain labels alone; that coarse pass stops at
``Settings.coarse_fragment_steps``, and the fragments found by then seed the search. When
a trim drops atoms for an inverted stereo element, the search is also
seeded from each dropped atom paired with every target atom of the fragment of its
color (``_shift_seeds``): the optimum then usually pairs the element's surroundings
otherwise than the fragment did (a chain shifted along a double bond, the arms of a centre
swapped). With inversions allowed, the search also starts from every seed it would start
from with them forbidden, so allowing them never gives a larger change (unless a budget or
the time limit stops the search). The seeds are kept whole, so the result is optimal
given them, not optimal: when the largest fragments break another rule, the optimum may
hold no seed, and the result can then be far from it (README, "Limits").

The time limit is counted in work, not read from a clock (``clock``): every stage advances
the call's virtual clock, and a stage stops when that clock, or the wall-clock guard behind
it, says so.
"""

import time
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import asdict
from typing import Any

from rdkit import Chem

from . import _mces
from . import clock as clocks
from .adapter import MolView, build_view, parse
from .clock import VirtualClock
from .coloring import coloring
from .cost import change_size, global_bound, kept_counts, lomap_weights
from .errors import UnsupportedInput
from .fragment_search import (
    ENGINE,
    _arrays,
    _ordered,
    pair_isolated,
    remaining,
    reversed_bond_pieces,
    single_pairs,
)
from .graph import canonical_key
from .preference import dedupe, preference_key
from .refine import EPS, RefineOutcome, refine
from .result import MappingResult, assemble, unsupported, versions
from .settings import Settings
from .stereo import _TETRAHEDRAL, _compare, _perceive

_RULES = (
    "around seeds, over correspondences that keep one connected piece, pair ring atoms "
    "with ring atoms only, keep every ring complete, lose no bond between kept atoms and "
    "invert no stereo element, a tetrahedral centre or a double bond (LOMAP's mapping rules, "
    "which name only chiral atoms); the seeds are kept whole (optimal_given_seed)"
)
OBJECTIVES = {
    "change": "smallest change " + _RULES,
    "bonds": "most bonds kept (LOMAP's objective), then most atoms, then smallest change " + _RULES,
}
_INVERSION_RULE = (
    "invert no stereo element, a tetrahedral centre or a double bond (LOMAP's mapping rules, "
    "which name only chiral atoms)"
)
_INVERSION_COUNTED = (
    "count each inverted stereo element, a tetrahedral centre or a double bond, as a change "
    "(LOMAP's other mapping rules)"
)


def _objective_text(objective: str, allow_stereo_inversion: bool) -> str:
    text = OBJECTIVES[objective]
    return text.replace(_INVERSION_RULE, _INVERSION_COUNTED) if allow_stereo_inversion else text


REASONS = {
    "complete": None,
    "timed_out": "a search budget ran out; the result is the best found so far "
    "(see metadata['budget']['stopped_by'])",
    "no_common_vertices": "no atom can be kept under the rules",
}


_BRIDGE_CACHE = 1 << 17
"""Potential graphs whose bridges ``_Rules`` remembers, per pair of molecules."""


class _Rules:
    """LOMAP's rules on one pair of molecule views, precomputed once."""

    def __init__(
        self, source: MolView, target: MolView, allow_stereo_inversion: bool = False
    ) -> None:
        self.allow_stereo_inversion = allow_stereo_inversion
        self.source, self.target = source.alignment, target.alignment
        # _trim's results by (correspondence, gentle, forbid_inversion): trimmed_pieces and
        # stereo_dropped trim the same fragments, often thousands of them on one pair.
        self._trims: dict[
            tuple[frozenset[tuple[int, int]], bool, bool], tuple[dict[int, int], set[int]]
        ] = {}
        self.adj = {v: list(self.source.neighbors(v)) for v in self.source}
        self.adj_set = {v: set(n) for v, n in self.adj.items()}
        self.source_lone = {v for v, n in self.adj.items() if not n}
        self.target_lone = {v for v in self.target if self.target.degree(v) == 0}
        self.target_adj = {v: set(self.target.neighbors(v)) for v in self.target}
        self.source_color = {v: self.source.nodes[v]["color"] for v in self.source}
        self.target_color = {v: self.target.nodes[v]["color"] for v in self.target}
        self.source_ring_atoms = {v for v in source.graph if source.graph.nodes[v]["cycles"]}
        self.target_ring_atoms = {v for v in target.graph if target.graph.nodes[v]["cycles"]}
        self.source_ring_bonds = _ring_bonds(source.mol)
        self.target_ring_bonds = _ring_bonds(target.mol)
        # Source bonds by index, for bond sets as bitmasks: the bonds at each atom, and
        # the bridges of each potential graph met so far (they repeat from node to node).
        self.bonds = list(self.source.edges)
        self.target_bonds = list(self.target.edges)
        self.incident = dict.fromkeys(self.adj, 0)
        for b, (u, v) in enumerate(self.bonds):
            self.incident[u] |= 1 << b
            self.incident[v] |= 1 << b
        self.bridge_masks: dict[int, int] = {}
        _, source_stereo = _perceive(source.mol)
        _, target_stereo = _perceive(target.mol)
        self.centres = {
            atoms[0]: info
            for (kind, atoms), info in source_stereo.items()
            if kind == "atom" and info.type == _TETRAHEDRAL
        }
        self.target_centres = {
            atoms[0]: info
            for (kind, atoms), info in target_stereo.items()
            if kind == "atom" and info.type == _TETRAHEDRAL
        }
        # Double bonds, by their atoms in the order RDKit gives them (kind "bond" is only
        # ever a double bond), and the molecules to find the partner bond in the target.
        self.double_bonds = {
            atoms: info for (kind, atoms), info in source_stereo.items() if kind == "bond"
        }
        self.target_double_bonds = {
            atoms: info for (kind, atoms), info in target_stereo.items() if kind == "bond"
        }
        self.target_mol = target.mol
        self.max_inversions = min(
            len(self.centres) + len(self.double_bonds),
            len(self.target_centres) + len(self.target_double_bonds),
        )

    def exempt(self, s: int, t: int) -> bool:
        return s in self.source_lone and t in self.target_lone

    def kept_adjacency(self, mapping: Mapping[int, int]) -> dict[int, list[int]]:
        """Kept atoms (exempt ones left out) and, for each, its neighbours across kept
        bonds."""
        return {
            s: [n for n in self.adj[s] if n in mapping and mapping[n] in self.target_adj[t]]
            for s, t in mapping.items()
            if not self.exempt(s, t)
        }

    def _is_ring(self, s: int, mapping: Mapping[int, int]) -> bool:
        return s in self.source_ring_atoms or mapping[s] in self.target_ring_atoms

    def _is_ring_bond(self, u: int, v: int, mapping: Mapping[int, int]) -> bool:
        return (u, v) in self.source_ring_bonds or (
            mapping[u],
            mapping[v],
        ) in self.target_ring_bonds

    def broken(self, mapping: Mapping[int, int], forbid_inversion: bool = False) -> set[int]:
        """Source atoms that break a rule other than one piece: ring atoms and ends of ring
        bonds off every cycle of the kept part, inverted stereocentres, and both ends of a
        bond lost between kept atoms. Inverted stereo elements break no rule when they are
        allowed; they cost a change instead (``inversion_cost``). ``forbid_inversion``
        applies the rule against them even then (the strict trims of ``trimmed_pieces``)."""
        out = self._off_cycles(mapping, self.kept_adjacency(mapping))
        if forbid_inversion or not self.allow_stereo_inversion:
            out |= self.inverted(mapping)
        out |= self.lost_between_kept(mapping)
        return out

    def lost_between_kept(self, mapping: Mapping[int, int]) -> set[int]:
        """Source atoms at either end of a bond that one side holds between two kept atoms
        and the other does not (LOMAP's ``trim_mcs_fix_broken_rdkit_code``, on both
        sides)."""
        out = set()
        target_adj = self.target_adj
        for u, v in self.bonds:
            if u in mapping and v in mapping and mapping[v] not in target_adj[mapping[u]]:
                out |= {u, v}
        if len(mapping) > 1:
            inverse = {t: s for s, t in mapping.items()}
            adj_set = self.adj_set
            for a, b in self.target_bonds:
                u, v = inverse.get(a), inverse.get(b)
                if u is not None and v is not None and v not in adj_set[u]:
                    out |= {u, v}
        return out

    def _off_cycles(self, mapping: Mapping[int, int], adj: Mapping[int, list[int]]) -> set[int]:
        """Kept ring atoms with no bond on a cycle of ``adj``, and both ends of kept ring
        bonds that are bridges of it."""
        if not any(self._is_ring(s, mapping) for s in adj):
            return set()  # no kept ring atom, so no ring bond either: nothing to drop
        bridges = _bridges(adj, mapping)
        target_adj = self.target_adj
        out = set()
        for s, t in mapping.items():
            neighbours = adj.get(s)
            if neighbours is None:
                continue
            if self._is_ring(s, mapping) and all((s, n) in bridges for n in neighbours):
                out.add(s)
            for n in neighbours:
                if (
                    s < n
                    and (s, n) in bridges
                    and n in mapping
                    and mapping[n] in target_adj[t]
                    and self._is_ring_bond(s, n, mapping)
                ):
                    out |= {s, n}
        return out

    def with_bonds_reversed(self, mapping: Mapping[int, int]) -> list[dict[int, int]]:
        """``fragment_search.reversed_bond_pieces`` on this pair, read from the cached lookups."""
        return reversed_bond_pieces(
            mapping, self.bonds, self.source_color, self.adj, self.target_adj
        )

    def hopeless(
        self,
        mapping: Mapping[int, int],
        inverse: Mapping[int, int],
        excluded: Mapping[int, frozenset[int]],
    ) -> bool:
        """Whether no completion of ``mapping`` can satisfy the rules on bonds.

        A bond lost between two paired atoms stays lost, whatever is paired below. And
        whatever is kept below this node is a subgraph of the potential graph: the bonds
        kept now, every bond between two unpaired atoms, and every bond from a paired atom
        to an unpaired one that can still pair across it (with a free, uncolored-out,
        unexcluded neighbour of the paired atom's partner). A bridge of a graph is a
        bridge of every subgraph that holds it, so a kept ring atom off every cycle of the
        potential graph, or a kept ring bond that is a bridge of it, stays so.
        """
        target_adj, target_color, source_color = (
            self.target_adj,
            self.target_color,
            self.source_color,
        )
        for a, b in self.target_bonds:
            u, v = inverse.get(a), inverse.get(b)
            if u is not None and v is not None and v not in self.adj_set[u]:
                return True
        free: dict[int, list[int]] = {}

        def open_across(s: int, n: int) -> bool:
            partners = free.get(s)
            if partners is None:
                partners = free[s] = [t for t in target_adj[mapping[s]] if t not in inverse]
            color, out = source_color[n], excluded.get(n, frozenset())
            return any(target_color[t] == color and t not in out for t in partners)

        potential = 0
        for b, (u, v) in enumerate(self.bonds):
            tu, tv = mapping.get(u), mapping.get(v)
            if tu is None:
                keep = tv is None or open_across(v, u)
            elif tv is None:
                keep = open_across(u, v)
            elif tv in target_adj[tu]:
                keep = True
            else:
                return True  # a bond lost between paired atoms
            if keep:
                potential |= 1 << b
        bridges = self.bridge_masks.get(potential)
        if bridges is None:
            if len(self.bridge_masks) >= _BRIDGE_CACHE:
                self.bridge_masks.clear()
            bridges = self.bridge_masks[potential] = self._bridge_mask(potential)
        for s, t in mapping.items():
            if (s in self.source_ring_atoms or t in self.target_ring_atoms) and not self.incident[
                s
            ] & potential & ~bridges:
                return True
        # Kept ring bonds that are bridges: a potential bond between paired atoms is kept.
        candidates = potential & bridges
        while candidates:
            low = candidates & -candidates
            candidates ^= low
            u, v = self.bonds[low.bit_length() - 1]
            if u in mapping and v in mapping and self._is_ring_bond(u, v, mapping):
                return True
        return False

    def _bridge_mask(self, bonds: int) -> int:
        """The bridges of the graph on the source atoms with the bonds in ``bonds``."""
        adj: dict[int, list[int]] = {v: [] for v in self.adj}
        index = {}
        for b, (u, v) in enumerate(self.bonds):
            if bonds >> b & 1:
                adj[u].append(v)
                adj[v].append(u)
                index[u, v] = index[v, u] = b
        mask = 0
        for pair in _bridges(adj, adj):
            mask |= 1 << index[pair]
        return mask

    def inverted_elements(self, mapping: Mapping[int, int]) -> list[tuple[int, ...]]:
        """The stereo elements, as their source atoms, that a correspondence keeps inverted:
        tetrahedral centres and double bonds with both ends kept."""
        out: list[tuple[int, ...]] = []
        for s, info in self.centres.items():
            target_info = self.target_centres.get(mapping.get(s, -1))
            if target_info is None:
                continue
            if _compare(info, target_info, (s,), (mapping[s],), mapping) == "inverted":
                out.append((s,))
        for atoms, info in self.double_bonds.items():
            u, v = atoms
            if u not in mapping or v not in mapping:
                continue
            bond = self.target_mol.GetBondBetweenAtoms(mapping[u], mapping[v])
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            target_atoms = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            target_info = self.target_double_bonds.get(target_atoms)
            if target_info is None:
                continue
            if _compare(info, target_info, atoms, target_atoms, mapping) == "inverted":
                out.append(atoms)
        return out

    def inverted(self, mapping: Mapping[int, int]) -> set[int]:
        """The atoms of the inverted stereo elements: the centre, or both ends of a bond."""
        return {a for atoms in self.inverted_elements(mapping) for a in atoms}

    def inversion_cost(self, mapping: Mapping[int, int]) -> int:
        """The inverted stereo elements a correspondence keeps, one change each (0 unless
        they are allowed, since otherwise no accepted correspondence keeps one)."""
        return len(self.inverted_elements(mapping)) if self.allow_stereo_inversion else 0

    def pieces(self, mapping: Mapping[int, int]) -> list[set[int]]:
        adj = self.kept_adjacency(mapping)
        seen: set[int] = set()
        out = []
        for start in adj:
            if start in seen:
                continue
            piece, frontier = {start}, [start]
            while frontier:
                for n in adj[frontier.pop()]:
                    if n not in piece:
                        piece.add(n)
                        frontier.append(n)
            seen |= piece
            out.append(piece)
        return out

    def accept(self, mapping: Mapping[int, int]) -> bool:
        return len(self.pieces(mapping)) <= 1 and not self.broken(mapping)

    def trimmed_pieces(self, mapping: Mapping[int, int]) -> list[dict[int, int]]:
        """``mapping`` cut to the rules two ways, then the largest remaining pieces of each
        (ties all kept), as LOMAP keeps the largest fragment after its trims. The smaller
        pieces a trim leaves are poor seeds: refining from a few atoms costs a large search
        and rarely finds better.

        Atoms that break a rule are dropped until none does, all at once each round, so
        the order of the atoms does not matter. The strict trim drops every such atom each
        round. The gentle one drops only kept ring atoms on no cycle while there are any
        (dropping them breaks no other cycle), and keeps more of a fragment that threads
        through rings. Both go: refinement keeps a seed whole, and the larger seed is not
        always the better start.

        With inversions allowed, the trims run twice: as the rules stand, which keeps an
        inverted element, and as if inversions were forbidden. A seed that holds an
        inverted element keeps it through refinement, so without the second run the search
        would miss every mapping that only a seed cut free of the inversion leads to, and
        could end worse than with inversions forbidden. With both, it starts from every seed
        the strict search starts from, and each leads at least as far as under the rule.
        """
        out = []
        for forbid in (False, True) if self.allow_stereo_inversion else (True,):
            for gentle in (False, True):
                kept, _ = self._trim(mapping, gentle, forbid)
                pieces = self.pieces(kept)
                largest = max((len(piece) for piece in pieces), default=0)
                out += [
                    {s: kept[s] for s in sorted(piece)} for piece in pieces if len(piece) == largest
                ]
        return out

    def stereo_dropped(self, mapping: Mapping[int, int]) -> set[int]:
        """Source atoms either trim of ``trimmed_pieces`` drops for an inverted stereo
        element, under the rule against them even when inversions are allowed (the strict
        trims, which seed that search too)."""
        return self._trim(mapping, False, True)[1] | self._trim(mapping, True, True)[1]

    def _trim(
        self, mapping: Mapping[int, int], gentle: bool, forbid_inversion: bool
    ) -> tuple[dict[int, int], set[int]]:
        """``mapping`` trimmed to the rules, and the atoms dropped for an inverted stereo
        element on the way (remembered: callers must not change them).
        ``forbid_inversion`` drops inverted elements even when inversions are allowed."""
        key = (frozenset(mapping.items()), gentle, forbid_inversion)
        if key not in self._trims:
            self._trims[key] = self._trim_uncached(mapping, gentle, forbid_inversion)
        return self._trims[key]

    def _trim_uncached(
        self, mapping: Mapping[int, int], gentle: bool, forbid_inversion: bool
    ) -> tuple[dict[int, int], set[int]]:
        kept = dict(mapping)
        inverted: set[int] = set()
        forbid = forbid_inversion or not self.allow_stereo_inversion
        while drop := self._to_drop(kept, gentle, forbid):
            if forbid:
                inverted |= drop & self.inverted(kept)
            for s in drop:
                del kept[s]
        return kept, inverted

    def _to_drop(
        self, mapping: Mapping[int, int], gentle: bool, forbid_inversion: bool
    ) -> set[int]:
        if gentle:
            adj = self.kept_adjacency(mapping)
            bridges = _bridges(adj, adj)
            lonely = {
                s
                for s, neighbours in adj.items()
                if self._is_ring(s, mapping) and all((s, n) in bridges for n in neighbours)
            }
            if lonely:
                return lonely
        return self.broken(mapping, forbid_inversion)


def _bridges(adj: Mapping[int, list[int]], roots: Iterable[int]) -> set[tuple[int, int]]:
    """Bridges, in both orientations, of the components of ``roots`` in a simple undirected
    graph given as adjacency lists (Tarjan, iterative)."""
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    out: set[tuple[int, int]] = set()
    for root in roots:
        if root in disc or root not in adj:
            continue
        disc[root] = low[root] = len(disc)
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            v, parent, neighbours = stack[-1]
            for w in neighbours:
                if w == parent:
                    continue
                if w in disc:
                    if disc[w] < low[v]:
                        low[v] = disc[w]
                else:
                    disc[w] = low[w] = len(disc)
                    stack.append((w, v, iter(adj[w])))
                    break
            else:
                stack.pop()
                if parent != -1:
                    if low[v] < low[parent]:
                        low[parent] = low[v]
                    if low[v] > disc[parent]:
                        out.add((parent, v))
                        out.add((v, parent))
    return out


def _ring_bonds(mol: Chem.Mol) -> set[tuple[int, int]]:
    """Ring bonds as atom pairs, in both orientations."""
    out = set()
    for bond in mol.GetBonds():
        if bond.IsInRing():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            out |= {(u, v), (v, u)}
    return out


def _connected_fragments(
    source: MolView,
    target: MolView,
    rules: _Rules,
    settings: Settings,
    label: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    max_steps: int,
    clock: VirtualClock,
    known_score: int = 0,
) -> tuple[list[dict[int, int]], int, str | None, bool, int]:
    """The connected common fragments on the atom labels ``label`` gives that seed the
    search: every largest one, and every one whose largest piece keeps the most bonds once
    trimmed to complete rings. On ring and chain labels alone, the largest fragments can
    thread through rings and lose most of their atoms to the trim. ``known_score`` is a
    trimmed score some common fragment is known to reach (the search starts from it, and
    starts again from zero if no fragment it scores reaches it); the last value returned
    is the best trimmed score found. The engine's work advances ``clock``, and what the
    clock has left is the engine's work limit."""
    src = _ordered(source.alignment, source.ranks)
    tgt = _ordered(target.alignment, target.ranks)
    keys = sorted(
        {label(source.alignment.nodes[v]) for v in src}
        | {label(target.alignment.nodes[v]) for v in tgt}
    )
    index = {key: i for i, key in enumerate(keys)}
    source_labels = [index[label(source.alignment.nodes[v])] for v in src]
    target_labels = [index[label(target.alignment.nodes[v])] for v in tgt]
    _, source_bonds = _arrays(source.alignment, src, {k: 0 for k in _colors(source)})
    _, target_bonds = _arrays(target.alignment, tgt, {k: 0 for k in _colors(target)})
    if not source_bonds or not target_bonds or max_steps <= 0:
        return [], 0, None if max_steps > 0 else "steps", False, known_score
    if clock.expired:
        return [], 0, "work", False, known_score
    left = remaining(clock.deadline)
    if left is not None and left <= 0:
        return [], 0, "time", False, known_score
    rates = clock.rates
    rings = (
        [v in rules.source_ring_atoms for v in src],
        [(src[a], src[b]) in rules.source_ring_bonds for a, b in source_bonds],
        [v in rules.target_ring_atoms for v in tgt],
        [(tgt[a], tgt[b]) in rules.target_ring_bonds for a, b in target_bonds],
    )
    fragments, steps, stop, truncated, _split, _pairs, best_score, work = _mces.fragments(
        source_labels,
        source_bonds,
        target_labels,
        target_bonds,
        max_steps=max_steps,
        max_fragments=settings.max_fragments,
        min_bonds=1,
        max_pairs=settings.fragment_bond_pairs,
        best_only=True,
        rings=rings,
        time_limit_s=left,
        known_score=known_score,
        work_limit=clock.left,
        work_costs=(rates.relation_pair, rates.step, rates.word, rates.leaf_bond),
    )
    for name, units in zip(("relation_pair", "step", "word", "leaf_bond"), work, strict=True):
        clock.charge(name, units)
    out = [{src[a]: tgt[b] for a, b in atoms} for atoms, _bonds in fragments]
    return out, steps, stop, truncated, best_score


def _shift_seeds(
    source: MolView,
    target: MolView,
    rules: _Rules,
    fragments: Iterable[Mapping[int, int]],
    clock: VirtualClock,
    size: int,
) -> tuple[list[dict[int, int]], bool]:
    """Single pairs that seed the search when a trim drops atoms of a fragment for an
    inverted stereo element: each dropped source atom with every target atom of the
    fragment of its color. The pieces the trim leaves hold the stereo element's
    surroundings paired as in the fragment, and the best correspondence under the rule
    often pairs them otherwise (a chain shifted along a double bond, two arms of a centre
    swapped); a correspondence grows from any one of its pairs, and these are its likeliest
    ones. With inversions allowed they come from the strict trims, which seed that search
    too (``_Rules.trimmed_pieces``). The second value says a clock stopped the stage before
    every fragment was read; then none is returned. ``size`` is the unit of the work."""
    out = []
    for fragment in fragments:
        if clock.late():
            # Shift seeds follow the others, and past the limit only the first seed is
            # refined: none of them would be.
            return [], True
        clock.charge("piece", size)
        dropped = rules.stereo_dropped(fragment)
        if not dropped:
            continue
        targets = sorted(set(fragment.values()))
        for s in sorted(dropped):
            color = source.alignment.nodes[s]["color"]
            for t in targets:
                pair = {s: t}
                if target.alignment.nodes[t]["color"] == color:
                    clock.charge("piece", size)
                    if rules.accept(pair):
                        out.append(pair)
    clock.charge("piece", len(out) * size)
    return dedupe(out, source, target), False


def _colors(view: MolView) -> set[Hashable]:
    return {d["color"] for _, d in view.alignment.nodes(data=True)}


def _color(atom: Mapping[str, Any]) -> tuple[Any, ...]:
    return atom["color"]


def _color_and_element(atom: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*atom["color"], atom["q"][0])


def map_atoms(
    source: str | Chem.Mol,
    target: str | Chem.Mol,
    *,
    element_change: bool = True,
    objective: str = "change",
    allow_stereo_inversion: bool = False,
    settings: Settings | None = None,
) -> MappingResult:
    """Find an atom correspondence between two molecules with a connected shared core.

    Parameters
    ----------
    source, target : str or rdkit.Chem.Mol
        SMILES strings or RDKit molecules. No coordinates are required. Molecules are
        copied and sanitized; existing atom-map numbers are ignored. Ordinary attached
        hydrogens are made implicit. Returned atom indices refer to the original input
        molecules, or to SMILES parsed with explicit hydrogens retained.
    element_change : bool, default True
        Allow mapped atoms to change element. If False, elements must match. Bond orders
        may change in either mode; ring membership and complete rings are always required.
    objective : {"change", "bonds"}, default "change"
        Minimize change size, or maximize kept bonds, then kept atoms, then minimize
        change size. Each inserted/deleted atom or bond, element change, and bond-order
        change counts as one. Other attribute changes are reported but not charged.
    allow_stereo_inversion : bool, default False
        Allow a mapped tetrahedral centre or double bond to invert and count each
        inversion as one change. Inversions appear in the change lists and stereo metadata.
    settings : Settings or None, default None
        Search budgets and limits. None uses Settings(), including a time limit of 60
        virtual seconds (counted in work, so the same calls stop on every machine; a
        virtual second is a second of the reference machine, see ``calibrate``).

    Returns
    -------
    MappingResult
        Mapped SMILES, input atom correspondence, changes, status, and metadata.
        Inspect ``status``: ``timed_out`` returns a partial result after a work or time
        limit; ``unsupported_input`` returns an explanation without mapped SMILES;
        ``no_common_vertices`` returns an empty mapping. A ``complete`` result is usually
        optimal only given the search seeds, not globally optimal.

    Raises
    ------
    InvalidInput
        An input has the wrong type, is empty, or cannot be parsed or sanitized.
    ValueError
        The objective is not "change" or "bonds".

    Notes
    -----
    ``metadata["size"]`` is the change size under either objective, not the number of
    entries in the change lists. ``metadata["cost"]`` is the objective's scalar cost;
    it equals size for "change". ``metadata["kept"]`` counts mapped atoms and kept bonds.
    See the module docstring for the constraints and search method.

    Examples
    --------
    >>> result = map_atoms("CCO", "CC=O")
    >>> result.status
    'complete'
    >>> result.metadata["size"]
    1
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; valid: {', '.join(sorted(OBJECTIVES))}")
    settings = settings or Settings()
    colors = coloring("ring" if element_change else "element-ring")
    start = time.perf_counter()
    clock = VirtualClock(
        settings.work_limit, clocks.REFERENCE_RATES, settings.wall_guard_factor, start
    )
    source_mol = parse(source, "source")
    target_mol = parse(target, "target")
    try:
        source_view = build_view(source_mol, "source", colors)
        target_view = build_view(target_mol, "target", colors)
        rules = _Rules(source_view, target_view, allow_stereo_inversion)
        mapping, tags, meta = _search(
            source_view,
            target_view,
            rules,
            settings,
            objective,
            colors.element_hard,
            clock,
        )
    except UnsupportedInput as error:
        return unsupported(str(error), settings)
    status = (
        "timed_out"
        if meta["budget"]["stopped_by"]
        else "complete"
        if mapping
        else "no_common_vertices"
    )
    metadata: dict[str, Any] = {
        "versions": versions(),
        "settings": asdict(settings),
        "objective": _objective_text(objective, allow_stereo_inversion),
        "objective_name": objective,
        "element_change": element_change,
        "allow_stereo_inversion": allow_stereo_inversion,
        "coloring": {"name": colors.name, "hard": list(colors.hard)},
        **meta,
    }
    result = assemble(
        source_view,
        target_view,
        mapping,
        tags,
        element_hard=not element_change,
        status=status,
        reason=REASONS[status],
        metadata=metadata,
    )
    result.metadata["timings"]["total"] = time.perf_counter() - start
    return result


def _search(
    source: MolView,
    target: MolView,
    rules: _Rules,
    settings: Settings,
    objective: str,
    element_hard: bool,
    clock: VirtualClock,
) -> tuple[dict[int, int], dict[int, str], dict[str, Any]]:
    allow = rules.allow_stereo_inversion
    # The unit of the Python stages' work: each step of theirs reads both molecules whole.
    size = sum(
        graph.number_of_nodes() + graph.number_of_edges()
        for graph in (source.alignment, target.alignment)
    )
    clock.charge("call", size)
    began = time.perf_counter()
    inversions = rules.max_inversions if allow else 0
    weights = (
        lomap_weights(source.alignment, target.alignment, inversions)
        if objective == "bonds"
        else (1, 1)
    )
    # With element soft, the fragments on ring and chain labels ignore the elements that
    # the objective charges for; the fragments on element labels too seed the search.
    labels = [_color] if element_hard else [_color, _color_and_element]
    # With element soft, the search on element labels runs first: a fragment on element
    # labels is one on ring and chain labels too, so the coarser search (most of the fragment
    # time) starts from its best score and cuts more. That search scores parts of maximal
    # cliques, which can score less than the fragment (the trim is not monotone); if none
    # reaches the score, the engine searches again from zero (``FragmentLimits``).
    # The fragments keep the order of ``labels``.
    # That coarser search (the coarse pass) has a step cap of its own
    # (``Settings.coarse_fragment_steps``): on polycyclic pairs it takes most of the time
    # and seldom finds a fragment that changes the result. Reaching the cap is part of the
    # method, not a stop: the fragments found so far seed the search.
    found_by_label: dict[Callable[[Mapping[str, Any]], tuple[Any, ...]], list] = {}
    steps, stops, truncated, known, coarse_capped = 0, set(), False, 0, False
    for label in reversed(labels):
        budget = settings.fragment_steps - steps
        coarse = label is _color and len(labels) > 1
        limit = min(budget, settings.coarse_fragment_steps) if coarse else budget
        found, used_steps, stop, capped, best_score = _connected_fragments(
            source,
            target,
            rules,
            settings,
            label,
            limit,
            clock,
            known_score=known,
        )
        found_by_label[label] = found
        known = max(known, best_score)
        steps += used_steps
        truncated |= capped
        if stop == "steps" and limit < budget:
            coarse_capped = True
        elif stop:
            stops.add(stop)
    fragments = [fragment for label in labels for fragment in found_by_label[label]]
    fragment_stop = ",".join(sorted(stops)) or None
    fragments_done = time.perf_counter()
    # Seeding advances the clock too: on pairs with thousands of fragments it takes seconds.
    # Past the limit, the seeds made so far are searched (refinement stops at once) and
    # the call reports which clock stopped it.
    late = False
    trims = 4 if allow else 2
    pieces: list[dict[int, int]] = []
    for fragment in fragments:
        # The first fragment that yields a piece is trimmed even past the limit: when the
        # clock stopped the fragment search, dropping it would leave only a single-atom seed.
        if pieces and clock.late():
            late = True
            break
        clock.charge("trim", trims * size)
        pieces += rules.trimmed_pieces(fragment)
    late = late or clock.late()
    turned_pieces = []
    for piece in pieces:
        if late:
            break
        if clock.late():
            late = True
            break
        turned = rules.with_bonds_reversed(piece)
        clock.charge("piece", (1 + len(turned)) * size)
        turned_pieces += [mapping for mapping in turned if rules.accept(mapping)]
    pieces += turned_pieces
    if late:
        # Past the limit only the first seed is refined: keep the first largest piece
        # rather than rank thousands, which takes a large part of a second.
        pieces = [max(pieces, key=len)] if pieces else []
    clock.charge("piece", len(pieces) * size)
    seeds = dedupe(pieces, source, target)
    if not seeds:
        # No common bond survives the rules: a single kept atom may still, any chain atom.
        # Past the limit, the first one is kept rather than rank them all.
        singles = single_pairs(source.alignment, target.alignment)
        accepted = (pair for pair in singles if rules.accept(pair))
        if late:
            first = next(accepted, None)
            seeds = [first] if first is not None else []
        else:
            clock.charge("piece", 2 * len(singles) * size)
            seeds = dedupe(list(accepted), source, target)
    shifts, shifts_late = (
        ([], True) if late else _shift_seeds(source, target, rules, fragments, clock, size)
    )
    late |= shifts_late
    keys = {canonical_key(seed, source.classes, target.classes) for seed in seeds}
    shifts = [
        seed for seed in shifts if canonical_key(seed, source.classes, target.classes) not in keys
    ]
    seeds += shifts
    if settings.k_seeds is not None:
        seeds = seeds[: settings.k_seeds]
    seeds = seeds or [{}]
    nodes_left, nodes_used = settings.node_budget, 0
    solved: list[tuple[RefineOutcome, dict[int, int]]] = []

    def attempt(anchors: dict[int, int], max_nodes: int) -> RefineOutcome:
        return refine(
            source.alignment,
            target.alignment,
            anchors,
            deadline=clock.deadline,
            max_nodes=max_nodes,
            max_alternatives=settings.max_alternatives,
            source_rank=source.ranks,
            target_rank=target.ranks,
            source_classes=source.classes,
            target_classes=target.classes,
            accept=rules.accept,
            prune=rules.hopeless,
            bound=min((done.cost for done, _ in solved), default=None),
            weights=weights,
            extra_cost=rules.inversion_cost if allow else None,
            clock=clock,
            node_units=size,
        )

    seeding_done = time.perf_counter()
    for index, seed in enumerate(seeds):
        # Past the limit, the seeds not yet refined are left out: each would still cost
        # a refinement set-up (a large part of a second over thousands of seeds).
        if solved and clock.late():
            late = True
            break
        clock.charge("seed", size)
        # Lone atoms (no bond on either side) are paired around each seed, with what the
        # seed leaves: a seed may hold a lone atom's best partner.
        anchors = {
            **seed,
            **pair_isolated(source.alignment, target.alignment, source.ranks, target.ranks, seed),
        }
        out = attempt(anchors, nodes_left // (len(seeds) - index))
        nodes_left -= out.nodes
        nodes_used += out.nodes
        solved.append((out, anchors))
    # A seed that ran out of its share tries again with what the others left: it starts
    # over, with the best cost found so far to match. A finished retry replaces the first
    # try; one that stops again replaces it only if it got as far.
    stopped = sorted(
        (k for k, (out, _) in enumerate(solved) if out.stopped_by == "nodes"),
        key=lambda k: solved[k][0].cost,
    )
    retries = 0
    for index, k in enumerate(stopped):
        if clock.late():
            # A retry past the limit would pay its set-up only to stop at once.
            late = True
            break
        first, anchors = solved[k]
        share = nodes_left // (len(stopped) - index)
        if share <= first.nodes:
            continue
        clock.charge("seed", size)
        out = attempt(anchors, share)
        retries += 1
        nodes_left -= out.nodes
        nodes_used += out.nodes
        if out.complete or out.cost <= first.cost + EPS:
            solved[k] = (out, anchors)
    reasons = set(stops) | clock.stops
    reasons |= {out.stopped_by for out, _ in solved if out.stopped_by}
    refinement_done = time.perf_counter()
    best = min(out.cost for out, _ in solved)
    tied = [(out, anchors) for out, anchors in solved if out.cost <= best + EPS]
    # The tie-break over every tied mapping, not only each seed's first in canonical order
    out, anchors, mapping = min(
        ((done, anchors, m) for done, anchors in tied for m in done.alternatives),
        key=lambda dam: (not dam[0].complete, preference_key(source, target, dam[2])),
    )
    tags = {s: "seed" if s in anchors else "refined" for s in mapping}
    distinct = {frozenset(m.items()) for done, _ in tied for m in done.alternatives}
    optimality = "not_established" if reasons else "optimal_given_seed"
    bound, bound_complete = None, None
    if settings.global_budget_s > 0:
        bound, bound_complete = global_bound(
            source.alignment, target.alignment, settings.global_budget_s, weights
        )
        if not reasons and bound_complete and bound is not None and out.cost <= bound + EPS:
            optimality = "optimal"
    meta = {
        "size": change_size(source.alignment, target.alignment, mapping)
        + rules.inversion_cost(mapping),
        "cost": out.cost,
        "kept": kept_counts(source.alignment, target.alignment, mapping),
        "optimality": optimality,
        "ambiguity": {
            "count": len(distinct),
            "lower_bound": any(done.alternatives_capped for done, _ in tied),
        },
        "global_check": {
            "bound": bound,
            "complete": bound_complete,
            "gap": None if bound is None else out.cost - bound,
        },
        "fragment_search": {
            "engine": ENGINE,
            "objective": "largest connected common edge subgraphs (Koch c-cliques), and "
            "those with the most bonds once trimmed to complete rings; on element labels "
            "too when element is soft",
            "fragments": len(fragments),
            "truncated": truncated,
            "stopped_by": fragment_stop,
            "coarse_capped": coarse_capped,
            "steps": steps,
        },
        "seeds": len(seeds),
        "shift_seeds": len(shifts),
        "budget": {
            "nodes_used": nodes_used,
            "steps_used": steps,
            "retries": retries,
            "stopped_by": ",".join(sorted(reasons)) or None,
            "virtual_s": clock.seconds,
            "work": dict(clock.work),
        },
        # Wall times of the stages, reported only: nothing reads them to decide anything.
        "timings": {
            "fragment_search": fragments_done - began,
            "seeding": seeding_done - fragments_done,
            "refinement": refinement_done - seeding_done,
        },
    }
    return mapping, tags, meta
