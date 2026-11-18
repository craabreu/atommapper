"""Refinement: anchored branch-and-bound over connected correspondences grown from anchors."""

import time
from collections.abc import Callable, Collection, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .clock import VirtualClock
from .cost import FORBIDDEN, change_size
from .graph import OrderKey, tie_order

EPS = 1e-9

Bond = tuple[int, int, Any]
"""A bond as the bounds read it: its two atoms and its order."""


@dataclass(frozen=True)
class RefineOutcome:
    mapping: dict[int, int]
    cost: float
    complete: bool
    alternatives: tuple[dict[int, int], ...]
    alternatives_capped: bool
    nodes: int
    stopped_by: str | None
    """``"nodes"`` when the node budget ran out, ``"work"`` when the virtual clock did,
    ``"time"`` when the wall clock did, else None."""


def refine(
    source: nx.Graph,
    target: nx.Graph,
    anchors: Mapping[int, int],
    *,
    deadline: float,
    max_nodes: int,
    max_alternatives: int,
    source_rank: Mapping[int, int],
    target_rank: Mapping[int, int],
    source_classes: Mapping[int, int],
    target_classes: Mapping[int, int],
    accept: Callable[[Mapping[int, int]], bool] | None = None,
    prune: Callable[[Mapping[int, int], Mapping[int, int], Mapping[int, frozenset[int]]], bool]
    | None = None,
    bound: float | None = None,
    weights: tuple[int, int] = (1, 1),
    extra_cost: Callable[[Mapping[int, int]], float] | None = None,
    clock: VirtualClock | None = None,
    node_units: int = 0,
) -> RefineOutcome:
    """Lowest-cost correspondence grown from the anchors through kept bonds.

    A pair is offered only when it is bonded to a paired pair on both sides (the bond is
    kept), so the kept part grows as one piece, and the search cuts branches with a lower
    bound that reads this growth (``_BranchAndBound._connected_bound``). ``accept``, when
    given, is a test every recorded leaf must pass (the anchors included); ``mapper`` uses
    it for its own rules. ``prune``, when given, is called with the correspondence, its
    inverse and the exclusions of the node, and cuts the node when no leaf below can
    satisfy ``accept``; it must never cut a node that has one.

    ``bound``, when given, is a cost to match (the best found from other anchors): only
    leaves that cost no more are recorded, and branches that cannot reach it are cut.
    When none does, the outcome is the anchors at their own cost.

    ``weights`` (atom, bond) are the cost of an atom and of a bond present on one side
    only (``cost.change_size``); ``map_atoms`` passes ``cost.lomap_weights`` for LOMAP's
    objective.

    ``extra_cost``, when given, is a cost added to every leaf and to the anchors on top of
    ``cost.change_size``, a function of the correspondence alone. It must never be
    negative: the lower bound does not read it, so a negative value would let the search
    cut a branch that holds the best leaf. ``map_atoms`` uses it to charge inverted
    stereocentres.

    The search stops after ``max_nodes`` branch-and-bound nodes, which makes the result
    independent of machine speed and load, and so does ``clock``, when given: each node
    advances it by ``node_units`` units of its ``node`` rate, and the search stops when it
    is past its limit. ``deadline`` is the call's wall-clock limit.
    """
    search = _BranchAndBound(
        source,
        target,
        deadline,
        max_alternatives,
        source_rank,
        target_rank,
        anchors,
        source_classes,
        target_classes,
        accept,
        prune,
        bound,
        weights,
        extra_cost,
    )
    search.clock, search.node_units = clock, node_units
    search.max_nodes = max_nodes
    search.run()
    if search.alternatives:
        alternatives, cost = sorted(search.alternatives, key=search._order), search.best_cost
    else:
        alternatives, cost = [dict(anchors)], search.anchor_cost
    return RefineOutcome(
        mapping=alternatives[0],
        cost=cost,
        complete=search.stopped_by is None,
        alternatives=tuple(alternatives),
        alternatives_capped=search.capped,
        nodes=search.nodes,
        stopped_by=search.stopped_by,
    )


def _largest_pairing(allowed: Mapping[int, Collection[int]]) -> int:
    """Size of a largest one-to-one pairing of the keys with atoms each allows (augmenting
    paths)."""
    partner: dict[int, int] = {}

    def augment(s: int, seen: set[int]) -> bool:
        for t in allowed[s]:
            if t not in seen:
                seen.add(t)
                if t not in partner or augment(partner[t], seen):
                    partner[t] = s
                    return True
        return False

    return sum(augment(s, set()) for s in allowed)


def _both_ways(bonds: list[tuple[int, int, Any]]) -> dict[tuple[int, int], Any]:
    return {(u, v): order for a, b, order in bonds for u, v in ((a, b), (b, a))}


class _BranchAndBound:
    def __init__(
        self,
        source: nx.Graph,
        target: nx.Graph,
        deadline: float,
        max_alternatives: int,
        source_rank: Mapping[int, int],
        target_rank: Mapping[int, int],
        anchors: Mapping[int, int],
        source_classes: Mapping[int, int],
        target_classes: Mapping[int, int],
        accept: Callable[[Mapping[int, int]], bool] | None = None,
        prune: Callable[[Mapping[int, int], Mapping[int, int], Mapping[int, frozenset[int]]], bool]
        | None = None,
        bound: float | None = None,
        weights: tuple[int, int] = (1, 1),
        extra_cost: Callable[[Mapping[int, int]], float] | None = None,
    ) -> None:
        self.source, self.target = source, target
        self.deadline, self.max_alternatives = deadline, max_alternatives
        self.source_rank, self.target_rank = source_rank, target_rank
        self.source_classes, self.target_classes = source_classes, target_classes
        self.mapping: dict[int, int] = dict(anchors)
        self.inverse: dict[int, int] = {t: s for s, t in anchors.items()}
        self.excluded: dict[int, frozenset[int]] = {}
        self.accept = accept
        self.prune = prune
        self.weights = weights
        self.extra_cost = extra_cost
        if accept is not None and not accept(anchors):
            raise ValueError("the anchors fail the accept test")
        self.anchor_cost = self._cost(self.mapping)
        self.best_cost: float = self.anchor_cost
        self.alternatives: list[dict[int, int]] = [dict(anchors)]
        self.seen: set[frozenset[tuple[int, int]]] = {frozenset(anchors.items())}
        if bound is not None and bound < self.anchor_cost - EPS:
            self.best_cost, self.alternatives, self.seen = bound, [], set()
        self.capped = False
        self.max_nodes = 0
        self.clock: VirtualClock | None = None
        self.node_units = 0
        self.nodes = 0
        self.stopped_by: str | None = None
        # Plain copies of what the lower bound reads at every node.
        self._source_atom = {v: (d["color"], d["q"][0]) for v, d in source.nodes(data=True)}
        self._target_atom = {v: (d["color"], d["q"][0]) for v, d in target.nodes(data=True)}
        self._source_bonds = [(u, v, d["order"]) for u, v, d in source.edges(data=True)]
        self._target_bonds = [(u, v, d["order"]) for u, v, d in target.edges(data=True)]
        self._source_order = _both_ways(self._source_bonds)
        self._target_order = _both_ways(self._target_bonds)
        self._source_adjacency = {v: list(source.neighbors(v)) for v in source}
        self._target_adjacency = {v: list(target.neighbors(v)) for v in target}

    def run(self) -> None:
        if self.stopped_by is not None:
            return
        if self.nodes >= self.max_nodes:
            self.stopped_by = "nodes"
            return
        if self.clock is not None:
            if self.clock.expired:
                self.stopped_by = "work"
                return
            self.clock.charge("node", self.node_units)
        if time.perf_counter() > self.deadline:
            self.stopped_by = "time"
            return
        self.nodes += 1
        if self.prune is not None and self.prune(self.mapping, self.inverse, self.excluded):
            return
        if self._connected_bound() > self.best_cost + EPS:
            return
        choice = self._next_vertex()
        if choice is None:
            self._record()
            return
        s, candidates = choice
        for t in candidates:
            self.mapping[s] = t
            self.inverse[t] = s
            self.run()
            del self.mapping[s]
            del self.inverse[t]
            if self.stopped_by is not None:
                return
        previous = self.excluded.get(s)
        self.excluded[s] = (previous or frozenset()) | frozenset(candidates)
        self.run()
        if previous is None:
            del self.excluded[s]
        else:
            self.excluded[s] = previous

    def _candidates(self, s: int) -> list[int]:
        """Target atoms of ``s``'s color bonded to the partner of a paired neighbor of
        ``s``: pairing them keeps that bond."""
        color = self.source.nodes[s]["color"]
        excluded = self.excluded.get(s, frozenset())
        found = {
            t
            for n in self.source.neighbors(s)
            if n in self.mapping
            for t in self.target.neighbors(self.mapping[n])
            if t not in self.inverse
            and t not in excluded
            and self.target.nodes[t]["color"] == color
        }
        return sorted(found, key=self.target_rank.__getitem__)

    def _next_vertex(self) -> tuple[int, list[int]] | None:
        best: tuple[int, list[int]] | None = None
        unmatched = sorted(
            (v for v in self.source if v not in self.mapping), key=self.source_rank.__getitem__
        )
        for s in unmatched:
            candidates = self._candidates(s)
            if candidates and (best is None or len(candidates) < len(best[1])):
                best = (s, candidates)
        return best

    def _decided_cost(self) -> tuple[float, list[Bond], list[Bond]]:
        """The cost of the decisions so far: each pair's element change, and each bond
        between two paired atoms (lost or changed in order). Also the source and target
        bonds with an end still unpaired, in bond order."""
        bond_weight = self.weights[1]
        m, inverse = self.mapping, self.inverse
        source_atom, target_atom = self._source_atom, self._target_atom
        cost = 0.0
        for s, t in m.items():
            (color, element), (other_color, other_element) = source_atom[s], target_atom[t]
            if color != other_color:
                cost += FORBIDDEN
            elif element != other_element:
                cost += 1
        open_source: list[Bond] = []
        open_target: list[Bond] = []
        target_order = self._target_order
        for bond in self._source_bonds:
            u, v, order = bond
            if u in m and v in m:
                other = target_order.get((m[u], m[v]))
                cost += bond_weight if other is None else int(other != order)
            else:
                open_source.append(bond)
        source_order = self._source_order
        for bond in self._target_bonds:
            u, v, _ = bond
            if u in inverse and v in inverse:
                if (inverse[u], inverse[v]) not in source_order:
                    cost += bond_weight
            else:
                open_target.append(bond)
        return cost, open_source, open_target

    def _connected_bound(self) -> float:
        """A lower bound that reads how the kept part grows. It replaced a bound that
        counted unpaired atoms and bonds per color only, which on the slowest pilot pairs
        cut almost no node while most leaves cost far more than the best (2026-11-16, in
        moldelta). Its terms are least costs of separate parts of ``cost.change_size``:

        - the decisions so far (``_decided_cost``);
        - reachability: an atom is paired only through a kept bond to a paired atom, so an
          atom paired later is reachable, through unpaired atoms that still have a partner
          they may take, from an atom that has a candidate now; on the target side, from
          such a candidate, through atoms of a color some reachable atom has. Every other
          unpaired atom is left out, with its bonds;
        - atoms, per color: the largest pairing the node's exclusions allow (augmenting
          paths from the atoms that have exclusions; the others take any atom left), then
          ``atom_weight`` per atom left out and one per pair beyond those that can keep
          their element;
        - open bonds, per class: a bond from a paired atom ``a`` to an unpaired one is kept
          only as a bond from ``a``'s partner to an unpaired atom, and a bond between two
          unpaired atoms only as one between unpaired atoms of the same two colors; of
          ``n`` and ``k`` bonds in a class, ``|n - k|`` are lost (``bond_weight`` each) and
          the others change order at least ``min(n, k)`` minus the orders in common times.
        """
        atom_weight, bond_weight = self.weights
        m, inverse, excluded = self.mapping, self.inverse, self.excluded
        source_atom, target_atom = self._source_atom, self._target_atom
        source_adjacency, target_adjacency = self._source_adjacency, self._target_adjacency
        cost, open_source, open_target = self._decided_cost()

        open_targets: dict[Hashable, list[int]] = {}
        for t, (color, _) in target_atom.items():
            if t not in inverse:
                open_targets.setdefault(color, []).append(t)
        blocked: dict[int, bool] = {}

        def is_blocked(s: int) -> bool:
            """No unpaired target atom ``s`` may take."""
            if s not in blocked:
                pool = open_targets.get(source_atom[s][0], ())
                ex = excluded.get(s)
                blocked[s] = not pool or (ex is not None and all(t in ex for t in pool))
            return blocked[s]

        reach_source: set[int] = set()
        start_target: set[int] = set()
        for a, b in m.items():
            for s in source_adjacency[a]:
                if s in m:
                    continue
                color, ex = source_atom[s][0], excluded.get(s, frozenset())
                for t in target_adjacency[b]:
                    if t not in inverse and target_atom[t][0] == color and t not in ex:
                        reach_source.add(s)
                        start_target.add(t)
        stack = [s for s in reach_source if not is_blocked(s)]
        reach_source = set(stack)
        while stack:
            for y in source_adjacency[stack.pop()]:
                if y not in m and y not in reach_source and not is_blocked(y):
                    reach_source.add(y)
                    stack.append(y)
        reached_colors = {source_atom[s][0] for s in reach_source}
        stack = [t for t in start_target if target_atom[t][0] in reached_colors]
        reach_target = set(stack)
        while stack:
            for y in target_adjacency[stack.pop()]:
                if (
                    y not in inverse
                    and y not in reach_target
                    and target_atom[y][0] in reached_colors
                ):
                    reach_target.add(y)
                    stack.append(y)
        gone_source = {s for s in source_atom if s not in m and s not in reach_source}
        gone_target = {t for t in target_atom if t not in inverse and t not in reach_target}
        cost += atom_weight * (len(gone_source) + len(gone_target))

        by_color: dict[Hashable, tuple[list[int], list[int]]] = {}
        for s in reach_source:
            by_color.setdefault(source_atom[s][0], ([], []))[0].append(s)
        for t in reach_target:
            by_color.setdefault(target_atom[t][0], ([], []))[1].append(t)
        for sources, targets in by_color.values():
            n, k = len(sources), len(targets)
            elements: dict[Hashable, int] = {}
            for t in targets:
                elements[target_atom[t][1]] = elements.get(target_atom[t][1], 0) + 1
            ours: dict[Hashable, int] = {}
            for s in sources:
                ours[source_atom[s][1]] = ours.get(source_atom[s][1], 0) + 1
            same = sum(min(c, elements.get(e, 0)) for e, c in ours.items())
            pairs = min(n, k)
            constrained = [s for s in sources if s in excluded]
            if constrained:
                allowed = {s: [t for t in targets if t not in excluded[s]] for s in constrained}
                gone_source.update(s for s in constrained if not allowed[s])
                pairs = min(_largest_pairing(allowed) + n - len(constrained), k)
            cost += atom_weight * (n + k - 2 * pairs) + max(0, pairs - same)

        classes: dict[Hashable, tuple[dict[Any, int], dict[Any, int]]] = {}
        for u, v, order in open_source:
            if u in gone_source or v in gone_source:
                cost += bond_weight
                continue
            if u in m:
                key: Hashable = ("paired", m[u])
            elif v in m:
                key = ("paired", m[v])
            else:
                key = ("open", frozenset((source_atom[u][0], source_atom[v][0])))
            tally = classes.setdefault(key, ({}, {}))[0]
            tally[order] = tally.get(order, 0) + 1
        for u, v, order in open_target:
            if u in gone_target or v in gone_target:
                cost += bond_weight
                continue
            if u in inverse:
                key = ("paired", u)
            elif v in inverse:
                key = ("paired", v)
            else:
                key = ("open", frozenset((target_atom[u][0], target_atom[v][0])))
            tally = classes.setdefault(key, ({}, {}))[1]
            tally[order] = tally.get(order, 0) + 1
        for ours_, theirs in classes.values():
            n, k = sum(ours_.values()), sum(theirs.values())
            same = sum(min(c, theirs.get(o, 0)) for o, c in ours_.items())
            cost += bond_weight * abs(n - k) + min(n, k) - same
        return cost

    def _order(self, mapping: Mapping[int, int]) -> OrderKey:
        """The tie-break's canonical order (``graph.tie_order``), the one the search uses."""
        return tie_order(
            mapping, self.source_classes, self.target_classes, self.source_rank, self.target_rank
        )

    def _cost(self, mapping: Mapping[int, int]) -> float:
        cost: float = change_size(self.source, self.target, mapping, self.weights)
        if self.extra_cost is not None:
            cost += self.extra_cost(mapping)
        return cost

    def _record(self) -> None:
        cost = self._cost(self.mapping)
        if cost > self.best_cost + EPS:
            return
        if self.accept is not None and not self.accept(self.mapping):
            return
        key = frozenset(self.mapping.items())
        if cost < self.best_cost - EPS:
            self.best_cost = cost
            self.alternatives = [dict(self.mapping)]
            self.seen = {key}
            self.capped = False
        elif abs(cost - self.best_cost) <= EPS and key not in self.seen:
            if len(self.alternatives) < self.max_alternatives:
                self.alternatives.append(dict(self.mapping))
                self.seen.add(key)
            else:
                # Keep the alternatives smallest under the canonical key, not the first
                # found, so the kept set does not depend on the search order or on which
                # molecule is the source.
                self.capped = True
                order = self._order(self.mapping)
                worst = max(
                    range(len(self.alternatives)), key=lambda i: self._order(self.alternatives[i])
                )
                if order < self._order(self.alternatives[worst]):
                    self.seen.discard(frozenset(self.alternatives[worst].items()))
                    self.alternatives[worst] = dict(self.mapping)
                    self.seen.add(key)
