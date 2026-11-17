"""Exhaustive optimum under map_atoms's rules, for small molecules, under either objective.

Every valid correspondence is one connected piece plus lone atoms, so it grows, one kept
bond at a time, from any one of its pairs, with the lone atoms it leaves paired at their
cheapest (``pair_isolated`` around that pair): refining from every single pair, with the
best cost so far to match, finds the optimum.
"""

import time

from atommapper.adapter import build_view, parse
from atommapper.coloring import coloring
from atommapper.cost import change_size, lomap_weights
from atommapper.fragment_search import pair_isolated
from atommapper.mapper import _Rules
from atommapper.refine import refine


def lomap_optimum(
    source: str,
    target: str,
    element_change: bool = True,
    objective: str = "change",
    allow_stereo_inversion: bool = False,
) -> int:
    """The least cost: the size for ``"change"``, ``lomap_weights`` for ``"bonds"``."""
    paint = coloring("ring" if element_change else "element-ring")
    s = build_view(parse(source, "s"), "s", paint)
    t = build_view(parse(target, "t"), "t", paint)
    rules = _Rules(s, t, allow_stereo_inversion)
    isolated = pair_isolated(s.alignment, t.alignment, s.ranks, t.ranks)
    centres = rules.max_inversions if allow_stereo_inversion else 0
    weights = lomap_weights(s.alignment, t.alignment, centres) if objective == "bonds" else (1, 1)
    best = change_size(s.alignment, t.alignment, isolated, weights)
    for a in s.alignment:
        for b in t.alignment:
            if s.alignment.nodes[a]["color"] != t.alignment.nodes[b]["color"]:
                continue
            anchors = {a: b, **pair_isolated(s.alignment, t.alignment, s.ranks, t.ranks, {a: b})}

            def accept(m, anchors=anchors):
                return m == anchors or rules.accept(m)

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
                accept=accept,
                prune=rules.hopeless,
                bound=best,
                weights=weights,
                extra_cost=rules.inversion_cost if allow_stereo_inversion else None,
            )
            assert out.stopped_by is None
            if rules.accept(out.mapping) and out.cost < best:
                best = int(out.cost)
    return best
