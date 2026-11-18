"""Fragment search helpers: input to the engine and seeds that need no engine.

The connected common fragments themselves come from the Rust engine (``rust/``, built as
``atommapper._mces``) within a budget of search steps, so a search stopped by its
budget returns the same fragments on every machine, and so does one stopped by the cost
of its work, which is how the call's virtual clock (``clock``) reaches the engine; it
reports ``"virtual"``. With a deadline the engine also reads the clock; a search the clock
stopped reports ``"time"`` and depends on machine speed and load.
"""

import time
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any

import networkx as nx

from . import _mces

ENGINE = f"atommapper._mces {_mces.__version__}"
MAX_BOND_PAIRS = 20_000
"""Default ceiling on compatible bond pairs (``Settings.fragment_bond_pairs``). The search
graph grows with its square; a pair above it is not searched and reports
``stopped_by == "pairs"``."""


def _ordered(vertices: Collection[int], rank: Mapping[int, int] | None) -> list[int]:
    """Vertices in canonical rank order, so the engine's tie-breaks do not depend on the
    input atom order."""
    if rank is None:
        return sorted(vertices)
    return sorted(vertices, key=lambda v: (rank[v], v))


def _arrays(
    g: nx.Graph, order: list[int], labels: Mapping[object, int]
) -> tuple[list[int], list[tuple[int, int]]]:
    position = {v: i for i, v in enumerate(order)}
    bonds = sorted(
        (min(position[u], position[v]), max(position[u], position[v]))
        for u, v in g.subgraph(order).edges
    )
    return [labels[g.nodes[v]["color"]] for v in order], bonds


def remaining(deadline: float | None) -> float | None:
    """Seconds left before ``deadline`` (a ``time.perf_counter`` reading), None without one."""
    return None if deadline is None else deadline - time.perf_counter()


def pair_isolated(
    source: nx.Graph,
    target: nx.Graph,
    source_rank: Mapping[int, int],
    target_rank: Mapping[int, int],
    taken: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Pair single-vertex components one to one in canonical order: first those with equal
    qualifiers, then the rest by element, so a protonated single atom keeps its atom, then
    the rest by color: when the coloring leaves element out, a sodium ion becomes a
    potassium ion (one change) rather than one leaving and one arriving (two). Atoms paired
    in ``taken`` (on either side) are left out. Each color pairs as many atoms as the
    smaller side has, the same elements first, which is the cheapest pairing."""
    taken = taken or {}
    used = set(taken.values())
    targets = sorted(
        (v for v in target if target.degree(v) == 0 and v not in used),
        key=target_rank.__getitem__,
    )
    sources = sorted(
        (v for v in source if source.degree(v) == 0 and v not in taken),
        key=source_rank.__getitem__,
    )
    pairs: dict[int, int] = {}
    keys: tuple[Callable[[Mapping[str, Any]], object], ...] = (
        lambda d: d["q"],
        lambda d: d["q"][0],
        lambda d: d["color"],
    )
    for key in keys:
        buckets: dict[object, list[int]] = {}
        for t in targets:
            if t not in pairs.values():
                buckets.setdefault(key(target.nodes[t]), []).append(t)
        for s in sources:
            if s in pairs:
                continue
            bucket = buckets.get(key(source.nodes[s]))
            if bucket:
                pairs[s] = bucket.pop(0)
    return pairs


def reversed_bond_pieces(
    mapping: Mapping[int, int],
    bonds: Iterable[tuple[int, int]],
    color: Mapping[int, Any],
    source_adj: Mapping[int, Collection[int]],
    target_adj: Mapping[int, Collection[int]],
) -> list[dict[int, int]]:
    """``mapping`` with one two-atom piece turned around, for each such piece whose ends
    share a color. Such a piece fixes no orientation: the engine lists one of the two, and
    when element is soft the other can cost less (C-B against N-C). ``bonds`` are the
    source bonds in graph order, ``color`` the source colors; the adjacencies support
    ``in`` and iteration."""
    out = []
    for u, v in bonds:
        if u not in mapping or v not in mapping or color[u] != color[v]:
            continue
        x, y = mapping[u], mapping[v]
        if y not in target_adj[x]:
            continue
        alone = all(
            n in (u, v) or n not in mapping or mapping[n] not in target_adj[mapping[a]]
            for a in (u, v)
            for n in source_adj[a]
        )
        if alone:
            out.append({**mapping, u: y, v: x})
    return out


def single_pairs(source: nx.Graph, target: nx.Graph) -> list[dict[int, int]]:
    """Every pair of one source and one target atom of the same color, each a mapping of
    its own, for a search that has no common bond to start from."""
    return [
        {s: t}
        for s in sorted(source)
        for t in sorted(target)
        if source.nodes[s]["color"] == target.nodes[t]["color"]
    ]
