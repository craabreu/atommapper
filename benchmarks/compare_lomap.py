"""AtomMapper against OpenFE's LOMAP: time and result on the same pairs, side by side.

Usage: python benchmarks/compare_lomap.py run PAIRS.csv OUT.jsonl [--workers N] [--limit N]
                                              [--reverse]
       python benchmarks/compare_lomap.py summary OUT.jsonl [OUT.jsonl ...]
       python benchmarks/compare_lomap.py symmetry FORWARD.jsonl REVERSE.jsonl
       python benchmarks/compare_lomap.py timeouts PAIRS.csv OUT.jsonl [OUT.jsonl ...]
                                              [--limits 5 60] [--workers N]

PAIRS.csv needs ``source`` and ``target`` SMILES columns (``group`` optional). Needs LOMAP
(``lomap2`` on conda-forge or PyPI) besides atommapper.

Each pair is mapped, in the same worker one after the other, by:

- ``lomap``: ``lomap.mcs.MCS`` with its defaults (2D, a 20 s MCS timeout), from SMILES:
  the molecule with hydrogens added and 2D coordinates, then the heavy-atom map;
- ``change``: ``map_atoms`` with its default objective, the smallest change;
- ``bonds``: ``map_atoms(objective="bonds")``, LOMAP's objective, the most bonds kept;

each with elements free (LOMAP's ``element_change=True``) and fixed. Times are wall clock
for the whole call from the SMILES strings. Every mapping is then judged the same way, by
AtomMapper's own code: the atoms and bonds it keeps, the size of the change, and which of
LOMAP's rules (as AtomMapper states them) it breaks.

Each worker imports LOMAP and maps one small pair with both tools before its first timed
call, so no timing includes an import or a first-call warm-up.

``--reverse`` maps each pair the other way round (target to source). ``symmetry`` compares
a forward run with a reverse run of the same pairs: how often each method's answer depends
on which molecule comes first. ``timeouts`` maps again, with other LOMAP timeouts, the cases
where LOMAP's MCS search timed out, and reports how often its mapping changes: what a slower
or faster machine would have returned.

"Same mapping, up to symmetry" compares ``graph.canonical_key``: the sorted pairs of
symmetry classes (RDKit canonical ranks without tie-breaking) the two atoms of each kept pair
belong to. Two mappings with the same key are, in principle, not always related by a
symmetry of the molecules, so the test is slightly coarse.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor

from atommapper import map_atoms
from atommapper.adapter import build_view, parse
from atommapper.coloring import coloring
from atommapper.cost import change_size, kept_counts
from atommapper.graph import canonical_key
from atommapper.mapper import _Rules

RDLogger.DisableLog("rdApp.*")
METHODS = ("lomap", "change", "bonds")
CANCELED: list[bool] = []
"""Whether each MCS search of the current LOMAP call hit LOMAP's 20 s timeout, in which
case LOMAP goes on with the MCS found so far (``_record_cancels``)."""
MODES = {"free": True, "fixed": False}


def canonical(smiles: str) -> str:
    """The same string for both tools: canonical SMILES, hydrogens implicit, so atom indices
    agree between LOMAP's molecule and AtomMapper's input."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _record_cancels() -> None:
    """Wrap RDKit's MCS search, which LOMAP calls once or twice per pair, so that each
    call's ``canceled`` flag (its timeout) is recorded in ``CANCELED``."""
    from rdkit.Chem import rdFMCS

    if getattr(rdFMCS.FindMCS, "_recording", False):
        return
    search = rdFMCS.FindMCS

    def recording(*args, **kwargs):
        found = search(*args, **kwargs)
        CANCELED.append(bool(found.canceled))
        return found

    recording._recording = True  # type: ignore[attr-defined]
    rdFMCS.FindMCS = recording


def lomap_map(
    source: str, target: str, element_change: bool, time_limit: int = 20
) -> list[tuple[int, int]]:
    """LOMAP's heavy-atom map, with ``time_limit`` seconds for each MCS search (its default
    is 20)."""
    from lomap import mcs  # already imported by ``_warm_up`` in the benchmark's workers

    _record_cancels()

    def prepared(smiles: str) -> Chem.Mol:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        rdDepictor.Compute2DCoords(mol)
        return mol

    found = mcs.MCS(
        prepared(source),
        prepared(target),
        time=time_limit,
        verbose="error",
        element_change=element_change,
    )
    return [(int(a), int(b)) for a, b in found.heavy_atom_mcs_map()]


def atommapper_map(
    source: str, target: str, element_change: bool, objective: str
) -> tuple[list[tuple[int, int]], str | None]:
    """The heavy-atom map, and the limits that stopped the search (None when it finished):
    ``"time"`` among them means the clock stopped it."""
    result = map_atoms(source, target, element_change=element_change, objective=objective)
    pairs = [(a.source_index, a.target_index) for a in result.mapping]
    return pairs, result.metadata["budget"]["stopped_by"]


def judged(source: str, target: str, pairs: list[tuple[int, int]], element_change: bool) -> dict:
    """A mapping in AtomMapper's terms: what it keeps, the size of the change, the rules it
    breaks and its key up to symmetry (``graph.canonical_key``; see the module docstring)."""
    colors = coloring("ring" if element_change else "element-ring")
    s = build_view(parse(source, "source"), "source", colors)
    t = build_view(parse(target, "target"), "target", colors)
    mapping = {s.input_to_vertex[a]: t.input_to_vertex[b] for a, b in pairs}
    rules = _Rules(s, t)
    try:
        size = change_size(s.alignment, t.alignment, mapping)
    except ValueError:  # a pair of atoms of different colors
        size = None
    adjacency = rules.kept_adjacency(mapping)
    return {
        **kept_counts(s.alignment, t.alignment, mapping),
        "size": size,
        "broken": {
            "pieces": len(rules.pieces(mapping)) > 1,
            "rings": bool(rules._off_cycles(mapping, adjacency)),
            "stereo": bool(rules.inverted(mapping)),
            "lost_bond": bool(rules.lost_between_kept(mapping)),
        },
        "key": [list(p) for p in canonical_key(mapping, s.classes, t.classes)],
    }


def run_pair(args: tuple[int, dict[str, str]]) -> dict:
    index, row = args
    source, target = canonical(row["source"]), canonical(row["target"])
    record: dict = {"i": index, "group": row.get("group") or "all"}
    for mode, element_change in MODES.items():
        for method in METHODS:
            CANCELED.clear()
            stopped_by = None
            start = time.perf_counter()
            try:
                if method == "lomap":
                    pairs = lomap_map(source, target, element_change)
                else:
                    pairs, stopped_by = atommapper_map(source, target, element_change, method)
                error = None
            except Exception as exc:  # recorded, not raised: one bad pair must not stop a run
                pairs, error = [], f"{type(exc).__name__}: {exc}"[:300]
            seconds = time.perf_counter() - start
            entry: dict = {"t": seconds, "error": error}
            if method == "lomap":
                entry["timed_out"] = any(CANCELED)
            else:
                entry["stopped_by"] = stopped_by
                entry["timed_out"] = "time" in (stopped_by or "").split(",")
            if error is None:
                entry |= judged(source, target, pairs, element_change)
            record[f"{method}_{mode}"] = entry
    return record


def _warm_up() -> None:
    """Import LOMAP and map one small pair with each tool, so that a worker's first timed
    call pays no import (about a second for LOMAP) or first-call cost."""
    for element_change in MODES.values():
        lomap_map("CCO", "CCN", element_change)
        atommapper_map("CCO", "CCN", element_change, "change")


def _rows(path: str, limit: int | None, reverse: bool) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    if reverse:
        rows = [{**row, "source": row["target"], "target": row["source"]} for row in rows]
    return rows


def run(args: argparse.Namespace) -> None:
    rows = _rows(args.pairs, args.limit, args.reverse)
    with Pool(args.workers, initializer=_warm_up) as pool, open(args.out, "w") as out:
        jobs = pool.imap_unordered(run_pair, enumerate(rows), chunksize=1)
        for done, record in enumerate(jobs, 1):
            record["reverse"] = args.reverse
            out.write(json.dumps(record) + "\n")
            out.flush()
            if done % 500 == 0:
                print(f"{done}/{len(rows)}", file=sys.stderr, flush=True)


def _remap(job: tuple[str, str, str, int]) -> dict:
    source, target, mode, limit = job
    CANCELED.clear()
    pairs = lomap_map(canonical(source), canonical(target), MODES[mode], limit)
    entry = judged(canonical(source), canonical(target), pairs, MODES[mode])
    return {"timed_out": any(CANCELED), "bonds": entry["bonds"], "key": entry["key"]}


def timeouts(pairs_path: str, paths: list[str], limits: list[int], workers: int) -> str:
    """LOMAP mapped again, with each of ``limits`` seconds for its MCS search, on every case
    (pair, direction and element mode) of the records where its search timed out: how often
    the mapping and the bonds it keeps differ from the 20 s result."""
    cases = []
    for path in paths:
        with open(path) as handle:
            for record in map(json.loads, handle):
                row = _rows(pairs_path, None, record.get("reverse", False))[record["i"]]
                for mode in MODES:
                    if _timed_out(record, "lomap", mode):
                        cases.append((row, mode, record[f"lomap_{mode}"]))
    jobs = [
        (row["source"], row["target"], mode, limit) for limit in limits for row, mode, _ in cases
    ]
    with Pool(workers, initializer=_warm_up) as pool:
        results = pool.map(_remap, jobs, chunksize=1)
    lines = [
        f"{len(cases)} cases where LOMAP's MCS search timed out at 20 s ({', '.join(paths)})",
        "",
        "| timeout | still timed out | different mapping | more bonds | fewer bonds |",
        "|---|---|---|---|---|",
    ]
    for k, limit in enumerate(limits):
        chunk = results[k * len(cases) : (k + 1) * len(cases)]
        pairs = [(new, old) for new, (_, _, old) in zip(chunk, cases, strict=True)]
        lines.append(
            f"| {limit} s | {sum(new['timed_out'] for new, _ in pairs)} | "
            f"{sum(new['key'] != old['key'] for new, old in pairs)} | "
            f"{sum(new['bonds'] > old['bonds'] for new, old in pairs)} | "
            f"{sum(new['bonds'] < old['bonds'] for new, old in pairs)} |"
        )
    return "\n".join(lines)


def _swapped(key: list[list[int]]) -> list[list[int]]:
    """A mapping's key read from the other side, as a reverse run writes it."""
    return sorted([b, a] for a, b in key)


def symmetry(forward_path: str, reverse_path: str) -> str:
    """How often each method's answer changes when the two molecules swap places, on the
    pairs where no run of that method was stopped by the clock."""
    runs = []
    for path in (forward_path, reverse_path):
        with open(path) as handle:
            runs.append({r["i"]: r for r in map(json.loads, handle)})
    forward, reverse = runs
    common = sorted(forward.keys() & reverse.keys())
    lines = [f"{len(common)} pairs in both directions ({forward_path}, {reverse_path})", ""]
    lines += [
        "| | method | pairs compared | different bonds kept | different mapping |",
        "|---|---|---|---|---|",
    ]
    for mode in MODES:
        for method in METHODS:
            rows = [
                (forward[i][f"{method}_{mode}"], reverse[i][f"{method}_{mode}"]) for i in common
            ]
            rows = [(a, b) for a, b in rows if a["error"] is None and b["error"] is None]
            rows = [(a, b) for a, b in rows if not (a.get("timed_out") or b.get("timed_out"))]
            bonds = sum(a["bonds"] != b["bonds"] for a, b in rows)
            mapping = sum(a["key"] != _swapped(b["key"]) for a, b in rows)
            lines.append(
                f"| elements {mode} | {method} | {len(rows)} | {_share(bonds, len(rows))} | "
                f"{_share(mapping, len(rows))} |"
            )
    return "\n".join(lines)


def _timed_out(record: dict, method: str, mode: str) -> bool:
    """Whether the clock stopped ``method``'s search on this pair (records written before the
    flag existed count as not stopped)."""
    return bool(record[f"{method}_{mode}"].get("timed_out"))


def _seconds(value: float) -> str:
    return f"{1000 * value:.0f} ms" if value < 1 else f"{value:.1f} s"


def _share(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.1f}%)" if total else "0"


def _versus(ok: list[dict], mode: str, method: str, field: str, better: str) -> str:
    """How often ``method`` does better, as well, or worse than LOMAP on ``field``, with how
    many of the worse ones keep a stereo element LOMAP's mapping inverts."""
    rows = [r for r in ok if r[f"{method}_{mode}"][field] is not None]
    rows = [r for r in rows if r[f"lomap_{mode}"][field] is not None]

    def outcome(r: dict) -> int:
        a, b = r[f"{method}_{mode}"][field], r[f"lomap_{mode}"][field]
        return (a > b) - (a < b) if better == "more" else (a < b) - (a > b)

    wins = [r for r in rows if outcome(r) > 0]
    losses = [r for r in rows if outcome(r) < 0]
    inverted = sum(r[f"lomap_{mode}"]["broken"]["stereo"] for r in losses)
    return (
        f"better {_share(len(wins), len(rows))}, "
        f"same {_share(len(rows) - len(wins) - len(losses), len(rows))}, "
        f"worse {_share(len(losses), len(rows))} ({inverted} where LOMAP's mapping inverts "
        f"a stereo element)"
    )


def summary(paths: list[str]) -> str:
    """The comparison as Markdown tables, one section per element mode."""
    records = []
    for path in paths:
        with open(path) as handle:
            records += [json.loads(line) for line in handle]
    n = len(records)
    lines = [f"{n} pairs ({', '.join(paths)})", ""]
    for mode in MODES:
        # A search the clock stopped returns what it found by then, which depends on the
        # machine's speed and load: a pair where any method was stopped so is timed but
        # not compared. (A search stopped by a work budget stops at the same point on every
        # machine, so it stays in.)
        mapped = [r for r in records if all(r[f"{m}_{mode}"]["error"] is None for m in METHODS)]
        ok = [r for r in mapped if not any(_timed_out(r, m, mode) for m in METHODS)]
        lines += [f"### Elements {mode}", ""]
        errors = {m: sum(r[f"{m}_{mode}"]["error"] is not None for r in records) for m in METHODS}
        clock = {m: sum(_timed_out(r, m, mode) for r in records) for m in METHODS}
        budget = {
            m: sum(bool(r[f"{m}_{mode}"].get("stopped_by")) for r in records) - clock[m]
            for m in METHODS[1:]
        }
        lines += [
            f"Errors: {errors}. Stopped by the clock (LOMAP's 20 s MCS timeout, AtomMapper's "
            f"time limit): {clock}; AtomMapper stopped by a work budget: {budget}. Times are "
            f"over all pairs; mappings are compared on the {len(ok)} pairs that all three "
            f"mapped and that no clock stopped.",
            "",
        ]
        lines += ["| | LOMAP | AtomMapper (change) | AtomMapper (bonds) |", "|---|---|---|---|"]
        for label, fn in (
            ("median time", lambda xs: _seconds(statistics.median(xs))),
            ("mean time", lambda xs: _seconds(statistics.fmean(xs))),
            ("95th percentile", lambda xs: _seconds(sorted(xs)[int(0.95 * len(xs))])),
            ("99th percentile", lambda xs: _seconds(sorted(xs)[int(0.99 * len(xs))])),
            ("slowest", lambda xs: _seconds(max(xs))),
            ("total", lambda xs: f"{sum(xs):.0f} s"),
        ):
            row = [fn([r[f"{m}_{mode}"]["t"] for r in records]) for m in METHODS]
            lines.append(f"| {label} | " + " | ".join(row) + " |")
        for label, field in (("mean atoms kept", "atoms"), ("mean bonds kept", "bonds")):
            row = [f"{statistics.fmean(r[f'{m}_{mode}'][field] for r in ok):.2f}" for m in METHODS]
            lines.append(f"| {label} | " + " | ".join(row) + " |")
        sized = [r for r in ok if all(r[f"{m}_{mode}"]["size"] is not None for m in METHODS)]
        row = [f"{statistics.fmean(r[f'{m}_{mode}']['size'] for r in sized):.2f}" for m in METHODS]
        lines.append("| mean size of the change | " + " | ".join(row) + " |")
        for rule in ("pieces", "rings", "lost_bond", "stereo"):
            row = [
                _share(sum(r[f"{m}_{mode}"]["broken"][rule] for r in ok), len(ok)) for m in METHODS
            ]
            lines.append(f"| breaks rule: {rule} | " + " | ".join(row) + " |")
        lines.append("")

        same = sum(r[f"bonds_{mode}"]["key"] == r[f"lomap_{mode}"]["key"] for r in ok)
        lines += [
            f"- Bonds kept (LOMAP's objective), AtomMapper (bonds) against LOMAP: "
            f"{_versus(ok, mode, 'bonds', 'bonds', 'more')}.",
            f"- Size of the change (AtomMapper's objective), AtomMapper (change) against "
            f"LOMAP: {_versus(ok, mode, 'change', 'size', 'fewer')}.",
            f"- Same mapping as LOMAP, up to symmetry, for AtomMapper (bonds): "
            f"{_share(same, len(ok))}.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("pairs")
    run_parser.add_argument("out")
    run_parser.add_argument("--workers", type=int, default=8)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--reverse", action="store_true", help="map target to source")
    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("records", nargs="+")
    symmetry_parser = commands.add_parser("symmetry")
    symmetry_parser.add_argument("forward")
    symmetry_parser.add_argument("reverse")
    timeouts_parser = commands.add_parser("timeouts")
    timeouts_parser.add_argument("pairs")
    timeouts_parser.add_argument("records", nargs="+")
    timeouts_parser.add_argument("--limits", type=int, nargs="+", default=[5, 60])
    timeouts_parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "summary":
        print(summary(args.records))
    elif args.command == "symmetry":
        print(symmetry(args.forward, args.reverse))
    else:
        print(timeouts(args.pairs, args.records, args.limits, args.workers))


if __name__ == "__main__":
    main()
