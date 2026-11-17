"""AtomMapper against OpenFE's LOMAP: time and result on the same pairs, side by side.

Usage: python benchmarks/compare_lomap.py run PAIRS.csv OUT.jsonl [--workers N] [--limit N]
       python benchmarks/compare_lomap.py summary OUT.jsonl [OUT.jsonl ...]

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


def lomap_map(source: str, target: str, element_change: bool) -> list[tuple[int, int]]:
    from lomap import mcs

    _record_cancels()

    def prepared(smiles: str) -> Chem.Mol:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        rdDepictor.Compute2DCoords(mol)
        return mol

    found = mcs.MCS(
        prepared(source), prepared(target), verbose="error", element_change=element_change
    )
    return [(int(a), int(b)) for a, b in found.heavy_atom_mcs_map()]


def atommapper_map(
    source: str, target: str, element_change: bool, objective: str
) -> list[tuple[int, int]]:
    result = map_atoms(source, target, element_change=element_change, objective=objective)
    return [(a.source_index, a.target_index) for a in result.mapping]


def judged(source: str, target: str, pairs: list[tuple[int, int]], element_change: bool) -> dict:
    """A mapping in AtomMapper's terms: what it keeps, the size of the change, the rules it
    breaks and its key up to symmetry."""
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
            start = time.perf_counter()
            try:
                if method == "lomap":
                    pairs = lomap_map(source, target, element_change)
                else:
                    pairs = atommapper_map(source, target, element_change, method)
                error = None
            except Exception as exc:  # recorded, not raised: one bad pair must not stop a run
                pairs, error = [], f"{type(exc).__name__}: {exc}"[:300]
            seconds = time.perf_counter() - start
            entry: dict = {"t": seconds, "error": error}
            if method == "lomap":
                entry["timed_out"] = any(CANCELED)
            if error is None:
                entry |= judged(source, target, pairs, element_change)
            record[f"{method}_{mode}"] = entry
    return record


def run(args: argparse.Namespace) -> None:
    with open(args.pairs, newline="") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]
    with Pool(args.workers) as pool, open(args.out, "w") as out:
        jobs = pool.imap_unordered(run_pair, enumerate(rows), chunksize=1)
        for done, record in enumerate(jobs, 1):
            out.write(json.dumps(record) + "\n")
            out.flush()
            if done % 500 == 0:
                print(f"{done}/{len(rows)}", file=sys.stderr, flush=True)


def _seconds(value: float) -> str:
    return f"{1000 * value:.0f} ms" if value < 1 else f"{value:.1f} s"


def _share(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.1f}%)" if total else "0"


def _versus(ok: list[dict], mode: str, method: str, field: str, better: str) -> str:
    """How often ``method`` does better, as well, or worse than LOMAP on ``field``, with how
    many of the better ones LOMAP's clock cut short and how many of the worse ones keep a
    stereo element LOMAP's mapping inverts."""
    rows = [r for r in ok if r[f"{method}_{mode}"][field] is not None]
    rows = [r for r in rows if r[f"lomap_{mode}"][field] is not None]

    def outcome(r: dict) -> int:
        a, b = r[f"{method}_{mode}"][field], r[f"lomap_{mode}"][field]
        return (a > b) - (a < b) if better == "more" else (a < b) - (a > b)

    wins = [r for r in rows if outcome(r) > 0]
    losses = [r for r in rows if outcome(r) < 0]
    timed_out = sum(r[f"lomap_{mode}"]["timed_out"] for r in wins)
    inverted = sum(r[f"lomap_{mode}"]["broken"]["stereo"] for r in losses)
    return (
        f"better {_share(len(wins), len(rows))} ({timed_out} where LOMAP timed out), "
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
        ok = [r for r in records if all(r[f"{m}_{mode}"]["error"] is None for m in METHODS)]
        lines += [f"### Elements {mode}", ""]
        errors = {m: sum(r[f"{m}_{mode}"]["error"] is not None for r in records) for m in METHODS}
        timeouts = sum(bool(r[f"lomap_{mode}"].get("timed_out")) for r in records)
        lines += [
            f"Errors: {errors}; compared on the {len(ok)} pairs that all three mapped. LOMAP's "
            f"MCS search reached its 20 s timeout on {_share(timeouts, n)} pairs.",
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
    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("records", nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        print(summary(args.records))


if __name__ == "__main__":
    main()
