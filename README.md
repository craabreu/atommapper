# AtomMapper

**Fast atom mapping for pairs of related molecules.** AtomMapper identifies a shared
core and reports which atoms and bonds are added, removed, or changed. It accepts SMILES
strings or RDKit molecules and returns an atom correspondence, a mapped reaction SMILES,
and a structured list of changes. No 3D coordinates are needed.

AtomMapper follows mapping constraints based on OpenFE's LOMAP: a connected core,
complete rings, and preserved stereochemistry by default. Its search engine is written
in Rust, with a Python API and command-line interface.

**In the included benchmarks, AtomMapper used 17–102× less total mapping time than
LOMAP's mapping step** across 3,529 molecular pairs. Most of the gain comes from avoiding
LOMAP's slowest searches; the median per-pair speedup was 1.1–2.3×. With the bond-maximizing
objective, 95–98% of mappings agreed with LOMAP up to symmetry, excluding LOMAP timeouts.
These measurements cover mapping, not similarity scoring or network planning.
See the [benchmark results and methodology](https://github.com/craabreu/atommapper/blob/main/benchmarks/README.md).

<p align="center">
  <img src="https://raw.githubusercontent.com/craabreu/atommapper/main/docs/images/example.png" width="820"
       alt="Two EG5 ligands with their shared core aligned and atoms and bonds highlighted as removed, added, or changed">
</p>

*Two EG5 ligands from the protein-ligand benchmark. The shared core is aligned; colors
show removed, added, and changed atoms and bonds. AtomMapper can generate these drawings
from either the command line or Python.*

## Installation

Requires **Python 3.11 or later**.

```bash
python -m pip install atommapper
```

Wheels are available for Linux (x86_64 and aarch64), macOS (Intel and Apple silicon), and
Windows (x64). RDKit, NetworkX, and SciPy are installed as dependencies. If pip builds from
source instead of installing a compatible wheel, you also need a
[Rust toolchain](https://rustup.rs). For an editable checkout, see [Development](#development).

## Quick start

### Python

```python
from atommapper import map_atoms

result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1")

print(result.status)            # complete
print(result.metadata["size"])  # 4: one atom and one bond removed, one of each added
print(result.reaction_smiles)

# Zero-based atom indices: source atom -> target atom
atom_map = {atom.source_index: atom.target_index for atom in result.mapping}
```

Each entry in `result.mapping` connects a source atom to a target atom. A mapped atom
can change element by default; use `element_change=False` to require matching elements.
Atoms outside the mapping are reported in `result.atom_changes`, and bond changes in
`result.bond_changes`.

You can also pass RDKit `Chem.Mol` objects directly. The input molecules are copied,
not modified, and existing atom-map numbers are ignored. Returned indices refer to the
input molecules; for SMILES, they follow RDKit's parsing order with explicit hydrogens
retained. Ordinary explicit hydrogens are then merged into their attached atoms for the
search. Pass `Chem.Mol` objects when you need indices tied to molecules already in your
workflow.

### Command line

```bash
atommapper "Cc1ccc(F)cc1" "Cc1cccc(F)c1"
```

```text
[CH3:1][c:6]1[cH:2][cH:4][c:7]([F:8])[cH:5][cH:3]1>>[CH3:1][c:6]1[cH:2][cH:4][cH:7][c:5]([F:9])[cH:3]1
```

The same atom-map number on both sides identifies a mapped pair. A number on only one
side identifies a removed or added atom: here, fluorine 8 is removed and fluorine 9 is
added. These **one-based identifiers are different from the zero-based input indices**.
The `source>>target` notation describes the correspondence; it does not predict a chemical
reaction or mechanism.

```bash
# Full result, including status, atom indices, changes, and search metadata
atommapper "CCO" "CC=O" --json

# Save a drawing as well as printing the mapped SMILES
atommapper "Cc1ccc(F)cc1" "Cc1cccc(F)c1" --draw walk.svg --numbers

atommapper --help
```

## Choosing a mapping

```text
map_atoms(source, target, *, element_change=True, objective="change",
          allow_stereo_inversion=False, settings=None)
```

| Option | Default behavior | Alternative |
|---|---|---|
| `objective` | `"change"`: minimize the size of the change | `"bonds"`: maximize kept bonds, then kept atoms, then minimize change |
| `element_change` | `True`: mapped atoms may change element | `False` / `--fixed-elements`: require matching elements |
| `allow_stereo_inversion` | `False`: reject mappings that invert a retained stereo element | `True` / `--allow-stereo-inversion`: allow inversions and count each as one change |
| `settings` | Default search budgets and time limit | A `Settings` instance; see [Search limits](#search-limits) |

The change size counts each inserted or deleted atom or bond, each element change, and
each bond-order change as one. Allowed stereo inversions also count as one. Other
attribute changes, such as charge and hydrogen count, are reported but do not add to this
objective. Consequently, `metadata["size"]` is **not the number of entries** in the change
lists, nor a free-energy or chemical-similarity score.

For example, pyridine and benzene can keep all six ring atoms with `element_change=True`.
With `element_change=False`, they have no valid shared core because preserving the five
carbons alone would leave an incomplete ring.

Use `objective="bonds"` (`--objective bonds`) for a bond-maximizing comparison with LOMAP.
AtomMapper applies constraints during its search and uses stricter stereo checks, so
identical mappings are not guaranteed. It does not provide LOMAP's hydrogen mapping,
similarity scores, or network planner.

## Drawings and notebooks

A `MappingResult` displays as an SVG drawing in notebooks. To save or render it explicitly:

```python
from atommapper import draw_png, draw_svg, map_atoms, save_image

result = map_atoms("Cc1ccc(F)cc1", "Cc1cccc(F)c1")
svg = draw_svg(result, width=400, height=300)  # SVG text
png = draw_png(result, numbers=True)          # PNG bytes
save_image(result, "walk.svg")               # .svg or .png selects the format
```

The source and target appear side by side with their shared atoms aligned. Each panel is
400 × 300 pixels by default, with a legend below. Set `width` and `height` in Python or
`--size 400x300` on the command line to change the panel size.

Removed atoms and bonds are vermillion, additions are blue, and changes are yellow.
Hydrogen-count changes are omitted from the highlights. Stereo inversions carry R/S or
E/Z labels; preserving double-bond geometry can require moving some mapped atoms out of
alignment. `numbers=True` (`--numbers`) labels mapped atoms with their identifiers.

## Working with results

`map_atoms` returns a `MappingResult`:

| Field | Meaning |
|---|---|
| `reaction_smiles` | Mapped `source>>target` SMILES; mapped atoms are numbered first, then source-only and target-only atoms |
| `mapping` | Tuple of `MappedAtom(identifier, source_index, target_index, tag)` records |
| `atom_changes`, `bond_changes` | Changes keyed by the identifiers in `reaction_smiles` |
| `status`, `reason` | Search outcome and an explanation when applicable |
| `metadata` | Change size, kept counts, optimality, stereo verdicts, checks, budgets, timings, settings, and dependency versions |

The mapping tag is `seed` or `refined`, indicating how an atom entered the search; it is
not a confidence score. `result.reverse()` swaps the two sides and renumbers the result
without searching again. `result.to_dict()` returns the public fields as plain Python
data for serialization.

Check `status` before using a result in an automated workflow:

| Status | Meaning |
|---|---|
| `complete` | The configured search finished; global optimality is not guaranteed |
| `timed_out` | A work budget, size ceiling, or time limit stopped the search; the result is the best found so far |
| `no_common_vertices` | No atom can be mapped under the constraints; the mapping is empty |
| `unsupported_input` | A valid input cannot be represented faithfully; `reason` explains why, and no mapped SMILES is returned |

```python
from atommapper import InvalidInput, Settings, map_atoms

try:
    result = map_atoms("CCO", "CC=O", settings=Settings(work_limit=10.0))
except InvalidInput as error:
    print(f"Invalid molecule: {error}")
else:
    if result.status == "unsupported_input":
        print(result.reason)
    elif result.status == "timed_out":
        print("Partial result:", result.metadata["budget"]["stopped_by"])
    else:
        print(result.reaction_smiles)
```

Invalid or empty molecules raise `InvalidInput` (a `ValueError` subclass). An unknown
objective raises `ValueError`; invalid `Settings` values raise `TypeError` or `ValueError`.
A charged hydrogen attached to a heavy atom is an example of an unsupported input.

The CLI exits with **0** for a completed search (including an empty mapping), **1** for
invalid or unsupported input, **2** for argument-parsing errors, and **3** for a search
stopped by a limit. A partial result is still printed; `--accept-partial` changes its exit
code to 0, but does not change the result's status.

## Mapping constraints

- **Connected core.** Mapped atoms joined by retained bonds form one connected component.
  Atoms isolated on both sides, such as counterions, are exempt.
- **Ring membership.** Ring atoms map only to ring atoms; chain atoms only to chain atoms.
- **Complete rings.** Every retained ring atom and ring bond must lie on a cycle in the
  shared core. A benzene ring can map to one ring of naphthalene, but cyclohexane and
  cyclopentane cannot share a partial ring.
- **Bonds between mapped atoms.** Two mapped atoms bonded on either side must also be
  bonded on the other side. Bond orders may change, even with fixed elements.
- **Stereochemistry.** By default, a mapping cannot invert a retained tetrahedral centre
  or double bond. The check uses the mapped neighbours, including attached hydrogens,
  rather than simply comparing R/S labels. Stereo can still be gained or lost and is
  reported in `metadata["stereo"]`.
- **Hydrogens.** Ordinary attached hydrogens are implicit; changes in their count are
  reported as atom attributes. This is intended for heavy-atom mapping, not a complete
  all-atom correspondence.

Ties favor fewer mapped atoms with changed heavy-atom neighbours, then the smaller
moving group, then canonical order. These choices are designed to give consistent
mappings when the inputs are reordered or reversed. All benchmark pairs agreed under
reversal up to symmetry; timeouts and search caps can affect that behavior.

## Search limits

```python
from atommapper import Settings, map_atoms

result = map_atoms(
    "Cc1ccc(F)cc1", "Cc1cccc(F)c1",
    settings=Settings(work_limit=30.0, node_budget=2_000_000),
)
```

| Setting | Default | Purpose | CLI option |
|---|---:|---|---|
| `fragment_steps` | 20,000,000 | Total fragment-search work budget | `--steps` |
| `node_budget` | 1,000,000 | Total branch-and-bound work budget | `--nodes` |
| `work_limit` | 60.0 | Limit on the work of a call, in virtual seconds (about one wall second of the reference machine; see below), counted without reading a clock | `--timeout` |
| `wall_guard_factor` | 10.0 | Wall-clock guard, as a multiple of `work_limit` | Python only |
| `fragment_bond_pairs` | 20,000 | Ceiling on compatible bond pairs in the fragment search | `--pairs` |
| `coarse_fragment_steps` | 100,000 | Work cap for the element-independent fragment pass | Python only |

The work budgets and the time limit make the stopping point reproducible. The time limit
is a virtual clock: each stage counts its work, and the clock advances by a fixed amount
per unit. The same calls therefore stop on every machine and under any load, with
`virtual` among the stop reasons, and a pair whose fragment-search set-up alone costs more
than the limit stops before the search. CPLEX's deterministic time limit and Gurobi's
`WorkLimit` are the same device.

The amounts per unit are a constant of the package, the reference clock
(`atommapper.clock.REFERENCE_RATES`): one virtual second is one second of the reference
machine, an AMD Ryzen Threadripper PRO 5975WX running eight calls at a time (with all 32
cores busy, the same calls took about 1.8 times as long). `work_limit` is the only knob.
Results made on two machines are comparable when it is the same; the name of the clock is
in `Settings().reference_clock` and in `metadata["versions"]["clock"]`, so results cached
by their settings do not mix if a later version fits the rates again.

The wall clock remains as a guard, at `wall_guard_factor` times the limit in real seconds:
a call it stops reports `time` and does depend on machine speed and load. Neither clock is
a hard process deadline: parsing, stereo perception, and result assembly are not
interrupted. The optional global check described below has its own additional budget.

To see what a virtual second is worth on your machine, calibrate:

```bash
atommapper calibrate --wall-seconds 10 --workers 1
```

or `atommapper.calibrate(10.0, workers=1)` in Python. It maps a benchmark shipped with
the package (300 calls on synthetic molecules, about two minutes on one core of the
reference machine; `benchmarks/make_calibration_pairs.py` builds it), reports the wall
seconds per virtual second (median, 5th and 95th percentiles, per stage too; on the
reference machine its slow calls give 1.02 with eight calls at a time and 0.97 with one)
and suggests the `work_limit` that takes the wall time you asked for. Run it with the
number of workers the real work will use. Results are comparable across machines only at the same
`work_limit`: a calibrated limit gives up exact comparability with results made at
another limit.

Reaching the coarse-pass cap is part of the normal method and does not set `timed_out`;
inspect `metadata["fragment_search"]["coarse_capped"]`. The bond-pair ceiling protects
against large search graphs; increase it if `stopped_by` includes `pairs`.

`Settings` also exposes `max_fragments`, `k_seeds`, and `max_alternatives` for advanced
control of candidate storage and refinement. Restricting seeds or truncating candidates
can affect mapping quality and consistency under input reversal.

## Optimality and limitations

**A completed search is not a proof of the globally best mapping.** AtomMapper starts
from shared connected fragments, trims them to the constraints, and grows mappings around
the remaining pieces using branch-and-bound. Each starting piece is held fixed, so the
usual result is `metadata["optimality"] == "optimal_given_seed"`.

The seeds can miss a better mapping, sometimes by a large margin. For example,
`NC1Cc2ccccc2CCC1=O` to `N[C@H]1CCc2ccccc2C[C@@H]1O` with fixed elements gives a change
size of 21, although a size of 8 is possible. Extra seeds address common stereo-related
failures, but do not make the search globally exact. Increasing budgets alone may not
fix a poor choice of seeds.

`Settings(global_budget_s=...)` enables an optional exact lower-bound calculation after
the search, using unconstrained graph edit distance. If it finishes and matches the
mapping cost, `optimality` becomes `optimal`. Otherwise, it does not establish optimality.
This check can be expensive and is intended for small molecules and validation.

## Development

Building from source requires Python 3.11+ and Rust:

```bash
git clone https://github.com/craabreu/atommapper.git
cd atommapper
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[test,dev]"

make check   # lint, types, Rust checks and tests, rebuild, Python tests
make format  # format Python and Rust sources
```

The `make` commands require Make. To run only the Python tests after installation, use
`python -m pytest -q`. Benchmark dependencies and reproduction commands are documented
[separately](https://github.com/craabreu/atommapper/blob/main/benchmarks/README.md#reproducing-the-benchmarks).

Please report bugs with a minimal pair of input molecules, the mapping options, and
`result.metadata["versions"]` on the
[issue tracker](https://github.com/craabreu/atommapper/issues).

## License

AtomMapper is released under the [MIT license](https://github.com/craabreu/atommapper/blob/main/LICENSE).
