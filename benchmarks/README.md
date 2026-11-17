# Benchmark: AtomMapper against LOMAP

`compare_lomap.py` maps the same pairs with OpenFE's LOMAP and with AtomMapper, one after
the other in the same worker, and judges every mapping with the same code: the atoms and
bonds it keeps, the size of the change, and which of LOMAP's rules it breaks.

## Methods compared

- **LOMAP**: `lomap.mcs.MCS` (LOMAP 3.3.0) with its defaults: 2D and a 20 s timeout on
  each MCS search. Its input is the molecule from SMILES with hydrogens added and 2D
  coordinates; its output is the heavy-atom map. Scoring is not run.
- **AtomMapper (change)**: `map_atoms` with its default objective, the smallest change.
- **AtomMapper (bonds)**: `map_atoms(objective="bonds")`, LOMAP's objective: the most
  bonds kept, then the most atoms, then the smallest change.

Each method runs with elements free (LOMAP's default `element_change=True`) and with
elements fixed, and every pair is mapped in both directions, source to target and target to
source, since LOMAP's answer can depend on which molecule comes first. Times are wall clock
for the whole call from the SMILES strings.

## Pairs

- **FEP networks** (529 pairs): the edges of the networks OpenFE's LOMAP planner built on
  the 15 targets of the [protein-ligand benchmark](https://github.com/openforcefield/protein-ligand-benchmark)
  (MIT license), the pairs a relative binding free energy study would run.
  `fep_pairs.py OUT.csv` writes them; `--all` writes every pair within each target
  instead (5,084 pairs).
- **AZ matched pairs** (3,000 pairs): a fixed random sample of the AstraZeneca matched
  molecular pairs on Zenodo (record 6319821), more varied than a congeneric series.
  `az_pairs.py AZ_CSV OUT.csv` draws it from the deposit.

## Results

Run on 2026-11-16 on an AMD Ryzen Threadripper PRO 5975WX (32 cores, 64 threads) with 32
workers, Python 3.12, RDKit 2026.03.6, LOMAP 3.3.0 and AtomMapper 0.1.0. No method failed
on any pair. Each result below counts both directions, so each pair counts twice.
AtomMapper (bonds) times the same as AtomMapper (change) to within a few milliseconds, so
the figure shows only the latter.

### Time per pair

![Fraction of pairs taking at least a given time, log-log: AtomMapper's curves fall off below a second, LOMAP's run on to its 20 s timeout](timing.png)

AtomMapper is 1.1-2.4 times faster at the median and takes at most 3.8 s on any pair.
LOMAP's slow tail comes from its MCS search: of the 1,058 FEP and 6,000 AZ mappings in each
element mode, it reached the 20 s timeout on 9 and 27 FEP ones (elements free and fixed)
and on 55 and 80 AZ ones, and LOMAP then goes on with the MCS found so far.

### Mappings

The mappings are compared only where LOMAP's MCS search finished. When it times out, LOMAP
goes on with the MCS found by then, which depends on the machine's speed and load. Rerun on
the 83 cases that timed out in one run of the benchmark, with the timeout at 5 s instead of
20 s (a machine about four times slower) LOMAP's mapping changed in 1 case, and at 60 s
(about three times faster) in 8; from one run to the next on the same machine, one AZ pair
close to 20 s timed out in one run and not in the other. AtomMapper finished every pair
within its work budgets and its default 60 s limit, so its results do not depend on the
machine.

| | FEP, free | FEP, fixed | AZ, free | AZ, fixed |
|---|---|---|---|---|
| mappings compared (both directions) | 1,049 | 1,031 | 5,945 | 5,920 |
| same mapping as LOMAP, up to symmetry (bonds) | 98.1% | 95.9% | 95.3% | 96.2% |
| more bonds kept than LOMAP (bonds) | 0 | 1 | 20 | 2 |
| fewer bonds kept than LOMAP (bonds) | 0 | 0 | 14 | 15 |
| smaller change than LOMAP (change) | 2 | 11 | 132 | 68 |
| larger change than LOMAP (change) | 0 | 0 | 14 | 15 |
| LOMAP's mapping inverts a stereo element | 0 | 0 | 30 | 27 |

- **Where AtomMapper keeps more bonds** (23 mappings), LOMAP lost bonds when it trimmed
  the MCS it had chosen to its rules: on each of them RDKit's MCS has at least as many bonds
  as AtomMapper's mapping, and LOMAP's trimmed mapping fewer. AtomMapper applies the rules
  during its search.
- **Where AtomMapper keeps fewer bonds** (14 and 15 AZ mappings), LOMAP's mapping keeps a
  stereo element inverted every time, as AtomMapper judges it. LOMAP's own check compares a
  centre's neighbours within its heavy-atom match only, so it can miss an inversion, for
  instance one that the centre's hydrogen decides (README, "The rules"). Among the mappings
  that satisfy the rules, AtomMapper never keeps fewer bonds than LOMAP.
- **Where AtomMapper makes a smaller change**, it mostly keeps the same bonds as LOMAP
  (2 of 2, 10 of 11, 112 of 132 and 66 of 68 mappings): LOMAP chose among mappings that
  keep as many bonds without looking at the change. In the others it keeps more bonds too.
- No mapping by either tool breaks the other rules: one piece, complete rings, no bond
  lost between kept atoms.

### Symmetry

Mapping B to A should give the reverse of mapping A to B. On the pairs where no run was
stopped by the clock, the two directions give:

| | LOMAP: different bonds kept | LOMAP: different mapping | AtomMapper: either |
|---|---|---|---|
| FEP, elements free | 0 of 524 | 4 (0.8%) | 0 of 529 |
| FEP, elements fixed | 1 of 515 | 6 (1.2%) | 0 of 529 |
| AZ, elements free | 0 of 2,969 | 13 (0.4%) | 0 of 3,000 |
| AZ, elements fixed | 1 of 2,957 | 22 (0.7%) | 0 of 3,000 |

AtomMapper returns the reverse mapping, up to symmetry, on every pair, with either
objective. Where LOMAP's two directions differ, they mostly keep the same bonds and choose
another of the tied mappings; on two pairs one direction keeps fewer bonds.

## Running it

```bash
pip install lomap2 pyyaml           # besides atommapper
python benchmarks/fep_pairs.py fep.csv
python benchmarks/compare_lomap.py run fep.csv fep.jsonl --workers 32
python benchmarks/compare_lomap.py run fep.csv fep_reverse.jsonl --workers 32 --reverse
python benchmarks/compare_lomap.py summary fep.jsonl fep_reverse.jsonl
python benchmarks/compare_lomap.py symmetry fep.jsonl fep_reverse.jsonl
```

`summary` prints the comparison for one or more record files and `symmetry` the table
above; `python benchmarks/plot_timing.py fep.jsonl,fep_reverse.jsonl az.jsonl,az_reverse.jsonl
benchmarks/timing` draws the figure (it needs Matplotlib).
