# Benchmark: atommapper against LOMAP

`compare_lomap.py` maps the same pairs with OpenFE's LOMAP and with atommapper, one after
the other in the same worker, and judges every mapping with the same code: the atoms and
bonds it keeps, the size of the change, and which of LOMAP's rules it breaks.

## Methods compared

- **LOMAP**: `lomap.mcs.MCS` (LOMAP 3.3.0) with its defaults: 2D and a 20 s timeout on
  each MCS search. Its input is the molecule from SMILES with hydrogens added and 2D
  coordinates; its output is the heavy-atom map. Scoring is not run.
- **atommapper (change)**: `map_atoms` with its default objective, the smallest change.
- **atommapper (bonds)**: `map_atoms(objective="bonds")`, LOMAP's objective: the most
  bonds kept, then the most atoms, then the smallest change.

Each method runs with elements free (LOMAP's default `element_change=True`) and with
elements fixed. Times are wall clock for the whole call from the SMILES strings.

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
workers, Python 3.12, RDKit 2026.03.6, LOMAP 3.3.0 and atommapper 0.1.0. No method failed
on any pair. atommapper (bonds) times the same as atommapper (change) to within a few
milliseconds, so only the latter is shown.

### Time per pair

| | | LOMAP | atommapper | speed-up |
|---|---|---|---|---|
| FEP networks, elements free | median | 16 ms | 14 ms | 1.1× |
| | mean | 366 ms | 19 ms | 19× |
| | 99th percentile | 6.6 s | 63 ms | |
| | slowest | 20.2 s | 209 ms | |
| FEP networks, elements fixed | median | 17 ms | 8 ms | 2.1× |
| | mean | 952 ms | 9 ms | 106× |
| | 99th percentile | 20.3 s | 23 ms | |
| | slowest | 22.0 s | 36 ms | |
| AZ pairs, elements free | median | 19 ms | 15 ms | 1.3× |
| | mean | 530 ms | 31 ms | 17× |
| | 99th percentile | 19.0 s | 244 ms | |
| | slowest | 21.6 s | 3.2 s | |
| AZ pairs, elements fixed | median | 19 ms | 8 ms | 2.4× |
| | mean | 699 ms | 11 ms | 64× |
| | 99th percentile | 20.5 s | 50 ms | |
| | slowest | 21.9 s | 1.6 s | |

On a typical pair the two take about the same time. LOMAP's mean is set by its tail:
its MCS search reached the 20 s timeout on 4 and 14 of the FEP pairs (elements free and
fixed) and on 24 and 41 of the AZ pairs, and then goes on with the MCS found so far.

### Mappings

| | FEP, free | FEP, fixed | AZ, free | AZ, fixed |
|---|---|---|---|---|
| same mapping as LOMAP, up to symmetry (bonds) | 97.7% | 95.8% | 95.0% | 95.9% |
| more bonds kept than LOMAP (bonds) | 0 | 2 | 17 | 12 |
| fewer bonds kept than LOMAP (bonds) | 0 | 0 | 7 | 8 |
| smaller change than LOMAP (change) | 1 | 7 | 74 | 45 |
| larger change than LOMAP (change) | 0 | 0 | 7 | 8 |
| LOMAP's mapping inverts a stereo element | 0 | 0 | 15 | 14 |

- **Where atommapper keeps more bonds**, LOMAP's MCS search had mostly timed out (2 of 2
  FEP pairs with elements fixed, 18 of the 29 AZ pairs).
- **Where atommapper keeps fewer bonds** (7 and 8 AZ pairs), LOMAP's mapping keeps a
  stereo element inverted every time, as atommapper judges it. LOMAP's own check compares a
  centre's neighbours within its heavy-atom match only, so it can miss an inversion, for
  instance one that the centre's hydrogen decides (README, "The rules"). Among the mappings
  that satisfy the rules, atommapper never keeps fewer bonds than LOMAP.
- **Where atommapper makes a smaller change**, it mostly keeps the same bonds as LOMAP
  (1 of 1, 5 of 7, 57 of 74 and 33 of 45 pairs): LOMAP chose among mappings that keep as
  many bonds without looking at the change. In the other pairs it keeps more bonds too.
- No mapping by either tool breaks the other rules: one piece, complete rings, no bond
  lost between kept atoms.

## Running it

```bash
pip install lomap2 pyyaml           # besides atommapper
python benchmarks/fep_pairs.py fep.csv
python benchmarks/compare_lomap.py run fep.csv fep.jsonl --workers 32
python benchmarks/compare_lomap.py summary fep.jsonl
```

The summary prints the tables above for one or more record files.
