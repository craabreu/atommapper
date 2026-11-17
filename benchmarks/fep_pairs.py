"""Pairs of ligands from the protein-ligand benchmark's free energy networks.

The benchmark (https://github.com/openforcefield/protein-ligand-benchmark, MIT license)
holds the ligand series that relative binding free energy methods are tested on, which is
what LOMAP maps. For each target this script downloads, at a pinned commit, the ligands
(``00_data/ligands.yml``) and the network OpenFE's LOMAP planner built on them
(``03_edges/lomap_mapper_lomap_network.yml``): its edges are the pairs a free energy study
would run. Each ligand is written as canonical SMILES without explicit hydrogens. With
``--all``, every unordered pair within a target is written instead.

Usage: python benchmarks/fep_pairs.py OUT.csv [--all]
"""

import argparse
import csv
import itertools
import sys
import urllib.request

import yaml
from rdkit import Chem

COMMIT = "fd88824f9114244f95a14b485e6d6c96c1de716d"
TARGETS = (
    "cdk2",
    "cdk8",
    "cmet",
    "eg5",
    "hif2a",
    "mcl1",
    "p38",
    "pde2",
    "pfkfb3",
    "ptp1b",
    "shp2",
    "syk",
    "thrombin",
    "tnks2",
    "tyk2",
)
URL = "https://raw.githubusercontent.com/openforcefield/protein-ligand-benchmark/{commit}/data/{target}/{path}"


def fetched(target: str, path: str) -> dict:
    url = URL.format(commit=COMMIT, target=target, path=path)
    with urllib.request.urlopen(url, timeout=60) as reply:
        return yaml.safe_load(reply.read())


def ligands(target: str) -> dict[str, str]:
    """Name to canonical SMILES, hydrogens implicit, for one target's series."""
    return {
        name: Chem.MolToSmiles(Chem.MolFromSmiles(entry["smiles"]))
        for name, entry in sorted(fetched(target, "00_data/ligands.yml").items())
    }


def network(target: str) -> list[tuple[str, str]]:
    """The edges of the LOMAP network on one target, in the file's order."""
    edges = fetched(target, "03_edges/lomap_mapper_lomap_network.yml")["edges"]
    return [(edge["ligand_a"], edge["ligand_b"]) for edge in edges.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--all", action="store_true", help="every pair within each target")
    args = parser.parse_args()
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "source_name", "target_name", "source", "target"])
        for target in TARGETS:
            series = ligands(target)
            pairs = itertools.combinations(sorted(series), 2) if args.all else network(target)
            count = 0
            for a, b in pairs:
                writer.writerow([target, a, b, series[a], series[b]])
                count += 1
            print(f"{target}: {len(series)} ligands, {count} pairs", file=sys.stderr)


if __name__ == "__main__":
    main()
