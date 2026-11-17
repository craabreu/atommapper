"""A fixed random sample of AstraZeneca matched molecular pairs.

The pairs are those of the AstraZeneca deposit on Zenodo (record 6319821), a CSV with
``Source_Mol`` and ``Target_Mol`` columns. The sample is drawn with a fixed seed, so the
same deposit gives the same pairs.

Usage: python benchmarks/az_pairs.py AZ_CSV OUT.csv [--n 3000] [--seed 20261106]
"""

import argparse
import csv
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("az_csv")
    parser.add_argument("out_csv")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    with open(args.az_csv, newline="") as handle:
        rows = [
            {"group": "az_mmp", "source": r["Source_Mol"], "target": r["Target_Mol"]}
            for r in csv.DictReader(handle)
        ]
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "source", "target"])
        writer.writeheader()
        writer.writerows(sample)


if __name__ == "__main__":
    main()
