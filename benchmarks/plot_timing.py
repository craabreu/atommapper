"""The time per pair of LOMAP and AtomMapper, as the fraction of pairs that take at least a
given time, on log-log axes, so that the slow tail shows.

Usage: python benchmarks/plot_timing.py FEP.jsonl[,FEP_REVERSE.jsonl] AZ.jsonl[,...] OUT_STEM
Writes OUT_STEM.png from the records of ``compare_lomap.py run``; several record files of
one dataset (a forward and a reverse run, say) are given separated by commas.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "plotting"))
from publication_plotting import (
    label_panels,
    save_publication_figure,
    use_publication_style,
)

LOMAP_TIMEOUT = 20.0
SERIES = (
    # (record key, legend, color, line style)
    ("change_free", "AtomMapper, elements free", "#0072B2", "-"),
    ("change_fixed", "AtomMapper, elements fixed", "#0072B2", "--"),
    ("lomap_free", "LOMAP, elements free", "#D55E00", "-"),
    ("lomap_fixed", "LOMAP, elements fixed", "#D55E00", "--"),
)


def records(paths: str) -> list[dict]:
    """The records of every file in the comma-separated ``paths``."""
    out = []
    for path in paths.split(","):
        with open(path) as handle:
            out += [json.loads(line) for line in handle]
    return out


def at_least(times: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Each time, sorted, and the fraction of pairs that take at least that long; drawn with
    ``where="pre"``, the curve holds that fraction up to the time and steps down after it."""
    xs = np.sort(np.asarray(times))
    return xs, 1.0 - np.arange(xs.size) / xs.size


def main() -> None:
    fep_path, az_path, stem = sys.argv[1:4]
    use_publication_style(preset=None, figsize=(7.0, 3.0), box=False)
    fig, axes = plt.subplots(1, 2, sharey=True)
    datasets = (("FEP networks", records(fep_path)), ("AZ matched pairs", records(az_path)))
    for ax, (title, rows) in zip(axes, datasets, strict=True):
        pairs = len({r["i"] for r in rows})
        runs = f", {len(rows) // pairs} directions" if len(rows) > pairs else ""
        for key, label, color, style in SERIES:
            xs, ys = at_least([r[key]["t"] for r in rows])
            ax.step(xs, ys, where="pre", color=color, linestyle=style, label=label)
        ax.axvline(LOMAP_TIMEOUT, color="0.55", linewidth=0.8, linestyle=":")
        ax.text(
            LOMAP_TIMEOUT / 1.15,
            0.5,
            "LOMAP's MCS timeout",
            rotation=90,
            ha="right",
            va="center",
            color="0.4",
            fontsize=7,
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(1e-3, 40.0),
            ylim=(1.0 / len(rows) / 1.5, 1.4),
            xlabel="Time per pair (s)",
            title=f"{title} ({pairs:,} pairs{runs})",
        )
    axes[0].set_ylabel("Fraction of pairs taking at least this long")
    # Outside the axes, in one row under both panels, so that it covers no curve.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    label_panels(axes)
    save_publication_figure(fig, stem, formats=("png",), png_dpi=200, close=True)


if __name__ == "__main__":
    main()
