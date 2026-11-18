"""The work limit stops the same calls, with the same results, on every machine.

Twenty calls of the packaged benchmark, across its size bands, are mapped at two work
limits with the wall-clock guard off, and what decides or reports the outcome (status,
mapping, stop reasons, steps, nodes, work and virtual seconds) is hashed. The hashes were
made on the reference machine (x86_64 Linux) and matched on an Intel one; continuous
integration checks them on the other platforms the wheels are built for. They depend on
RDKit, whose perception the search starts from, so the test runs only on the RDKit version
they were made with; a change of the mapper that changes any outcome changes them too, and
they are then made again with ``python tests/test_determinism.py``.
"""

import hashlib
import json
import math

import pytest
import rdkit

from atommapper import Settings, map_atoms
from atommapper.calibration import benchmark

RDKIT_VERSION = "2026.09.2"
CALLS = range(0, 300, 15)
EXPECTED = {
    0.02: "cbcda41a539345189fbf5cfebe4539ea20625566330d7f8755b52cfa610a5bad",
    0.2: "bf60e812e780c4f9f0d95d85633464702b7c1b925138d5f613294c336238e493",
}


def outcomes(limit: float) -> list[dict]:
    calls = benchmark()
    settings = Settings(work_limit=limit, wall_guard_factor=math.inf)
    found = []
    for index in CALLS:
        source, target, allow = calls[index]
        result = map_atoms(source, target, allow_stereo_inversion=allow, settings=settings)
        meta = result.metadata
        found.append(
            {
                "call": index,
                "status": result.status,
                "reaction_smiles": result.reaction_smiles,
                "mapping": [(m.source_index, m.target_index) for m in result.mapping],
                "budget": meta["budget"],
                "fragment_search": {
                    k: meta["fragment_search"][k] for k in ("stopped_by", "steps", "fragments")
                },
                "kept": meta["kept"],
            }
        )
    return found


def digest(limit: float) -> str:
    text = json.dumps(outcomes(limit), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.skipif(
    rdkit.__version__ != RDKIT_VERSION, reason=f"the hashes were made with RDKit {RDKIT_VERSION}"
)
@pytest.mark.parametrize("limit", sorted(EXPECTED))
def test_the_work_limit_gives_the_same_results_on_every_machine(limit):
    assert digest(limit) == EXPECTED[limit]


if __name__ == "__main__":
    for limit in sorted(EXPECTED):
        print(f"    {limit}: {digest(limit)!r},")
