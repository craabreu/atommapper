"""The calibration of the virtual clock against this machine, and the benchmark it maps."""

import importlib.util
from importlib import resources
from pathlib import Path

import pytest
from rdkit import Chem

from atommapper import calibrate, calibration
from atommapper.calibration import GUIDANCE, REFERENCE, benchmark
from atommapper.cli import main
from atommapper.clock import CLOCK_ID, STAGES

# Two calls of a few milliseconds and one of a tenth of a virtual second, in refinement.
LONGER = (
    "CO[C@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
    "CO[C@@H](CN(c1ccc(Oc2ccc(C)cc2)cc1)S(C)(=O)=O)C(=O)NO",
    False,
)
CALLS = [("CCO", "CC=O", False), ("Cc1ccc(F)cc1", "Cc1cccc(F)c1", True), LONGER]


GENERATOR = Path(__file__).resolve().parent.parent / "benchmarks" / "make_calibration_pairs.py"


def packaged() -> str:
    return resources.files("atommapper").joinpath("calibration_pairs.tsv").read_text()


def test_the_benchmark_is_packaged_and_every_molecule_of_it_is_valid():
    assert resources.files("atommapper").joinpath("calibration_pairs.tsv").is_file()
    calls = benchmark()
    assert len(calls) == 300 == len(set(calls))
    assert {allow for _, _, allow in calls} == {False, True}
    sizes, molecules = [], set()
    for source, target, _ in calls:
        mols = [Chem.MolFromSmiles(smiles) for smiles in (source, target)]
        assert all(mol is not None for mol in mols)  # parsed and sanitized
        canonical = [Chem.MolToSmiles(mol) for mol in mols]
        assert canonical[0] != canonical[1]
        molecules.update(canonical)
        sizes.append(sum(mol.GetNumHeavyAtoms() for mol in mols))
    # Many molecules, not a few against each other; small pairs first.
    assert len(molecules) > 500
    assert sizes == sorted(sizes)
    # Fifty calls in each band of the pair's heavy atoms.
    bands = [sum(size > top for top in (50, 70, 90, 120, 200)) for size in sizes]
    assert [bands.count(band) for band in range(6)] == [50] * 6


def test_the_benchmark_is_synthetic_and_says_where_it_comes_from():
    header = [line for line in packaged().splitlines() if line.startswith("#")]
    assert any("Synthetic" in line for line in header)
    assert any("benchmarks/make_calibration_pairs.py" in line for line in header)


@pytest.mark.skipif(not GENERATOR.is_file(), reason="the generator is not in an installed copy")
def test_the_generator_writes_the_packaged_benchmark_again():
    spec = importlib.util.spec_from_file_location("make_calibration_pairs", GENERATOR)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    # Byte for byte: the molecules are text the script writes, not RDKit's canonical SMILES.
    assert generator.build() == packaged()


def test_calibrate_compares_wall_and_virtual_time_and_suggests_a_limit():
    report = calibrate(4.0, calls=CALLS)
    assert (report.calls, report.workers, report.clock) == (3, 1, CLOCK_ID)
    assert report.wall_s > 0 and report.virtual_s > 0.1
    # No call of half a virtual second: no ratio over slow calls, and one stage long
    # enough for a ratio of its own.
    assert set(report.ratios) == {"all calls", "refinement"}
    assert report.ratios["all calls"].calls == 3 and report.ratios["refinement"].calls == 1
    for ratio in report.ratios.values():
        assert 0 < ratio.p5 <= ratio.median <= ratio.p95
    # Fewer than five slow calls: the factor is the median over all of them.
    assert report.factor == report.ratios["all calls"].median
    assert report.work_limit == pytest.approx(4.0 / report.factor)
    text = str(report)
    assert "work_limit = " in text and GUIDANCE in text and CLOCK_ID in text
    assert REFERENCE in text
    assert "For about 4 s of this machine per call" in text


def test_the_slow_calls_set_the_factor_when_there_are_enough(monkeypatch):
    monkeypatch.setattr(calibration, "SLOW_S", 0.0)
    report = calibrate(calls=[*CALLS, *CALLS])
    assert report.ratios["slow calls"].calls == 6
    assert report.factor == report.ratios["slow calls"].median
    assert set(report.ratios) >= {"all calls", "slow calls", "refinement"}
    assert set(report.ratios) <= {"all calls", "slow calls", *STAGES}


def test_the_virtual_time_of_a_calibration_does_not_depend_on_the_workers():
    assert calibrate(calls=CALLS, workers=2).virtual_s == calibrate(calls=CALLS).virtual_s


def test_calibrate_rejects_what_it_cannot_measure():
    with pytest.raises(ValueError, match="wall_seconds"):
        calibrate(0.0, calls=CALLS)
    with pytest.raises(ValueError, match="workers"):
        calibrate(calls=CALLS, workers=0)
    with pytest.raises(ValueError, match="no call"):
        calibrate(calls=[])


def test_the_calibrate_command_prints_the_report(capsys):
    assert main(["calibrate", "--calls", "3", "--wall-seconds", "5"]) == 0
    out = capsys.readouterr().out
    assert "3 calls, 1 at a time" in out
    assert "For about 5 s of this machine per call" in out
    assert out.rstrip().endswith(GUIDANCE)
    with pytest.raises(SystemExit) as error:
        main(["calibrate", "--wall-seconds", "-1", "--calls", "1"])
    assert error.value.code == 2
    # The mapping command is as before.
    assert main(["CCO", "CC=O"]) == 0
