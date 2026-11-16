import json

import pytest

from atommapper.adapter import parse_reaction
from atommapper.cli import main


def test_prints_the_reaction_smiles(capsys):
    assert main(["CCO", "CC=O"]) == 0
    out = capsys.readouterr().out.strip()
    assert ">>" in out
    parse_reaction(out)


def test_json_output(capsys):
    assert main(["CCO", "CC=O", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["metadata"]["objective_name"] == "change"


def test_invalid_input_exits_one(capsys):
    assert main(["C1CC", "CC"]) == 1
    assert "invalid SMILES" in capsys.readouterr().err


def test_unsupported_input_exits_one(capsys):
    assert main(["C[H+]", "C"]) == 1
    assert "unsupported" in capsys.readouterr().err


def test_a_timeout_exits_three_unless_partial_results_are_accepted(capsys):
    assert main(["Cc1ccc(F)cc1", "Cc1cccc(F)c1", "--timeout", "0"]) == 3
    assert main(["Cc1ccc(F)cc1", "Cc1cccc(F)c1", "--timeout", "0", "--accept-partial"]) == 0


def test_an_exhausted_step_budget_exits_three(capsys):
    assert main(["Cc1ccc(F)cc1", "Cc1cccc(F)c1", "--steps", "0"]) == 3
    assert main(["Cc1ccc(F)cc1", "Cc1cccc(F)c1", "--steps", "0", "--accept-partial"]) == 0


def test_no_common_vertices_is_a_normal_result(capsys):
    assert main(["C1CCCCC1", "CCCCCC", "--fixed-elements"]) == 0


def test_fixed_elements_keep_every_element(capsys):
    assert main(["c1ccncc1", "c1ccccc1", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["mapping"]) == 6
    assert main(["c1ccncc1", "c1ccccc1", "--fixed-elements", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["mapping"] == []


def test_stereo_inversion_can_be_allowed(capsys):
    assert main(["C[C@H](N)O", "C[C@@H](N)O", "--fixed-elements", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["mapping"]) == 2
    args = ["C[C@H](N)O", "C[C@@H](N)O", "--fixed-elements", "--allow-stereo-inversion"]
    assert main([*args, "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["mapping"]) == 4


def test_the_bond_pair_ceiling_flag(capsys):
    assert main(["CCCCCC", "CCCCCC", "--pairs", "10"]) == 3
    assert main(["CCCCCC", "CCCCCC", "--pairs", "10000"]) == 0


def test_an_unknown_objective_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["CC", "CC", "--objective", "atoms"])
    assert exit_info.value.code == 2
