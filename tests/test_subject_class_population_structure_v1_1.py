"""Regression and immutable-contract tests for the V1.1 technical amendment."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.subject_class_population_structure_v1 as v1
from src.interaction_provenance_v0 import sha256_file
from src.subject_class_population_structure_v1_1 import (
    bnci_report_sentence,
    load_amendment_config,
    output_path,
    run_nonvoting_control,
    run_synthetic_recovery_gates,
    run_voting_component,
    scientific_equivalence_rows,
    terminal_next_question,
    validate_final_outputs,
    validate_immutable_v1_history,
)


ROOT = Path(__file__).resolve().parents[1]


def _raise(error: BaseException) -> None:
    raise error


def test_v1_history_is_immutable_and_v1_1_namespace_is_separate() -> None:
    amendment, _, _, _ = load_amendment_config(ROOT)
    observed = validate_immutable_v1_history(ROOT, amendment)
    assert observed == {
        name: record["sha256"]
        for name, record in amendment["immutable_v1_artifacts"].items()
    }
    assert output_path(ROOT, amendment).name == "subject_class_population_structure_v1_1"
    assert output_path(ROOT, amendment) != ROOT / "outputs/subject_class_population_structure_v1"


def test_v1_v1_1_scientific_contract_has_only_orchestration_change() -> None:
    amendment, _, scientific, _ = load_amendment_config(ROOT)
    rows = scientific_equivalence_rows(scientific, amendment)
    changed = [row for row in rows if row["changed"]]
    assert changed == [rows[-1]]
    assert changed[0]["field"] == "secondary failure isolation"
    assert all(not row["changed"] for row in rows if row["change_class"] == "SCIENTIFIC")


def test_component_roles_are_explicit_and_primary_roles_vote() -> None:
    amendment, _, _, _ = load_amendment_config(ROOT)
    roles = amendment["component_roles"]
    assert roles["reliability"] == "VOTING_REQUIRED"
    assert roles["openbmi_sensor_primary"] == "VOTING_REQUIRED"
    assert roles["subject_pairing_null"] == "VOTING_REQUIRED"
    assert roles["class_semantics_null"] == "VOTING_REQUIRED"
    assert roles["equal_rank_random_subspace"] == "VOTING_REQUIRED"
    assert roles["full_space_sensor_baseline"] == "VOTING_REQUIRED"
    assert roles["ordered_z_eigenspectrum"] == "NON_VOTING_OPTIONAL_DIAGNOSTIC"
    assert roles["bnci_multiclass"] == "NON_VOTING_OPTIONAL_DIAGNOSTIC"


def test_case_a_primary_and_nulls_survive_ordered_spectrum_degeneracy() -> None:
    primary = run_voting_component("primary", lambda: {"statistic": 1.0})
    nulls = run_voting_component("nulls", lambda: {"pairing_p": 0.01})
    spectrum = run_nonvoting_control(
        "ordered_z_eigenspectrum",
        lambda: _raise(v1.NumericalContractError("projected training-score scale is degenerate")),
    )
    terminal = v1.terminal_decision(
        data_contract_pass=True, reliability_pass=True, statistic=1.0,
        forward_median=1.0, reverse_median=1.0, pairing_p=0.01,
        class_p=0.01, random_p=0.01, influence_positive=True,
        full_space_stable=True, selected_ranks=[2] * 6, low_cap=8,
    )
    assert primary["status"] == "VOTING_COMPLETED"
    assert nulls["status"] == "VOTING_COMPLETED"
    assert spectrum["status"] == "CONTROL_UNASSESSED_NUMERICAL_DEGENERACY"
    assert terminal == "GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION"


def test_case_b_primary_sensor_degeneracy_remains_fail_closed() -> None:
    primary = run_voting_component(
        "openbmi_sensor_primary",
        lambda: _raise(v1.NumericalContractError("primary scale degeneracy")),
    )
    assert primary["status"] == "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE"


def test_case_c_generalized_eigen_failure_is_nonvoting() -> None:
    primary = run_voting_component("primary", lambda: 1)
    diagnostic = run_nonvoting_control(
        "generalized_eigen_signature",
        lambda: _raise(v1.DataContractError("generalized eigen contract")),
    )
    assert primary["status"] == "VOTING_COMPLETED"
    assert diagnostic["status"] == "CONTROL_UNASSESSED_DATA_CONTRACT_FAILURE"


def test_case_d_bnci_failure_retains_existing_openbmi_terminal() -> None:
    openbmi_terminal = "STOP_RANDOM_SUBSPACE_EQUIVALENT"
    diagnostic = run_nonvoting_control(
        "bnci_multiclass", lambda: _raise(RuntimeError("BNCI synthetic failure"))
    )
    assert openbmi_terminal == "STOP_RANDOM_SUBSPACE_EQUIVALENT"
    assert diagnostic["status"] == "CONTROL_UNASSESSED_EXECUTION_FAILURE"
    sentence = bnci_report_sentence({
        "executed": False, "status": diagnostic["status"],
        "message": diagnostic["message"],
        "openbmi_terminal_retained": openbmi_terminal,
    })
    assert "explicitly unassessed" in sentence
    assert openbmi_terminal in sentence


def test_all_four_synthetic_recovery_gates_pass() -> None:
    result = run_synthetic_recovery_gates(ROOT)
    assert result["passed"]
    assert len(result["cases"]) == 4


@pytest.mark.parametrize(
    ("terminal", "fragment"),
    [
        ("GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION", "unseen subject's coordinates"),
        ("STOP_NO_HELDOUT_POPULATION_STRUCTURE", "additional supervision or physiological anchor"),
        ("UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE", "hypothesis remains unassessed"),
    ],
)
def test_report_next_question_is_terminal_aware(terminal: str, fragment: str) -> None:
    assert fragment in terminal_next_question(terminal)


def test_historical_v1_report_still_has_original_hash() -> None:
    amendment, _, _, _ = load_amendment_config(ROOT)
    record = amendment["immutable_v1_artifacts"]["report"]
    assert sha256_file(ROOT / record["path"]) == record["sha256"]


def test_final_v1_1_report_consistency_when_present() -> None:
    report = ROOT / "outputs/subject_class_population_structure_v1_1/report/subject_class_population_structure_v1_1.md"
    if report.is_file():
        validate_final_outputs(ROOT)
    else:
        assert not report.exists()
