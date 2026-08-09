"""Synthetic contract tests for the frozen V2 summary, figures, and report."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_v2 import load_config
from src.reporting_v2 import (
    CLASS_ORDER,
    FIGURE_STEMS,
    GEOMETRIES,
    MDM_SPECS,
    PREREQUISITE_FILENAMES,
    REPORT_HEADINGS,
    REPORT_TITLE,
    ReportingContractError,
    compute_frozen_verdicts,
    create_reporting_outputs,
    validate_reporting_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA = "c" * 64
PROTOCOL_SHA = "p" * 64


def _config() -> dict[str, object]:
    config = copy.deepcopy(
        load_config(ROOT / "configs" / "bnci2014_001_geometry_v2.yaml")
    )
    config["protocol"]["protocol_sha256"] = PROTOCOL_SHA
    return config


def _metric_fields(score: float, evaluation_n: int) -> dict[str, object]:
    per_class = evaluation_n // 4
    result: dict[str, object] = {
        "balanced_accuracy": float(score),
        "accuracy": float(score),
        "macro_f1": float(score),
        "confusion_matrix_json": "[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]",
    }
    for label in CLASS_ORDER:
        result[f"recall_{label}"] = float(score)
    for truth in CLASS_ORDER:
        for prediction in CLASS_ORDER:
            result[f"confusion_{truth}__{prediction}"] = (
                per_class if truth == prediction else 0
            )
    return result


def _result_row(
    subject: int,
    geometry: str,
    native_metric: str,
    protocol: str,
    split: str,
    decoder: str,
    score: float,
) -> dict[str, object]:
    evaluation_n = 288 if split in {"ALL", "AGGREGATE"} else 144
    return {
        "protocol_version": "2.0",
        "config_sha256": CONFIG_SHA,
        "seed": 20260809,
        "subject": subject,
        "target_subject": subject,
        "geometry": geometry,
        "protocol": protocol,
        "split": split,
        "decoder": decoder,
        "native_metric": native_metric,
        "source_n": 2304,
        "evaluation_n": evaluation_n,
        "transductive_overlap": protocol == "T1" and geometry != "RAW",
        "status": "PASS",
        "convergence_warning": False,
        "warning_messages": "[]",
        **_metric_fields(score, evaluation_n),
    }


def _primary_scores() -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    t1: dict[int, dict[str, float]] = {}
    t2: dict[int, dict[str, float]] = {}
    for subject in range(1, 10):
        raw = 0.43 + subject * 0.006
        t1[subject] = {
            "RAW": raw,
            "LE": raw + 0.020,
            "AIRM": raw + 0.018,
            "EA": raw + 0.005,
        }
        t2[subject] = {
            "RAW": raw - 0.002,
            "LE": raw + 0.015,
            "AIRM": raw + 0.013,
            "EA": raw + 0.003,
        }
    return t1, t2


def _loso_tables() -> dict[str, pd.DataFrame]:
    t1_scores, t2_scores = _primary_scores()
    logistic_t1: list[dict[str, object]] = []
    logistic_t2: list[dict[str, object]] = []
    mdm_t1: list[dict[str, object]] = []
    mdm_t2: list[dict[str, object]] = []
    for subject in range(1, 10):
        for geometry in GEOMETRIES:
            logistic_t1.append(
                _result_row(
                    subject,
                    geometry,
                    "euclidean_log_svec",
                    "T1",
                    "ALL",
                    "logistic",
                    t1_scores[subject][geometry],
                )
            )
            for split in ("A", "B", "AGGREGATE"):
                logistic_t2.append(
                    _result_row(
                        subject,
                        geometry,
                        "euclidean_log_svec",
                        "T2",
                        split,
                        "logistic",
                        t2_scores[subject][geometry],
                    )
                )
        for spec_index, (geometry, metric) in enumerate(MDM_SPECS):
            score_t1 = 0.36 + 0.01 * spec_index + 0.002 * subject
            score_t2 = score_t1 - 0.004
            mdm_t1.append(
                _result_row(subject, geometry, metric, "T1", "ALL", "MDM", score_t1)
            )
            for split in ("A", "B", "AGGREGATE"):
                mdm_t2.append(
                    _result_row(subject, geometry, metric, "T2", split, "MDM", score_t2)
                )
    return {
        "loso_logistic_transductive.csv": pd.DataFrame(logistic_t1),
        "loso_logistic_calibration.csv": pd.DataFrame(logistic_t2),
        "loso_mdm_transductive.csv": pd.DataFrame(mdm_t1),
        "loso_mdm_calibration.csv": pd.DataFrame(mdm_t2),
    }


def _mean_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in range(1, 10):
        for protocol, splits in (("T1", ("ALL",)), ("T2", ("A", "B"))):
            for split in splits:
                rows.append(
                    {
                        "subject": subject,
                        "protocol": protocol,
                        "split": split,
                        "fit_scope": f"synthetic_{protocol}_{split}",
                        "fit_n": 288 if protocol == "T1" else 144,
                        "normalized_d_le_le_airm": 0.020 + subject * 0.001,
                        "normalized_d_airm_le_airm": 0.024 + subject * 0.001,
                        "le_airm_coordinate_difference_mean_l2": 0.10 + subject * 0.002,
                        "airm_normalized_karcher_residual": 1e-10,
                    }
                )
    return pd.DataFrame(rows)


def _leakage_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, base, overlap in (
        ("v1_all_sample", 0.610, 1),
        ("fold_safe", 0.605, 0),
    ):
        for fold in range(5):
            score = base + (fold - 2) * 0.002
            evaluation_n = 504 if fold >= 2 else 540
            rows.append(
                {
                    "protocol_version": "2.0",
                    "config_sha256": CONFIG_SHA,
                    "seed": 20260809,
                    "condition": condition,
                    "row_type": "fold",
                    "fold": fold,
                    "classifier_status": "PASS",
                    "convergence_warning": False,
                    "center_fit_n": 2592 if condition == "v1_all_sample" else 2070,
                    "evaluation_n": evaluation_n,
                    "center_evaluation_overlap_n": evaluation_n * overlap,
                    "original_v1_benchmark_accuracy": 0.6119,
                    "actual_accuracy_difference_from_benchmark": np.nan,
                    **_metric_fields(score, evaluation_n),
                }
            )
        pooled_score = base
        rows.append(
            {
                "protocol_version": "2.0",
                "config_sha256": CONFIG_SHA,
                "seed": 20260809,
                "condition": condition,
                "row_type": "pooled_oof",
                "fold": pd.NA,
                "classifier_status": "PASS",
                "convergence_warning": False,
                "center_fit_n": 2592,
                "evaluation_n": 2592,
                "center_evaluation_overlap_n": 2592 * overlap,
                "original_v1_benchmark_accuracy": 0.6119,
                "actual_accuracy_difference_from_benchmark": pooled_score - 0.6119,
                **_metric_fields(pooled_score, 2592),
            }
        )
    return pd.DataFrame(rows)


def _domain_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in range(1, 10):
        for geometry, metric in MDM_SPECS:
            for protocol, splits in (("T1", ("ALL",)), ("T2", ("A", "B"))):
                for split in splits:
                    rows.append(
                        {
                            "protocol_version": "2.0",
                            "config_sha256": CONFIG_SHA,
                            "seed": 20260809,
                            "subject": subject,
                            "target_subject": subject,
                            "geometry": geometry,
                            "protocol": protocol,
                            "split": split,
                            "reference_metric": metric,
                            "source_target_mean_distance": 0.1 + 0.01 * subject,
                            "absolute_dispersion_difference": 0.02 + 0.001 * subject,
                            "all_subject_subject_silhouette": 0.12 if protocol == "T1" else np.nan,
                            "all_subject_subject_between_within_rms_ratio": 1.2 if protocol == "T1" else np.nan,
                            "uses_class_labels": False,
                            "status": "PASS",
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, object]]:
    config = _config()
    correctness = pd.DataFrame(
        {
            "subject": [1, 1],
            "protocol": ["T1", "T1"],
            "split": ["ALL", "ALL"],
            "required": [True, False],
            "passed": [True, False],
            "status": ["PASS", "SKIPPED_API_UNAVAILABLE"],
        }
    )
    tables = {
        "geometry_correctness.csv": correctness,
        "geometry_mean_comparison.csv": _mean_table(),
        "v1_leakage_audit.csv": _leakage_table(),
        "domain_shift_diagnostics.csv": _domain_table(),
        **_loso_tables(),
    }
    assert set(tables) == set(PREREQUISITE_FILENAMES)
    gate = {
        "classification_gate_pass": True,
        "protocol_version": "2.0",
        "protocol_sha256": PROTOCOL_SHA,
        "config_sha256": CONFIG_SHA,
        "required_rows": 1,
        "passed_required_rows": 1,
        "failed_required_rows": 0,
    }
    return tables, gate, config


def _primary_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    tables = _loso_tables()
    return (
        tables["loso_logistic_transductive.csv"],
        tables["loso_logistic_calibration.csv"],
        _config(),
    )


def test_01_frozen_q1_q2_q3_default_rules_and_next_experiment() -> None:
    t1, t2, config = _primary_frames()
    result = compute_frozen_verdicts(t1, t2, config)
    assert result.q1 == "ROBUSTLY SUPPORTED"
    assert result.q2 == "NOT SUPPORTED"
    assert result.q3 == "SMALL IN THIS PILOT"
    assert result.next_experiment == "subject × class conditional geometry anatomy"
    assert result.operands["Q1"]["geometries"]["LE"]["improved_subjects"] == 9
    assert result.operands["Q2"]["subject_sign_category_disagreements"] == 0


def test_02_q_threshold_boundaries_are_inclusive() -> None:
    t1, t2, config = _primary_frames()
    for subject in range(1, 10):
        raw = float(t1.loc[(t1.target_subject == subject) & (t1.geometry == "RAW"), "balanced_accuracy"].iloc[0])
        for geometry in ("LE", "AIRM"):
            t1.loc[(t1.target_subject == subject) & (t1.geometry == geometry), "balanced_accuracy"] = raw + 0.01
            t2.loc[(t2.target_subject == subject) & (t2.geometry == geometry), "balanced_accuracy"] = raw - 0.01
    result = compute_frozen_verdicts(t1, t2, config)
    assert result.q1 == "ROBUSTLY SUPPORTED"  # delta exactly 0.01, 9/9 improved
    assert result.q3 == "POTENTIALLY IMPORTANT"  # absolute T1/T2 difference exactly 0.02

    # Q2 exact 0.02 mean-delta difference is inclusive.
    for subject in range(1, 10):
        raw = float(t1.loc[(t1.target_subject == subject) & (t1.geometry == "RAW"), "balanced_accuracy"].iloc[0])
        t1.loc[(t1.target_subject == subject) & (t1.geometry == "LE"), "balanced_accuracy"] = raw + 0.02
        t1.loc[(t1.target_subject == subject) & (t1.geometry == "AIRM"), "balanced_accuracy"] = raw
    result = compute_frozen_verdicts(t1, t2, config)
    assert result.q2 == "SUPPORTED"
    assert result.operands["Q2"]["mean_delta_difference_pass"] is True


def test_03_q1_requires_mean_and_six_subjects_and_q2_counts_categories() -> None:
    t1, t2, config = _primary_frames()
    raw = t1[t1.geometry == "RAW"].sort_values("target_subject").balanced_accuracy.to_numpy()
    # Five improved and four mildly worsened gives mean > 0.01 but fails 6/9.
    le_delta = np.array([0.03] * 5 + [-0.005] * 4)
    # Four category disagreements, with close positive means and no opposite sign.
    airm_delta = np.array([0.025] * 6 + [0.0] * 3)
    for geometry, values in (("LE", le_delta), ("AIRM", airm_delta)):
        indices = t1[t1.geometry == geometry].sort_values("target_subject").index
        t1.loc[indices, "balanced_accuracy"] = raw + values
    result = compute_frozen_verdicts(t1, t2, config)
    assert result.operands["Q1"]["geometries"]["LE"]["passes"] is False
    assert result.q1 == "GEOMETRY-SENSITIVE-MIXED"
    assert result.q2 == "SUPPORTED"
    assert result.operands["Q2"]["subject_sign_category_disagreements"] == 4


def test_04_gate_and_complete_input_grid_are_hard_requirements(synthetic_inputs) -> None:
    tables, gate, config = synthetic_inputs
    validated = validate_reporting_inputs(tables, gate, config, config_sha256=CONFIG_SHA)
    assert len(validated["loso_logistic_transductive.csv"]) == 36
    assert len(validated["loso_logistic_calibration.csv"]) == 108
    assert len(validated["loso_mdm_transductive.csv"]) == 45
    assert len(validated["loso_mdm_calibration.csv"]) == 135
    assert len(validated["domain_shift_diagnostics.csv"]) == 135

    failed_gate = dict(gate, classification_gate_pass="true")
    with pytest.raises(ReportingContractError, match="not exactly true"):
        validate_reporting_inputs(tables, failed_gate, config, config_sha256=CONFIG_SHA)
    incomplete = dict(tables)
    incomplete["loso_logistic_transductive.csv"] = tables[
        "loso_logistic_transductive.csv"
    ].iloc[:-1]
    with pytest.raises(ReportingContractError, match="grid mismatch"):
        validate_reporting_inputs(incomplete, gate, config, config_sha256=CONFIG_SHA)


def test_05_outputs_have_exact_five_figure_pairs_summary_and_report_contract(
    synthetic_inputs, tmp_path: Path
) -> None:
    tables, gate, config = synthetic_inputs
    tables_dir = tmp_path / "outputs" / "bnci2014_001_geometry_v2" / "tables"
    figures_dir = tmp_path / "outputs" / "bnci2014_001_geometry_v2" / "figures"
    report_path = (
        tmp_path
        / "outputs"
        / "bnci2014_001_geometry_v2"
        / "report"
        / "geometry_audit_v2.md"
    )
    result = create_reporting_outputs(
        tables,
        gate,
        config,
        config_sha256=CONFIG_SHA,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_path=report_path,
    )
    assert (tables_dir / "geometry_v2_summary.csv").is_file()
    decision = result.summary[result.summary.row_type == "decision_verdict"]
    assert dict(zip(decision.question, decision.verdict, strict=True)) == {
        "Q1": "ROBUSTLY SUPPORTED",
        "Q2": "NOT SUPPORTED",
        "Q3": "SMALL IN THIS PILOT",
    }
    assert decision.operands_json.notna().all()
    assert decision.formula.notna().all()
    assert (result.summary[result.summary.row_type == "aggregate_metric"].std_ddof1 >= 0).all()

    names = {path.name for path in figures_dir.iterdir()}
    expected = {
        *(f"{stem}.png" for stem in FIGURE_STEMS),
        *(f"{stem}.csv" for stem in FIGURE_STEMS),
    }
    assert names == expected
    for stem in FIGURE_STEMS:
        assert (figures_dir / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (figures_dir / f"{stem}.png").stat().st_size > 5_000
    assert len(pd.read_csv(figures_dir / "figure_1_loso_ba_by_subject.csv")) == 9
    assert len(pd.read_csv(figures_dir / "figure_2_paired_delta_vs_raw.csv")) == 9
    assert len(pd.read_csv(figures_dir / "figure_3_t1_vs_t2_ba.csv")) == 9
    assert len(pd.read_csv(figures_dir / "figure_4_le_vs_airm_centers.csv")) == 9
    assert len(pd.read_csv(figures_dir / "figure_5_v1_leakage_audit.csv")) == 12

    report = report_path.read_text(encoding="utf-8")
    assert report.splitlines()[0] == REPORT_TITLE
    assert tuple(
        line[3:] for line in report.splitlines() if line.startswith("## ")
    ) == REPORT_HEADINGS
    for phrase in (
        "conditional-alignment method",
        "temporal or trajectory model",
        "WINDOW5 conclusion",
        "target-label-free conditional structure is identifiable",
    ):
        assert phrase in report
    assert report.count("Run exactly one next experiment:") == 1
    for stem in FIGURE_STEMS:
        assert f"../figures/{stem}.png" in report
        assert f"../figures/{stem}.csv" in report


def test_06_secondary_mdm_failure_and_missing_rows_do_not_block_primary_report(
    synthetic_inputs, tmp_path: Path
) -> None:
    tables, gate, config = synthetic_inputs
    tables = {name: frame.copy() for name, frame in tables.items()}
    mdm_t1 = tables["loso_mdm_transductive.csv"]
    failed_index = mdm_t1.index[
        (mdm_t1.target_subject == 1)
        & (mdm_t1.geometry == "RAW")
        & (mdm_t1.native_metric == "riemann")
    ][0]
    mdm_t1.loc[failed_index, "status"] = "FAILED"
    mdm_t1.loc[failed_index, "convergence_warning"] = True
    mdm_t1.loc[failed_index, "warning_messages"] = '["synthetic convergence"]'
    for column in ("balanced_accuracy", "accuracy", "macro_f1"):
        mdm_t1.loc[failed_index, column] = np.nan
    # Simulate the producer's immediate return: a prefix exists and later
    # secondary logical rows were never generated.
    tables["loso_mdm_transductive.csv"] = mdm_t1.iloc[:6].copy()
    tables["loso_mdm_calibration.csv"] = tables[
        "loso_mdm_calibration.csv"
    ].iloc[:0].copy()

    output_root = tmp_path / "outputs" / "bnci2014_001_geometry_v2"
    result = create_reporting_outputs(
        tables,
        gate,
        config,
        config_sha256=CONFIG_SHA,
        tables_dir=output_root / "tables",
        figures_dir=output_root / "figures",
        report_path=output_root / "report" / "geometry_audit_v2.md",
    )
    assert result.verdicts.q1 == "ROBUSTLY SUPPORTED"
    failures = result.summary[result.summary.row_type == "secondary_mdm_failure"]
    assert not failures.empty
    assert failures.notes.str.contains("EXPLICIT_FAILED").any()
    assert failures.notes.str.contains("MISSING_AFTER_SECONDARY_FAILURE").any()
    assert "synthetic convergence" in result.report_text
    assert "MDM is secondary and does not vote in Q1–Q3" in result.report_text


def test_07_primary_failed_row_generates_technical_report_without_available_case_verdict(
    synthetic_inputs, tmp_path: Path
) -> None:
    tables, gate, config = synthetic_inputs
    tables = {name: frame.copy() for name, frame in tables.items()}
    t1 = tables["loso_logistic_transductive.csv"]
    failed_mask = (t1.target_subject == 3) & (t1.geometry == "AIRM")
    assert failed_mask.sum() == 1
    t1.loc[failed_mask, "status"] = "FAILED"
    t1.loc[failed_mask, "convergence_warning"] = True
    t1.loc[failed_mask, "warning_messages"] = '["STOP: synthetic primary convergence"]'
    failed_numeric = [
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        *(f"recall_{label}" for label in CLASS_ORDER),
        *(
            f"confusion_{truth}__{prediction}"
            for truth in CLASS_ORDER
            for prediction in CLASS_ORDER
        ),
    ]
    t1.loc[failed_mask, failed_numeric] = np.nan
    t1.loc[failed_mask, "confusion_matrix_json"] = pd.NA

    output_root = tmp_path / "outputs" / "bnci2014_001_geometry_v2"
    result = create_reporting_outputs(
        tables,
        gate,
        config,
        config_sha256=CONFIG_SHA,
        tables_dir=output_root / "tables",
        figures_dir=output_root / "figures",
        report_path=output_root / "report" / "geometry_audit_v2.md",
    )
    technical = config["verdicts"]["technical_failure_verdict"]
    assert result.verdicts.technical_failure is True
    assert (result.verdicts.q1, result.verdicts.q2, result.verdicts.q3) == (
        technical,
        technical,
        technical,
    )
    assert len(result.verdicts.primary_failures) == 1
    assert result.verdicts.primary_failures.iloc[0].to_dict() == {
        "source_table": "loso_logistic_transductive.csv",
        "subject": 3,
        "geometry": "AIRM",
        "protocol": "T1",
        "split": "ALL",
        "convergence_warning": True,
        "warning_messages": '["STOP: synthetic primary convergence"]',
    }
    assert not (result.summary.row_type == "paired_delta_aggregate").any()
    decisions = result.summary[result.summary.row_type == "decision_verdict"]
    assert decisions.verdict.eq(technical).all()
    assert decisions.formula.str.contains("NOT COMPUTED").all()
    assert len(result.summary[result.summary.row_type == "primary_logistic_failure"]) == 1
    airm_ba = result.summary[
        (result.summary.row_type == "aggregate_metric")
        & (result.summary.source_table == "loso_logistic_transductive.csv")
        & (result.summary.geometry == "AIRM")
        & (result.summary.metric == "balanced_accuracy")
    ].iloc[0]
    assert airm_ba["n"] == 8
    assert "(8/9)" in airm_ba.notes

    figure_1 = result.figure_sources["figure_1_loso_ba_by_subject"]
    failed_source = figure_1[figure_1.subject == 3].iloc[0]
    assert failed_source.AIRM_status == "FAILED"
    assert pd.isna(failed_source.AIRM)
    figure_2 = result.figure_sources["figure_2_paired_delta_vs_raw"]
    failed_pair = figure_2[figure_2.subject == 3].iloc[0]
    assert failed_pair.pair_status_AIRM_vs_RAW == "FAILED"
    assert pd.isna(failed_pair.delta_AIRM_vs_RAW)
    assert "STOP: synthetic primary convergence" in result.report_text
    assert "no 8/9 available-case verdict is permitted" in result.report_text
    assert "UNASSESSED — TECHNICAL FAILURE" in result.report_text
    assert (
        result.verdicts.next_experiment
        == "preregistered numerical-convergence audit of the fixed unscaled logistic decoder"
    )
