"""Synthetic contracts for Trajectory Anatomy v0 reporting.

No official BNCI data or official output directory is touched by these tests.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting_trajectory_v0 import (
    CLASS_ORDER,
    COMMON_COLUMNS,
    FAILURE_NUMERICAL,
    FAILURE_TECHNICAL,
    FIGURE_STEMS,
    GEOMETRY_GATE_FILENAME,
    MASTER_SEED,
    PREREQUISITE_FILENAMES,
    REPORT_HEADINGS,
    REPORT_TITLE,
    REQUIRED_TABLE_FILENAMES,
    SCALARS_11,
    SUMMARY_COLUMNS,
    SUMMARY_FILENAME,
    TABLE_REQUIRED_COLUMNS,
    VERDICT_GO_BAG,
    VERDICT_GO_ORDER,
    VERDICT_MIXED,
    VERDICT_STOP,
    VERDICT_UNASSESSED,
    compute_frozen_verdicts,
    compute_terminal_verdict,
    create_reporting_outputs,
    expected_null_seeds,
    expected_seed_json,
    stable_frame_sha256,
    validate_geometry_gate,
    validate_reporting_inputs,
)


PROTOCOL_SHA = "a" * 64
CONFIG_SHA = "b" * 64
GENERATED = "2026-08-09T10:00:00Z"


def _prov(status: str = "PASS", generated: str = GENERATED) -> dict[str, object]:
    return {
        "protocol_version": "0.0",
        "protocol_sha256": PROTOCOL_SHA,
        "config_sha256": CONFIG_SHA,
        "seed": MASTER_SEED,
        "session": "0train",
        "generated_at_utc": generated,
        "status": status,
    }


def _gate() -> dict[str, object]:
    return {
        "protocol_version": "0.0",
        "protocol_sha256": PROTOCOL_SHA,
        "config_sha256": CONFIG_SHA,
        "seed": MASTER_SEED,
        "session": "0train",
        "generated_at_utc": GENERATED,
        "status": "PASS",
        "gate_passed": True,
        "scientific_classification_allowed": True,
        "n_trials": 2592,
        # Current official stage-20 semantics: total window-state rows.
        "n_windows": 12960,
        "required_failure_counts": {
            "dataset_contract": 0,
            "covariance_sanity": 0,
            "trajectory_geometry_correctness": 0,
        },
    }


def _table(rows: list[dict[str, object]], generated: str = GENERATED) -> pd.DataFrame:
    return pd.DataFrame([{**_prov(generated=generated), **row} for row in rows])


def _trial_identity(index: int, subject: int, class_label: str) -> dict[str, object]:
    return {
        "sample_index": index,
        "subject": subject,
        "run": str((index - 1) % 6),
        "trial_id": index,
        "trial_uid": f"synthetic-{index:04d}",
        "class_label": class_label,
    }


def _metric(score: float) -> dict[str, object]:
    return {
        "balanced_accuracy": score,
        "accuracy": score - 0.002,
        "macro_f1": score - 0.004,
        "recall_left_hand": score - 0.01,
        "recall_right_hand": score,
        "recall_feet": score + 0.005,
        "recall_tongue": score + 0.005,
        "confusion_matrix_json": "[[50,10,6,6],[9,51,6,6],[7,7,51,7],[6,6,8,52]]",
        "prediction_sha256": hashlib.sha256(f"pred-{score}".encode()).hexdigest(),
        "convergence_warning": False,
        "warning_messages": "[]",
    }


def _path_feature_row(index: int, geometry: str) -> dict[str, object]:
    subject = (index - 1) // 4 + 1
    class_label = CLASS_ORDER[(index - 1) % 4]
    base = 0.4 + 0.01 * subject + 0.005 * ((index - 1) % 4)
    return {
        **_trial_identity(index, subject, class_label),
        "geometry": geometry,
        "s1": base,
        "s2": base + 0.01,
        "s3": base + 0.02,
        "s4": base + 0.03,
        "total_path_length": 4 * base + 0.06,
        "endpoint_distance": 2 * base,
        "efficiency": 0.5,
        "excess": 2 * base + 0.06,
        "theta2": 0.7,
        "theta3": 0.8,
        "theta4": 0.9,
        "mean_turn": 0.8,
        "max_turn": 0.9,
        "dev2": 0.05,
        "dev3": 0.06,
        "dev4": 0.07,
        "mean_geodesic_deviation": 0.06,
        "max_geodesic_deviation": 0.07,
        "frechet_variance": 0.04,
        "frechet_radius_mean": 0.18,
        "diameter": 0.8,
        "degenerate": False,
    }


def _representation_rows(geometry: str, representation: str) -> list[dict[str, object]]:
    identity = _trial_identity(1, 1, CLASS_ORDER[0])
    if representation == "PATH_D10":
        values = {name: 0.1 * i for i, name in enumerate(
            ("d12", "d13", "d14", "d15", "d23", "d24", "d25", "d34", "d35", "d45"), 1)}
    elif representation == "BAG_CANON_D10":
        values = {f"bag{i:02d}": 0.1 * i for i in range(1, 11)}
        values["canonical_permutation"] = "[1,2,3,4,5]"
    else:
        values = {f"sorted{i:02d}": 0.1 * i for i in range(1, 11)}
    return [{**identity, "geometry": geometry, "representation": representation, **values}]


def _class_loso_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offsets = {
        ("AIRM", "PATH_D10"): 0.15,
        ("AIRM", "BAG_CANON_D10"): 0.10,
        ("AIRM", "BAG_SORTED_D10"): 0.09,
        ("AIRM", "SCALARS_11"): 0.08,
        ("LE", "PATH_D10"): 0.14,
        ("LE", "BAG_CANON_D10"): 0.095,
        ("LE", "SCALARS_11"): 0.075,
    }
    for (geometry, representation), offset in offsets.items():
        for subject in range(1, 10):
            score = 0.45 + offset + 0.002 * subject
            rows.append({
                "geometry": geometry,
                "representation": representation,
                "target_subject": subject,
                "source_subjects": "[" + ",".join(str(value) for value in range(1, 10) if value != subject) + "]",
                "train_n": 2304,
                "test_n": 288,
                "train_uid_sha256": hashlib.sha256(f"train-{subject}".encode()).hexdigest(),
                "test_uid_sha256": hashlib.sha256(f"test-{subject}".encode()).hexdigest(),
                "scaler_fit_uid_sha256": hashlib.sha256(f"train-{subject}".encode()).hexdigest(),
                "classifier_config_sha256": "c" * 64,
                **_metric(score),
            })
    return rows


def _runhalf_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for representation_index, representation in enumerate(("PATH_D10", "BAG_CANON_D10", "SCALARS_11")):
        scores = (0.78 - 0.03 * representation_index, 0.76 - 0.03 * representation_index)
        for split, score in zip(("A_TO_B", "B_TO_A"), scores):
            rows.append({
                "geometry": "AIRM",
                "representation": representation,
                "split": split,
                "train_runs": "[0,1,2]" if split == "A_TO_B" else "[3,4,5]",
                "evaluation_runs": "[3,4,5]" if split == "A_TO_B" else "[0,1,2]",
                "train_n": 1296,
                "test_n": 1296,
                "train_uid_sha256": hashlib.sha256(f"subject-train-{split}".encode()).hexdigest(),
                "test_uid_sha256": hashlib.sha256(f"subject-test-{split}".encode()).hexdigest(),
                "chance_level": 1.0 / 9.0,
                "balanced_accuracy": score,
                "accuracy": score - 0.01,
                "direction_average_ba": float(np.mean(scores)),
                "direction_average_accuracy": float(np.mean(scores)) - 0.01,
                "prediction_sha256": hashlib.sha256(f"subject-{representation}-{split}".encode()).hexdigest(),
                "classifier_config_sha256": "d" * 64,
                "convergence_warning": False,
                "warning_messages": "[]",
            })
    return rows


def _factor_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for geometry in ("AIRM", "LE"):
        for scalar_index, scalar in enumerate(SCALARS_11):
            ss_total = 100.0 + scalar_index
            components = (0.45, 0.15, 0.10, 0.30)
            rows.append({
                "geometry": geometry,
                "scalar": scalar,
                "n_subjects": 9,
                "n_classes": 4,
                "n_per_cell": 72,
                "grand_mean": 1.0 + 0.1 * scalar_index,
                "ss_subject": ss_total * components[0],
                "ss_class": ss_total * components[1],
                "ss_interaction": ss_total * components[2],
                "ss_residual": ss_total * components[3],
                "ss_total": ss_total,
                "eta2_subject": components[0],
                "eta2_class": components[1],
                "eta2_interaction": components[2],
                "eta2_residual": components[3],
                "ss_reconstruction_relative_error": 0.0,
                "degenerate": False,
                "uses_p_value": False,
            })
    return rows


def _group_row(
    geometry: str,
    representation: str,
    values: np.ndarray,
    observed: float,
    *,
    path_minus_bag: float | None = None,
) -> dict[str, object]:
    exceedance = int(np.count_nonzero(values >= observed))
    row = {
        "geometry": geometry,
        "representation": representation,
        "observed_median_subject_ba": observed,
        "null_replicates": 199,
        "null_median": float(np.median(values)),
        "null_mean": float(np.mean(values)),
        "null_sd_ddof1": float(np.std(values, ddof=1)),
        "null_min": float(np.min(values)),
        "null_max": float(np.max(values)),
        "effect": observed - float(np.median(values)),
        "p_value": (1 + exceedance) / 200.0,
        "exceedance_count": exceedance,
        "hypothesis_operand_pass": bool(
            observed - float(np.median(values)) > 0 and (1 + exceedance) / 200.0 <= 0.05
        ),
    }
    if path_minus_bag is not None:
        row["median_subject_path_minus_bag"] = path_minus_bag
    return row


def _null_subject_rows(
    kind: str,
    conditions: tuple[tuple[str, str], ...],
    arrays: dict[str, np.ndarray],
    observed_by_condition: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    seeds = expected_null_seeds(kind)
    rows: list[dict[str, object]] = []
    for geometry, representation in conditions:
        values = arrays[f"{geometry.lower()}__{representation.lower()}__median_subject_ba"]
        observed = observed_by_condition[(geometry, representation)]
        for replicate in range(1, 200):
            for subject in range(1, 10):
                score = float(np.clip(values[replicate - 1] + 0.001 * (subject - 5), 0, 1))
                rows.append({
                    "geometry": geometry,
                    "representation": representation,
                    "replicate": replicate,
                    "replicate_seed": int(seeds[replicate - 1]),
                    "target_subject": subject,
                    "balanced_accuracy": score,
                    "accuracy": score,
                    "macro_f1": score,
                    "observed_ba": observed + 0.001 * (subject - 5),
                    "subject_null_median_ba": float(np.median(values)) + 0.001 * (subject - 5),
                    "subject_effect": observed - float(np.median(values)),
                    "train_uid_sha256": hashlib.sha256(f"train-{subject}".encode()).hexdigest(),
                    "test_uid_sha256": hashlib.sha256(f"test-{subject}".encode()).hexdigest(),
                    "classifier_status": "PASS",
                    "convergence_warning": False,
                    "warning_messages": "[]",
                })
    return rows


def _mdm_rows(kind: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject in range(1, 10):
        score = (0.57 if kind == "local" else 0.55) + 0.002 * subject
        row = {
            "representation": "LOCAL_BARYCENTER" if kind == "local" else "WHOLE-1000",
            "target_subject": subject,
            "train_n": 2304,
            "test_n": 288,
            "train_uid_sha256": hashlib.sha256(f"train-{subject}".encode()).hexdigest(),
            "test_uid_sha256": hashlib.sha256(f"test-{subject}".encode()).hexdigest(),
            "metric": "riemann",
            **_metric(score),
        }
        if kind == "whole":
            row.update({
                "covariance_samples_per_estimate": 1000,
                "estimator_regime_confounded": True,
                "interpretation_limit": "not an unconfounded local-vs-whole method comparison",
            })
        rows.append(row)
    return rows


def _fixture(generated: str = GENERATED) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    order_arrays = {
        "replicate": np.arange(1, 200, dtype=np.int64),
        "replicate_seed": expected_null_seeds("order"),
        "airm__path_d10__median_subject_ba": np.linspace(0.46, 0.54, 199, dtype=np.float64),
        "le__path_d10__median_subject_ba": np.linspace(0.45, 0.53, 199, dtype=np.float64),
    }
    label_arrays = {
        "replicate": np.arange(1, 200, dtype=np.int64),
        "replicate_seed": expected_null_seeds("label"),
        "airm__path_d10__median_subject_ba": np.linspace(0.23, 0.31, 199, dtype=np.float64),
        "airm__bag_canon_d10__median_subject_ba": np.linspace(0.22, 0.30, 199, dtype=np.float64),
    }
    class_rows = _class_loso_rows()
    path_observed = float(np.median([row["balanced_accuracy"] for row in class_rows if row["geometry"] == "AIRM" and row["representation"] == "PATH_D10"]))
    bag_observed = float(np.median([row["balanced_accuracy"] for row in class_rows if row["geometry"] == "AIRM" and row["representation"] == "BAG_CANON_D10"]))
    path_minus_bag = 0.05
    order_observed = {("AIRM", "PATH_D10"): path_observed, ("LE", "PATH_D10"): path_observed - 0.01}
    label_observed = {("AIRM", "PATH_D10"): path_observed, ("AIRM", "BAG_CANON_D10"): bag_observed}

    tables: dict[str, pd.DataFrame] = {
        "dataset_contract.csv": _table([{
            "check": "session0_counts", "observed": "2592", "expected": "2592",
            "comparator": "==", "required": True, "passed": True, "failure_message": "",
        }], generated),
        "covariance_sanity.csv": _table([{
            **_trial_identity(1, 1, CLASS_ORDER[0]), "window_index": 1,
            "symmetry_relative_error": 0.0, "min_eigenvalue": 1e-8,
            "max_eigenvalue": 1e-5, "condition_number": 1000.0,
            "has_nan": False, "has_inf": False, "is_spd": True,
            "required": True, "passed": True,
        }], generated),
        "trajectory_geometry_correctness.csv": _table([{
            "geometry": "AIRM", "subject": 1, "sample_index": 1,
            "trial_uid": "synthetic-0001", "check": "all_gates", "statistic": "max",
            "value": 0.0, "threshold": 1e-10, "comparator": "<=",
            "absolute_error": 0.0, "relative_error": 0.0, "required": True,
            "passed": True, "failure_message": "",
        }], generated),
        "trial_airm_path_features.csv": _table([_path_feature_row(i, "AIRM") for i in range(1, 37)], generated),
        "trial_le_path_features.csv": _table([_path_feature_row(i, "LE") for i in range(1, 37)], generated),
        "airm_path_d10.csv": _table(_representation_rows("AIRM", "PATH_D10"), generated),
        "airm_bag_canon_d10.csv": _table(_representation_rows("AIRM", "BAG_CANON_D10"), generated),
        "airm_bag_sorted_d10.csv": _table(_representation_rows("AIRM", "BAG_SORTED_D10"), generated),
        "le_path_d10.csv": _table(_representation_rows("LE", "PATH_D10"), generated),
        "le_bag_canon_d10.csv": _table(_representation_rows("LE", "BAG_CANON_D10"), generated),
        "class_loso_metrics.csv": _table(class_rows, generated),
        "subject_runhalf_probe.csv": _table(_runhalf_rows(), generated),
        "scalar_factor_decomposition.csv": _table(_factor_rows(), generated),
        "order_shuffle_subject_metrics.csv": _table(_null_subject_rows(
            "order", (("AIRM", "PATH_D10"), ("LE", "PATH_D10")), order_arrays, order_observed
        ), generated),
        "order_shuffle_group_metrics.csv": _table([
            _group_row("AIRM", "PATH_D10", order_arrays["airm__path_d10__median_subject_ba"], path_observed, path_minus_bag=path_minus_bag),
            _group_row("LE", "PATH_D10", order_arrays["le__path_d10__median_subject_ba"], path_observed - 0.01, path_minus_bag=path_minus_bag),
        ], generated),
        "label_null_subject_metrics.csv": _table(_null_subject_rows(
            "label", (("AIRM", "PATH_D10"), ("AIRM", "BAG_CANON_D10")), label_arrays, label_observed
        ), generated),
        "label_null_group_metrics.csv": _table([
            _group_row("AIRM", "PATH_D10", label_arrays["airm__path_d10__median_subject_ba"], path_observed),
            _group_row("AIRM", "BAG_CANON_D10", label_arrays["airm__bag_canon_d10__median_subject_ba"], bag_observed),
        ], generated),
        "local_barycenter_mdm.csv": _table(_mdm_rows("local"), generated),
        "whole_context_mdm.csv": _table(_mdm_rows("whole"), generated),
        "airm_le_robustness.csv": _table([{
            "analysis": "class_loso", "representation": "PATH_D10", "subject": subject,
            "scalar": "", "airm_value": 0.60 + 0.002 * subject,
            "le_value": 0.59 + 0.002 * subject, "paired_delta": -0.01,
            "delta_category": "worsened", "airm_status": "PASS", "le_status": "PASS",
            "agreement_category": "AGREE", "interpretation": "secondary robustness only",
        } for subject in range(1, 10)], generated),
    }
    assert tuple(tables) == PREREQUISITE_FILENAMES
    nulls: dict[str, object] = {
        "order_shuffle_seeds.json": expected_seed_json("order"),
        "order_shuffle_group_stats.npz": order_arrays,
        "label_permutation_seeds.json": expected_seed_json("label"),
        "label_null_group_stats.npz": label_arrays,
    }
    return tables, nulls


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_static_contract() -> None:
    assert len(REQUIRED_TABLE_FILENAMES) == 21
    assert REQUIRED_TABLE_FILENAMES[-1] == SUMMARY_FILENAME
    assert len(FIGURE_STEMS) == 8
    assert len(REPORT_HEADINGS) == 18
    assert tuple(SUMMARY_COLUMNS[:7]) == COMMON_COLUMNS
    assert set(PREREQUISITE_FILENAMES) == set(TABLE_REQUIRED_COLUMNS)
    assert REPORT_TITLE == "# BNCI2014_001 Trajectory Anatomy v0"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({
            "effect_label_path": 0.02, "p_label_path": 0.05,
            "effect_label_bag": 0.02, "p_label_bag": 0.05,
            "effect_order_path": 0.01, "p_order_path": 0.05,
            "median_subject_path_minus_bag": 1e-15,
        }, VERDICT_GO_ORDER),
        ({
            "effect_label_path": 0.0, "p_label_path": 0.005,
            "effect_label_bag": 0.02, "p_label_bag": 0.05,
            "effect_order_path": 0.0, "p_order_path": 0.005,
            "median_subject_path_minus_bag": 0.1,
        }, VERDICT_GO_BAG),
        ({
            "effect_label_path": 0.02, "p_label_path": 0.05,
            "effect_label_bag": -0.01, "p_label_bag": 0.005,
            "effect_order_path": 0.0, "p_order_path": 0.005,
            "median_subject_path_minus_bag": 0.1,
        }, VERDICT_MIXED),
        ({
            "effect_label_path": -0.01, "p_label_path": 0.005,
            "effect_label_bag": 0.0, "p_label_bag": 0.005,
            "effect_order_path": 0.02, "p_order_path": 0.05,
            "median_subject_path_minus_bag": 0.1,
        }, VERDICT_STOP),
    ],
)
def test_frozen_verdict_boundaries(kwargs: dict[str, float], expected: str) -> None:
    verdict = compute_terminal_verdict(**kwargs)
    assert verdict.verdict == expected
    expected_next_steps = {
        VERDICT_GO_ORDER: (
            "Conduct one rigorous literature/novelty review and representation-study design "
            "for SPD trajectories."
        ),
        VERDICT_GO_BAG: "Conduct one rigorous trial-as-SPD-distribution comparison.",
        VERDICT_MIXED: (
            "Run one preregistered diagnostic designed solely to explain the "
            "PATH/BAG/order contradiction."
        ),
        VERDICT_STOP: "Return to the frozen Conditional-Geometry Anatomy preregistration.",
    }
    assert verdict.next_experiment == expected_next_steps[expected]


def test_failure_precedence_and_no_available_case() -> None:
    operands = {
        "effect_label_path": 0.1, "p_label_path": 0.005,
        "effect_label_bag": 0.1, "p_label_bag": 0.005,
        "effect_order_path": 0.1, "p_order_path": 0.005,
        "median_subject_path_minus_bag": 0.1,
    }
    technical = compute_terminal_verdict(**operands, technical_failures=["target 9/PATH FAILED"])
    assert technical.verdict == VERDICT_UNASSESSED
    assert technical.failure_status == FAILURE_TECHNICAL
    assert technical.h_path_class is None
    numerical = compute_terminal_verdict(
        **operands,
        numerical_failures=["SPD failure"],
        technical_failures=["target 9/PATH FAILED"],
    )
    assert numerical.verdict == VERDICT_UNASSESSED
    assert numerical.failure_status == FAILURE_NUMERICAL
    assert numerical.h_order is None


def test_geometry_gate_accepts_current_and_explicit_window_semantics() -> None:
    current = _gate()
    assert validate_geometry_gate(
        current,
        protocol_version="0.0",
        protocol_sha256=PROTOCOL_SHA,
        config_sha256=CONFIG_SHA,
    ) == ()

    explicit = copy.deepcopy(current)
    explicit["window_state_rows"] = 12960
    explicit["windows_per_trial"] = 5
    explicit["n_windows"] = 5
    assert validate_geometry_gate(
        explicit,
        protocol_version="0.0",
        protocol_sha256=PROTOCOL_SHA,
        config_sha256=CONFIG_SHA,
    ) == ()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda gate: gate.update(gate_passed=False),
        lambda gate: gate.update(scientific_classification_allowed=False),
        lambda gate: gate.update(status="FAILED"),
        lambda gate: gate.update(protocol_sha256="wrong"),
        lambda gate: gate.update(config_sha256="wrong"),
        lambda gate: gate.update(seed=1),
        lambda gate: gate.update(session="1test"),
        lambda gate: gate.update(n_trials=2591),
        lambda gate: gate.update(n_windows=5),
        lambda gate: gate["required_failure_counts"].update(covariance_sanity=1),
    ],
)
def test_geometry_gate_inconsistency_is_numerical_failure(mutator) -> None:
    gate = _gate()
    mutator(gate)
    failures = validate_geometry_gate(
        gate,
        protocol_version="0.0",
        protocol_sha256=PROTOCOL_SHA,
        config_sha256=CONFIG_SHA,
    )
    assert failures


def test_missing_gate_forces_numerical_unassessed() -> None:
    tables, nulls = _fixture()
    validation = validate_reporting_inputs(
        tables,
        nulls,
        None,
        protocol_sha256=PROTOCOL_SHA,
        config_sha256=CONFIG_SHA,
        strict_counts=False,
    )
    assert validation.numerical_failures == (
        f"{GEOMETRY_GATE_FILENAME}: missing or invalid JSON object",
    )
    verdict = compute_frozen_verdicts(tables, validation)
    assert verdict.verdict == VERDICT_UNASSESSED
    assert verdict.failure_status == FAILURE_NUMERICAL
    assert verdict.h_path_class is None


def test_full_synthetic_validation_and_frozen_operands() -> None:
    tables, nulls = _fixture()
    validation = validate_reporting_inputs(
        tables, nulls, _gate(), protocol_sha256=PROTOCOL_SHA, config_sha256=CONFIG_SHA,
        strict_counts=False,
    )
    assert validation.numerical_failures == ()
    assert validation.technical_failures == ()
    verdict = compute_frozen_verdicts(tables, validation)
    assert verdict.verdict == VERDICT_GO_ORDER
    assert verdict.h_path_class is True
    assert verdict.h_bag_class is True
    assert verdict.h_order is True
    assert verdict.operands["median_subject_PATH_minus_BAG"] == pytest.approx(0.05)


def test_failed_required_classifier_row_is_technical_unassessed() -> None:
    tables, nulls = _fixture()
    frame = tables["class_loso_metrics.csv"].copy()
    mask = (
        frame["geometry"].eq("AIRM")
        & frame["representation"].eq("PATH_D10")
        & frame["target_subject"].eq(9)
    )
    frame.loc[mask, "status"] = "FAILED"
    frame.loc[mask, "convergence_warning"] = True
    frame.loc[mask, "warning_messages"] = '["ConvergenceWarning"]'
    frame.loc[mask, ["balanced_accuracy", "accuracy", "macro_f1"]] = np.nan
    tables["class_loso_metrics.csv"] = frame
    validation = validate_reporting_inputs(
        tables, nulls, _gate(), protocol_sha256=PROTOCOL_SHA, config_sha256=CONFIG_SHA,
        strict_counts=False,
    )
    assert not validation.numerical_failures
    assert any("target_subject=9" in failure for failure in validation.technical_failures)
    verdict = compute_frozen_verdicts(tables, validation)
    assert verdict.verdict == VERDICT_UNASSESSED
    assert verdict.failure_status == FAILURE_TECHNICAL


def test_numerical_gate_failure_wins() -> None:
    tables, nulls = _fixture()
    covariance = tables["covariance_sanity.csv"].copy()
    covariance.loc[0, "passed"] = False
    covariance.loc[0, "is_spd"] = False
    covariance.loc[0, "min_eigenvalue"] = -1e-12
    covariance.loc[0, "status"] = "FAILED"
    tables["covariance_sanity.csv"] = covariance
    validation = validate_reporting_inputs(
        tables, nulls, _gate(), protocol_sha256=PROTOCOL_SHA, config_sha256=CONFIG_SHA,
        strict_counts=False,
    )
    verdict = compute_frozen_verdicts(tables, validation)
    assert validation.numerical_failures
    assert verdict.verdict == VERDICT_UNASSESSED
    assert verdict.failure_status == FAILURE_NUMERICAL


def test_exact_outputs_report_and_deterministic_rerun(tmp_path: Path) -> None:
    first_tables, first_nulls = _fixture("2026-08-09T10:00:00Z")
    second_tables, second_nulls = _fixture("2026-08-09T11:00:00Z")
    roots = (tmp_path / "first", tmp_path / "second")
    results = []
    for root, tables, nulls in zip(roots, (first_tables, second_tables), (first_nulls, second_nulls)):
        artifacts = create_reporting_outputs(
            tables,
            nulls,
            _gate(),
            protocol_sha256=PROTOCOL_SHA,
            config_sha256=CONFIG_SHA,
            tables_dir=root / "tables",
            figures_dir=root / "figures",
            report_path=root / "report" / "trajectory_anatomy_v0.md",
            strict_counts=False,
        )
        results.append(artifacts)
        assert tuple(artifacts.summary.columns) == SUMMARY_COLUMNS
        figure_1_series = set(artifacts.figure_sources[FIGURE_STEMS[0]]["series"])
        assert "WHOLE-1000" in figure_1_series
        assert "WHOLE_1000" not in figure_1_series
        assert (root / "tables" / SUMMARY_FILENAME).is_file()
        for stem in FIGURE_STEMS:
            for suffix in (".png", ".pdf", ".csv"):
                path = root / "figures" / f"{stem}{suffix}"
                assert path.is_file() and path.stat().st_size > 0
        report = (root / "report" / "trajectory_anatomy_v0.md").read_text(encoding="utf-8")
        assert report.splitlines()[0] == REPORT_TITLE
        assert tuple(line[3:] for line in report.splitlines() if line.startswith("## ")) == tuple(
            f"{index}. {heading}" for index, heading in enumerate(REPORT_HEADINGS, start=1)
        )
        assert "t-SNE" not in report and "tsne" not in report.lower()
        assert report.count("## ") == 18
        assert "estimator-regime-confounded" in report
        assert "GO_TRAJECTORY_ORDER" in report
        assert "AIRM+LE CONSISTENT" in report
        assert "averaged held-out class probabilities" in report
        agreement_mask = tables["airm_le_robustness.csv"]["agreement_category"].astype(str).str.upper().isin(
            ["AGREE", "AGREEMENT", "SAME", "CONCORDANT"]
        )
        expected_disagreements = int((~agreement_mask).sum())
        assert f"{expected_disagreements} rows were not marked as agreement" in report

    assert results[0].decision_sha256 == results[1].decision_sha256
    assert stable_frame_sha256(results[0].summary) == stable_frame_sha256(results[1].summary)
    for stem in FIGURE_STEMS:
        assert stable_frame_sha256(results[0].figure_sources[stem]) == stable_frame_sha256(results[1].figure_sources[stem])
        assert _file_sha(roots[0] / "figures" / f"{stem}.png") == _file_sha(roots[1] / "figures" / f"{stem}.png")
        assert _file_sha(roots[0] / "figures" / f"{stem}.pdf") == _file_sha(roots[1] / "figures" / f"{stem}.pdf")


def test_null_seed_manifests_are_exact_and_distinct() -> None:
    order = expected_null_seeds("order")
    label = expected_null_seeds("label")
    assert order.dtype == np.uint64 and order.shape == (199,)
    assert label.dtype == np.uint64 and label.shape == (199,)
    assert len(set(order.tolist())) == 199
    assert len(set(label.tolist())) == 199
    assert not np.array_equal(order, label)
    assert expected_seed_json("order")["replicates"][0] == {
        "replicate": 1,
        "seed": int(order[0]),
    }
