"""Synthetic-only evaluation, RNG, leakage, and verdict tests for v0."""

from __future__ import annotations

import copy
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.exceptions import ConvergenceWarning

import src.evaluation_trajectory_v0 as evaluation
from src.evaluation_trajectory_v0 import (
    CLASS_LOSO_COLUMNS,
    FACTOR_COLUMNS,
    LABEL_GROUP_COLUMNS,
    NULL_SUBJECT_COLUMNS,
    ORDER_GROUP_COLUMNS,
    SUBJECT_PROBE_COLUMNS,
    apply_label_permutation,
    apply_order_permutation,
    balanced_factor_decomposition,
    evaluate_frozen_verdict,
    fit_source_scaled_logistic,
    make_label_permutation_plan,
    make_null_seed_plan,
    make_order_permutation_plan,
    replay_label_permutation_plan,
    replay_null_seed_plan,
    replay_order_permutation_plan,
    run_class_loso,
    run_label_destruction_null,
    run_mdm_loso,
    run_order_shuffle_null,
    run_subject_runhalf_probe,
    summarize_null_distribution,
)


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("left_hand", "right_hand")


def _config() -> dict[str, object]:
    config = copy.deepcopy(
        yaml.safe_load((ROOT / "configs/bnci2014_001_trajectory_v0.yaml").read_text())
    )
    config["dataset"].update(
        {
            "subjects": [1, 2, 3],
            "runs": [0, 1, 2, 3, 4, 5],
            "classes": list(CLASSES),
            "expected_trials": 72,
            "expected_trials_per_subject": 24,
            "expected_trials_per_subject_class": 12,
            "expected_trials_per_subject_run": 4,
            "expected_trials_per_subject_run_class": 2,
        }
    )
    config["nulls"]["order_shuffle"]["replicates"] = 3
    config["nulls"]["label_destruction"]["replicates"] = 3
    config["subject_probe"]["trials_per_subject_half"] = 12
    config["subject_probe"]["chance"] = 1 / 3
    config["factor_decomposition"].update(
        {"n_subjects": 3, "n_classes": 2, "n_per_cell": 12}
    )
    return config


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    path_rows: list[np.ndarray] = []
    bag_rows: list[np.ndarray] = []
    covariance_rows: list[np.ndarray] = []
    rng = np.random.default_rng(731)
    sample = 0
    for subject in (1, 2, 3):
        trial_id = 0
        for run in range(6):
            run_trial_id = 0
            for label_index, label in enumerate(CLASSES):
                for repetition in range(2):
                    trial_id += 1
                    run_trial_id += 1
                    rows.append(
                        {
                            "sample_index": sample,
                            "subject": subject,
                            "session": "0train",
                            "run": run,
                            "trial_id": trial_id,
                            "run_trial_id": run_trial_id,
                            "trial_uid": f"S{subject:02d}_0train_T{trial_id:03d}",
                            "class_label": label,
                        }
                    )
                    base = rng.normal(scale=0.08, size=10)
                    base += (2 * label_index - 1) * np.linspace(0.2, 0.8, 10)
                    base += 0.04 * subject
                    base += 0.005 * repetition
                    path_rows.append(base)
                    bag_rows.append(np.sort(base))
                    covariance_rows.append(
                        np.diag(
                            [
                                np.exp(0.5 * label_index + 0.03 * subject),
                                np.exp(-0.2 * label_index + 0.02 * repetition),
                            ]
                        )
                    )
                    sample += 1
    return (
        pd.DataFrame(rows),
        np.asarray(path_rows),
        np.asarray(bag_rows),
        np.asarray(covariance_rows),
    )


def test_class_loso_is_source_scaled_and_schema_exact(synthetic) -> None:
    metadata, path, _, _ = synthetic
    result = run_class_loso(
        path,
        metadata,
        _config(),
        geometry="AIRM",
        representation="PATH_D10",
        provenance={"config_sha256": "c" * 64},
    )
    assert tuple(result.columns) == CLASS_LOSO_COLUMNS
    assert len(result) == 3
    assert set(result.status) == {"PASS"}
    assert set(result.train_n) == {48}
    assert set(result.test_n) == {24}
    assert (result.train_uid_sha256 == result.scaler_fit_uid_sha256).all()
    assert result.test_uid_sha256.nunique() == 3
    assert result.classifier_config_sha256.nunique() == 1
    assert result.balanced_accuracy.between(0, 1).all()
    forbidden = {"target_labels", "target_y", "test_labels"}
    assert forbidden.isdisjoint(inspect.signature(fit_source_scaled_logistic).parameters)


def test_convergence_warning_is_failed_and_model_is_never_scored(
    synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, path, _, _ = synthetic

    class WarningModel:
        def __init__(self, **kwargs):
            self.n_iter_ = np.asarray([5000])

        def fit(self, x, y):
            self.classes_ = np.unique(y)
            self.coef_ = np.zeros((len(self.classes_), x.shape[1]))
            self.intercept_ = np.zeros(len(self.classes_))
            warnings.warn("synthetic convergence failure", ConvergenceWarning)
            return self

        def predict(self, x):
            raise AssertionError("FAILED estimator must not be scored")

    monkeypatch.setattr(evaluation, "LogisticRegression", WarningModel)
    result = run_class_loso(
        path,
        metadata,
        _config(),
        geometry="AIRM",
        representation="PATH_D10",
        target_subjects=(1,),
    )
    row = result.iloc[0]
    assert row.status == "FAILED"
    assert bool(row.convergence_warning)
    assert pd.isna(row.balanced_accuracy)
    assert "synthetic convergence failure" in row.warning_messages


def test_seedsequence_and_order_plan_exact_replay_and_no_identity(synthetic) -> None:
    metadata, path, _, _ = synthetic
    seeds = make_null_seed_plan("order", replicates=3)
    children = np.random.SeedSequence(
        [20260809, 0x4F52444552]
    ).spawn(3)
    expected = np.asarray(
        [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children],
        dtype=np.uint64,
    )
    assert np.array_equal(seeds.seeds, expected)
    assert replay_null_seed_plan(seeds)
    plan = make_order_permutation_plan(metadata, seed_plan=seeds)
    assert plan.permutation_indices.shape == (3, 72)
    assert np.all((plan.permutation_indices >= 1) & (plan.permutation_indices <= 119))
    assert replay_order_permutation_plan(metadata, plan)
    shuffled = apply_order_permutation(path, plan, 1)
    assert shuffled.shape == path.shape
    assert not shuffled.flags.writeable
    first_map = evaluation.EDGE_REINDEX_TABLE[plan.permutation_indices[0, 0]]
    assert np.array_equal(shuffled[0], path[0, first_map])


def test_label_plan_is_shared_replayable_and_preserves_each_cell(synthetic) -> None:
    metadata, path, bag, _ = synthetic
    seeds = make_null_seed_plan("label", replicates=3)
    plan = make_label_permutation_plan(metadata, seed_plan=seeds)
    assert replay_label_permutation_plan(metadata, plan)
    labels = metadata.class_label.to_numpy()
    first = apply_label_permutation(labels, plan, 1)
    second_use_same_plan = apply_label_permutation(labels, plan, 1)
    assert np.array_equal(first, second_use_same_plan)
    assert plan.source_row_indices.shape == (3, 72)
    for _, group in metadata.groupby(["subject", "run"], sort=True):
        rows = group.index.to_numpy()
        assert sorted(first[rows]) == sorted(labels[rows])
    # Features are never modified by the label plan and PATH/BAG replay the same y.
    assert path.shape == bag.shape == (72, 10)
    assert not np.shares_memory(first, path)


def test_null_evaluation_has_full_grid_and_fixed_plus_one_summary(synthetic) -> None:
    metadata, path, bag, _ = synthetic
    config = _config()
    observed_path = run_class_loso(
        path, metadata, config, geometry="AIRM", representation="PATH_D10"
    )
    observed_bag = run_class_loso(
        bag, metadata, config, geometry="AIRM", representation="BAG_CANON_D10"
    )
    order_plan = make_order_permutation_plan(metadata, replicates=3)
    order_rows = run_order_shuffle_null(
        path,
        metadata,
        config,
        order_plan,
        observed_path,
        geometry="AIRM",
    )
    assert tuple(order_rows.columns) == NULL_SUBJECT_COLUMNS
    assert len(order_rows) == 9
    assert set(order_rows.replicate) == {1, 2, 3}
    assert order_rows.subject_null_median_ba.notna().all()
    order_summary = summarize_null_distribution(
        order_rows,
        observed_path,
        config,
        family="order",
        geometry="AIRM",
        representation="PATH_D10",
        median_subject_path_minus_bag=0.01,
    )
    assert order_summary.complete
    assert tuple(order_summary.table.columns) == ORDER_GROUP_COLUMNS
    row = order_summary.table.iloc[0]
    assert row.p_value == (1 + int(row.exceedance_count)) / 4
    assert order_summary.replicate_statistics.shape == (3,)

    label_plan = make_label_permutation_plan(metadata, replicates=3)
    path_null = run_label_destruction_null(
        path,
        metadata,
        config,
        label_plan,
        observed_path,
        geometry="AIRM",
        representation="PATH_D10",
    )
    bag_null = run_label_destruction_null(
        bag,
        metadata,
        config,
        label_plan,
        observed_bag,
        geometry="AIRM",
        representation="BAG_CANON_D10",
    )
    assert np.array_equal(path_null.replicate_seed, bag_null.replicate_seed)
    label_summary = summarize_null_distribution(
        path_null,
        observed_path,
        config,
        family="label",
        geometry="AIRM",
        representation="PATH_D10",
    )
    assert label_summary.complete
    assert tuple(label_summary.table.columns) == LABEL_GROUP_COLUMNS


def test_failed_null_row_forbids_available_case_inference(synthetic) -> None:
    metadata, path, _, _ = synthetic
    config = _config()
    observed = run_class_loso(
        path, metadata, config, geometry="AIRM", representation="PATH_D10"
    )
    plan = make_order_permutation_plan(metadata, replicates=3)
    rows = run_order_shuffle_null(
        path, metadata, config, plan, observed, geometry="AIRM"
    )
    rows.loc[0, ["status", "classifier_status", "balanced_accuracy"]] = [
        "FAILED",
        "FAILED",
        np.nan,
    ]
    summary = summarize_null_distribution(
        rows,
        observed,
        config,
        family="order",
        geometry="AIRM",
        representation="PATH_D10",
        median_subject_path_minus_bag=0.1,
    )
    assert not summary.complete
    assert summary.table.iloc[0].status == "FAILED"
    assert pd.isna(summary.table.iloc[0].p_value)
    assert summary.replicate_statistics.size == 0


def test_subject_probe_run_halves_are_disjoint_and_averaged(synthetic) -> None:
    metadata, path, _, _ = synthetic
    result = run_subject_runhalf_probe(
        path,
        metadata,
        _config(),
        geometry="AIRM",
        representation="PATH_D10",
    )
    assert tuple(result.columns) == SUBJECT_PROBE_COLUMNS
    assert list(result.split) == ["A_TO_B", "B_TO_A"]
    assert set(result.train_n) == set(result.test_n) == {36}
    assert result.train_uid_sha256.nunique() == 2
    assert result.direction_average_ba.nunique() == 1
    assert result.direction_average_ba.iloc[0] == pytest.approx(
        result.balanced_accuracy.mean()
    )


def test_balanced_factor_decomposition_closes_without_p_values(synthetic) -> None:
    metadata, path, _, _ = synthetic
    config = _config()
    names = config["representations"]["scalar_columns"]
    values = np.empty((len(metadata), len(names)), dtype=np.float64)
    subject = metadata.subject.to_numpy()
    label = (metadata.class_label.to_numpy() == "right_hand").astype(float)
    run = metadata.run.to_numpy()
    for index in range(len(names)):
        values[:, index] = (
            (index + 1) * 0.1 * subject
            + (index + 2) * 0.2 * label
            + 0.01 * run
            + 0.001 * path[:, index % 10]
        )
    result = balanced_factor_decomposition(
        values, metadata, config, geometry="AIRM"
    )
    assert tuple(result.columns) == FACTOR_COLUMNS
    assert len(result) == 11
    assert set(result.status) == {"PASS"}
    assert (result.uses_p_value == False).all()  # noqa: E712
    assert (result.ss_reconstruction_relative_error <= 1e-10).all()
    eta_sum = result[
        ["eta2_subject", "eta2_class", "eta2_interaction", "eta2_residual"]
    ].sum(axis=1)
    assert np.allclose(eta_sum, 1.0, atol=1e-12)


def test_mdm_loso_is_trial_separated_and_schema_complete(synthetic) -> None:
    metadata, _, _, covariances = synthetic
    result = run_mdm_loso(
        covariances,
        metadata,
        _config(),
        representation="LOCAL_BARYCENTER",
    )
    assert len(result) == 3
    assert set(result.status) == {"PASS"}
    assert set(result.train_n) == {48}
    assert set(result.test_n) == {24}
    assert set(result.metric) == {"riemann"}
    assert result.train_uid_sha256.nunique() == 3


@pytest.mark.parametrize(
    ("operands", "expected"),
    [
        (
            dict(
                label_path_effect=0.1,
                label_path_p=0.05,
                label_bag_effect=0.1,
                label_bag_p=0.05,
                order_path_effect=0.1,
                order_path_p=0.05,
                median_subject_path_minus_bag=1e-12,
            ),
            "GO_TRAJECTORY_ORDER",
        ),
        (
            dict(
                label_path_effect=-0.1,
                label_path_p=1.0,
                label_bag_effect=0.1,
                label_bag_p=0.05,
                order_path_effect=-0.1,
                order_path_p=1.0,
                median_subject_path_minus_bag=-0.1,
            ),
            "GO_UNORDERED_DISTRIBUTION",
        ),
        (
            dict(
                label_path_effect=-0.1,
                label_path_p=1.0,
                label_bag_effect=-0.1,
                label_bag_p=1.0,
                order_path_effect=-0.1,
                order_path_p=1.0,
                median_subject_path_minus_bag=0.0,
            ),
            "STOP_LOCAL_TRAJECTORY_V0",
        ),
    ],
)
def test_frozen_verdict_boundaries(operands, expected) -> None:
    result = evaluate_frozen_verdict(
        numerical_gate_pass=True, technical_grid_pass=True, **operands
    )
    assert result.verdict == expected


def test_failure_precedence_uses_exact_tokens() -> None:
    numerical = evaluate_frozen_verdict(
        numerical_gate_pass=False, technical_grid_pass=False
    )
    assert numerical.verdict == "UNASSESSED"
    assert numerical.failure_status == "UNASSESSED — NUMERICAL/DATA FAILURE"
    technical = evaluate_frozen_verdict(
        numerical_gate_pass=True, technical_grid_pass=False
    )
    assert technical.failure_status == "UNASSESSED—PROTOCOL/TECHNICAL FAILURE"
