"""Synthetic-only freeze tests for trajectory within-subject audit v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.trajectory_geometry_v0 import bag_canon_d10, path_d10, permute_distance_matrix
from src.trajectory_within_subject_analysis_v1 import _fast_statistics, _split_indices
from src.trajectory_within_subject_v1 import (
    ALL_PERMUTATIONS_5,
    CLASS_ORDER,
    HALF_A,
    HALF_B,
    IncompleteRequiredGrid,
    LABEL_STREAM_TAG,
    NULL_REPLICATES,
    ORDER_STREAM_TAG,
    SESSION_ORDER,
    apply_order_shuffle,
    assert_bag_invariance,
    compare_reproduction,
    load_frozen_config,
    make_seed_vector,
    monte_carlo_result,
    order_permutation_indices,
    permute_labels,
    run_stage_w,
    run_stage_x,
    sha256_array,
    stage_subject_statistics,
    terminal_decision,
    validate_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "bnci2014_001_trajectory_within_subject_v1.yaml"


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return load_frozen_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def synthetic(config) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(41)
    rows: list[dict[str, object]] = []
    path_rows: list[np.ndarray] = []
    bag_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    global_index = 0
    session_index = {session: 0 for session in SESSION_ORDER}
    class_patterns = np.asarray(
        [
            np.linspace(-1.0, 0.8, 10),
            np.linspace(0.9, -0.7, 10),
            np.sin(np.linspace(0, np.pi, 10)),
            np.cos(np.linspace(0, np.pi, 10)),
        ]
    )
    for subject in range(1, 10):
        for session_number, session in enumerate(SESSION_ORDER):
            trial_id = 0
            for run in range(6):
                for class_index, label in enumerate(CLASS_ORDER):
                    for repetition in range(12):
                        trial_id += 1
                        sample_index = session_index[session]
                        session_index[session] += 1
                        rows.append(
                            {
                                "global_sample_index": global_index,
                                "sample_index": sample_index,
                                "subject": subject,
                                "session": session,
                                "run": run,
                                "trial_id": trial_id,
                                "trial_uid": f"S{subject:02d}_{session}_T{trial_id:03d}",
                                "class_label": label,
                            }
                        )
                        noise = rng.normal(scale=0.12, size=10)
                        path = (
                            class_patterns[class_index]
                            + 0.025 * subject
                            + 0.015 * session_number
                            + 0.003 * run
                            + noise
                        )
                        path_rows.append(path)
                        bag_rows.append(np.sort(path))
                        scalar_rows.append(
                            np.concatenate(
                                [path, [float(np.mean(path) + 0.01 * repetition)]]
                            )
                        )
                        global_index += 1
    metadata = pd.DataFrame.from_records(rows)
    arrays = {
        "PATH_D10": np.asarray(path_rows, dtype=np.float64),
        "BAG_CANON_D10": np.asarray(bag_rows, dtype=np.float64),
        "SCALARS_11": np.asarray(scalar_rows, dtype=np.float64),
    }
    assert len(metadata) == 5184
    validate_metadata(metadata, config)
    return metadata, arrays


@pytest.fixture(scope="module")
def observed_w(config, synthetic) -> pd.DataFrame:
    metadata, arrays = synthetic
    return run_stage_w(
        arrays["PATH_D10"], metadata, config, representation="PATH_D10"
    )


@pytest.fixture(scope="module")
def observed_x(config, synthetic) -> pd.DataFrame:
    metadata, arrays = synthetic
    return run_stage_x(
        arrays["PATH_D10"], metadata, config, representation="PATH_D10"
    )


def test_protocol_config_freezes_exact_existing_method(config) -> None:
    assert config["protocol"]["reference_commit"] == (
        "fcb55ccdccdd4613290b8e8d93be91ea256edd45"
    )
    assert config["representations"]["primary"] == "PATH_D10"
    assert config["representations"]["mandatory_unordered_control"] == "BAG_CANON_D10"
    assert config["representations"]["descriptive_scalar_control"] == "SCALARS_11"
    assert config["window5"]["n_windows"] == 5
    assert config["classifier"]["c"] == 1.0
    assert config["classifier"]["tuning"] is False
    assert config["nulls"]["order_shuffle"]["replicates"] == 1999


def test_run_halves_are_disjoint_and_exhaustive() -> None:
    assert set(HALF_A).isdisjoint(HALF_B)
    assert set(HALF_A) | set(HALF_B) == set(range(6))
    assert len(HALF_A) == len(HALF_B) == 3


def test_stage_w_is_one_subject_one_session_and_has_no_trial_overlap(observed_w) -> None:
    assert len(observed_w) == 36
    assert set(observed_w.status) == {"PASS"}
    assert set(observed_w.train_n) == set(observed_w.test_n) == {144}
    assert (observed_w.train_subjects == observed_w.subject.map(lambda x: f"[{x}]")).all()
    assert (observed_w.test_subjects == observed_w.subject.map(lambda x: f"[{x}]")).all()
    assert (observed_w.train_uid_sha256 != observed_w.test_uid_sha256).all()
    for row in observed_w.itertuples(index=False):
        assert json.loads(row.train_session) == [row.session]
        assert json.loads(row.test_session) == [row.session]
        assert set(json.loads(row.train_runs)).isdisjoint(json.loads(row.test_runs))
        assert set(json.loads(row.train_runs)) | set(json.loads(row.test_runs)) == set(range(6))


def test_scaler_is_fit_on_training_rows_only(config, synthetic, observed_w) -> None:
    metadata, arrays = synthetic
    row = observed_w.iloc[0]
    train = (
        metadata["subject"].eq(int(row.subject))
        & metadata["session"].eq(str(row.session))
        & metadata["run"].isin(json.loads(row.train_runs))
    ).to_numpy()
    expected_mean = arrays["PATH_D10"][train].mean(axis=0)
    expected_scale_hash_input = arrays["PATH_D10"][train].std(axis=0, ddof=0)
    assert row.scaler_fit_uid_sha256 == row.train_uid_sha256
    assert row.scaler_mean_sha256 == sha256_array(expected_mean)
    # StandardScaler's scale equals the population SD for nonconstant columns.
    assert row.scaler_scale_sha256 == sha256_array(expected_scale_hash_input)


def test_stage_x_sessions_are_disjoint_and_no_other_subject_leaks(observed_x) -> None:
    assert len(observed_x) == 18
    assert set(observed_x.status) == {"PASS"}
    assert set(observed_x.train_n) == set(observed_x.test_n) == {288}
    for row in observed_x.itertuples(index=False):
        train_sessions = json.loads(row.train_session)
        test_sessions = json.loads(row.test_session)
        assert len(train_sessions) == len(test_sessions) == 1
        assert set(train_sessions).isdisjoint(test_sessions)
        assert json.loads(row.train_subjects) == [row.subject]
        assert json.loads(row.test_subjects) == [row.subject]
        assert row.train_uid_sha256 != row.test_uid_sha256


def test_class_counts_are_frozen_in_every_required_split(synthetic) -> None:
    metadata, _ = synthetic
    for subject in range(1, 10):
        for session in SESSION_ORDER:
            scope = metadata["subject"].eq(subject) & metadata["session"].eq(session)
            for runs in (HALF_A, HALF_B):
                counts = metadata.loc[scope & metadata["run"].isin(runs), "class_label"].value_counts()
                assert counts.to_dict() == {label: 36 for label in CLASS_ORDER}
            counts = metadata.loc[scope, "class_label"].value_counts()
            assert counts.to_dict() == {label: 72 for label in CLASS_ORDER}


def test_label_permutation_is_deterministic_and_preserves_each_run_multiset(
    synthetic,
) -> None:
    metadata, _ = synthetic
    labels = metadata.class_label.to_numpy()
    seed = int(make_seed_vector("label", replicates=2)[0])
    first = permute_labels(labels, metadata, seed)
    second = permute_labels(labels, metadata, seed)
    np.testing.assert_array_equal(first, second)
    assert not first.flags.writeable
    assert not np.array_equal(first, labels)
    for _, group in metadata.groupby(["subject", "session", "run"], sort=True):
        positions = group.index.to_numpy()
        assert sorted(first[positions]) == sorted(labels[positions])
        assert pd.Series(first[positions]).value_counts().to_dict() == {
            label: 12 for label in CLASS_ORDER
        }


def test_rng_seed_vectors_are_exact_replayable_and_family_separated() -> None:
    label = make_seed_vector("label", replicates=3)
    order = make_seed_vector("order", replicates=3)
    expected_label = np.asarray(
        [
            int(child.generate_state(1, dtype=np.uint64)[0])
            for child in np.random.SeedSequence([20260810, LABEL_STREAM_TAG]).spawn(3)
        ],
        dtype=np.uint64,
    )
    expected_order = np.asarray(
        [
            int(child.generate_state(1, dtype=np.uint64)[0])
            for child in np.random.SeedSequence([20260810, ORDER_STREAM_TAG]).spawn(3)
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(label, expected_label)
    np.testing.assert_array_equal(order, expected_order)
    assert not np.array_equal(label, order)


def test_order_shuffle_is_nonidentity_bag_invariant_and_path_changes() -> None:
    coordinates = np.asarray([0.0, 1.0, 3.0, 7.0, 12.0])
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    original_path = path_d10(distance)
    indices = order_permutation_indices(32, int(make_seed_vector("order", replicates=1)[0]))
    assert np.all((indices >= 1) & (indices <= 119))
    tiled = np.tile(original_path, (32, 1))
    shuffled = apply_order_shuffle(tiled, indices)
    assert np.all(np.any(shuffled != tiled, axis=1))
    for index in np.unique(indices):
        assert_bag_invariance(distance, int(index))
        permuted = permute_distance_matrix(distance, ALL_PERMUTATIONS_5[int(index)])
        np.testing.assert_array_equal(
            bag_canon_d10(permuted).vector, bag_canon_d10(distance).vector
        )


def test_v0_reproduction_gate_requires_exact_identity_and_machine_tolerance(synthetic) -> None:
    metadata, arrays = synthetic
    session0 = metadata[metadata.session.eq("0train")].reset_index(drop=True)
    observed = {
        "airm_path_d10": arrays["PATH_D10"][:2592],
        "airm_bag_canon_d10": arrays["BAG_CANON_D10"][:2592],
        "airm_scalars_11": arrays["SCALARS_11"][:2592],
    }
    passed = compare_reproduction(session0, session0.copy(), observed, observed)
    assert passed.passed.all()
    changed = {name: value.copy() for name, value in observed.items()}
    changed["airm_path_d10"][0, 0] += 2e-12
    failed = compare_reproduction(session0, session0.copy(), changed, observed)
    assert not bool(failed.loc[failed.check.eq("airm_path_d10_machine_precision"), "passed"].iloc[0])


def test_no_available_case_silent_drop(observed_w) -> None:
    _, statistic = stage_subject_statistics(observed_w, "W")
    assert 0.0 <= statistic <= 1.0
    failed = observed_w.copy()
    failed.loc[0, ["status", "balanced_accuracy"]] = ["FAILED", np.nan]
    with pytest.raises(IncompleteRequiredGrid, match="FAILED"):
        stage_subject_statistics(failed, "W")
    with pytest.raises(IncompleteRequiredGrid, match="incomplete"):
        stage_subject_statistics(observed_w.iloc[:-1], "W")


def test_fast_null_statistic_matches_public_observed_definition(
    config, synthetic, observed_w, observed_x
) -> None:
    metadata, arrays = synthetic
    w_splits, x_splits = _split_indices(metadata)
    labels = metadata.class_label.to_numpy(dtype=str)
    fast_w_subject, fast_w_group = _fast_statistics(
        arrays["PATH_D10"], labels, w_splits, 4, config
    )
    fast_x_subject, fast_x_group = _fast_statistics(
        arrays["PATH_D10"], labels, x_splits, 2, config
    )
    public_w_subject, public_w_group = stage_subject_statistics(observed_w, "W")
    public_x_subject, public_x_group = stage_subject_statistics(observed_x, "X")
    np.testing.assert_array_equal(fast_w_subject, public_w_subject)
    np.testing.assert_array_equal(fast_x_subject, public_x_subject)
    assert fast_w_group == public_w_group
    assert fast_x_group == public_x_group


def test_plus_one_monte_carlo_uses_all_1999_replicates() -> None:
    null = np.linspace(0.20, 0.30, NULL_REPLICATES)
    result = monte_carlo_result(0.31, null)
    assert result.replicates == 1999
    assert result.null_median == pytest.approx(0.25)
    assert result.effect == pytest.approx(0.06)
    assert result.exceedance_count == 0
    assert result.p_value == pytest.approx(1 / 2000)
    assert result.passed is True
    with pytest.raises(IncompleteRequiredGrid):
        monte_carlo_result(0.31, null[:-1])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                reproduction_gate_pass=False,
                technical_grid_pass=False,
                stage_w_pass=None,
                stage_x_pass=None,
                stage_o_pass=None,
            ),
            "UNASSESSED_TRAJECTORY_REPRODUCTION_FAILURE",
        ),
        (
            dict(
                reproduction_gate_pass=True,
                technical_grid_pass=False,
                stage_w_pass=None,
                stage_x_pass=None,
                stage_o_pass=None,
            ),
            "UNASSESSED_TECHNICAL_FAILURE",
        ),
        (
            dict(
                reproduction_gate_pass=True,
                technical_grid_pass=True,
                stage_w_pass=False,
                stage_x_pass=None,
                stage_o_pass=None,
            ),
            "STOP_WITHIN_SUBJECT_TRAJECTORY_CLASS_POOR",
        ),
        (
            dict(
                reproduction_gate_pass=True,
                technical_grid_pass=True,
                stage_w_pass=True,
                stage_x_pass=False,
                stage_o_pass=None,
            ),
            "STOP_SESSION_SPECIFIC_TRAJECTORY_ONLY",
        ),
        (
            dict(
                reproduction_gate_pass=True,
                technical_grid_pass=True,
                stage_w_pass=True,
                stage_x_pass=True,
                stage_o_pass=False,
            ),
            "GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY",
        ),
        (
            dict(
                reproduction_gate_pass=True,
                technical_grid_pass=True,
                stage_w_pass=True,
                stage_x_pass=True,
                stage_o_pass=True,
            ),
            "GO_STABLE_SUBJECT_SPECIFIC_ORDERED_TRAJECTORY_COMPONENT",
        ),
    ],
)
def test_terminal_decision_mapping(kwargs, expected) -> None:
    assert terminal_decision(**kwargs).decision == expected
