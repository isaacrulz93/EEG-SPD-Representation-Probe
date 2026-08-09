"""Synthetic-only tests for the frozen Conditional-Geometry v1 null engines."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.conditional_nulls_v1 import (
    FAMILY_LABEL,
    FAMILY_ORACLE,
    PHASE_COMMON,
    PHASE_DISCOVERY,
    all_derangements,
    all_s4_permutations,
    create_null_checkpoint,
    load_null_checkpoint,
    oracle_rank_null,
    oracle_random_true_indices,
    pending_checkpoint_indices,
    permutation_matrix,
    permute_class_object,
    permuted_shape_bank,
    record_checkpoint_batch,
    run_label_destruction_null,
    save_null_checkpoint,
    semantic_confirmatory_null,
    semantic_discovery_null,
    semantic_permutation_indices,
    shuffle_labels_within_strata,
    tagged_replicate_rng,
    unrelated_derangement_statistics,
)


_D_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _d_vectorizer(objects: np.ndarray) -> np.ndarray:
    array = np.asarray(objects, dtype=np.float64)
    return np.stack([array[..., i, j] for i, j in _D_PAIRS], axis=-1)


def _synthetic_d_objects(n_subjects: int = 4) -> np.ndarray:
    objects = np.zeros((n_subjects, 3, 4, 4), dtype=np.float64)
    base = np.asarray([0.7, 1.1, 1.8, 1.4, 2.2, 1.6], dtype=np.float64)
    for subject in range(n_subjects):
        for split in range(3):
            values = base + 0.03 * subject + 0.01 * split * np.arange(1, 7)
            for value, (i, j) in zip(values, _D_PAIRS, strict=True):
                objects[subject, split, i, j] = value
                objects[subject, split, j, i] = value
    return objects


def _label_fixture() -> tuple[np.ndarray, ...]:
    labels: list[int] = []
    subjects: list[int] = []
    sessions: list[str] = []
    runs: list[str] = []
    uids: list[str] = []
    for subject in (1, 2):
        for run in ("0", "1"):
            for trial in range(12):
                labels.append(trial % 4)
                subjects.append(subject)
                sessions.append("0train")
                runs.append(run)
                uids.append(f"S{subject}_R{run}_T{trial:02d}")
    return tuple(
        np.asarray(value)
        for value in (labels, subjects, sessions, runs, uids)
    )


def test_tagged_rng_is_exact_pcg64dxsm_and_replicate_indexed() -> None:
    actual = tagged_replicate_rng(
        family_tag=FAMILY_ORACLE,
        phase_tag=PHASE_COMMON,
        replicate_index=17,
    ).integers(0, 2**31, size=12)
    expected = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence([20260809, 1301, 2, 17]))
    ).integers(0, 2**31, size=12)
    np.testing.assert_array_equal(actual, expected)
    different = tagged_replicate_rng(
        family_tag=FAMILY_ORACLE,
        phase_tag=PHASE_COMMON,
        replicate_index=18,
    ).integers(0, 2**31, size=12)
    assert not np.array_equal(actual, different)
    with pytest.raises(ValueError, match="unregistered RNG family"):
        tagged_replicate_rng(family_tag=999, phase_tag=2, replicate_index=0)


def test_all_24_s4_actions_are_unique_equivariant_and_identity_first() -> None:
    permutations = all_s4_permutations()
    assert permutations.shape == (24, 4)
    assert len({tuple(row) for row in permutations.tolist()}) == 24
    np.testing.assert_array_equal(permutations[0], np.arange(4))
    object_matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    for permutation in permutations:
        matrix = permutation_matrix(permutation)
        expected = matrix @ object_matrix @ matrix.T
        actual = permute_class_object(object_matrix, permutation)
        np.testing.assert_array_equal(actual, expected)


def test_label_null_preserves_each_stratum_multiset_and_is_row_order_invariant() -> None:
    labels, subjects, sessions, runs, uids = _label_fixture()
    shuffled = shuffle_labels_within_strata(
        labels,
        subjects,
        sessions,
        runs,
        uids,
        replicate_index=3,
        phase_tag=PHASE_DISCOVERY,
    )
    for subject in (1, 2):
        for run in ("0", "1"):
            mask = (subjects == subject) & (runs == run)
            np.testing.assert_array_equal(np.sort(shuffled[mask]), np.sort(labels[mask]))

    order = np.arange(len(labels))[::-1]
    reordered = shuffle_labels_within_strata(
        labels[order],
        subjects[order],
        sessions[order],
        runs[order],
        uids[order],
        replicate_index=3,
        phase_tag=PHASE_DISCOVERY,
    )
    by_uid = dict(zip(uids.tolist(), shuffled.tolist(), strict=True))
    reordered_by_uid = dict(zip(uids[order].tolist(), reordered.tolist(), strict=True))
    assert by_uid == reordered_by_uid


def test_generic_label_null_replays_indexed_refit_callback() -> None:
    labels, subjects, sessions, runs, uids = _label_fixture()

    def statistic_fn(permuted: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                np.mean(permuted[subjects == subject] == labels[subjects == subject])
                for subject in (1, 2)
            ],
            dtype=np.float64,
        )

    first = run_label_destruction_null(
        labels,
        subjects,
        sessions,
        runs,
        uids,
        statistic_fn,
        replicate_indices=[7, 2, 19],
        phase_tag=PHASE_DISCOVERY,
    )
    second = run_label_destruction_null(
        labels,
        subjects,
        sessions,
        runs,
        uids,
        statistic_fn,
        replicate_indices=[19, 7, 2],
        phase_tag=PHASE_DISCOVERY,
    )
    lookup = {int(index): row for index, row in zip(first.replicate_indices, first.subject_statistics)}
    for index, row in zip(second.replicate_indices, second.subject_statistics):
        np.testing.assert_array_equal(row, lookup[int(index)])
    np.testing.assert_array_equal(first.group_statistics, np.median(first.subject_statistics, axis=1))


def test_semantic_plan_is_full_s4_subjectwise_and_common_across_calls() -> None:
    subjects = np.arange(1, 10)
    indices = np.asarray([0, 5, 999])
    first = semantic_permutation_indices(indices, subjects)
    second = semantic_permutation_indices(indices[::-1], subjects)[::-1]
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 9)
    assert np.all((first >= 0) & (first < 24))


def test_semantic_discovery_and_confirmatory_nulls_are_batch_invariant() -> None:
    discovery = _synthetic_d_objects()
    confirmatory = discovery.copy()
    confirmatory[:, 0, 0, 1] += 0.02
    confirmatory[:, 0, 1, 0] += 0.02
    indices = np.asarray([0, 1, 4, 9, 25, 100])
    discovery_one = semantic_discovery_null(
        discovery, _d_vectorizer, replicate_indices=indices, batch_size=1
    )
    discovery_many = semantic_discovery_null(
        discovery, _d_vectorizer, replicate_indices=indices, batch_size=20
    )
    confirm_one = semantic_confirmatory_null(
        discovery, confirmatory, _d_vectorizer, replicate_indices=indices, batch_size=2
    )
    confirm_many = semantic_confirmatory_null(
        discovery, confirmatory, _d_vectorizer, replicate_indices=indices, batch_size=20
    )
    np.testing.assert_array_equal(discovery_one.subject_statistics, discovery_many.subject_statistics)
    np.testing.assert_array_equal(discovery_one.group_statistics, discovery_many.group_statistics)
    np.testing.assert_array_equal(confirm_one.subject_statistics, confirm_many.subject_statistics)
    np.testing.assert_array_equal(confirm_one.group_statistics, confirm_many.group_statistics)


def test_permuted_shape_bank_reconstructs_and_normalizes_after_action() -> None:
    objects = _synthetic_d_objects(n_subjects=3)
    bank = permuted_shape_bank(objects, _d_vectorizer)
    assert bank.shape == (3, 3, 24, 6)
    np.testing.assert_allclose(np.linalg.norm(bank, axis=-1), 1.0, rtol=0.0, atol=5e-16)
    expected_identity = _d_vectorizer(objects)
    expected_identity /= np.linalg.norm(expected_identity, axis=-1, keepdims=True)
    np.testing.assert_allclose(bank[:, :, 0], expected_identity, rtol=0.0, atol=1e-16)


def test_oracle_null_uses_common_indexed_plan_and_is_batch_invariant() -> None:
    score_sets = np.vstack(
        [np.linspace(1.0 + subject * 0.01, 0.0, 24) for subject in range(5)]
    )
    indices = np.asarray([0, 1, 17, 300, 999])
    first = oracle_rank_null(score_sets, replicate_indices=indices, batch_size=1)
    second = oracle_rank_null(score_sets, replicate_indices=indices, batch_size=100)
    np.testing.assert_array_equal(first.subject_statistics, second.subject_statistics)
    plans = oracle_random_true_indices(indices, np.arange(1, 6))
    expected = (24.0 - (plans + 1)) / 23.0
    # Scores are strictly descending, so lexicographic candidate i has rank i+1.
    np.testing.assert_array_equal(first.subject_statistics, expected)


def test_all_9_subject_derangements_are_exhaustive_fixed_point_free() -> None:
    derangements = all_derangements(9)
    assert derangements.shape == (133_496, 9)
    assert np.all(derangements != np.arange(9)[None, :])
    assert np.all(np.sort(derangements, axis=1) == np.arange(9)[None, :])
    shapes = np.eye(9, dtype=np.float64)
    statistics = unrelated_derangement_statistics(shapes, shapes)
    assert statistics.shape == (133_496,)
    np.testing.assert_array_equal(statistics, np.zeros(133_496))


def test_checkpoint_resume_skips_completed_indices_and_validates_identity(tmp_path) -> None:
    checkpoint = create_null_checkpoint(
        total_replicates=6,
        n_subjects=3,
        protocol_sha256="protocol-hash",
        config_sha256="config-hash",
        code_commit="commit",
        input_hash="input-hash",
        family_tag=FAMILY_LABEL,
        phase_tag=PHASE_DISCOVERY,
    )
    updated = record_checkpoint_batch(
        checkpoint,
        [4, 1],
        np.asarray([[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]),
    )
    with pytest.raises(ValueError, match="already completed"):
        record_checkpoint_batch(updated, [1], np.asarray([[0.0, 0.0, 0.0]]))
    path = tmp_path / "checkpoint.npz"
    save_null_checkpoint(path, updated)
    loaded = load_null_checkpoint(
        path,
        expected_metadata={"protocol_sha256": "protocol-hash", "family_tag": 1101},
    )
    np.testing.assert_array_equal(pending_checkpoint_indices(loaded), [0, 2, 3, 5])
    np.testing.assert_array_equal(loaded.completed, updated.completed)
    np.testing.assert_array_equal(loaded.subject_statistics, updated.subject_statistics)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_null_checkpoint(path, expected_metadata={"config_sha256": "wrong"})


def test_checkpoint_payload_tampering_is_rejected(tmp_path) -> None:
    checkpoint = create_null_checkpoint(
        total_replicates=2,
        n_subjects=2,
        protocol_sha256="p",
        config_sha256="c",
        code_commit="g",
        input_hash="i",
        family_tag=FAMILY_LABEL,
        phase_tag=PHASE_DISCOVERY,
    )
    checkpoint = record_checkpoint_batch(checkpoint, [0], np.asarray([[0.2, 0.4]]))
    path = tmp_path / "valid.npz"
    save_null_checkpoint(path, checkpoint)
    with np.load(path, allow_pickle=False) as archive:
        metadata_json = archive["metadata_json"].copy()
        replicate_indices = archive["replicate_indices"].copy()
        completed = archive["completed"].copy()
        subject_statistics = archive["subject_statistics"].copy()
        group_statistics = archive["group_statistics"].copy()
        payload_hash = archive["payload_hash"].copy()
    group_statistics[0] = 123.0
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(
        tampered,
        metadata_json=metadata_json,
        replicate_indices=replicate_indices,
        completed=completed,
        subject_statistics=subject_statistics,
        group_statistics=group_statistics,
        payload_hash=payload_hash,
    )
    with pytest.raises(ValueError, match="group medians|payload hash"):
        load_null_checkpoint(tampered)
