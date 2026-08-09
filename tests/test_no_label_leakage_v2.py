"""Synthetic identity/leakage tests for the preregistered V2 LOSO protocols."""

from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from src.alignment_v2 import (
    FROZEN_CLASSES,
    IdentityAuditError,
    assert_t1_overlap,
    assert_t2_disjoint,
    canonical_stable_sort,
    label_free_metadata_view,
    make_calibration_splits,
    make_loso_partition,
    make_sample_id_audit_rows,
    trial_uid_sha256,
)


def _frozen_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_index = 0
    for subject in range(1, 10):
        trial_id = 0
        for run in range(6):
            for class_label in FROZEN_CLASSES:
                for _ in range(12):
                    trial_id += 1
                    rows.append(
                        {
                            "sample_index": sample_index,
                            "subject": subject,
                            "session": "0train",
                            "run": str(run),
                            "trial_id": trial_id,
                            "trial_uid": f"S{subject:02d}_0train_T{trial_id:03d}",
                            "class_label": class_label,
                            # Deliberate label-like nuisance columns demonstrate
                            # that the center-fit view uses a strict whitelist.
                            "y": class_label,
                            "true_label": class_label,
                        }
                    )
                    sample_index += 1
    return pd.DataFrame.from_records(rows)


@pytest.fixture(scope="module")
def metadata() -> pd.DataFrame:
    return _frozen_metadata()


def test_01_canonical_sort_is_stable_and_row_order_independent(
    metadata: pd.DataFrame,
) -> None:
    expected = canonical_stable_sort(metadata)
    shuffled = metadata.sample(frac=1.0, random_state=44).reset_index(drop=True)
    observed = canonical_stable_sort(shuffled)
    assert expected["trial_uid"].tolist() == observed["trial_uid"].tolist()
    assert expected["run"].dtype.kind in "iu"
    first_subject_runs = expected.loc[expected.subject == 1, "run"].drop_duplicates()
    assert first_subject_runs.tolist() == list(range(6))


def test_02_canonical_sort_rejects_non_whole_duplicate_identity(
    metadata: pd.DataFrame,
) -> None:
    duplicated = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="trial_uid must be globally unique"):
        canonical_stable_sort(duplicated)


def test_03_trial_uid_hash_is_order_independent_and_identity_sensitive() -> None:
    uids = ["trial-c", "trial-a", "trial-b"]
    expected = trial_uid_sha256(uids)
    assert trial_uid_sha256(list(reversed(uids))) == expected
    assert trial_uid_sha256(["trial-c", "trial-a", "trial-x"]) != expected
    with pytest.raises(ValueError, match="duplicates"):
        trial_uid_sha256(["trial-a", "trial-a"])


def test_04_label_free_view_is_immutable_and_strictly_whitelisted(
    metadata: pd.DataFrame,
) -> None:
    positions = [100, 2, 50]
    view = label_free_metadata_view(metadata, positions)
    assert view.columns == ("subject", "session", "run", "trial_id", "trial_uid")
    assert not hasattr(view, "class_label")
    assert "class_label" not in view.to_frame()
    assert "y" not in view.to_frame()
    assert "true_label" not in view.to_frame()
    assert set(view.row_positions) == set(positions)
    with pytest.raises(FrozenInstanceError):
        view.subject = (99,)  # type: ignore[misc]
    parameters = inspect.signature(label_free_metadata_view).parameters
    assert not {"class_label", "labels", "target", "y"}.intersection(parameters)


def test_05_loso_has_exactly_eight_sources_and_disjoint_target(
    metadata: pd.DataFrame,
) -> None:
    partition = make_loso_partition(metadata, target_subject=4)
    assert partition.target_subject == 4
    assert partition.source_subjects == (1, 2, 3, 5, 6, 7, 8, 9)
    assert partition.n_source_trials == 2304
    assert partition.n_target_trials == 288
    assert set(partition.source_trial_uids).isdisjoint(partition.target_trial_uids)
    assert set(metadata.iloc[list(partition.target_row_positions)].subject) == {4}
    assert set(metadata.iloc[list(partition.source_row_positions)].subject) == {
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
    }


def test_06_loso_hashes_are_invariant_to_metadata_row_order(
    metadata: pd.DataFrame,
) -> None:
    original = make_loso_partition(metadata, 7)
    shuffled = metadata.sample(frac=1.0, random_state=20260809).reset_index(drop=True)
    repeated = make_loso_partition(shuffled, 7)
    assert repeated.source_trial_uid_sha256 == original.source_trial_uid_sha256
    assert repeated.target_trial_uid_sha256 == original.target_trial_uid_sha256
    assert repeated.source_trial_uids == original.source_trial_uids
    assert repeated.target_trial_uids == original.target_trial_uids


def test_07_loso_rejects_missing_subject_and_invalid_target(
    metadata: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="subjects"):
        make_loso_partition(metadata[metadata.subject != 9], 1)
    with pytest.raises(ValueError, match="target_subject"):
        make_loso_partition(metadata, 10)
    with pytest.raises(ValueError, match="integer"):
        make_loso_partition(metadata, 1.5)


def test_08_calibration_splits_are_exact_frozen_run_halves(
    metadata: pd.DataFrame,
) -> None:
    split_a, split_b = make_calibration_splits(metadata, target_subject=3)
    assert split_a.name == "A"
    assert split_a.calibration_runs == (0, 1, 2)
    assert split_a.evaluation_runs == (3, 4, 5)
    assert split_b.name == "B"
    assert split_b.calibration_runs == (3, 4, 5)
    assert split_b.evaluation_runs == (0, 1, 2)
    assert split_a.n_calibration_trials == split_a.n_evaluation_trials == 144
    assert split_b.n_calibration_trials == split_b.n_evaluation_trials == 144
    assert set(split_a.calibration_trial_uids).isdisjoint(
        split_a.evaluation_trial_uids
    )
    assert set(split_a.calibration_trial_uids) == set(
        split_b.evaluation_trial_uids
    )
    assert set(split_a.evaluation_trial_uids) == set(
        split_b.calibration_trial_uids
    )


def test_09_calibration_hashes_are_row_order_invariant(
    metadata: pd.DataFrame,
) -> None:
    original = make_calibration_splits(metadata, 5)
    shuffled = metadata.sample(frac=1.0, random_state=8).reset_index(drop=True)
    repeated = make_calibration_splits(shuffled, 5)
    for expected, observed in zip(original, repeated, strict=True):
        assert observed.calibration_trial_uid_sha256 == (
            expected.calibration_trial_uid_sha256
        )
        assert observed.evaluation_trial_uid_sha256 == (
            expected.evaluation_trial_uid_sha256
        )


def test_10_calibration_rejects_missing_run_or_bad_run_count(
    metadata: pd.DataFrame,
) -> None:
    target = metadata[metadata.subject == 2].copy()
    missing_run = target[target.run != "5"]
    with pytest.raises(ValueError, match="288 trials|runs must be exactly"):
        make_calibration_splits(missing_run)

    short_run = target.drop(target[target.run == "0"].index[0]).reset_index(drop=True)
    with pytest.raises(ValueError, match="288 trials|48 trials"):
        make_calibration_splits(short_run)


def test_11_partition_and_calibration_are_label_permutation_and_drop_invariant(
    metadata: pd.DataFrame,
) -> None:
    original_partition = make_loso_partition(metadata, 6)
    original_splits = make_calibration_splits(metadata, 6)

    permuted = metadata.copy()
    target_mask = permuted.subject == 6
    permuted.loc[target_mask, "class_label"] = (
        permuted.loc[target_mask, "class_label"].iloc[::-1].to_numpy()
    )
    repeated_partition = make_loso_partition(permuted, 6)
    repeated_splits = make_calibration_splits(permuted, 6)
    assert repeated_partition.source_trial_uid_sha256 == (
        original_partition.source_trial_uid_sha256
    )
    assert repeated_partition.target_trial_uid_sha256 == (
        original_partition.target_trial_uid_sha256
    )
    for expected, observed in zip(original_splits, repeated_splits, strict=True):
        assert observed.calibration_trial_uid_sha256 == (
            expected.calibration_trial_uid_sha256
        )
        assert observed.evaluation_trial_uid_sha256 == (
            expected.evaluation_trial_uid_sha256
        )

    identity_only = metadata.drop(columns=["class_label", "y", "true_label"])
    dropped_partition = make_loso_partition(identity_only, 6)
    dropped_splits = make_calibration_splits(identity_only, 6)
    assert dropped_partition.target_trial_uids == original_partition.target_trial_uids
    for expected, observed in zip(original_splits, dropped_splits, strict=True):
        assert observed.calibration_trial_uids == expected.calibration_trial_uids
        assert observed.evaluation_trial_uids == expected.evaluation_trial_uids


def test_12_t1_overlap_assertion_rejects_partial_or_duplicate_sets() -> None:
    uids = ["a", "b", "c"]
    assert_t1_overlap(uids, list(reversed(uids)))
    with pytest.raises(IdentityAuditError, match="to be equal"):
        assert_t1_overlap(uids, ["a", "b", "x"])
    with pytest.raises(IdentityAuditError, match="internally unique"):
        assert_t1_overlap(["a", "a"], ["a", "b"])


def test_13_t2_disjoint_assertion_rejects_any_overlap() -> None:
    assert_t2_disjoint(["a", "b"], ["c", "d"])
    with pytest.raises(IdentityAuditError, match="overlap"):
        assert_t2_disjoint(["a", "b"], ["b", "c"])
    with pytest.raises(IdentityAuditError, match="cannot be empty"):
        assert_t2_disjoint([], ["b"])


def test_14_t1_audit_rows_encode_roles_label_policy_and_overlap(
    metadata: pd.DataFrame,
) -> None:
    loso = make_loso_partition(metadata, 1)
    identity_only = metadata.drop(columns=["class_label", "y", "true_label"])
    audit = make_sample_id_audit_rows(identity_only, loso, protocol="T1")
    assert len(audit) == 11  # eight source centers + classifier + target center + eval
    assert (audit.transductive_overlap == True).all()  # noqa: E712
    assert (audit.relation_assertion_pass == True).all()  # noqa: E712
    assert (audit[audit.role == "source_center_fit"].n_trials == 288).all()
    assert len(audit[audit.role == "source_center_fit"]) == 8
    classifier = audit[audit.role == "classifier_train"].iloc[0]
    assert classifier.n_trials == 2304
    assert classifier.label_access == "source_labels_only"
    target_center = audit[audit.role == "target_center_fit"].iloc[0]
    evaluation = audit[audit.role == "evaluation"].iloc[0]
    assert target_center.n_trials == evaluation.n_trials == 288
    assert len(json.loads(target_center.trial_uids_json)) == 288
    assert json.loads(target_center.trial_uids_json) == json.loads(
        evaluation.trial_uids_json
    )
    assert target_center.trial_uid_sha256 == evaluation.trial_uid_sha256
    assert target_center.overlap_with_evaluation_n == 288
    assert target_center.label_access == "forbidden"
    assert bool(target_center.label_free_metadata)
    assert "class_label" not in target_center.metadata_columns
    assert evaluation.label_access == "target_labels_only"


def test_15_t2_audit_rows_are_disjoint_and_source_identity_is_split_invariant(
    metadata: pd.DataFrame,
) -> None:
    loso = make_loso_partition(metadata, 9)
    identity_only = metadata.drop(columns=["class_label", "y", "true_label"])
    split_a, split_b = make_calibration_splits(identity_only, 9)
    audit_a = make_sample_id_audit_rows(
        identity_only, loso, protocol="T2", calibration_split=split_a
    )
    audit_b = make_sample_id_audit_rows(
        identity_only, loso, protocol="T2", calibration_split=split_b
    )
    assert (audit_a.transductive_overlap == False).all()  # noqa: E712
    assert (audit_b.transductive_overlap == False).all()  # noqa: E712
    for audit in (audit_a, audit_b):
        center = audit[audit.role == "target_center_fit"].iloc[0]
        evaluation = audit[audit.role == "evaluation"].iloc[0]
        assert center.n_trials == evaluation.n_trials == 144
        assert set(json.loads(center.trial_uids_json)).isdisjoint(
            json.loads(evaluation.trial_uids_json)
        )
        assert center.overlap_with_evaluation_n == 0
        assert center.expected_relation_to_evaluation == "disjoint"
        assert evaluation.overlap_with_evaluation_n == 144
    classifier_a = audit_a[audit_a.role == "classifier_train"].iloc[0]
    classifier_b = audit_b[audit_b.role == "classifier_train"].iloc[0]
    assert classifier_a.trial_uid_sha256 == classifier_b.trial_uid_sha256
    center_a = audit_a[audit_a.role == "target_center_fit"].iloc[0]
    evaluation_b = audit_b[audit_b.role == "evaluation"].iloc[0]
    assert center_a.trial_uid_sha256 == evaluation_b.trial_uid_sha256


def test_16_audit_rejects_protocol_split_mismatch(metadata: pd.DataFrame) -> None:
    loso = make_loso_partition(metadata, 2)
    wrong_target_split = make_calibration_splits(metadata, 3)[0]
    with pytest.raises(ValueError, match="target subjects differ"):
        make_sample_id_audit_rows(
            metadata,
            loso,
            protocol="T2",
            calibration_split=wrong_target_split,
        )
    with pytest.raises(ValueError, match="does not accept"):
        make_sample_id_audit_rows(
            metadata,
            loso,
            protocol="T1",
            calibration_split=make_calibration_splits(metadata, 2)[0],
        )
    valid_split = make_calibration_splits(metadata, 2)[0]
    forged_hash = replace(valid_split, calibration_trial_uid_sha256="0" * 64)
    with pytest.raises(IdentityAuditError, match="hash is inconsistent"):
        make_sample_id_audit_rows(
            metadata,
            loso,
            protocol="T2",
            calibration_split=forged_hash,
        )
