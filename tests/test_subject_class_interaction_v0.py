"""Synthetic-only tests for the frozen subject×class interaction pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.conditional_geometry_v1 import airm_mean_official, spd_invsqrt, spd_sqrt, spd_log, symmetric_exp
from src.interaction_nulls_v0 import (
    canonical_final_summary_bytes,
    create_checkpoint,
    load_checkpoint,
    make_key,
    pending_indices,
    record_batch,
    refit_label_null_once,
    replicate_rng,
    save_checkpoint,
    seed_words,
    shuffle_labels_subject_session,
)
from src.interaction_statistics_v0 import all_derangements, primary_outcome
from src.openbmi_protocol_v0 import OpenBMILockError, validate_scientific_unlock
from src.spd_utils import svec
from src.subject_class_interaction_v0 import (
    build_interactions_from_means,
    compute_interactions,
    geometry_thresholds,
    load_frozen_config,
    split_masks,
)


ROOT = Path(__file__).resolve().parents[1]


def _symmetric(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values + values.T)


def _orthogonal(seed: int, p: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(p, p)))
    return q @ np.diag(np.sign(np.diag(r)))


def _synthetic_means(n_subjects: int = 4, n_sessions: int = 2, n_classes: int = 4, p: int = 3):
    marginal = np.tile(np.eye(p), (n_subjects, n_sessions, 1, 1))
    class_means = np.empty((n_subjects, n_sessions, n_classes, p, p))
    for subject in range(n_subjects):
        for session in range(n_sessions):
            for class_index in range(n_classes):
                diagonal = np.asarray([
                    0.07 * (class_index - 1.5) + 0.01 * subject,
                    -0.04 * (class_index - 1.5) + 0.006 * session,
                    0.02 * (subject - 1.5) * (class_index - 1.5),
                ])
                tangent = np.diag(diagonal)
                tangent[0, 1] = tangent[1, 0] = 0.008 * (subject + 1) * (class_index - 1.5)
                class_means[subject, session, class_index] = symmetric_exp(tangent)
    counts = np.full((n_subjects, n_sessions, n_classes), 12, dtype=np.int64)
    return marginal, class_means, counts


def test_frozen_protocol_hash_is_valid() -> None:
    config, config_hash = load_frozen_config(ROOT)
    assert len(config_hash) == 64
    assert config["protocol"]["base_commit"] == "6124adb907369bc8f76733e880ebb8c43db38e94"


def test_airm_marginal_centering_identity_and_congruence_gives_orthogonal_U() -> None:
    config, _ = load_frozen_config(ROOT)
    thresholds = geometry_thresholds(config)
    rng = np.random.default_rng(41)
    covariances = np.stack([matrix @ matrix.T + np.eye(3) for matrix in rng.normal(size=(18, 3, 3))])
    marginal = airm_mean_official(covariances, thresholds=thresholds).matrix
    class_mean = airm_mean_official(covariances[:9], thresholds=thresholds).matrix
    whitening = spd_invsqrt(marginal)
    u = spd_log(whitening @ class_mean @ whitening)
    np.testing.assert_allclose(whitening @ marginal @ whitening, np.eye(3), rtol=0.0, atol=2e-14)

    a = rng.normal(size=(3, 3))
    while abs(np.linalg.det(a)) < 0.2:
        a = rng.normal(size=(3, 3))
    transformed = a @ covariances @ a.T
    marginal_prime = airm_mean_official(transformed, thresholds=thresholds).matrix
    class_prime = airm_mean_official(transformed[:9], thresholds=thresholds).matrix
    u_prime = spd_log(spd_invsqrt(marginal_prime) @ class_prime @ spd_invsqrt(marginal_prime))
    q = spd_invsqrt(marginal_prime) @ a @ spd_sqrt(marginal)
    np.testing.assert_allclose(q @ q.T, np.eye(3), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(u_prime, q @ u @ q.T, rtol=2e-9, atol=2e-9)


def test_Z_weighted_mean_is_zero_and_loso_excludes_target() -> None:
    config, _ = load_frozen_config(ROOT)
    marginal, classes, counts = _synthetic_means()
    first = build_interactions_from_means(
        marginal_means=marginal, class_means=classes, class_counts=counts,
        geometry="AIRM", template="session_specific", subjects=[1, 2, 3, 4],
        sessions=["s0", "s1"], classes=["a", "b", "c", "d"], config=config,
    )
    weighted = np.einsum("sqc,sqcij->sqij", first.class_proportions, first.Z)
    np.testing.assert_allclose(weighted, 0.0, rtol=0.0, atol=2e-16)

    changed = classes.copy()
    changed[0] = np.stack([symmetric_exp(np.eye(3) * (index + 1) * 0.2) for index in range(4)])
    second = build_interactions_from_means(
        marginal_means=marginal, class_means=changed, class_counts=counts,
        geometry="AIRM", template="session_specific", subjects=[1, 2, 3, 4],
        sessions=["s0", "s1"], classes=["a", "b", "c", "d"], config=config,
    )
    np.testing.assert_array_equal(first.population_templates[0], second.population_templates[0])
    assert not np.array_equal(first.population_templates[1], second.population_templates[1])
    assert first.classes == ("a", "b", "c", "d")


def test_svec_is_deterministic_frobenius_isometric() -> None:
    matrix = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    expected = np.asarray([1.0, 2.0 * np.sqrt(2), 3.0 * np.sqrt(2), 4.0, 5.0 * np.sqrt(2), 6.0])
    np.testing.assert_array_equal(svec(matrix), expected)
    np.testing.assert_array_equal(svec(matrix), svec(matrix.copy()))
    assert np.linalg.norm(svec(matrix)) == pytest.approx(np.linalg.norm(matrix))


def test_spectrum_is_invariant_under_orthogonal_conjugation() -> None:
    rng = np.random.default_rng(2)
    z = _symmetric(rng.normal(size=(5, 5)))
    q = _orthogonal(3, 5)
    np.testing.assert_allclose(np.linalg.eigvalsh(q @ z @ q.T), np.linalg.eigvalsh(z), rtol=1e-14, atol=1e-14)


def test_derangements_have_no_fixed_points() -> None:
    values = all_derangements(9)
    assert values.shape == (133496, 9)
    assert np.all(values != np.arange(9)[None, :])


def _label_fixture() -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    labels = []
    for subject in (1, 2):
        for session in ("s0", "s1"):
            for trial in range(24):
                rows.append({"subject": subject, "session": session, "run": str(trial // 4), "trial_uid": f"S{subject}_{session}_T{trial:02d}", "class_label": str(trial % 3)})
                labels.append(str(trial % 3))
    return np.asarray(labels), pd.DataFrame(rows)


def test_label_shuffle_preserves_counts_is_reproducible_and_refits() -> None:
    labels, metadata = _label_fixture()
    key = make_key(dataset="synthetic", geometry="AIRM", stage="R", signature="sensor_Z", template="session_specific", replicate_index=7)
    first = shuffle_labels_subject_session(labels, metadata, key=key)
    second = shuffle_labels_subject_session(labels, metadata, key=key)
    np.testing.assert_array_equal(first, second)
    for _, indices in metadata.groupby(["subject", "session"]).groups.items():
        np.testing.assert_array_equal(np.sort(first[list(indices)]), np.sort(labels[list(indices)]))
    seen = []
    result = refit_label_null_once(labels, metadata, key=key, fit_statistic=lambda shuffled: seen.append(shuffled.copy()) or np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(result, [1.0, 2.0])
    np.testing.assert_array_equal(seen[0], first)
    assert not np.array_equal(first, labels)


def test_split_halves_are_disjoint_exhaustive_and_have_no_trial_leakage() -> None:
    labels, metadata = _label_fixture()
    masks = split_masks(metadata, "bnci2014_001")
    assert not np.any(masks["A"] & masks["B"])
    assert np.all(masks["A"] | masks["B"])
    a_ids = set(metadata.loc[masks["A"], "trial_uid"])
    b_ids = set(metadata.loc[masks["B"], "trial_uid"])
    assert a_ids.isdisjoint(b_ids)
    assert a_ids | b_ids == set(metadata["trial_uid"])
    assert len(labels) == len(a_ids | b_ids)
    assert a_ids and b_ids


def test_label_override_recomputes_class_means_U_and_Z() -> None:
    config, _ = load_frozen_config(ROOT)
    rng = np.random.default_rng(9)
    rows = []
    covariances = []
    classes = config["datasets"]["bnci2014_001"]["classes"]
    for subject in range(1, 5):
        for session in ("s0", "s1"):
            for class_index, class_name in enumerate(classes):
                for trial in range(3):
                    base = np.diag([1.0 + 0.1 * class_index, 1.1 + 0.02 * subject, 1.2 + 0.01 * trial])
                    noise = rng.normal(scale=0.01, size=(3, 3))
                    covariances.append(base + noise @ noise.T)
                    rows.append({"subject": subject, "session": session, "class_label": class_name, "run": str(trial), "trial_uid": f"S{subject}_{session}_{class_name}_{trial}"})
    metadata = pd.DataFrame(rows)
    covariances = np.asarray(covariances)
    original = compute_interactions(covariances, metadata, config=config, geometry="LE")
    key = make_key(dataset="synthetic", geometry="LE", stage="C", signature="sensor_Z", template="session_specific", replicate_index=3)
    shuffled = shuffle_labels_subject_session(metadata["class_label"].to_numpy(), metadata, key=key)
    refit = compute_interactions(covariances, metadata, config=config, geometry="LE", labels=shuffled)
    np.testing.assert_array_equal(original.marginal_means, refit.marginal_means)
    assert not np.array_equal(original.class_means, refit.class_means)
    assert not np.array_equal(original.U, refit.U)
    assert not np.array_equal(original.Z, refit.Z)


def test_openbmi_lock_blocks_scientific_access_before_manifest_freeze(tmp_path: Path) -> None:
    config = {"project": {"openbmi_manifest_path": "manifest.json", "openbmi_unlock_path": "unlock.json"}}
    with pytest.raises(OpenBMILockError, match="locked"):
        validate_scientific_unlock(tmp_path, config)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"hard_gates_pass": False, "openbmi_unlocked": False, "gate_r": None, "gate_i": None, "gate_c": None, "r_stable": False, "spectrum_supportive": False, "bnci_directions_positive": True}, "UNASSESSED_NUMERICAL_OR_DATA_FAILURE"),
        ({"hard_gates_pass": True, "openbmi_unlocked": False, "gate_r": None, "gate_i": None, "gate_c": None, "r_stable": False, "spectrum_supportive": False, "bnci_directions_positive": False}, "STOP_BNCI_DIRECTION_FAILURE"),
        ({"hard_gates_pass": True, "openbmi_unlocked": True, "gate_r": False, "gate_i": None, "gate_c": None, "r_stable": True, "spectrum_supportive": False, "bnci_directions_positive": True}, "UNASSESSED_CURRENT_OBJECT"),
        ({"hard_gates_pass": True, "openbmi_unlocked": True, "gate_r": True, "gate_i": False, "gate_c": None, "r_stable": False, "spectrum_supportive": False, "bnci_directions_positive": True}, "STOP_NO_STABLE_INDIVIDUAL_COMPONENT"),
        ({"hard_gates_pass": True, "openbmi_unlocked": True, "gate_r": True, "gate_i": True, "gate_c": False, "r_stable": False, "spectrum_supportive": False, "bnci_directions_positive": True}, "STOP_GENERIC_SUBJECT_FINGERPRINT"),
        ({"hard_gates_pass": True, "openbmi_unlocked": True, "gate_r": True, "gate_i": True, "gate_c": True, "r_stable": True, "spectrum_supportive": False, "bnci_directions_positive": True}, "GO_SENSOR_SPACE_ONLY"),
        ({"hard_gates_pass": True, "openbmi_unlocked": True, "gate_r": True, "gate_i": True, "gate_c": True, "r_stable": True, "spectrum_supportive": True, "bnci_directions_positive": True}, "GO_STABLE_SUBJECT_CLASS_INTERACTION"),
    ],
)
def test_primary_outcome_mapping(arguments, expected) -> None:
    assert primary_outcome(**arguments) == expected


def test_master_seed_and_checkpoint_resume_are_exact(tmp_path: Path) -> None:
    key = make_key(dataset="synthetic", geometry="AIRM", stage="C", signature="sensor_Z", template="session_specific", replicate_index=4)
    np.testing.assert_array_equal(seed_words(key), seed_words(dict(reversed(list(key.items())))))
    np.testing.assert_array_equal(replicate_rng(key).integers(0, 2**31, 20), replicate_rng(key).integers(0, 2**31, 20))
    identity = {"protocol_sha256": "p", "config_sha256": "c", "code_sha": "code", "input_hashes": {"x": "y"}}
    checkpoint = create_checkpoint(total_replicates=4, n_subjects=2, identity=identity, dataset="synthetic", geometry="AIRM", stage="C", signature="sensor_Z", template="session_specific")
    checkpoint = record_batch(checkpoint, [2, 0], np.asarray([[0.2, 0.4], [0.0, 0.1]]))
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, checkpoint)
    resumed = load_checkpoint(path, expected_identity={"config_sha256": "c"})
    np.testing.assert_array_equal(pending_indices(resumed), [1, 3])
    resumed = record_batch(resumed, [1, 3], np.asarray([[0.1, 0.3], [0.5, 0.7]]))
    uninterrupted = create_checkpoint(total_replicates=4, n_subjects=2, identity=identity, dataset="synthetic", geometry="AIRM", stage="C", signature="sensor_Z", template="session_specific")
    uninterrupted = record_batch(uninterrupted, [0, 1, 2, 3], np.asarray([[0.0, 0.1], [0.1, 0.3], [0.2, 0.4], [0.5, 0.7]]))
    assert canonical_final_summary_bytes(resumed) == canonical_final_summary_bytes(uninterrupted)
