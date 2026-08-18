"""Synthetic and contract tests for population-structure V1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.subject_class_population_structure_v1 as module
from src.subject_class_population_structure_v1 import (
    DatasetObjects,
    _haar_basis,
    _inner_rank_selection_openbmi,
    deterministic_rng,
    evaluate_openbmi_nested,
    fixed_point_free,
    fit_two_view,
    generate_openbmi_class_mappings,
    generate_openbmi_pairing_mappings,
    helmert_matrix,
    load_config,
    load_parent_dataset,
    openbmi_fold_indices,
    reconstruct_fold_z,
    select_rank_one_se,
    signature_from_z,
    svec,
    terminal_decision,
    validate_helmert,
    validate_no_nonfinite_outputs,
    validate_parent_hashes,
    validate_report_consistency,
)


ROOT = Path(__file__).resolve().parents[1]


def _fake_openbmi(seed: int = 7) -> DatasetObjects:
    rng = np.random.default_rng(seed)
    n, q, k, d = 54, 2, 2, 3
    base = rng.normal(size=(n, q, d, d))
    base = 0.5 * (base + np.swapaxes(base, -1, -2))
    interaction = np.stack([-base, base], axis=2)
    proportions = np.full((n, q, k), 0.5)
    counts = np.full((n, q, k), 10, dtype=np.int64)
    means = np.tile(np.eye(d), (n, q, k, 1, 1))
    marginal = np.tile(np.eye(d), (n, q, 1, 1))
    return DatasetObjects(
        name="openbmi", subjects=tuple(range(1, 55)), sessions=("0", "1"),
        classes=("left_hand", "right_hand"), channels=("a", "b", "c"),
        U={split: interaction.copy() for split in ("A", "B", "F")},
        proportions={split: proportions.copy() for split in ("A", "B", "F")},
        counts={split: counts.copy() for split in ("A", "B", "F")},
        class_means={split: means.copy() for split in ("A", "B", "F")},
        marginal_means={split: marginal.copy() for split in ("A", "B", "F")},
    )


def test_protocol_config_and_parent_hashes_are_frozen() -> None:
    config, digest = load_config(ROOT)
    assert len(digest) == 64
    assert config["protocol"]["base_commit"] == "d9a67a130aeeea7eb8a93d76878e43f636802e93"
    observed = validate_parent_hashes(ROOT, config)
    assert observed["openbmi_objects"] == "f7e2fd7517fe1f55f84ef7729823b2d3f10452833ec2399a4b7014f769c98572"


def test_parent_object_ordering_contracts() -> None:
    config, _ = load_config(ROOT)
    openbmi = load_parent_dataset(ROOT, config, "openbmi")
    bnci = load_parent_dataset(ROOT, config, "bnci")
    assert openbmi.subjects == tuple(range(1, 55))
    assert openbmi.sessions == ("0", "1")
    assert openbmi.classes == ("left_hand", "right_hand")
    assert bnci.subjects == tuple(range(1, 10))
    assert bnci.sessions == ("0train", "1test")
    assert bnci.classes == ("left_hand", "right_hand", "feet", "tongue")


def test_svec_is_frobenius_isometric() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(8, 5, 5))
    matrix = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
    np.testing.assert_allclose(np.linalg.norm(svec(matrix), axis=-1), np.linalg.norm(matrix, axis=(-2, -1)), rtol=2e-16, atol=2e-16)


def test_helmert_is_literal_orthonormal_and_sum_zero() -> None:
    config, _ = load_config(ROOT)
    matrix = validate_helmert(config)
    np.testing.assert_allclose(matrix, helmert_matrix(4), rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=3e-16)
    np.testing.assert_allclose(matrix @ np.ones(4), 0.0, rtol=0.0, atol=2e-16)


def test_binary_contrast_removes_redundant_concatenation() -> None:
    matrix = np.asarray([[1.0, 2.0], [2.0, -1.0]])
    z = np.stack([-matrix, matrix])[None, None]
    raw, norms = signature_from_z(z, "openbmi", normalize=False)
    np.testing.assert_allclose(raw[0, 0], svec(matrix), rtol=0.0, atol=0.0)
    assert raw.shape[-1] == 3
    assert norms[0, 0] == pytest.approx(np.linalg.norm(matrix))


def test_outer_fold_coverage_exactly_once_and_inner_coverage() -> None:
    config, _ = load_config(ROOT)
    folds, inner = openbmi_fold_indices(_fake_openbmi(), config)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(54))
    for fold_index, test in enumerate(folds):
        train = set(range(54)) - set(test)
        assert set(np.concatenate(inner[fold_index])) == train


def test_outer_test_never_changes_training_template_or_z() -> None:
    data = _fake_openbmi()
    train = np.arange(45)
    test = np.arange(45, 54)
    first, held_first, audit = reconstruct_fold_z(data.U["F"], data.proportions["F"], train, test)
    changed = data.U["F"].copy()
    changed[test] += 1e6
    second, held_second, _ = reconstruct_fold_z(changed, data.proportions["F"], train, test)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(held_first, held_second)
    assert all(not set(sources) & set(test) for sources in audit["train_template_sources"])
    assert not set(audit["held_template_sources"]) & set(test)


def test_outer_test_never_changes_feature_center_or_basis() -> None:
    rng = np.random.default_rng(13)
    train0 = rng.normal(size=(30, 12))
    train1 = rng.normal(size=(30, 12))
    test0 = rng.normal(size=(5, 12))
    test1 = rng.normal(size=(5, 12))
    first = fit_two_view(train0, train1, 5)
    test0 += 1e9
    test1 -= 1e9
    second = fit_two_view(train0, train1, 5)
    for field in ("mean0", "mean1", "left", "right", "scale0", "scale1", "singular_values"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_outer_test_never_changes_inner_rank_selection() -> None:
    config, _ = load_config(ROOT)
    data = _fake_openbmi()
    outer, inner = openbmi_fold_indices(data, config)
    test = outer[0]
    train = np.setdiff1d(np.arange(54), test)
    first = _inner_rank_selection_openbmi(
        data, config, 0, train, inner[0], kind="sensor", helmert=None,
        class_mapping=None, inner_pairing_mappings=None,
    )
    changed_u = {key: value.copy() for key, value in data.U.items()}
    changed_u["F"][test] += 1e8
    changed = DatasetObjects(**{**data.__dict__, "U": changed_u})
    second = _inner_rank_selection_openbmi(
        changed, config, 0, train, inner[0], kind="sensor", helmert=None,
        class_mapping=None, inner_pairing_mappings=None,
    )
    assert first["selected_rank"] == second["selected_rank"]
    np.testing.assert_array_equal(first["fold_scores"], second["fold_scores"])


def test_compact_cross_covariance_svd_matches_direct() -> None:
    rng = np.random.default_rng(21)
    x0 = rng.normal(size=(28, 17))
    x1 = rng.normal(size=(28, 17))
    fit = fit_two_view(x0, x1, 8)
    covariance = (x0 - x0.mean(0)).T @ (x1 - x1.mean(0)) / 27
    left, singular, right_t = np.linalg.svd(covariance, full_matrices=False)
    np.testing.assert_allclose(fit.singular_values, singular[:8], rtol=2e-13, atol=2e-13)
    reconstructed = fit.left @ np.diag(fit.singular_values) @ fit.right.T
    expected = left[:, :8] @ np.diag(singular[:8]) @ right_t[:8]
    np.testing.assert_allclose(reconstructed, expected, rtol=3e-13, atol=3e-13)


def test_rank_one_se_selects_smallest_eligible() -> None:
    scores = np.asarray([[1.0, 1.12, 1.12], [1.0, 1.11, 1.15], [1.0, 1.13, 1.10], [1.0, 1.12, 1.13], [1.0, 1.12, 1.12]])
    selected = select_rank_one_se(scores, [1, 2, 3])
    assert selected["best_rank"] == 3
    assert selected["selected_rank"] == 2


def test_derangements_have_no_fixed_point_and_are_deterministic() -> None:
    values = np.arange(17)
    first = fixed_point_free(values, deterministic_rng(20260818, "unit", 1))
    second = fixed_point_free(values, deterministic_rng(20260818, "unit", 1))
    np.testing.assert_array_equal(first, second)
    assert np.all(first != values)


def test_pairing_mappings_are_partition_local_and_reproducible() -> None:
    config, _ = load_config(ROOT)
    data = _fake_openbmi()
    first_outer, first_inner = generate_openbmi_pairing_mappings(data, config, 2)
    second_outer, second_inner = generate_openbmi_pairing_mappings(data, config, 2)
    np.testing.assert_array_equal(first_outer, second_outer)
    np.testing.assert_array_equal(first_inner, second_inner)
    outer, inner = openbmi_fold_indices(data, config)
    train = np.setdiff1d(np.arange(54), outer[0])
    assert set(first_outer[0, 0, train]) == set(train)
    assert set(first_outer[0, 0, outer[0]]) == set(outer[0])
    assert np.all(first_outer[0, 0] != np.arange(54))


def test_class_mapping_is_independent_not_global_swap() -> None:
    config, _ = load_config(ROOT)
    mapping = generate_openbmi_class_mappings(_fake_openbmi(), config, 4)
    for replicate in mapping:
        swaps = replicate[..., 0]
        assert len(np.unique(swaps)) == 2
        np.testing.assert_array_equal(np.sort(replicate, axis=-1), np.broadcast_to([0, 1], replicate.shape))


def test_null_worker_reruns_rank_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake = {
        "statistic": 1.0, "forward_median": 1.0, "reverse_median": 1.0,
        "full_space_statistic": 1.0, "selected_ranks": np.asarray([1, 2, 3, 1, 2, 3]),
    }
    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return fake
    monkeypatch.setattr(module, "evaluate_openbmi_nested", evaluate)
    module._NULL_STATE = {
        "data": object(), "config": {},
        "pairing_outer": np.zeros((1, 6, 54), dtype=int),
        "pairing_inner": np.zeros((1, 6, 5, 54), dtype=int),
        "class_mappings": np.zeros((1, 54, 2, 2), dtype=int),
        "selected_ranks": np.ones(6, dtype=int),
    }
    _, _, ranks = module._null_worker(("pairing", 0))
    assert calls and "outer_pairing_mappings" in calls[0] and "inner_pairing_mappings" in calls[0]
    np.testing.assert_array_equal(ranks, fake["selected_ranks"])


def test_random_basis_is_orthonormal_and_deterministic() -> None:
    first = _haar_basis(31, 7, deterministic_rng(88, "basis"))
    second = _haar_basis(31, 7, deterministic_rng(88, "basis"))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(7), rtol=0.0, atol=5e-16)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"data_contract_pass": False, "reliability_pass": True}, "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE"),
        ({"data_contract_pass": True, "reliability_pass": False}, "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE"),
        ({"data_contract_pass": True, "reliability_pass": True, "pairing_p": 0.2}, "STOP_NO_HELDOUT_POPULATION_STRUCTURE"),
        ({"data_contract_pass": True, "reliability_pass": True, "random_p": 0.2}, "STOP_RANDOM_SUBSPACE_EQUIVALENT"),
        ({"data_contract_pass": True, "reliability_pass": True, "selected_ranks": [13] * 6}, "GO_STRUCTURED_BUT_NOT_LOW_DIMENSIONAL"),
        ({"data_contract_pass": True, "reliability_pass": True}, "GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION"),
    ],
)
def test_expected_terminal_labels(kwargs: dict, expected: str) -> None:
    defaults = dict(
        data_contract_pass=True, reliability_pass=True, statistic=1.0,
        forward_median=1.0, reverse_median=1.0, pairing_p=0.01, class_p=0.01,
        random_p=0.01, influence_positive=True, full_space_stable=True,
        selected_ranks=[3] * 6, low_cap=8,
    )
    defaults.update(kwargs)
    assert terminal_decision(**defaults) == expected


def test_no_nan_inf_output_validator(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "good.npz", x=np.asarray([1.0, 2.0]))
    (tmp_path / "good.csv").write_text("x\n1\n2\n", encoding="utf-8")
    validate_no_nonfinite_outputs(tmp_path)
    np.savez_compressed(tmp_path / "bad.npz", x=np.asarray([np.nan]))
    with pytest.raises(module.DataContractError, match="nonfinite"):
        validate_no_nonfinite_outputs(tmp_path)


def test_final_report_table_consistency_when_present() -> None:
    path = ROOT / "outputs/subject_class_population_structure_v1/report/subject_class_population_structure_v1.md"
    if path.is_file():
        validate_report_consistency(ROOT)
    else:
        assert not path.exists()
