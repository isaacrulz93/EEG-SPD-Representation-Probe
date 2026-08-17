from __future__ import annotations

import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.alignment_v2 import make_calibration_splits, make_loso_partition
from src.local_mean_movement_v0 import anti_develop_sequence, conjugate_movement
from src.trial_movement_utility_v0 import (
    CLASS_ORDER,
    CONDITIONS,
    DIMENSIONS,
    _manifest,
    condition_features,
    delta_summary,
    exact_signflip_test,
    exact_subject_bootstrap_interval,
    feature_functions_are_label_free,
    fit_fixed_source_decoder,
    holm_adjust,
    movement_feature_tuple,
    terminal_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _orthogonal(seed: int, dimension: int = 22) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    return q * np.sign(np.diag(r))[None, :]


def _sequence() -> np.ndarray:
    q = _orthogonal(10)
    base = np.linspace(-0.8, 0.8, 22)
    return np.asarray(
        [q @ np.diag(np.exp(base + step * np.linspace(-0.04, 0.06, 22))) @ q.T for step in range(5)]
    )


def _full_metadata() -> pd.DataFrame:
    rows = []
    sample = 0
    for subject in range(1, 10):
        trial_id = 0
        for run in range(6):
            for run_trial in range(48):
                trial_id += 1
                rows.append(
                    {
                        "sample_index": sample,
                        "subject": subject,
                        "session": "0train",
                        "run": run,
                        "trial_id": trial_id,
                        "run_trial_id": run_trial + 1,
                        "trial_uid": f"S{subject:02d}_0train_T{trial_id:03d}",
                        "class_label": CLASS_ORDER[run_trial % 4],
                    }
                )
                sample += 1
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "decoder": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 20000,
            "tol": 1e-6,
            "random_state": 20260817,
        }
    }


def test_five_window_alignment_and_identity_contract() -> None:
    whole = _full_metadata()
    window = whole.loc[whole.index.repeat(5)].reset_index(drop=True)
    window["window_index"] = np.tile(np.arange(1, 6), len(whole))
    grouped = window.groupby("trial_uid", sort=False)
    assert len(whole) == 2592
    assert len(window) == 12960
    assert (grouped.size() == 5).all()
    assert all(tuple(frame["window_index"]) == (1, 2, 3, 4, 5) for _, frame in grouped)
    identity = ["subject", "session", "run", "trial_uid", "class_label"]
    assert all(all(frame[column].nunique() == 1 for column in identity) for _, frame in grouped)
    assert set(whole["trial_uid"]) == set(window["trial_uid"])


def test_v2_loso_and_complementary_splits_are_reused() -> None:
    metadata = _full_metadata().drop(columns=["class_label"])
    partition = make_loso_partition(metadata, 4)
    assert partition.source_subjects == (1, 2, 3, 5, 6, 7, 8, 9)
    assert not (set(partition.source_trial_uids) & set(partition.target_trial_uids))
    split_a, split_b = make_calibration_splits(metadata, 4)
    assert split_a.calibration_runs == (0, 1, 2)
    assert split_a.evaluation_runs == (3, 4, 5)
    assert split_b.calibration_runs == (3, 4, 5)
    assert split_b.evaluation_runs == (0, 1, 2)
    assert not (set(split_a.calibration_trial_uids) & set(split_a.evaluation_trial_uids))
    assert set(split_a.calibration_trial_uids) == set(split_b.evaluation_trial_uids)
    assert set(split_a.evaluation_trial_uids) == set(split_b.calibration_trial_uids)


def test_anti_development_norm_transport_symmetry_and_finiteness() -> None:
    sequence = _sequence()
    anti = anti_develop_sequence(sequence, delta_t=0.8)
    assert anti.z.shape == (4, 22, 22)
    assert np.isfinite(anti.z).all()
    assert np.allclose(anti.z, anti.z.transpose(0, 2, 1), atol=1e-12)
    assert anti.diagnostics["maximum_norm_absolute_error"].max() <= 1e-8
    assert anti.diagnostics["maximum_edge_transport_relative_error"].max() <= 1e-8
    assert anti.diagnostics["passed"].all()


def test_movement_features_dimensions_gram_psd_diagonal_and_common_o() -> None:
    anti = anti_develop_sequence(_sequence(), delta_t=0.8)
    lengths, gram, gram_feature, sensor = movement_feature_tuple(anti.z)
    assert lengths.shape == (4,)
    assert gram_feature.shape == (10,)
    assert sensor.shape == (1012,)
    assert np.min(np.linalg.eigvalsh(gram)) >= -1e-10
    assert np.allclose(np.diag(gram), lengths**2, atol=1e-10)
    _, conjugated_gram, _, _ = movement_feature_tuple(conjugate_movement(anti.z, _orthogonal(11)))
    assert np.allclose(conjugated_gram, gram, rtol=1e-10, atol=1e-10)


def test_gram_is_invariant_to_common_gl_congruence_of_original_sequence() -> None:
    sequence = _sequence()
    q = _orthogonal(12)
    action = q @ np.diag(np.linspace(0.7, 1.3, 22)) @ q.T
    transformed = np.einsum("ij,kjl,ml->kim", action, sequence, action, optimize=True)
    original_z = anti_develop_sequence(sequence, delta_t=0.8).z
    transformed_z = anti_develop_sequence(transformed, delta_t=0.8).z
    _, original_gram, _, _ = movement_feature_tuple(original_z)
    _, transformed_gram, _, _ = movement_feature_tuple(transformed_z)
    assert np.allclose(original_gram, transformed_gram, rtol=1e-8, atol=1e-8)


def test_all_seven_feature_dimensions_and_temporal_sensor_order() -> None:
    rng = np.random.default_rng(13)
    n = 7
    static = rng.normal(size=(n, 253))
    length = rng.normal(size=(n, 4))
    gram = rng.normal(size=(n, 10))
    sensor = rng.normal(size=(n, 1012))
    for condition in CONDITIONS:
        result = condition_features(condition, static, length, gram, sensor)
        assert result.shape == (n, DIMENSIONS[condition])
    combined = condition_features("STATIC_PLUS_SENSOR", static, length, gram, sensor)
    assert np.array_equal(combined[:, :253], static)
    assert np.array_equal(combined[:, 253:], sensor)


def test_feature_apis_have_no_label_input() -> None:
    assert feature_functions_are_label_free()


def test_scaler_and_classifier_fit_source_rows_only() -> None:
    rng = np.random.default_rng(14)
    source = rng.normal(size=(80, 10))
    labels = np.asarray(CLASS_ORDER * 20)
    target = np.full((12, 10), 1e6)
    uids = [f"source-{index}" for index in range(len(source))]
    scaler, model, audit = fit_fixed_source_decoder(
        source,
        labels,
        target_subject=1,
        condition="MOV_GRAM",
        source_trial_uids=uids,
        config=_config(),
    )
    assert np.allclose(scaler.mean_, source.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.concatenate([source, target]).mean(axis=0))
    assert audit.scaler_fit_n == len(source)
    assert audit.model_fit_n == len(source)
    assert audit.n_features == 10
    assert set(model.classes_) == set(CLASS_ORDER)


def test_target_labels_enter_only_scoring_boundary_by_signature() -> None:
    from src import trial_movement_utility_v0 as module

    assert "source_labels" in inspect.signature(module.fit_fixed_source_decoder).parameters
    assert "target_labels" not in inspect.signature(module.fit_fixed_source_decoder).parameters
    assert "class_label" not in inspect.signature(module._precompute_static).parameters
    assert "labels" not in inspect.signature(module._precompute_static).parameters


def test_exact_signflip_has_512_patterns_and_known_pvalue() -> None:
    result = exact_signflip_test(np.ones(9))
    assert result["n_patterns"] == 512
    assert result["p_raw_one_sided"] == 1 / 512


def test_holm_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_subject_aggregation_precedes_inference_and_loo_is_exact() -> None:
    deltas = np.arange(1.0, 10.0)
    summary = delta_summary(deltas)
    expected = [np.delete(deltas, index).mean() for index in range(9)]
    assert np.array_equal(summary["loo_means"], expected)
    assert summary["positive_subjects"] == 9
    assert summary["every_loo_mean_positive"]


def test_exact_bootstrap_constant_vector_collapses() -> None:
    interval = exact_subject_bootstrap_interval(np.full(9, 0.125))
    assert interval == (0.125, 0.125)


def test_terminal_precedence_and_breadth() -> None:
    primary = pd.DataFrame(
        [
            {"comparison": name, "mean_delta": 0.1, "median_delta": 0.1, "positive_subjects": 9, "every_loo_mean_positive": True}
            for name in ("DELTA_LEN", "DELTA_GRAM", "DELTA_SENSOR")
        ]
    )
    decision, states = terminal_decision(primary, {name: 0.01 for name in primary["comparison"]})
    assert decision == "GO_INVARIANT_MOVEMENT_INCREMENTAL_UTILITY"
    assert states["DELTA_GRAM"]["broad"]


def test_parent_manifest_excludes_new_output_namespace(tmp_path: Path) -> None:
    for directory in ("configs", "docs", "outputs/old", "outputs/bnci2014_001_trial_movement_utility_v0"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/a").write_text("a")
    (tmp_path / "docs/b").write_text("b")
    (tmp_path / "outputs/old/c").write_text("c")
    output = tmp_path / "outputs/bnci2014_001_trial_movement_utility_v0"
    (output / "new").write_text("new")
    manifest = _manifest(tmp_path, output)
    assert sorted(manifest) == ["configs/a", "docs/b", "outputs/old/c"]


def test_frozen_cache_loader_contains_no_write_call() -> None:
    from src.trial_movement_utility_v0 import load_frozen_inputs

    source = inspect.getsource(load_frozen_inputs)
    for forbidden in ("to_csv", "np.save", "write_text", "write_bytes", "open(\"w"):
        assert forbidden not in source


def test_no_forbidden_model_or_feature_family_in_frozen_config() -> None:
    text = (ROOT / "configs/bnci2014_001_trial_movement_utility_v0.yaml").read_text()
    lowered = text.lower()
    assert "logisticregression" in lowered
    assert "pca: false" in lowered
    for forbidden in ("spdnet", "transformer", "attention", "pseudo-label", "wasserstein"):
        assert forbidden not in lowered
