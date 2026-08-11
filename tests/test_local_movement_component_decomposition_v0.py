from __future__ import annotations

import numpy as np
import pytest

import src.local_movement_component_decomposition_v0 as decomposition
from src.local_temporal_sequence_v0 import classbreak_mappings, subjectbreak_mappings


def _component_matrices(seed: int = 1) -> decomposition.ComponentMatrices:
    rng = np.random.default_rng(seed)
    length = rng.uniform(0.0, 0.4, size=(36, 36))
    angular = rng.uniform(0.0, 0.6, size=(36, 36))
    orientation = rng.uniform(0.0, 0.8, size=(36, 36))
    full = length + angular
    sensor = full + orientation
    return decomposition.ComponentMatrices(
        sensor=sensor,
        full=full,
        length=length,
        angular=angular,
        orientation=orientation,
    )


def test_c_full_equals_c_len_plus_c_ang() -> None:
    values = _component_matrices()
    assert values.full == pytest.approx(values.length + values.angular, abs=0.0, rel=0.0)


def test_c_sensor_equals_c_full_plus_c_ori() -> None:
    values = _component_matrices()
    assert values.sensor == pytest.approx(values.full + values.orientation, abs=0.0, rel=0.0)


def test_full_three_way_reconstruction_and_pairwise_nonnegativity() -> None:
    values = _component_matrices()
    records = decomposition.pairwise_numerical_gates(
        values,
        independently_computed_sensor=values.sensor,
        independently_computed_length=values.length,
        d_mov=np.sqrt(values.full),
        d_len=np.sqrt(values.length),
        d_direct=np.sqrt(values.sensor),
    )
    assert values.sensor == pytest.approx(
        values.length + values.angular + values.orientation, abs=0.0, rel=0.0
    )
    assert all(int(row.get("meaningful_negative_count", 0)) == 0 for row in records)


def test_meaningful_negative_component_fails_without_clipping() -> None:
    values = _component_matrices()
    bad = decomposition.ComponentMatrices(
        sensor=values.sensor,
        full=values.full,
        length=values.length,
        angular=values.angular.copy(),
        orientation=values.orientation,
    )
    bad.angular[0, 0] = -1.0e-4
    with pytest.raises(
        decomposition.ComponentDecompositionNumericalError,
        match="meaningful negative c_ang",
    ):
        decomposition.pairwise_numerical_gates(
            bad,
            independently_computed_sensor=bad.sensor,
            independently_computed_length=bad.length,
            d_mov=np.sqrt(bad.full),
            d_len=np.sqrt(bad.length),
            d_direct=np.sqrt(bad.sensor),
        )
    assert bad.angular[0, 0] == -1.0e-4


def test_relation_S_C_J_linearity_and_J_level_exact_reconstruction() -> None:
    values = _component_matrices()
    statistics = {
        key: decomposition.relation_statistics(matrix)
        for key, matrix in values.as_dict().items()
    }
    records = decomposition.statistic_reconstruction_gates(statistics)
    assert len(records) == 18
    for field in ("s_sc", "c_specific_sc", "j_sc", "s_s", "c_s", "j_s"):
        assert getattr(statistics["full"], field) == pytest.approx(
            getattr(statistics["len"], field) + getattr(statistics["ang"], field),
            abs=1.0e-14,
            rel=1.0e-14,
        )
        assert getattr(statistics["sensor"], field) == pytest.approx(
            getattr(statistics["len"], field)
            + getattr(statistics["ang"], field)
            + getattr(statistics["ori"], field),
            abs=1.0e-14,
            rel=1.0e-14,
        )


def test_identical_indexed_null_permutations_and_null_reconstruction() -> None:
    values = _component_matrices()
    subject_maps = subjectbreak_mappings(replicates=7)
    class_maps = classbreak_mappings(replicates=7)
    inferences = decomposition.evaluate_all_components(
        values.as_dict(),
        subject_mappings=subject_maps,
        class_mappings=class_maps,
        replicates=7,
    )
    records = decomposition.null_reconstruction_gates(inferences)
    assert len(records) == 8
    assert inferences["full"].subjectbreak_t_j == pytest.approx(
        inferences["len"].subjectbreak_t_j + inferences["ang"].subjectbreak_t_j,
        abs=1.0e-14,
        rel=1.0e-14,
    )
    changed = subject_maps.copy()
    changed[0, 0], changed[0, 4] = changed[0, 4], changed[0, 0]
    with pytest.raises(ValueError, match="saved subject-break mappings differ"):
        decomposition.evaluate_all_components(
            values.as_dict(),
            subject_mappings=changed,
            class_mappings=class_maps,
            replicates=7,
        )


def test_root_distance_squared_reproduction_with_independent_sensor_and_length() -> None:
    rng = np.random.default_rng(4)
    session0 = rng.normal(scale=0.02, size=(36, 4, 22, 22))
    session1 = rng.normal(scale=0.02, size=(36, 4, 22, 22))
    session0 = 0.5 * (session0 + session0.transpose(0, 1, 3, 2))
    session1 = 0.5 * (session1 + session1.transpose(0, 1, 3, 2))
    sensor, length, _, _ = decomposition.sensor_and_length_squared_costs(session0, session1)
    angular = rng.uniform(0.0, 0.1, size=(36, 36))
    full = length + angular
    orientation = sensor - full
    shift = max(0.0, -float(np.min(orientation))) + 0.2
    sensor = sensor + shift
    orientation = sensor - full
    matrices = decomposition.build_component_matrices(
        np.sqrt(full), np.sqrt(length), np.sqrt(sensor)
    )
    decomposition.pairwise_numerical_gates(
        matrices,
        independently_computed_sensor=sensor,
        independently_computed_length=length,
        d_mov=np.sqrt(full),
        d_len=np.sqrt(length),
        d_direct=np.sqrt(sensor),
    )


def test_frozen_ordering_of_36_cells() -> None:
    assert decomposition.canonical_cell_subjects().tolist() == [
        subject for subject in range(1, 10) for _ in range(4)
    ]
    assert decomposition.canonical_cell_classes().tolist() == list(
        decomposition.CLASS_ORDER
    ) * 9
    assert decomposition.cell_index(0, 0) == 0
    assert decomposition.cell_index(8, 3) == 35


def test_split_half_A_B_isolation() -> None:
    values = np.empty((2, 2, 9, 4, 4, 22, 22), dtype=np.float64)
    for half in range(2):
        for session in range(2):
            values[half, session].fill(10.0 * half + session)
    banks = decomposition.split_replicate_banks(values)
    assert len(banks) == 2
    assert np.all(banks[0][0] == 0.0)
    assert np.all(banks[0][1] == 1.0)
    assert np.all(banks[1][0] == 10.0)
    assert np.all(banks[1][1] == 11.0)


def test_terminal_hierarchy_and_split_reliability_gate() -> None:
    kwargs = {
        "t_j_ang": 1.0,
        "p_j_ang_subjectbreak": 0.01,
        "p_j_ang_classbreak": 0.01,
        "t_j_ori": 1.0,
        "p_j_ori_subjectbreak": 0.01,
        "p_j_ori_classbreak": 0.01,
        "t_j_len": 1.0,
        "p_j_len_subjectbreak": 0.01,
        "p_j_len_classbreak": 0.01,
        "split_half_ang_sign_stable": True,
    }
    assert decomposition.terminal_decision(**kwargs) == (
        "BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS"
    )
    kwargs["split_half_ang_sign_stable"] = False
    assert decomposition.terminal_decision(**kwargs) == (
        "UNASSESSED_COMPONENT_DECOMPOSITION_UNRELIABLE"
    )
    kwargs["t_j_ang"] = -1.0
    assert decomposition.terminal_decision(**kwargs) == (
        "SENSOR_FRAME_ORIENTATION_ONLY_SUPPORT"
    )
