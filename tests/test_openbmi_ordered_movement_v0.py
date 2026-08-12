from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

import src.openbmi_ordered_movement_v0 as replication


def _movement_bank(seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.random.default_rng(seed).normal(
        scale=0.1,
        size=(2, replication.N_SUBJECTS, replication.N_CLASSES, 4, 20, 20),
    )
    values = 0.5 * (values + values.swapaxes(-1, -2))
    return values[0], values[1]


def _spd_sequence(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.05, size=(5, 20, 20))
    values = 0.5 * (values + values.swapaxes(-1, -2))
    return np.asarray([expm(value) for value in values])


def test_temporal_contract_is_exact_five_by_half_second() -> None:
    assert 250 % replication.N_STATES == 0
    assert 250 // replication.N_STATES == 50
    assert (250 // replication.N_STATES) / 100.0 == replication.DELTA_T_SECONDS == 0.5


def test_canonical_108_cell_order_is_subject_then_binary_class() -> None:
    assert replication.cell_index(0, 0) == 0
    assert replication.cell_index(0, 1) == 1
    assert replication.cell_index(1, 0) == 2
    assert replication.cell_index(53, 1) == 107


def test_exact_squared_component_reconstruction_and_nonnegativity() -> None:
    first, second = _movement_bank(1)
    sensor, length = replication.sensor_and_length_costs(first, second)
    # Identity is a feasible common action, while the radial norm bound is exact.
    full = length + 0.4 * (sensor - length)
    components = replication.component_matrices(sensor, full, length)
    assert np.min(components["ang"]) >= -1.0e-12
    assert np.min(components["ori"]) >= -1.0e-12
    assert components["sensor"] == pytest.approx(
        components["len"] + components["ang"] + components["ori"], abs=1e-12, rel=1e-12
    )


def test_meaningful_negative_component_is_never_clipped() -> None:
    shape = (replication.N_CELLS, replication.N_CELLS)
    length = np.ones(shape)
    full = length.copy(); full[0, 0] = 0.5
    sensor = np.full(shape, 2.0)
    with pytest.raises(replication.OpenBMINumericalError):
        replication.component_matrices(sensor, full, length)


def test_binary_relation_statistics_use_exact_other_class_and_53_subject_means() -> None:
    matrix = np.empty((replication.N_CELLS, replication.N_CELLS))
    for subject in range(replication.N_SUBJECTS):
        for class_index in range(replication.N_CLASSES):
            row = replication.cell_index(subject, class_index)
            for other_subject in range(replication.N_SUBJECTS):
                for other_class in range(replication.N_CLASSES):
                    column = replication.cell_index(other_subject, other_class)
                    if subject == other_subject and class_index == other_class:
                        matrix[row, column] = 0.0
                    elif subject == other_subject:
                        matrix[row, column] = 1.0
                    elif class_index == other_class:
                        matrix[row, column] = 2.0
                    else:
                        matrix[row, column] = 2.5
    result = replication.relation_statistics(matrix)
    assert result.t_subject == pytest.approx(2.0)
    assert result.t_class == pytest.approx(1.0)
    assert result.t_j == pytest.approx(0.5)


def test_null_streams_are_exact_deterministic_and_preserve_binary_semantics() -> None:
    subject = replication.subjectbreak_mappings()
    classes = replication.classbreak_mappings()
    assert subject.shape == classes.shape == (1999, 108)
    assert np.array_equal(subject, replication.subjectbreak_mappings())
    assert np.array_equal(classes, replication.classbreak_mappings())
    for draw in (0, 1, 1998):
        for class_index in range(2):
            mapped = subject[draw, class_index::2]
            assert np.array_equal(np.sort(mapped), np.arange(class_index, 108, 2))
        for cell in range(108):
            assert classes[draw, cell] // 2 == cell // 2
        for subject_index in range(54):
            mapped = classes[draw, 2 * subject_index : 2 * subject_index + 2]
            assert set(mapped.tolist()) == {2 * subject_index, 2 * subject_index + 1}


def test_d20_antidevelopment_inherits_frozen_geometry_and_half_second_scaling() -> None:
    result = replication.anti_develop_sequence(_spd_sequence(2), delta_t=0.5)
    assert result.z.shape == (4, 20, 20)
    assert result.diagnostics["passed"].all()
    assert np.max(np.abs(result.speeds - result.diagnostics["expected_airm_speed"])) < 1e-10


def test_primary_terminal_cannot_be_rescued_by_secondary_results() -> None:
    assert replication.terminal(1.0, 0.01, 0.01) == "REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION"
    assert replication.terminal(1.0, 0.01, 0.05) == "NOT_REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION"
    assert replication.terminal(-1.0, 0.001, 0.001) == "NOT_REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION"
