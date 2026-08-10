from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.conditional_nulls_v1 import all_s4_permutations
from src.dg_decomposition_v1 import (
    ABS_TOL,
    GEOMETRIES,
    OBJECTS,
    REL_TOL,
    class_weights_from_contract,
    derive_objects,
    distance_from_gram,
    distance_vector,
    gram_vector,
    identity_gate_table,
    load_source_matrices,
    relative_error,
    semantic_analysis,
    source_tree_snapshot,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/bnci2014_001_dg_decomposition_v1.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _synthetic_bundle():
    coordinates = np.asarray(
        [
            [-1.2, -0.2, 0.1],
            [0.8, -0.9, -0.3],
            [0.1, 1.1, -0.5],
            [0.3, 0.0, 0.7],
        ],
        dtype=np.float64,
    )
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    G = coordinates @ coordinates.T
    Dtan, _ = distance_from_gram(G)
    D = np.empty((2, 2, 9, 3, 4, 4), dtype=np.float64)
    grams = np.empty_like(D)
    for qi in range(2):
        for gi, geometry in enumerate(GEOMETRIES):
            for si in range(9):
                for hi in range(3):
                    factor = 1.0 + 0.01 * qi + 0.005 * si + 0.002 * hi
                    grams[qi, gi, si, hi] = factor * G
                    local, _ = distance_from_gram(factor * G)
                    D[qi, gi, si, hi] = local if geometry == "LE" else 1.03 * local
    weights = np.full((2, 9, 3, 4), 0.25)
    return derive_objects(D, grams, weights)


def test_d_tan_squared_reconstructs_from_g() -> None:
    bundle = _synthetic_bundle()
    diagonal = np.diagonal(bundle.G, axis1=-2, axis2=-1)
    expected = diagonal[..., :, None] + diagonal[..., None, :] - 2.0 * bundle.G
    expected = np.maximum(expected, 0.0)
    assert relative_error(bundle.D_tan**2, expected) <= REL_TOL


def test_le_exact_and_tangent_squared_identity() -> None:
    bundle = _synthetic_bundle()
    assert relative_error(bundle.D_exact[:, 1] ** 2, bundle.D_tan[:, 1] ** 2) <= REL_TOL


def test_g0_is_weighted_double_centering() -> None:
    bundle = _synthetic_bundle()
    for qi in range(2):
        H = bundle.H[qi, 0, 0]
        assert relative_error(bundle.G0[qi, 0, 0, 0], H @ bundle.G[qi, 0, 0, 0] @ H.T) <= REL_TOL


def test_tangent_distance_is_translation_invariant() -> None:
    bundle = _synthetic_bundle()
    reconstructed, _ = distance_from_gram(bundle.G0)
    assert relative_error(reconstructed, bundle.D_tan) <= REL_TOL


def test_balanced_le_g_equals_g0() -> None:
    bundle = _synthetic_bundle()
    assert relative_error(bundle.G[:, 1], bundle.G0[:, 1]) <= REL_TOL


def test_le_k_exact_equals_g0() -> None:
    bundle = _synthetic_bundle()
    assert relative_error(bundle.K_exact[:, 1], bundle.G0[:, 1]) <= REL_TOL


def test_derived_objects_are_permutation_equivariant() -> None:
    bundle = _synthetic_bundle()
    permutation = all_s4_permutations()[17]
    G = bundle.G[0, 0, 0, 0]
    D = bundle.D_exact[0, 0, 0, 0]
    pi = bundle.weights[0, 0, 0][permutation]
    H = np.eye(4) - np.ones((4, 1)) * pi[None, :]
    Gp = G[np.ix_(permutation, permutation)]
    Dp = D[np.ix_(permutation, permutation)]
    Dtanp, _ = distance_from_gram(Gp)
    assert relative_error(Dtanp, bundle.D_tan[0, 0, 0, 0][np.ix_(permutation, permutation)]) <= REL_TOL
    assert relative_error(H @ Gp @ H.T, bundle.G0[0, 0, 0, 0][np.ix_(permutation, permutation)]) <= REL_TOL
    assert relative_error(-0.5 * H @ (Dp**2) @ H.T, bundle.K_exact[0, 0, 0, 0][np.ix_(permutation, permutation)]) <= REL_TOL


def test_common_scaling_leaves_unit_shapes_unchanged() -> None:
    bundle = _synthetic_bundle()
    for name in OBJECTS:
        matrix = bundle.matrix(name)[0, 0, 0, 0]
        vectorize = distance_vector if name.startswith("D_") else gram_vector
        first = vectorize(matrix)
        second = vectorize(4.75 * matrix)
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)
        assert relative_error(first, second) <= REL_TOL


def test_k_exact_is_not_psd_clipped() -> None:
    bundle = _synthetic_bundle()
    direct = -0.5 * bundle.H[:, None] @ (bundle.D_exact**2) @ np.swapaxes(bundle.H[:, None], -1, -2)
    assert np.array_equal(bundle.K_exact, direct)
    assert not identity_gate_table(bundle)["K_exact_psd_clipped"].any()


def test_airm_curvature_residual_is_nonnegative() -> None:
    bundle = _synthetic_bundle()
    assert float(np.min(bundle.D_exact[:, 0] ** 2 - bundle.D_tan[:, 0] ** 2)) >= -ABS_TOL


def test_saved_original_shapes_reproduce_v1_tables() -> None:
    config = _config()
    D, G = load_source_matrices(REPO, config)
    weights = class_weights_from_contract(REPO, config)
    bundle = derive_objects(D, G, weights)
    source_root = REPO / config["project"]["source_output_dir"] / "tables"
    for name, filename, source_name in (
        ("D_exact", "D_shape_vectors.csv", "D"),
        ("G", "G_shape_vectors.csv", "G"),
    ):
        table = pd.read_csv(source_root / filename)
        for qi, phase in enumerate(("discovery", "confirmatory")):
            for gi, geometry in enumerate(GEOMETRIES):
                for si, subject in enumerate(range(1, 10)):
                    for hi, split in enumerate(("A", "B", "F")):
                        row = table[
                            (table.phase == phase)
                            & (table.geometry == geometry)
                            & (table.subject == subject)
                            & (table.split == split)
                        ].iloc[0]
                        expected = row[[column for column in table.columns if column.startswith("z_")]].to_numpy(dtype=float)
                        assert np.allclose(bundle.shapes_unit[name][qi, gi, si, hi], expected, rtol=REL_TOL, atol=ABS_TOL)


def test_small_semantic_null_uses_one_common_index_plan() -> None:
    bundle = _synthetic_bundle()
    result = semantic_analysis(bundle, replicates=7)
    assert result.null_subject.shape == (2, 2, 5, 7, 9)
    assert result.null_group.shape == (2, 2, 5, 7)
    # In synthetic LE, equal-information distance objects and Gram objects are
    # internally identical within their respective encoding families.
    assert np.array_equal(result.null_subject[:, 1, 0], result.null_subject[:, 1, 1])
    assert np.allclose(result.null_subject[:, 1, 2], result.null_subject[:, 1, 3], rtol=0.0, atol=2e-15)
    assert np.allclose(result.null_subject[:, 1, 3], result.null_subject[:, 1, 4], rtol=0.0, atol=2e-15)


def test_source_output_snapshot_is_unchanged_and_new_output_is_distinct() -> None:
    config = _config()
    source = REPO / config["project"]["source_output_dir"]
    records, aggregate = source_tree_snapshot(source)
    expected = config["source_contract"]["source_output_snapshot"]
    assert len(records) == expected["file_count"]
    assert sum(record["bytes"] for record in records) == expected["total_bytes"]
    assert aggregate == expected["aggregate_sha256"]
    assert config["project"]["output_dir"] != config["project"]["source_output_dir"]
