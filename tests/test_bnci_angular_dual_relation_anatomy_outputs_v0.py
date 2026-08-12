from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/bnci2014_001_angular_dual_relation_anatomy_v0"
PARENT = ROOT / "outputs/bnci2014_001_local_movement_component_decomposition_v0"


def test_saved_parent_matrix_is_exact_and_reconstruction_passes() -> None:
    with np.load(OUTPUT / "arrays/dual_relation_anatomy.npz", allow_pickle=False) as saved, np.load(
        PARENT / "arrays/component_cost_matrices.npz", allow_pickle=False
    ) as parent:
        assert np.array_equal(saved["D"], parent["c_ang_matrix"])
        assert np.array_equal(saved["A"], saved["A"].T)
    checks = pd.read_csv(OUTPUT / "tables/six_pair_reconstruction_checks.csv")
    assert checks["absolute_error"].max() <= 1e-12


def test_saved_G_H_diagonals_and_required_shapes() -> None:
    with np.load(OUTPUT / "arrays/dual_relation_anatomy.npz", allow_pickle=False) as saved:
        assert saved["G"].shape == (9, 4, 4)
        assert saved["H"].shape == (4, 9, 9)
        assert saved["J_subject_pair"].shape == (9, 6)
        for subject in range(9):
            for cls in range(4):
                assert saved["G"][subject, cls, cls] == saved["H"][cls, subject, subject]


def test_parent_artifacts_unchanged_and_manifest_valid() -> None:
    immutability = json.loads((OUTPUT / "provenance/parent_artifact_immutability.json").read_text())
    assert immutability["unchanged"] is True
    with (OUTPUT / "provenance/artifact_manifest.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = OUTPUT / row["relative_path"]
            data = path.read_bytes()
            assert len(data) == int(row["size_bytes"])
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
