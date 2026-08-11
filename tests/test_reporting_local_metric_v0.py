from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.local_metric_interaction_v0 import (
    CLASS_ORDER,
    evaluate_interaction_nulls,
    synthetic_additive_cell_matrix,
)
from src.local_metric_pipeline_v0 import CellMetricMatrices, DecodingResult
from src.reporting_local_metric_v0 import (
    write_artifact_manifest,
    write_report,
    write_scientific_outputs,
)


def test_synthetic_output_contract_and_figures(tmp_path: Path) -> None:
    raw_matrix = synthetic_additive_cell_matrix(
        subject_effect=0.7, class_effect=0.4, interaction_effect=0.3
    )
    size_matrix = synthetic_additive_cell_matrix(
        subject_effect=0.2, class_effect=0.1, interaction_effect=0.05
    )
    normalized_matrix = synthetic_additive_cell_matrix(
        subject_effect=0.5, class_effect=0.3, interaction_effect=0.2
    )
    results = {
        "raw": evaluate_interaction_nulls(raw_matrix, replicates=19),
        "size": evaluate_interaction_nulls(size_matrix, replicates=19),
        "normalized": evaluate_interaction_nulls(normalized_matrix, replicates=19),
    }
    matrices = CellMetricMatrices(
        raw_m01=raw_matrix,
        raw_m=raw_matrix,
        size_m01=size_matrix,
        size_m=size_matrix,
        normalized_m01=normalized_matrix,
        normalized_m=normalized_matrix,
    )
    reliability = pd.DataFrame(
        [
            {
                "subject": subject,
                "session": session,
                "class_label": class_label,
                "n_trials": 72,
                "median_within_cell_delta_raw": 0.5,
                "q25_within_cell_delta_raw": 0.4,
                "q75_within_cell_delta_raw": 0.6,
                "iqr_within_cell_delta_raw": 0.2,
                "medoid_objective": 0.45,
                "medoid_global_sample_index": 0,
                "medoid_trial_uid": "synthetic",
            }
            for subject in range(1, 10)
            for session in ("0train", "1test")
            for class_label in CLASS_ORDER
        ]
    )
    subject_scores = pd.DataFrame(
        {
            "subject": np.arange(1, 10),
            "ba_session0_to_session1": np.full(9, 0.5),
            "ba_session1_to_session0": np.full(9, 0.5),
            "balanced_accuracy": np.full(9, 0.5),
        }
    )
    decoding = DecodingResult(
        subject_scores=subject_scores,
        group_mean_ba=0.5,
        group_median_ba=0.5,
        null_group_median_ba=np.linspace(0.2, 0.3, 19),
        p_value=0.05,
    )
    provenance = {
        "protocol_freeze_sha": "f" * 40,
        "protocol_sha256": "a" * 64,
        "scientific_source_sha": "b" * 40,
        "config_sha256": "c" * 64,
        "implementation_source_sha256": "d" * 64,
    }
    decision = write_scientific_outputs(
        tmp_path,
        matrices=matrices,
        results=results,
        reliability=reliability,
        decoding=decoding,
        provenance=provenance,
        total_runtime_seconds=1.25,
    )
    write_report(
        tmp_path / "report" / "local_metric_interaction_v0.md",
        branch="pilot/synthetic",
        result_commit="pending",
        reproduction={
            "trial_count": 5184,
            "upper_triangle_path_max_abs_diff": 0.0,
            "degenerate_trial_count": 0,
        },
        results=results,
        reliability=reliability,
        decoding=decoding,
        decision=decision,
        tests_summary="PASS",
        repository_tests_summary="PASS",
        git_status="clean",
    )
    write_artifact_manifest(tmp_path, provenance)
    assert (tmp_path / "decisions" / "terminal_decision.json").is_file()
    assert (tmp_path / "tables" / "cell_distances.csv").is_file()
    assert (tmp_path / "report" / "local_metric_interaction_v0.md").is_file()
    assert (tmp_path / "protocol" / "artifact_manifest.csv").is_file()
    for index in range(1, 9):
        assert len(list((tmp_path / "figures").glob(f"figure_{index}_*.csv"))) == 1
        assert len(list((tmp_path / "figures").glob(f"figure_{index}_*.png"))) == 1
        assert len(list((tmp_path / "figures").glob(f"figure_{index}_*.pdf"))) == 1
