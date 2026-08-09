"""Synthetic integration contract for the discovery-only v1 producers."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.conditional_geometry_v1 import (
    DegenerateClassGeometryError,
    NumericalGateError,
)
from src.conditional_pipeline_v1 import (
    COMMON_COLUMNS,
    FAILURE_DECISION_KEYS,
    FAILURE_MANIFEST_KEYS,
    ConditionalPipelineError,
    classify_recognized_phase_failure,
    compute_phase_geometry,
    run_confirmatory_oracle_and_finalize,
    run_confirmatory_rs_producer,
    run_discovery_null_producer,
    run_discovery_label_dry_run,
    run_label_null_checkpointed,
    validate_confirmatory_snapshot_contract,
    validate_label_null_dry_run,
    validate_discovery_snapshot_contract,
    write_unassessed_failure_artifacts,
    write_phase_geometry_outputs,
)
from src.conditional_nulls_v1 import PHASE_DISCOVERY
from src.conditional_provenance_v1 import payload_sha256
from src.data_conditional_v1 import ConditionalWholeData, DataContractError


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_discovery(tmp_path: Path) -> tuple[ConditionalWholeData, dict[str, object]]:
    config = deepcopy(
        yaml.safe_load(
            (ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    config["project"]["output_dir"] = "outputs/synthetic_conditional"
    config["project"]["cache_dir"] = "cache/synthetic_conditional"
    config["statistics"]["bootstrap"]["replicates"] = 8
    config["nulls"]["label_destruction"]["batch_size"] = 1
    config["nulls"]["semantic_permutation"]["batch_size"] = 2
    config["nulls"]["oracle_rank"]["batch_size"] = 3
    config["nulls"]["label_destruction"]["replicates"] = 2
    config["nulls"]["semantic_permutation"]["replicates"] = 3
    config["nulls"]["oracle_rank"]["replicates"] = 5
    config["expected_data"]["covariance_shape_per_session"] = [2592, 2, 2]

    classes = list(config["dataset"]["classes"])
    class_logs = np.asarray(
        [[0.24, -0.03], [-0.08, 0.21], [-0.18, -0.07], [0.06, -0.19]],
        dtype=np.float64,
    )
    covariance_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    sample_index = 0
    for subject in range(1, 10):
        trial_id = 0
        for run in range(6):
            for run_trial in range(48):
                class_index = run_trial // 12
                trial_id += 1
                nuisance = np.asarray(
                    [0.015 * subject + 0.002 * run, -0.011 * subject + 0.001 * run]
                )
                within = 0.0002 * ((run_trial % 12) - 5.5) * np.asarray([1.0, -0.7])
                covariance_rows.append(np.diag(np.exp(nuisance + class_logs[class_index] + within)))
                metadata_rows.append(
                    {
                        "covariance_index": sample_index,
                        "sample_index": sample_index,
                        "subject": subject,
                        "session": "0train",
                        "run": str(run),
                        "trial_id": trial_id,
                        "run_trial_id": run_trial + 1,
                        "trial_uid": f"S{subject:02d}_0train_T{trial_id:03d}",
                        "class_label": classes[class_index],
                    }
                )
                sample_index += 1
    data = ConditionalWholeData(
        session="0train",
        covariances=np.asarray(covariance_rows, dtype=np.float64),
        metadata=pd.DataFrame(metadata_rows),
        channel_names=np.asarray(["synthetic_1", "synthetic_2"]),
        provenance={"synthetic": True},
    )
    return data, config


def _as_synthetic_confirmatory(data: ConditionalWholeData) -> ConditionalWholeData:
    metadata = data.metadata.copy()
    metadata["session"] = "1test"
    metadata["trial_uid"] = metadata["trial_uid"].str.replace(
        "_0train_", "_1test_", regex=False
    )
    return ConditionalWholeData(
        session="1test",
        covariances=data.covariances.copy(),
        metadata=metadata,
        channel_names=data.channel_names.copy(),
        provenance={"synthetic": True, "phase": "confirmatory"},
    )


def _produce_complete_discovery_snapshot(
    tmp_path: Path,
) -> tuple[dict[str, object], Path]:
    data, config = _synthetic_discovery(tmp_path)
    bundle, geometry_tables = compute_phase_geometry(
        data,
        config,
        config_sha256="c" * 64,
        code_commit="a" * 40,
        phase="discovery",
        phase_tag=0,
    )
    write_phase_geometry_outputs(bundle, geometry_tables, config, tmp_path)
    run_discovery_label_dry_run(
        data, bundle, config, tmp_path, replicates=1, workers=1
    )
    run_discovery_null_producer(
        data,
        bundle,
        config,
        tmp_path,
        workers=1,
        label_replicates=2,
        semantic_replicates=3,
        oracle_replicates=5,
    )
    return config, tmp_path / "outputs/synthetic_conditional"


def test_discovery_producer_exact_snapshot_contract_and_resume(tmp_path: Path) -> None:
    data, config = _synthetic_discovery(tmp_path)
    bundle, geometry_tables = compute_phase_geometry(
        data,
        config,
        config_sha256="c" * 64,
        code_commit="a" * 40,
        phase="discovery",
        phase_tag=0,
    )
    assert bundle.all_gates_passed
    write_phase_geometry_outputs(bundle, geometry_tables, config, tmp_path)
    run_discovery_label_dry_run(
        data, bundle, config, tmp_path, replicates=1, workers=1
    )
    assert validate_label_null_dry_run(config, tmp_path, bundle)["replicates"] == 1
    first = run_discovery_null_producer(
        data,
        bundle,
        config,
        tmp_path,
        workers=1,
        label_replicates=2,
        semantic_replicates=3,
        oracle_replicates=5,
    )
    second = run_discovery_null_producer(
        data,
        bundle,
        config,
        tmp_path,
        workers=1,
        label_replicates=2,
        semantic_replicates=3,
        oracle_replicates=5,
    )
    np.testing.assert_array_equal(first.label_group_statistics, second.label_group_statistics)
    np.testing.assert_array_equal(first.semantic_group_statistics, second.semantic_group_statistics)
    np.testing.assert_array_equal(first.oracle_group_statistics, second.oracle_group_statistics)

    output = tmp_path / "outputs/synthetic_conditional"
    object_files = sorted(path.name for path in (output / "objects/discovery").iterdir())
    assert object_files == sorted(
        [
            "airm_marginal_means.npz",
            "airm_class_means.npz",
            "le_marginal_means.npz",
            "le_class_means.npz",
            "D_matrices.npz",
            "G_matrices.npz",
        ]
    )
    assert len(list((output / "tables/discovery").glob("*.csv"))) == 23
    null_files = sorted(path.name for path in (output / "nulls/discovery").iterdir())
    assert null_files == [
        "label_destruction_group_statistics.npz",
        "oracle_rank_null.npz",
        "semantic_permutation_group_statistics.npz",
    ]
    expected_replicates = {
        "label_destruction_group_statistics.npz": 2,
        "semantic_permutation_group_statistics.npz": 3,
        "oracle_rank_null.npz": 5,
    }
    for filename, count in expected_replicates.items():
        with np.load(output / "nulls/discovery" / filename, allow_pickle=False) as archive:
            assert set(archive.files) == {
                "replicate_indices",
                "geometries",
                "objects",
                "group_statistics",
            }
            assert archive["group_statistics"].shape == (2, 2, count)
    group = pd.read_csv(output / "tables/discovery/label_destruction_group_summary.csv")
    assert group.columns[: len(COMMON_COLUMNS)].tolist() == list(COMMON_COLUMNS)
    assert group.shape[0] == 4
    assert set(group["phase"]) == {"discovery"}
    assert set(group["session"]) == {"0train"}
    dry_audit = yaml.safe_load(
        (output / "protocol/label_null_dry_run.json").read_text(encoding="utf-8")
    )["airm_scalar_crosscheck"]
    assert dry_audit["authoritative_solver"] == "pyriemann.geometry.mean.mean_riemann"
    assert dry_audit["official_all_72_pass"] is True
    assert dry_audit["groups_checked"] == 72
    validated = validate_discovery_snapshot_contract(
        config,
        tmp_path,
        config_sha256="c" * 64,
        code_commit="a" * 40,
    )
    assert validated["status"] == "PASS"
    assert validated["table_file_count"] == 23


def test_discovery_scripts_do_not_name_or_import_confirmatory_access() -> None:
    for filename in (
        "21_discovery_conditional_geometry.py",
        "22_discovery_nulls.py",
    ):
        source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "1test" not in source
        assert "load_confirmatory" not in source
        assert "prepare_confirmatory" not in source


def test_confirmatory_snapshot_finalizer_and_paired_outputs(tmp_path: Path) -> None:
    discovery_data, config = _synthetic_discovery(tmp_path)
    discovery, discovery_tables = compute_phase_geometry(
        discovery_data,
        config,
        config_sha256="c" * 64,
        code_commit="a" * 40,
        phase="discovery",
        phase_tag=0,
    )
    write_phase_geometry_outputs(discovery, discovery_tables, config, tmp_path)
    run_discovery_label_dry_run(
        discovery_data, discovery, config, tmp_path, replicates=1, workers=1
    )
    run_discovery_null_producer(
        discovery_data,
        discovery,
        config,
        tmp_path,
        workers=1,
        label_replicates=2,
        semantic_replicates=3,
        oracle_replicates=5,
    )

    confirmatory_data = _as_synthetic_confirmatory(discovery_data)
    confirmatory, confirmatory_tables = compute_phase_geometry(
        confirmatory_data,
        config,
        config_sha256="c" * 64,
        code_commit="a" * 40,
        phase="confirmatory",
        phase_tag=1,
    )
    write_phase_geometry_outputs(confirmatory, confirmatory_tables, config, tmp_path)
    run_confirmatory_rs_producer(
        confirmatory_data,
        discovery,
        confirmatory,
        config,
        tmp_path,
        workers=1,
        label_replicates=2,
        semantic_replicates=3,
    )
    result = run_confirmatory_oracle_and_finalize(
        discovery,
        confirmatory,
        config,
        tmp_path,
        config_sha256="c" * 64,
        code_commit="a" * 40,
        oracle_replicates=5,
    )
    assert result["status"] == "PASS"
    assert result["terminal_decision"] in {
        "GO_STRONG",
        "GO_METRIC_ONLY",
        "STOP_TANGENT_ONLY",
        "STOP_NO_SHARED_GEOMETRY",
    }

    output = tmp_path / "outputs/synthetic_conditional"
    validated = validate_confirmatory_snapshot_contract(
        config,
        tmp_path,
        config_sha256="c" * 64,
        code_commit="a" * 40,
    )
    assert validated["table_file_count"] == 23
    assert len(list((output / "objects/confirmatory").glob("*.npz"))) == 6
    assert len(list((output / "nulls/confirmatory").glob("*.npz"))) == 3
    comparison = pd.read_csv(output / "tables/discovery_confirmatory_comparison.csv")
    assert len(comparison) == 12
    assert set(comparison["phase"]) == {"combined"}
    assert set(comparison["session"]) == {"0train+1test"}

    bootstrap = pd.read_csv(output / "tables/subject_bootstrap_summary.csv")
    confirm_bootstrap = bootstrap.loc[bootstrap["phase"] == "confirmatory"]
    assert confirm_bootstrap[
        "discovery_confirmatory_effect_delta_median"
    ].notna().all()
    assert confirm_bootstrap.loc[
        confirm_bootstrap["geometry"] == "AIRM",
        "airm_minus_le_effect_delta_median",
    ].notna().all()
    assert confirm_bootstrap.loc[
        confirm_bootstrap["geometry"] == "LE",
        "airm_minus_le_effect_delta_median",
    ].isna().all()
    deltas = pd.read_csv(output / "tables/leave_one_subject_out_influence.csv")
    confirm_deltas = deltas.loc[deltas["phase"] == "confirmatory"]
    assert len(confirm_deltas) == 108
    assert confirm_deltas["discovery_confirmatory_effect_delta"].notna().all()
    np.testing.assert_allclose(
        confirm_deltas["discovery_confirmatory_effect_delta"],
        confirm_deltas["confirmatory_subject_effect"]
        - confirm_deltas["discovery_subject_effect"],
        rtol=0.0,
        atol=1.0e-15,
    )

    # Script-level barrier: unlock validation must textually precede the only
    # confirmatory loader call in every raw-data entry point.
    for filename in (
        "24_confirmatory_conditional_geometry.py",
        "25_confirmatory_nulls.py",
        "26_oracle_semantic_test.py",
    ):
        source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        unlock_at = source.index("validate_confirmatory_unlock(", source.index("def main"))
        loader_positions = [
            source.index(token, source.index("def main"))
            for token in ("load_confirmatory_whole(",)
            if token in source[source.index("def main") :]
        ]
        if loader_positions:
            assert unlock_at < min(loader_positions)


def test_unassessed_failure_artifacts_are_exact_and_null_free(tmp_path: Path) -> None:
    _data, config = _synthetic_discovery(tmp_path)
    decision = write_unassessed_failure_artifacts(
        config,
        tmp_path,
        config_sha256="c" * 64,
        code_commit="a" * 40,
        phase="discovery",
        session="0train",
        failure_class="data",
        reason_code="DataContractError",
        reason="synthetic count mismatch",
    )
    output = tmp_path / "outputs/synthetic_conditional"
    manifest_path = output / "protocol/discovery_failure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root_decision = json.loads(
        (output / "confirmatory_decision.json").read_text(encoding="utf-8")
    )
    assert set(manifest) == set(FAILURE_MANIFEST_KEYS)
    assert set(root_decision) == set(FAILURE_DECISION_KEYS)
    assert manifest["terminal_decision"] == "UNASSESSED_DATA_CONTRACT_FAILURE"
    assert manifest["scientific_nulls_executed"] is False
    assert manifest["downstream_phase_permitted"] is False
    assert root_decision == decision
    assert not (output / "nulls").exists()


@pytest.mark.parametrize(
    ("error", "expected_class"),
    [
        (DataContractError("bad data"), "data"),
        (NumericalGateError("bad mean"), "numerical"),
        (DegenerateClassGeometryError("DEGENERATE_CLASS_GEOMETRY:D"), "degenerate"),
        (ConditionalPipelineError("geometry gate failed"), "numerical"),
    ],
)
def test_recognized_failure_classification(error: BaseException, expected_class: str) -> None:
    assert classify_recognized_phase_failure(error) == expected_class


def _load_script21_module() -> object:
    path = ROOT / "scripts/21_discovery_conditional_geometry.py"
    spec = importlib.util.spec_from_file_location("conditional_script21_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script21_records_recognized_failure_but_reraises_unexpected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _data, config = _synthetic_discovery(tmp_path)
    module = _load_script21_module()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(config=tmp_path / "config.yaml", repo_root=tmp_path),
    )
    monkeypatch.setattr(
        module,
        "load_conditional_config",
        lambda *_args, **_kwargs: (
            config,
            tmp_path / "config.yaml",
            "c" * 64,
            str(config["protocol"]["protocol_sha256"]),
        ),
    )
    monkeypatch.setattr(module, "validate_frozen_protocol_outputs", lambda *_a, **_k: {})
    monkeypatch.setattr(module, "producer_code_commit", lambda *_a, **_k: "a" * 40)
    monkeypatch.setattr(
        module,
        "load_discovery_whole",
        lambda *_a, **_k: (_ for _ in ()).throw(DataContractError("synthetic bad data")),
    )
    assert module.main() == 2
    failure = json.loads(
        (
            tmp_path
            / "outputs/synthetic_conditional/protocol/discovery_failure_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["failure_class"] == "data"

    monkeypatch.setattr(
        module,
        "load_discovery_whole",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("unexpected bug")),
    )
    with pytest.raises(RuntimeError, match="unexpected bug"):
        module.main()


def test_checkpoint_rejects_changed_frozen_code_snapshot(tmp_path: Path) -> None:
    data, config = _synthetic_discovery(tmp_path)
    bundle, tables = compute_phase_geometry(
        data,
        config,
        config_sha256="c" * 64,
        code_commit="a" * 40,
        phase="discovery",
        phase_tag=0,
    )
    write_phase_geometry_outputs(bundle, tables, config, tmp_path)
    output = tmp_path / "outputs/synthetic_conditional"
    implementation = tmp_path / "src/frozen_impl.py"
    implementation.parent.mkdir(parents=True)
    original_bytes = b"VALUE = 'frozen'\n"
    implementation.write_bytes(original_bytes)
    record = {
        "path": "src/frozen_impl.py",
        "bytes": len(original_bytes),
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    snapshot = {
        "algorithm": "sha256_canonical_json_sorted_file_records_v1",
        "file_count": 1,
        "total_bytes": len(original_bytes),
        "files": [record],
        "aggregate_sha256": payload_sha256([record]),
    }
    (output / "git_provenance.json").write_text(
        json.dumps({"frozen_code_snapshot": snapshot}),
        encoding="utf-8",
    )
    run_label_null_checkpointed(
        data,
        bundle,
        config,
        tmp_path,
        phase_tag=PHASE_DISCOVERY,
        total_replicates=1,
        workers=1,
    )
    implementation.write_text("VALUE = 'dirty poison'\n", encoding="utf-8")
    with pytest.raises(
        ConditionalPipelineError, match="current code bytes differ from frozen snapshot"
    ):
        run_label_null_checkpointed(
            data,
            bundle,
            config,
            tmp_path,
            phase_tag=PHASE_DISCOVERY,
            total_replicates=1,
            workers=1,
        )
    # Reverting the source permits the original, correctly bound checkpoint to
    # resume; the dirty call never opened or mutated it.
    implementation.write_bytes(original_bytes)
    groups, subjects = run_label_null_checkpointed(
        data,
        bundle,
        config,
        tmp_path,
        phase_tag=PHASE_DISCOVERY,
        total_replicates=1,
        workers=1,
    )
    assert groups.shape == (2, 2, 1)
    assert subjects.shape == (2, 2, 1, 9)


@pytest.mark.parametrize("tamper", ["D", "G", "class_mean", "D_shape"])
def test_snapshot_validator_recomputes_geometry_linkage(
    tmp_path: Path, tamper: str
) -> None:
    config, output = _produce_complete_discovery_snapshot(tmp_path)
    if tamper in {"D", "G"}:
        path = output / f"objects/discovery/{tamper}_matrices.npz"
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.array(archive[key], copy=True) for key in archive.files}
        payload["matrices"][0, 0, 0, 0, 1] += 0.01
        payload["matrices"][0, 0, 0, 1, 0] += 0.01
    elif tamper == "class_mean":
        path = output / "objects/discovery/airm_class_means.npz"
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.array(archive[key], copy=True) for key in archive.files}
        payload["class_means"][0, 0, 0] *= 1.01
    else:
        path = output / "tables/discovery/D_shape_vectors.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "raw_0"] = float(frame.loc[0, "raw_0"]) + 0.01
        frame.to_csv(path, index=False)
        payload = None
    if payload is not None:
        np.savez_compressed(path, **payload)
    with pytest.raises(
        ConditionalPipelineError,
        match=(
            "shape vector mismatch"
            if tamper == "D_shape"
            else "inconsistent with saved means"
        ),
    ):
        validate_discovery_snapshot_contract(
            config,
            tmp_path,
            config_sha256="c" * 64,
            code_commit="a" * 40,
        )
