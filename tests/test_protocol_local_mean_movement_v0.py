from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import yaml
from pymanopt.optimizers import TrustRegions

import src.local_mean_movement_v0 as movement
from src.reporting_local_mean_movement_v0 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bnci2014_001_local_mean_movement_v0.yaml"
PROTOCOL = ROOT / "docs" / "PROTOCOL_LOCAL_MEAN_MOVEMENT_V0.md"
RUNNER = ROOT / "scripts" / "26_run_local_mean_movement_v0.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("local_mean_movement_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_hash_lineage_namespace_and_delta_t_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["sha256"] == sha256_file(PROTOCOL)
    assert config["protocol"]["branch"] == "pilot/local-mean-movement-antidevelopment-v0"
    assert config["project"]["output_dir"] == "outputs/bnci2014_001_local_mean_movement_v0"
    assert config["lineage"]["temporal_protocol_freeze_sha"] == "70981aa89ddbadceca42f354c3c51d05bf6dbf0c"
    assert config["lineage"]["temporal_scientific_result_sha"] == "43e926073fab0ba76fd5baa881804538f0d7beee"
    assert config["lineage"]["temporal_final_sha"] == "6d5ad6a0bdd4f2d19bfee8ce6fcbb97a5c499a5d"
    assert config["movement"]["delta_t_seconds"] == movement.DELTA_T_SECONDS == 0.8


def test_runtime_optimizer_is_frozen_trustregions_with_six_two_sector_starts() -> None:
    settings = movement.FROZEN_OPTIMIZER_SETTINGS
    optimizer = movement._movement_optimizer(settings)
    assert isinstance(optimizer, TrustRegions)
    assert settings.total_starts == 6
    assert optimizer._max_iterations == 250
    assert optimizer._max_time == 120.0
    assert optimizer._min_gradient_norm == 1.0e-6
    assert optimizer._min_step_size == 1.0e-12
    assert optimizer._max_cost_evaluations == 5000
    assert optimizer.miniter == 3
    assert optimizer.kappa == 0.1
    assert optimizer.theta == 1.0
    assert optimizer.rho_prime == 0.1
    assert optimizer.use_rand is False
    assert optimizer.rho_regularization == 1000.0


def test_prepare_cannot_compute_a_bnci_movement_representation_or_statistic() -> None:
    source = inspect.getsource(_runner_module().prepare)
    assert "reproduce_frozen_temporal_means" in source
    assert "run_synthetic_numerical_gates" in source
    assert "anti_develop_sequence" not in source
    assert "_compute_primary_quotient" not in source
    assert "evaluate_movement_inference" not in source


def test_protocol_forbids_step_permutation_and_freezes_split_nulls_controls() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "No S4" in text
    assert "Independent radial transport to `M1` is forbidden" in text
    assert "exactly 1,999 inherited subject-break" in text
    assert "exactly 1,999 inherited" in text
    assert "Half A runs `{0,2,4}`" in text
    assert "non-gating" in text
    assert "magnitude-only control" in text
    assert "direct montage-registered control" in text


def test_terminal_and_claim_restriction_text_is_complete() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for terminal in (
        "GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT",
        "GO_REPRODUCIBLE_ORDERED_MOVEMENT_WITHOUT_INTERACTION",
        "STOP_NO_REPRODUCIBLE_ORDERED_MOVEMENT_V0",
        "UNASSESSED_MOVEMENT_GEOMETRY_NUMERICAL_FAILURE",
        "UNASSESSED_MOVEMENT_QUOTIENT_NUMERICAL_FAILURE",
        "UNASSESSED_TECHNICAL_FAILURE",
    ):
        assert terminal in text
    for phrase in (
        "individual-trial neural velocity",
        "continuous-time dynamics",
        "source-space dynamics",
        "absolute subject pose",
        "biological privilege of AIRM",
    ):
        assert phrase in text


def test_clean_room_implementation_does_not_reference_cancelled_module_or_namespace() -> None:
    paths = (
        ROOT / "src" / "local_mean_movement_v0.py",
        ROOT / "src" / "reporting_local_mean_movement_v0.py",
        RUNNER,
        CONFIG,
        PROTOCOL,
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "ordered_airm_movement_v0" not in combined
    assert "bnci2014_001_ordered_airm_movement_v0" not in combined
