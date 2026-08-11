"""V2 optimizer-only wrapper around the frozen pairwise scientific pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pymanopt
import pandas as pd
import yaml
from pymanopt.manifolds import Stiefel

from src import pairwise_common_action_pipeline_v1 as pipeline_v1
from src.common_action_pipeline_v0 import sha256_file
from src.common_action_solver_v0 import build_pymanopt_trust_regions_optimizer
from src.common_action_solver_v0 import ActionSolverError
from src.pairwise_common_action_v2 import (
    NULL_REPLICATES,
    PAIRWISE_V2_SETTINGS,
    PairwiseContractError,
    fit_pairwise_action_v2,
)


CONFIG_PATH = "configs/bnci2014_001_pairwise_common_action_v2.yaml"
PROTOCOL_PATH = "docs/PROTOCOL_COMMON_SUBJECT_ACTION_PAIRWISE_V2.md"
# Filled after final pre-data content review and before the amendment commit.
EXPECTED_CONFIG_SHA256 = "5980cdb336358602b7c45d21dfa18c01e4c6bbf26cfaafb6e812f35d09d92ad6"
EXPECTED_PROTOCOL_SHA256 = "c6856ed0dd16c3b76ef5232735e0d894f561ac9cd84a758c4d3e8d62c365677d"
SOURCE_PATHS = (
    "src/common_action_solver_v0.py",
    "src/pairwise_common_action_v1.py",
    "src/pairwise_common_action_v2.py",
    "src/pairwise_common_action_pipeline_v1.py",
    "src/pairwise_common_action_pipeline_v2.py",
)
OPTIMIZER_IDENTITY = "pymanopt_2.2.1_stiefel_trust_regions"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def combined_source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_config(root: Path) -> dict[str, Any]:
    config_path = root / CONFIG_PATH
    protocol_path = root / PROTOCOL_PATH
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise PairwiseContractError("frozen V2 config content hash mismatch")
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise PairwiseContractError("frozen V2 protocol content hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    optimizer = config["optimizer"]
    expected = {
        "library": "pymanopt",
        "version": pymanopt.__version__,
        "optimizer_class": "pymanopt.optimizers.TrustRegions",
        "total_starts": PAIRWISE_V2_SETTINGS.starts,
        "max_iterations": PAIRWISE_V2_SETTINGS.max_iterations,
        "min_gradient_norm": PAIRWISE_V2_SETTINGS.gradient_tolerance,
        "min_step_size": PAIRWISE_V2_SETTINGS.optimizer_min_step_size,
        "max_cost_evaluations": PAIRWISE_V2_SETTINGS.max_iterations + 1,
        "max_time_seconds": PAIRWISE_V2_SETTINGS.optimizer_max_time_seconds,
        "log_verbosity": 0,
    }
    for key, value in expected.items():
        if optimizer[key] != value:
            raise PairwiseContractError(f"runtime/config V2 optimizer mismatch: {key}")
    trust = optimizer["trust_regions"]
    expected_trust = {
        "miniter": 3,
        "kappa": 0.1,
        "theta": 1.0,
        "rho_prime": 0.1,
        "use_rand": False,
        "rho_regularization": 1000.0,
        "run_mininner": 1,
        "run_maxinner": 231,
        "run_Delta_bar": math.sqrt(22.0),
        "run_Delta0": math.sqrt(22.0) / 8.0,
    }
    for key, value in expected_trust.items():
        if trust[key] != value:
            raise PairwiseContractError(
                f"runtime/config TrustRegions mismatch: {key}"
            )
    if config["nulls"]["replicates"] != NULL_REPLICATES:
        raise PairwiseContractError("runtime/config V2 null-replicate mismatch")
    if config["project"]["v1_cache_reuse_allowed"] is not False:
        raise PairwiseContractError("V1 cache reuse must be forbidden")
    manifold = Stiefel(22, 22, retraction="polar")
    if manifold.dim != trust["run_maxinner"]:
        raise PairwiseContractError("Pymanopt manifold dimension changed")
    if manifold.typical_dist != trust["run_Delta_bar"]:
        raise PairwiseContractError("Pymanopt trust-radius default changed")
    return config


def checkpoint_identity(root: Path) -> dict[str, str]:
    return {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "source_sha256": combined_source_sha256(root),
        "optimizer_identity": OPTIMIZER_IDENTITY,
        "protocol_amendment_sha": _git(root, "rev-parse", "HEAD"),
    }


def _assert_preserved_history(root: Path, config: dict[str, Any]) -> None:
    for key in (
        "original_pairwise_freeze_commit",
        "v1_technical_failure_commit",
        "optimizer_audit_precommit",
        "optimizer_audit_final_commit",
    ):
        commit = str(config["protocol"][key])
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=root,
            check=False,
        )
        if completed.returncode != 0:
            raise PairwiseContractError(f"required history is not preserved: {key}")


def _configure_v1_engine(root: Path, config: dict[str, Any]) -> None:
    identity = checkpoint_identity(root)
    pipeline_v1.CONFIG_PATH = CONFIG_PATH
    pipeline_v1.PROTOCOL_PATH = PROTOCOL_PATH
    pipeline_v1.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
    pipeline_v1.EXPECTED_PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
    pipeline_v1.OUTPUT_PROTOCOL_FILENAME = Path(PROTOCOL_PATH).name
    pipeline_v1.REPORT_TITLE = "Pairwise Common Action V2"
    pipeline_v1.REPORT_FILENAME = "pairwise_common_action_v2.md"
    pipeline_v1.CHECKPOINT_IDENTITY = identity
    pipeline_v1.PAIRWISE_SETTINGS = PAIRWISE_V2_SETTINGS
    pipeline_v1.fit_pairwise_action = fit_pairwise_action_v2
    pipeline_v1.load_config = lambda ignored_root: config


def runtime_contract(root: Path) -> dict[str, Any]:
    """Return reproducible V2 identities without reading any BNCI object."""

    config = load_config(root)
    optimizer = build_pymanopt_trust_regions_optimizer(PAIRWISE_V2_SETTINGS)
    trust = config["optimizer"]["trust_regions"]
    return {
        "config": config,
        "optimizer_class": type(optimizer).__name__,
        "optimizer_parameters": {
            "miniter": optimizer.miniter,
            "kappa": optimizer.kappa,
            "theta": optimizer.theta,
            "rho_prime": optimizer.rho_prime,
            "use_rand": optimizer.use_rand,
            "rho_regularization": optimizer.rho_regularization,
            "max_iterations": optimizer._max_iterations,
            "min_gradient_norm": optimizer._min_gradient_norm,
            "min_step_size": optimizer._min_step_size,
            "max_cost_evaluations": optimizer._max_cost_evaluations,
            "max_time": optimizer._max_time,
            "log_verbosity": optimizer._log_verbosity,
            "run_mininner": trust["run_mininner"],
            "run_maxinner": trust["run_maxinner"],
            "run_Delta_bar": trust["run_Delta_bar"],
            "run_Delta0": trust["run_Delta0"],
        },
        "checkpoint_identity": checkpoint_identity(root),
    }


def run_all(root: Path, *, workers: int) -> dict[str, Any]:
    config = load_config(root)
    _assert_preserved_history(root, config)
    _configure_v1_engine(root, config)
    return pipeline_v1.run_all(root, workers=workers)


def record_technical_failure(
    root: Path, error: ActionSolverError, *, runtime_seconds: float
) -> dict[str, Any]:
    """Preserve a V2 optimizer failure without computing scientific summaries."""

    config = load_config(root)
    output = root / str(config["project"]["output_dir"])
    for relative in ("tables", "decisions", "report"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    message = str(error)
    pd.DataFrame(
        [
            {
                "terminal_decision": "UNASSESSED_TECHNICAL_FAILURE",
                "optimizer": OPTIMIZER_IDENTITY,
                "runtime_seconds": float(runtime_seconds),
                "scientific_statistic_computed": False,
                "error": message,
            }
        ]
    ).to_csv(output / "tables/technical_failure.csv", index=False, lineterminator="\n")
    chain = [
        {"order": 1, "gate": "U_reproduction", "status": "PASS_IF_TABLE_PRESENT"},
        {"order": 2, "gate": "optimizer", "status": "FAIL"},
        {"order": 3, "gate": "Stage_A", "status": "UNASSESSED"},
        {"order": 4, "gate": "terminal", "status": "UNASSESSED_TECHNICAL_FAILURE"},
    ]
    pd.DataFrame(chain).to_csv(
        output / "decisions/decision_chain.csv", index=False, lineterminator="\n"
    )
    terminal = {
        "decision": "UNASSESSED_TECHNICAL_FAILURE",
        "optimizer": OPTIMIZER_IDENTITY,
        "runtime_seconds": float(runtime_seconds),
        "scientific_setting_changed_after_v2_data_access": False,
        "scientific_statistics": None,
        "error": message,
    }
    pipeline_v1.atomic_json(output / "decisions/terminal_decision.json", terminal)
    (output / "report/pairwise_common_action_v2.md").write_text(
        "# Pairwise Common Action V2\n\n"
        "The V2 run stopped at **UNASSESSED_TECHNICAL_FAILURE**. The "
        "TrustRegions component-wise certification contract failed, so no "
        "available-sector or available-case scientific inference is reported.\n\n"
        "No V3 was created and no frozen scientific setting was changed.\n",
        encoding="utf-8",
    )
    return terminal


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_PROTOCOL_SHA256",
    "OPTIMIZER_IDENTITY",
    "PROTOCOL_PATH",
    "SOURCE_PATHS",
    "checkpoint_identity",
    "combined_source_sha256",
    "load_config",
    "record_technical_failure",
    "run_all",
    "runtime_contract",
]
