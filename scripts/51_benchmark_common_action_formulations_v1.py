#!/usr/bin/env python3
"""Run the formulation-v1 benchmark on generated symmetric matrices only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common_action_formulation_audit_v1 import (
    bnci_stage_a_call_budget,
    fit_joint_latent_reference,
    fit_pairwise_action,
    fit_profiled_product_reference,
)
from src.common_action_solver_v0 import CANDIDATE_SOLVER_SETTINGS, conjugate


def orthogonal(seed: int, dimension: int, determinant: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if int(np.sign(np.linalg.det(q))) != int(determinant):
        q[:, 0] *= -1.0
    return q


def bank(seed: int, count: int, dimension: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(count, dimension, dimension))
    return 0.5 * (values + values.transpose(0, 2, 1))


def latent_fixture(
    *, seed: int, subjects: int, classes: int, dimension: int, noise: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    templates = bank(seed, classes, dimension)
    actions = np.empty((subjects, dimension, dimension))
    actions[0] = np.eye(dimension)
    for subject in range(1, subjects):
        actions[subject] = orthogonal(
            seed + 10 * subject,
            dimension,
            -1 if subject % 2 else 1,
        )
    objects = np.stack([conjugate(action, templates) for action in actions])
    if noise:
        rng = np.random.default_rng(seed + 999)
        perturbation = rng.normal(scale=noise, size=objects.shape)
        objects = objects + 0.5 * (
            perturbation + perturbation.transpose(0, 1, 3, 2)
        )
    return objects, templates, actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    seed = int(args.seed)
    output: dict[str, object] = {
        "contract": "synthetic_only_no_dataset_loader",
        "seed": seed,
    }

    single_action = []
    for starts in (4, 6, 8):
        settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=starts)
        for determinant in (1, -1):
            templates = bank(seed + 100 * starts + determinant, 4, 22)
            truth = orthogonal(seed + 200 * starts + determinant, 22, determinant)
            rng = np.random.default_rng(seed + 300 * starts + determinant)
            noise = rng.normal(scale=2.0e-4, size=(3, 22, 22))
            noise = 0.5 * (noise + noise.transpose(0, 2, 1))
            targets = conjugate(truth, templates[:3]) + noise
            started = time.perf_counter()
            fit = fit_pairwise_action(
                targets,
                templates[:3],
                seed=seed + 400 * starts + determinant,
                settings=settings,
            )
            elapsed = time.perf_counter() - started
            prediction = conjugate(fit.matrix, templates[3])
            expected = conjugate(truth, templates[3])
            single_action.append(
                {
                    "starts": starts,
                    "truth_determinant": determinant,
                    "elapsed_seconds": elapsed,
                    "iterations_best": fit.starts[fit.best_start_index].iterations,
                    "heldout_relative_error": float(
                        np.linalg.norm(prediction - expected) / np.linalg.norm(expected)
                    ),
                    "converged_starts": sum(result.converged for result in fit.starts),
                }
            )
    output["single_action_d22_noisy"] = single_action

    exact, _, exact_actions = latent_fixture(
        seed=seed + 10,
        subjects=8,
        classes=3,
        dimension=22,
        noise=0.0,
    )
    noisy, _, _ = latent_fixture(
        seed=seed + 20,
        subjects=8,
        classes=3,
        dimension=22,
        noise=2.0e-4,
    )
    settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=4)
    product_rows = []
    for name, objects in (("exact", exact), ("noisy", noisy)):
        fit = fit_profiled_product_reference(objects, seed=seed + 30, settings=settings)
        reconstruction = np.stack(
            [conjugate(fit.actions[index], fit.templates) for index in range(len(objects))]
        )
        product_rows.append(
            {
                "fixture": name,
                "elapsed_seconds": fit.elapsed_seconds,
                "iterations": fit.iterations,
                "gradient_norm": fit.gradient_norm,
                "objective": fit.objective,
                "relative_reconstruction": float(
                    np.linalg.norm(reconstruction - objects) / np.linalg.norm(objects)
                ),
                "converged": fit.converged,
            }
        )
    output["profiled_product_d22_R8_one_start"] = product_rows

    small, _, small_actions = latent_fixture(
        seed=seed + 40,
        subjects=4,
        classes=4,
        dimension=4,
        noise=1.0e-5,
    )
    joint = fit_joint_latent_reference(
        small,
        seed=seed + 50,
        settings=settings,
        initial_actions=small_actions,
    )
    profiled = fit_profiled_product_reference(
        small,
        seed=seed + 50,
        settings=settings,
        initial_actions=small_actions,
    )
    output["small_A_B_reference"] = {
        "joint_A_objective": joint.objective,
        "joint_A_elapsed_seconds": joint.elapsed_seconds,
        "profiled_B_objective": profiled.objective,
        "profiled_B_elapsed_seconds": profiled.elapsed_seconds,
        "absolute_objective_difference": abs(joint.objective - profiled.objective),
    }
    output["bnci_stage_a_call_budget_four_total_starts"] = bnci_stage_a_call_budget(4)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
