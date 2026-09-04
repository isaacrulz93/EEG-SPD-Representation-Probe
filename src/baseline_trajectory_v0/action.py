"""Zero-label common-action bridge for baseline trajectory V0.

Audited lineage, copied rather than merged:
- source branch: pilot/common-subject-action-falsification-v0
- source commit: d9a67a130aeeea7eb8a93d76878e43f636802e93
- source paths: src/common_action_solver_v0.py,
  src/pairwise_common_action_v2.py

The frozen four deterministic starts cover two starts in each determinant sector.
Every semantic candidate is evaluated by four leave-one-component-out fits.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score

from .geometry import (airm_mean, baseline_relative_logs, spd_invsqrt,
                       tangent_logs, load_bank)
from .identifiability import source_transform
from .io import append_status, atomic_csv


PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _spectral_start(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ea, ua = np.linalg.eigh(np.einsum("i,ijk->jk", weights, a))
    eb, ub = np.linalg.eigh(np.einsum("i,ijk->jk", weights, b))
    return ua[:, np.argsort(ea)] @ ub[:, np.argsort(eb)].T


def deterministic_starts(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    first = _spectral_start(a, b, np.ones(len(a)))
    second = _spectral_start(a, b, np.arange(1, len(a) + 1, dtype=float))
    starts = []
    for base in (first, second):
        for desired in (-1.0, 1.0):
            q = base.copy()
            if np.sign(np.linalg.det(q)) != desired:
                q[:, 0] *= -1
            starts.append(q)
    return starts


def action_error(a: np.ndarray, b: np.ndarray, q: np.ndarray) -> float:
    residual = a - q @ b @ q.T
    return float(np.sum(residual * residual) / max(np.sum(a * a), 1e-30))


def fit_common_action(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    manifold = Stiefel(a.shape[-1], a.shape[-1], retraction="polar")

    @pymanopt.function.numpy(manifold)
    def cost(q):
        return np.sum((a - q @ b @ q.T) ** 2)

    @pymanopt.function.numpy(manifold)
    def gradient(q):
        return 4.0 * np.sum(q[None] @ b @ b - a @ q[None] @ b, axis=0)

    @pymanopt.function.numpy(manifold)
    def hessian(q, h):
        return 4.0 * np.sum(h[None] @ b @ b - a @ h[None] @ b, axis=0)

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=gradient,
                               euclidean_hessian=hessian)
    optimizer = TrustRegions(min_iterations=3, kappa=0.1, theta=1.0,
                             rho_prime=0.1, use_rand=False,
                             rho_regularization=1000.0,
                             max_iterations=1000, max_time=3600,
                             min_gradient_norm=1e-5, min_step_size=1e-12,
                             verbosity=0)
    records = []
    best = None
    for index, initial in enumerate(deterministic_starts(a, b)):
        result = optimizer.run(problem, initial_point=initial)
        record = {"start": index, "determinant": float(np.linalg.det(result.point)),
                  "cost": float(result.cost), "iterations": int(result.iterations),
                  "stopping_criterion": str(result.stopping_criterion)}
        records.append(record)
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    return np.asarray(best.point), {"starts": records, "best_cost": float(best.cost)}


def select_semantic_permutation(source: np.ndarray, target: np.ndarray):
    """Select component->source mapping without a target-label argument."""
    rows, audits = [], []
    for permutation in PERMUTATIONS:
        total = 0.0
        for heldout in range(4):
            train_labels = [label for label in range(4) if label != heldout]
            a = np.concatenate([source[label] for label in train_labels])
            b = np.concatenate([target[permutation[label]] for label in train_labels])
            q, audit = fit_common_action(a, b)
            error = action_error(source[heldout], target[permutation[heldout]], q)
            total += error
            audits.append({"permutation": ",".join(map(str, permutation)),
                           "heldout": heldout, "error": error,
                           "best_cost": audit["best_cost"],
                           "determinants": ",".join(f"{r['determinant']:.8g}" for r in audit["starts"]),
                           "iterations": ",".join(str(r["iterations"]) for r in audit["starts"])})
        rows.append({"permutation": permutation, "score": total})
    rows.sort(key=lambda row: (row["score"], row["permutation"]))
    return rows[0]["permutation"], rows, audits


def _mean_components(matrices, labels, *, spd):
    result = []
    for label in range(4):
        selected = matrices[labels == label]
        if not spd:
            result.append(selected.mean(axis=0))
        elif selected.ndim == 3:
            result.append(airm_mean(selected))
        else:
            result.append(np.stack([airm_mean(selected[:, t]) for t in range(selected.shape[1])]))
    return np.stack(result)


def _representations(bank, indices, labels, reference):
    static = bank.full[indices]
    absolute = tangent_logs(bank.local[indices], reference)
    relative = baseline_relative_logs(bank.c0[indices], bank.local[indices])
    return {"STATIC": _mean_components(static, labels, spd=True)[:, None],
            "F1": _mean_components(absolute, labels, spd=False),
            "F2-S": _mean_components(relative, labels, spd=False)}


def _true_mapping(hidden_order):
    mapping = np.empty(4, dtype=int)
    for component, label in enumerate(hidden_order):
        mapping[component] = label
    return mapping


def _selected_component_mapping(permutation):
    mapping = np.empty(4, dtype=int)
    for label, component in enumerate(permutation):
        mapping[component] = label
    return mapping


def run(cache: str | Path, output: str | Path, resume: bool = True) -> None:
    started = time.time(); output = Path(output)
    bank = load_bank(cache); metadata = bank.metadata
    classes = sorted(metadata.class_label.unique())
    encoding = {name: i for i, name in enumerate(classes)}
    y = metadata.class_label.map(encoding).to_numpy()
    subjects = metadata.subject.to_numpy(); sessions = metadata.session.to_numpy()
    oracle_rows, zero_rows, rank_rows, optimizer_rows = [], [], [], []
    chunks = output / "action_bridge/chunks"; chunks.mkdir(parents=True, exist_ok=True)
    for target_subject in range(1, 10):
        for source_session in ("0train", "1test"):
            source_index = np.flatnonzero((subjects != target_subject) & (sessions == source_session))
            reference = airm_mean(bank.full[source_index])
            source = _representations(bank, source_index, y[source_index], reference)
            for target_session in ("0train", "1test"):
                target_index = np.flatnonzero((subjects == target_subject) & (sessions == target_session))
                target_named = _representations(bank, target_index, y[target_index], reference)
                hidden_order = np.random.default_rng(1000 + target_subject * 10 + (target_session == "1test")).permutation(4)
                truth = _true_mapping(hidden_order)
                for representation in ("STATIC", "F1", "F2-S"):
                    chunk = chunks / f"oracle_{target_subject}_{source_session}_{target_session}_{representation}.csv"
                    if resume and chunk.exists():
                        frame = pd.read_csv(chunk); oracle_rows.extend(frame.to_dict("records")); continue
                    hidden = target_named[representation][hidden_order]
                    selected, scores, audits = select_semantic_permutation(source[representation], hidden)
                    mapping = _selected_component_mapping(selected)
                    true_perm = tuple(np.argsort(hidden_order).tolist())
                    true_score = next(r["score"] for r in scores if r["permutation"] == true_perm)
                    rank = 1 + sum(r["score"] < true_score - 1e-12 for r in scores)
                    row = {"subject": target_subject, "source_session": source_session,
                           "target_session": target_session, "representation": representation,
                           "semantic_accuracy": float((mapping == truth).mean()),
                           "exact_permutation": bool(np.array_equal(mapping, truth)),
                           "true_permutation_rank": rank, "selected": ",".join(map(str, selected)),
                           "truth": ",".join(map(str, truth))}
                    atomic_csv(chunk, pd.DataFrame([row])); oracle_rows.append(row)
                    rank_rows.extend({"mode": "oracle", "subject": target_subject,
                                      "source_session": source_session, "target_session": target_session,
                                      "representation": representation, "permutation": ",".join(map(str, r["permutation"])),
                                      "score": r["score"]} for r in scores)
                    optimizer_rows.extend({"mode": "oracle", "subject": target_subject,
                                           "source_session": source_session, "target_session": target_session,
                                           "representation": representation, **a} for a in audits)
        source_index = np.flatnonzero(subjects != target_subject)
        target_index = np.flatnonzero(subjects == target_subject)
        reference = airm_mean(bank.full[source_index])
        source_repr = _representations(bank, source_index, y[source_index], reference)
        trial_matrices = {"STATIC": bank.full[target_index][:, None],
                          "F1": tangent_logs(bank.local[target_index], reference),
                          "F2-S": baseline_relative_logs(bank.c0[target_index], bank.local[target_index])}
        source_trial = {"STATIC": tangent_logs(bank.full, reference),
                        "F1": tangent_logs(bank.local, reference),
                        "F2-S": baseline_relative_logs(bank.c0, bank.local)}
        for representation in ("STATIC", "F1", "F2-S"):
            vectors = source_trial[representation].reshape(len(bank.full), -1)
            xs, xt = source_transform(vectors[source_index], vectors[target_index])
            for seed in range(20):
                chunk = chunks / f"zero_{target_subject}_{representation}_{seed}.csv"
                if resume and chunk.exists():
                    zero_rows.extend(pd.read_csv(chunk).to_dict("records")); continue
                clusters = KMeans(n_clusters=4, n_init=50, random_state=seed).fit_predict(xt)
                target_components = _mean_components(
                    trial_matrices[representation], clusters,
                    spd=(representation == "STATIC"),
                )
                if representation == "STATIC": target_components = target_components[:, None]
                selected, scores, audits = select_semantic_permutation(source_repr[representation], target_components)
                mapping = _selected_component_mapping(selected); prediction = mapping[clusters]
                row = {"subject": target_subject, "representation": representation, "seed": seed,
                       "mapping_accuracy": float((prediction == y[target_index]).mean()),
                       "mapping_balanced_accuracy": balanced_accuracy_score(y[target_index], prediction),
                       "selected": ",".join(map(str, selected))}
                atomic_csv(chunk, pd.DataFrame([row])); zero_rows.append(row)
                rank_rows.extend({"mode": "zero", "subject": target_subject, "source_session": "both",
                                  "target_session": "both", "representation": representation, "seed": seed,
                                  "permutation": ",".join(map(str, r["permutation"])), "score": r["score"]} for r in scores)
                optimizer_rows.extend({"mode": "zero", "subject": target_subject,
                                       "source_session": "both", "target_session": "both",
                                       "representation": representation, "seed": seed, **a} for a in audits)
    atomic_csv(output / "action_bridge/oracle_components.csv", pd.DataFrame(oracle_rows))
    atomic_csv(output / "action_bridge/zero_label_clusters.csv", pd.DataFrame(zero_rows))
    atomic_csv(output / "action_bridge/permutation_ranks.csv", pd.DataFrame(rank_rows))
    atomic_csv(output / "action_bridge/optimizer_audit.csv", pd.DataFrame(optimizer_rows))
    append_status(output, "Phase 0-C action bridge", "COMPLETE", time.time() - started,
                  "action_bridge/*.csv", "PENDING_DECISION", "Decision calculation")
