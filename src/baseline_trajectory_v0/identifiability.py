"""Source-fitted, target-label-free identifiability diagnostics."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .geometry import (airm_mean, fixed_nonidentity_permutations,
                       invariant_feature_map, load_bank, referenced_feature_map)
from .io import append_status, atomic_csv


FEATURES = ("F0", "F1", "F2-S", "F2-V", "F3-G")


def source_transform(x_source: np.ndarray, x_target: np.ndarray):
    scaler = StandardScaler().fit(x_source)
    source_scaled = scaler.transform(x_source)
    target_scaled = scaler.transform(x_target)
    count = min(20, x_source.shape[1], len(x_source) - 1)
    pca = PCA(n_components=count, whiten=False, random_state=20260904).fit(source_scaled)
    return pca.transform(source_scaled), pca.transform(target_scaled)


def assign_clusters_to_source(cluster_centres: np.ndarray,
                              source_centres: np.ndarray) -> np.ndarray:
    """Return mapping cluster->source class; target labels cannot enter this API."""
    cost = np.linalg.norm(cluster_centres[:, None] - source_centres[None, :], axis=2)
    rows, columns = linear_sum_assignment(cost)
    mapping = np.empty(4, dtype=np.int64)
    mapping[rows] = columns
    return mapping


def oracle_cluster_scores(cluster: np.ndarray, y_target: np.ndarray) -> dict[str, float]:
    counts = np.zeros((4, 4), dtype=np.int64)
    for a, b in zip(cluster, y_target):
        counts[int(a), int(b)] += 1
    rows, columns = linear_sum_assignment(-counts)
    matched = counts[rows, columns].sum() / len(cluster)
    purity = counts.max(axis=1).sum() / len(cluster)
    sizes = counts.sum(axis=1)
    probabilities = sizes[sizes > 0] / sizes.sum()
    entropy = -np.sum(probabilities * np.log(probabilities)) / np.log(4)
    return {"nmi": normalized_mutual_info_score(y_target, cluster),
            "ari": adjusted_rand_score(y_target, cluster),
            "hungarian_accuracy": matched, "purity": purity,
            "cluster_entropy": entropy,
            "cluster_size_imbalance": (sizes.max() - sizes.min()) / sizes.sum()}


def source_class_centres(x_source: np.ndarray, y_source: np.ndarray,
                         subjects: np.ndarray) -> np.ndarray:
    result = []
    for label in range(4):
        per_subject = [x_source[(subjects == subject) & (y_source == label)].mean(axis=0)
                       for subject in np.unique(subjects)]
        result.append(np.mean(per_subject, axis=0))
    return np.asarray(result)


def neighbour_scores(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    indices = NearestNeighbors(n_neighbors=6).fit(x).kneighbors(return_distance=False)
    same1 = (y[indices[:, 0]] == y).mean()
    purity5 = (y[indices[:, :5]] == y[:, None]).mean()
    return {"nn1_same_class": same1, "nn5_class_purity": purity5,
            "different_class_neighbour_fraction": 1.0 - purity5}


def run(cache: str | Path, output: str | Path, resume: bool = True) -> None:
    started = time.time()
    output = Path(output)
    chunks = output / "identifiability/chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    bank = load_bank(cache)
    encoder = LabelEncoder().fit(bank.metadata.class_label)
    y = encoder.transform(bank.metadata.class_label)
    subjects = bank.metadata.subject.to_numpy()
    permutations = fixed_nonidentity_permutations(len(y), 20261905)
    invariant, _ = invariant_feature_map(bank.c0, bank.local, permutations)
    cluster_rows, assignment_rows, neighbour_rows = [], [], []
    for target in range(1, 10):
        source_index = np.flatnonzero(subjects != target)
        target_index = np.flatnonzero(subjects == target)
        reference = airm_mean(bank.full[source_index])
        referenced, _ = referenced_feature_map(bank.full, bank.local, reference)
        features = {**referenced, **invariant}
        for feature in FEATURES:
            chunk = chunks / f"subject_{target}_{feature.replace('-', '_')}.npz"
            if resume and chunk.exists():
                with np.load(chunk, allow_pickle=False) as z:
                    cluster_rows.extend(pd.DataFrame(z["cluster"], columns=z["cluster_columns"].astype(str)).to_dict("records"))
                    assignment_rows.extend(pd.DataFrame(z["assignment"], columns=z["assignment_columns"].astype(str)).to_dict("records"))
                    neighbour_rows.extend(pd.DataFrame(z["neighbour"], columns=z["neighbour_columns"].astype(str)).to_dict("records"))
                continue
            xs, xt = source_transform(features[feature][source_index], features[feature][target_index])
            source_centres = source_class_centres(xs, y[source_index], subjects[source_index])
            nrow = {"subject": target, "feature": feature, **neighbour_scores(xt, y[target_index])}
            local_clusters, local_assignments = [], []
            for seed in range(20):
                model = KMeans(n_clusters=4, n_init=50, random_state=seed).fit(xt)
                cluster = model.labels_
                scores = oracle_cluster_scores(cluster, y[target_index])
                local_clusters.append({"subject": target, "feature": feature, "seed": seed, **scores})
                mapping = assign_clusters_to_source(model.cluster_centers_, source_centres)
                prediction = mapping[cluster]
                local_assignments.append({"subject": target, "feature": feature, "seed": seed,
                                          "mapping_accuracy": (prediction == y[target_index]).mean(),
                                          "mapping_balanced_accuracy": balanced_accuracy_score(y[target_index], prediction),
                                          "mapping": ",".join(map(str, mapping.tolist()))})
            cframe, aframe, nframe = pd.DataFrame(local_clusters), pd.DataFrame(local_assignments), pd.DataFrame([nrow])
            partial = chunk.with_suffix(".npz.partial")
            with partial.open("wb") as handle:
                np.savez(handle, cluster=cframe.to_numpy(dtype=str), cluster_columns=cframe.columns.to_numpy(),
                         assignment=aframe.to_numpy(dtype=str), assignment_columns=aframe.columns.to_numpy(),
                         neighbour=nframe.to_numpy(dtype=str), neighbour_columns=nframe.columns.to_numpy())
            partial.replace(chunk)
            cluster_rows.extend(local_clusters); assignment_rows.extend(local_assignments); neighbour_rows.append(nrow)
    clusters = pd.DataFrame(cluster_rows)
    for name in ["subject", "seed", "nmi", "ari", "hungarian_accuracy", "purity", "cluster_entropy", "cluster_size_imbalance"]:
        clusters[name] = pd.to_numeric(clusters[name])
    assignments = pd.DataFrame(assignment_rows)
    for name in ["subject", "seed", "mapping_accuracy", "mapping_balanced_accuracy"]:
        assignments[name] = pd.to_numeric(assignments[name])
    neighbours = pd.DataFrame(neighbour_rows)
    for name in ["subject", "nn1_same_class", "nn5_class_purity", "different_class_neighbour_fraction"]:
        neighbours[name] = pd.to_numeric(neighbours[name])
    atomic_csv(output / "identifiability/clustering_seed_results.csv", clusters)
    atomic_csv(output / "identifiability/source_assignment.csv", assignments)
    atomic_csv(output / "identifiability/neighbour_overlap.csv", neighbours)
    summary = (clusters.groupby(["subject", "feature"], as_index=False).median(numeric_only=True)
               .merge(assignments.groupby(["subject", "feature"], as_index=False).median(numeric_only=True),
                      on=["subject", "feature"], suffixes=("", "_assignment")))
    atomic_csv(output / "identifiability/subject_summary.csv", summary)
    append_status(output, "Phase 0-B identifiability", "COMPLETE", time.time() - started,
                  "identifiability/*.csv", "PENDING_DECISION", "Phase 0-C")
