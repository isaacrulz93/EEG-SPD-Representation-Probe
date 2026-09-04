"""Frozen linear task-headroom evaluation for baseline trajectory V0."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .geometry import (FEATURE_DIMENSIONS, FEATURE_ORDER, airm_mean,
                       fixed_nonidentity_permutations, invariant_feature_map,
                       load_bank, referenced_feature_map, subject_ra)
from .io import append_status, atomic_csv, atomic_json
from .statistics import holm_adjust, paired_summary


def classifier() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=np.arange(4)).tolist()),
    }


def split_manifest(metadata: pd.DataFrame, y: np.ndarray, seed: int = 20260904):
    subjects = metadata.subject.to_numpy()
    sessions = metadata.session.to_numpy()
    for subject in range(1, 10):
        for session in ("0train", "1test"):
            pool = np.flatnonzero((subjects == subject) & (sessions == session))
            splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
            for fold, (a, b) in enumerate(splitter.split(pool, y[pool])):
                yield "P0", subject, f"{session}_fold{fold}", pool[a], pool[b]
        for train_session, test_session in (("0train", "1test"), ("1test", "0train")):
            train = np.flatnonzero((subjects == subject) & (sessions == train_session))
            test = np.flatnonzero((subjects == subject) & (sessions == test_session))
            yield "P1", subject, f"{train_session}_to_{test_session}", train, test
    for target in range(1, 10):
        train = np.flatnonzero(subjects != target)
        test = np.flatnonzero(subjects == target)
        yield "P2", target, "LOSO", train, test
        yield "P3", target, "LOSO_RA", train, test


def _run_split(bank, invariant, ra_bank, ra_invariant, y, protocol, unit, split, train, test):
    active_bank = ra_bank if protocol == "P3" else bank
    active_invariant = ra_invariant if protocol == "P3" else invariant
    reference = airm_mean(active_bank.full[train])
    referenced, _ = referenced_feature_map(active_bank.full, active_bank.local, reference)
    features = {**referenced, **active_invariant}
    rows = []
    for name in FEATURE_ORDER:
        model = classifier().fit(features[name][train], y[train])
        pred = model.predict(features[name][test])
        rows.append({"protocol": protocol, "subject": unit, "split": split,
                     "feature": name, "n_train": len(train), "n_test": len(test),
                     **_score(y[test], pred)})
    return rows


def summarize(rows: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = rows[rows.protocol.isin(["P1", "P2", "P3"])].copy()
    by = (base.groupby(["protocol", "subject", "feature"], as_index=False)
          [["balanced_accuracy", "accuracy", "macro_f1"]].mean())
    atomic_csv(output / "task/subject_summary.csv", by)
    comparisons = []
    for protocol in ("P1", "P2", "P3"):
        table = by[by.protocol == protocol].pivot(index="subject", columns="feature", values="balanced_accuracy")
        raw = {}
        interim = {}
        for candidate in ("F2-S", "F3-G"):
            summary = paired_summary((table[candidate] - table.F0).to_numpy())
            raw[candidate] = float(summary["p_raw"])
            interim[candidate] = summary
        adjusted = holm_adjust(raw)
        for candidate in ("F2-S", "F3-G"):
            comparisons.append({"protocol": protocol, "candidate": candidate,
                                "reference": "F0", **interim[candidate],
                                "p_holm": adjusted[candidate]})
        order = paired_summary((table["F2-S"] - table["F2-S-SHUFFLE"]).to_numpy())
        comparisons.append({"protocol": protocol, "candidate": "F2-S",
                            "reference": "F2-S-SHUFFLE", **order, "p_holm": np.nan})
    tests = pd.DataFrame(comparisons)
    atomic_csv(output / "task/paired_tests.csv", tests)
    return by, tests


def run(cache: str | Path, output: str | Path, resume: bool = True) -> None:
    started = time.time()
    output = Path(output)
    chunks = output / "task/chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    bank = load_bank(cache)
    labels = LabelEncoder().fit(bank.metadata.class_label)
    y = labels.transform(bank.metadata.class_label)
    permutations = fixed_nonidentity_permutations(len(y), 20261905)
    invariant, _ = invariant_feature_map(bank.c0, bank.local, permutations)
    ra_bank, ra_audit = subject_ra(bank)
    ra_invariant, _ = invariant_feature_map(ra_bank.c0, ra_bank.local, permutations)
    atomic_csv(output / "task/ra_audit.csv", ra_audit)
    atomic_csv(output / "features/shuffle_manifest.csv",
               pd.DataFrame(permutations, columns=[f"position_{i}" for i in range(5)]).assign(trial_uid=bank.metadata.trial_uid))
    atomic_csv(output / "features/feature_manifest.csv",
               pd.DataFrame([{"feature": k, "dimension": v} for k, v in FEATURE_DIMENSIONS.items()]))
    all_rows = []
    for protocol, unit, split, train, test in split_manifest(bank.metadata, y):
        chunk = chunks / f"{protocol}_{unit}_{split}.csv"
        if resume and chunk.exists():
            all_rows.append(pd.read_csv(chunk))
            continue
        result = pd.DataFrame(_run_split(bank, invariant, ra_bank, ra_invariant, y,
                                         protocol, unit, split, train, test))
        atomic_csv(chunk, result)
        all_rows.append(result)
    rows = pd.concat(all_rows, ignore_index=True)
    atomic_csv(output / "task/fold_results.csv", rows)
    summarize(rows, output)
    atomic_json(output / "task/class_encoding.json",
                {name: int(i) for i, name in enumerate(labels.classes_)})
    append_status(output, "Phase 0-A task", "COMPLETE", time.time() - started,
                  "task/*.csv, features/*.csv", "PENDING_DECISION", "Phase 0-B")
