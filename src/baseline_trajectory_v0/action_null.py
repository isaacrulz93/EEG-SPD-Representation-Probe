"""Unrelated-target common-action null using the frozen V2 optimizer contract."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .action import _representations, action_error, fit_common_action
from .geometry import airm_mean, load_bank
from .io import append_status, atomic_csv


def _cell(target_subject, unrelated_subject, source_session, target_session,
          representation, source, unrelated, target):
    errors = []
    rows = []
    for heldout in range(4):
        training = [label for label in range(4) if label != heldout]
        q, audit = fit_common_action(
            np.concatenate([source[label] for label in training]),
            np.concatenate([unrelated[label] for label in training]),
        )
        error = action_error(source[heldout], target[heldout], q)
        errors.append(error)
        rows.append({"heldout": heldout, "error": error,
                     "best_cost": audit["best_cost"],
                     "determinants": ",".join(f"{x['determinant']:.8g}" for x in audit["starts"])})
    return {"subject": target_subject, "unrelated_subject": unrelated_subject,
            "source_session": source_session, "target_session": target_session,
            "representation": representation, "summed_heldout_error": sum(errors)}, rows


def run(cache: str | Path, output: str | Path):
    started = time.time(); output = Path(output); bank = load_bank(cache)
    metadata = bank.metadata; subjects = metadata.subject.to_numpy(); sessions = metadata.session.to_numpy()
    names = sorted(metadata.class_label.unique()); encoding = {name: i for i, name in enumerate(names)}
    y = metadata.class_label.map(encoding).to_numpy(); jobs = []
    for target_subject in range(1, 10):
        unrelated_subject = target_subject % 9 + 1
        for source_session in ("0train", "1test"):
            source_index = np.flatnonzero((subjects != target_subject) & (subjects != unrelated_subject) & (sessions == source_session))
            reference = airm_mean(bank.full[source_index])
            source = _representations(bank, source_index, y[source_index], reference)
            for target_session in ("0train", "1test"):
                unrelated_index = np.flatnonzero((subjects == unrelated_subject) & (sessions == target_session))
                target_index = np.flatnonzero((subjects == target_subject) & (sessions == target_session))
                unrelated = _representations(bank, unrelated_index, y[unrelated_index], reference)
                target = _representations(bank, target_index, y[target_index], reference)
                for representation in ("STATIC", "F1", "F2-S"):
                    jobs.append((target_subject, unrelated_subject, source_session, target_session,
                                 representation, source[representation], unrelated[representation], target[representation]))
    results = Parallel(n_jobs=8, prefer="processes")(delayed(_cell)(*job) for job in jobs)
    summary = pd.DataFrame([x[0] for x in results])
    details = []
    for (cell, rows) in results:
        details.extend({**{k: cell[k] for k in ("subject", "unrelated_subject", "source_session", "target_session", "representation")}, **row} for row in rows)
    atomic_csv(output / "action_bridge/unrelated_target_null.csv", summary)
    atomic_csv(output / "action_bridge/unrelated_target_optimizer_audit.csv", pd.DataFrame(details))
    append_status(output, "Phase 0-C unrelated-target null", "COMPLETE", time.time() - started,
                  "action_bridge/unrelated_target_null.csv", "NON_GATING_CONTROL", "Finalization")
