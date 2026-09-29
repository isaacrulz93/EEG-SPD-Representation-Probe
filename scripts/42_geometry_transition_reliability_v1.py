#!/usr/bin/env python3
"""Cross-session reliability diagnostic for trial-local covariance-geometry timing.

This is a diagnostic, not a classifier experiment. It reuses the exact
0.5 s / 0.1 s whitened-log covariance probe and 5-segment DP from the frozen
adaptive-segmentation v0 implementation, but evaluates both BNCI2014_001
sessions to ask whether geometry-change timing is reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/geometry_transition_reliability_v1"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)

from src.adaptive_segmentation_v0 import adaptive_boundaries, exact_sign_flip, probe_sequence

SESSIONS = ("0train", "1test")
SUBJECTS = tuple(range(1, 10))
FIXED5 = np.asarray([0.8, 1.6, 2.4, 3.2], dtype=np.float64)
PRIMARY_NAMES = (
    "profile_same_minus_wrongclass",
    "profile_same_minus_wrongsubject",
    "boundary_wrongclass_minus_same",
    "boundary_wrongsubject_minus_same",
    "crosspop_profile_margin",
    "crosspop_boundary_margin",
)


def arrhash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def load_session(session: str):
    cfg = yaml.safe_load((ROOT / "configs/bnci2014_001_5window.yaml").read_text())
    cache = ROOT / "cache/geometry_transition_reliability_v1/moabb_data"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(cache)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(cache)

    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    mne.set_log_level("WARNING")
    classes = [str(x) for x in cfg["dataset"]["classes"]]
    channels = [str(x) for x in cfg["dataset"]["eeg_channels"]]
    ds = BNCI2014_001(subjects=list(SUBJECTS), sessions=[session])
    paradigm = MotorImagery(
        n_classes=4,
        events=classes,
        fmin=8.0,
        fmax=32.0,
        tmin=0.0,
        tmax=3.996,
        baseline=None,
        channels=channels,
        resample=None,
    )
    epochs, labels, meta = paradigm.get_data(dataset=ds, subjects=list(SUBJECTS), return_epochs=True)
    if epochs.ch_names != channels:
        epochs.reorder_channels(channels)
    X = epochs.get_data(copy=True).astype(np.float32, copy=False)
    y = np.asarray(labels).astype(str)
    meta = meta.reset_index(drop=True).copy()
    meta["subject"] = pd.to_numeric(meta["subject"]).astype(int)
    meta["session"] = meta["session"].astype(str)
    meta["class_label"] = y
    if X.shape != (2592, 22, 1000):
        raise RuntimeError("unexpected session shape: {}".format(X.shape))
    if set(meta["session"]) != {session}:
        raise RuntimeError("session barrier failed: {}".format(sorted(set(meta["session"]))))
    counts = meta.groupby(["subject", "class_label"], observed=True).size()
    if not (counts == 72).all():
        raise RuntimeError("expected 72 trials per subject/class: {}".format(counts.to_dict()))
    return X, y, meta


def trial_geometry(x: np.ndarray):
    _, z, centers = probe_sequence(x, window=125, stride=25)
    change = np.linalg.norm(np.diff(z, axis=0), axis=1)
    total = float(change.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("nonpositive geometry-change mass")
    profile = change / total
    b = adaptive_boundaries(z, centers, n_samples=1000, k=5, grid=25, min_length=125)
    internal = b[1:-1].astype(np.float64) / 250.0
    return profile.astype(np.float32), internal.astype(np.float32), float(total)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def holm_adjust(ps):
    order = np.argsort(ps)
    out = np.empty(len(ps), dtype=float)
    running = 0.0
    m = len(ps)
    for rank, idx in enumerate(order):
        raw = min(1.0, (m - rank) * float(ps[idx]))
        running = max(running, raw)
        out[idx] = running
    return out


def main():
    t0 = time.time()
    classes = None
    session_data = {}
    session_meta = {}
    trial_profiles = {}
    trial_bounds = {}
    trial_mass = {}

    for session in SESSIONS:
        X, y, meta = load_session(session)
        if classes is None:
            classes = tuple(sorted(set(y.tolist())))
        if session == "0train":
            observed = arrhash(X)
            expected = "0ea4250cd215c0049fac050ae0dbae60d81a4ab3e8d0335b12b33e29092b3972"
            if observed != expected:
                raise RuntimeError("0train preprocessing hash mismatch: {} != {}".format(observed, expected))
        profiles = np.empty((len(X), 35), dtype=np.float32)
        bounds = np.empty((len(X), 4), dtype=np.float32)
        masses = np.empty(len(X), dtype=np.float64)
        for i, x in enumerate(X):
            p, b, m = trial_geometry(x)
            profiles[i] = p
            bounds[i] = b
            masses[i] = m
            if i % 250 == 0:
                print("{} geometry {}/{}".format(session, i, len(X)), flush=True)
        session_data[session] = (X, y)
        session_meta[session] = meta
        trial_profiles[session] = profiles
        trial_bounds[session] = bounds
        trial_mass[session] = masses
        del X

    cell_profile = {}
    cell_bound = {}
    cell_jitter = {}
    cell_mass = {}
    rows = []
    for session in SESSIONS:
        meta = session_meta[session]
        profiles = trial_profiles[session]
        bounds = trial_bounds[session]
        masses = trial_mass[session]
        for s, c in product(SUBJECTS, classes):
            idx = np.flatnonzero((meta["subject"].to_numpy() == s) & (meta["class_label"].to_numpy() == c))
            p = profiles[idx].mean(axis=0).astype(np.float64)
            p = p / p.sum()
            b = bounds[idx].mean(axis=0).astype(np.float64)
            j = float(bounds[idx].std(axis=0, ddof=1).mean())
            cell_profile[(session, s, c)] = p
            cell_bound[(session, s, c)] = b
            cell_jitter[(session, s, c)] = j
            cell_mass[(session, s, c)] = float(masses[idx].mean())
            rows.append({
                "session": session, "subject": s, "class_label": c,
                "boundary1_s": b[0], "boundary2_s": b[1], "boundary3_s": b[2], "boundary4_s": b[3],
                "mean_boundary_jitter_s": j, "mean_total_change_mass": float(masses[idx].mean()),
            })
    pd.DataFrame(rows).to_csv(OUT / "tables/cell_summary.csv", index=False)

    subject_rows = []
    primary = {name: [] for name in PRIMARY_NAMES}
    class_match_correct = 0
    class_match_total = 0
    for s in SUBJECTS:
        same_prof = []
        wrongc_prof = []
        wrongs_prof = []
        same_b = []
        wrongc_b = []
        wrongs_b = []
        cross_prof_margin = []
        cross_b_margin = []

        for c in classes:
            pt = cell_profile[("0train", s, c)]
            pe = cell_profile[("1test", s, c)]
            bt = cell_bound[("0train", s, c)]
            be = cell_bound[("1test", s, c)]
            same_prof.append(cosine(pt, pe))
            same_b.append(mae(bt, be))
            wrongc_prof.append(np.mean([cosine(pt, cell_profile[("1test", s, c2)]) for c2 in classes if c2 != c]))
            wrongc_b.append(np.mean([mae(bt, cell_bound[("1test", s, c2)]) for c2 in classes if c2 != c]))
            wrongs_prof.append(np.mean([cosine(pt, cell_profile[("1test", s2, c)]) for s2 in SUBJECTS if s2 != s]))
            wrongs_b.append(np.mean([mae(bt, cell_bound[("1test", s2, c)]) for s2 in SUBJECTS if s2 != s]))

        for target_session, source_session in (("0train", "1test"), ("1test", "0train")):
            pop_prof = {}
            pop_bound = {}
            for c in classes:
                pp = np.mean([cell_profile[(source_session, s2, c)] for s2 in SUBJECTS if s2 != s], axis=0)
                pp = pp / pp.sum()
                pop_prof[c] = pp
                pop_bound[c] = np.mean([cell_bound[(source_session, s2, c)] for s2 in SUBJECTS if s2 != s], axis=0)
            for c in classes:
                tp = cell_profile[(target_session, s, c)]
                tb = cell_bound[(target_session, s, c)]
                sims = {c2: cosine(tp, pop_prof[c2]) for c2 in classes}
                dists = {c2: mae(tb, pop_bound[c2]) for c2 in classes}
                cross_prof_margin.append(sims[c] - np.mean([v for k, v in sims.items() if k != c]))
                cross_b_margin.append(np.mean([v for k, v in dists.items() if k != c]) - dists[c])
                class_match_correct += int(max(sims, key=sims.get) == c)
                class_match_total += 1

        rec = {
            "subject": s,
            "same_profile_cos": float(np.mean(same_prof)),
            "wrongclass_profile_cos": float(np.mean(wrongc_prof)),
            "wrongsubject_profile_cos": float(np.mean(wrongs_prof)),
            "same_boundary_mae_s": float(np.mean(same_b)),
            "wrongclass_boundary_mae_s": float(np.mean(wrongc_b)),
            "wrongsubject_boundary_mae_s": float(np.mean(wrongs_b)),
            "crosspop_profile_margin": float(np.mean(cross_prof_margin)),
            "crosspop_boundary_margin_s": float(np.mean(cross_b_margin)),
        }
        subject_rows.append(rec)
        primary["profile_same_minus_wrongclass"].append(rec["same_profile_cos"] - rec["wrongclass_profile_cos"])
        primary["profile_same_minus_wrongsubject"].append(rec["same_profile_cos"] - rec["wrongsubject_profile_cos"])
        primary["boundary_wrongclass_minus_same"].append(rec["wrongclass_boundary_mae_s"] - rec["same_boundary_mae_s"])
        primary["boundary_wrongsubject_minus_same"].append(rec["wrongsubject_boundary_mae_s"] - rec["same_boundary_mae_s"])
        primary["crosspop_profile_margin"].append(rec["crosspop_profile_margin"])
        primary["crosspop_boundary_margin"].append(rec["crosspop_boundary_margin_s"])

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(OUT / "tables/subject_reliability.csv", index=False)

    tests = []
    ps = []
    for name in PRIMARY_NAMES:
        result = exact_sign_flip(primary[name])
        tests.append({
            "test": name,
            "mean_effect": result["mean_difference"],
            "median_effect": result["median_difference"],
            "p_one_sided": result["p_one_sided"],
        })
        ps.append(result["p_one_sided"])
    adj = holm_adjust(np.asarray(ps))
    for row, q in zip(tests, adj):
        row["p_holm_6"] = float(q)
    pd.DataFrame(tests).to_csv(OUT / "tables/primary_tests.csv", index=False)

    all_bounds = np.concatenate([trial_bounds[s] for s in SESSIONS], axis=0).astype(np.float64)
    nearest = np.min(np.abs(all_bounds[..., None] - FIXED5[None, None, :]), axis=-1)
    fixed10 = float(np.mean(nearest <= 0.10 + 1e-12))
    fixed20 = float(np.mean(nearest <= 0.20 + 1e-12))
    jitter_values = np.asarray(list(cell_jitter.values()), dtype=float)
    same_boundary = subject_df["same_boundary_mae_s"].to_numpy(float)

    test_by_name = {r["test"]: r for r in tests}
    alpha = 0.05
    within_profile = test_by_name["profile_same_minus_wrongclass"]["p_holm_6"] < alpha and test_by_name["profile_same_minus_wrongclass"]["mean_effect"] > 0
    cross_profile = test_by_name["crosspop_profile_margin"]["p_holm_6"] < alpha and test_by_name["crosspop_profile_margin"]["mean_effect"] > 0
    hard_boundary = (
        test_by_name["boundary_wrongclass_minus_same"]["p_holm_6"] < alpha
        and test_by_name["boundary_wrongclass_minus_same"]["mean_effect"] > 0
        and test_by_name["crosspop_boundary_margin"]["p_holm_6"] < alpha
        and test_by_name["crosspop_boundary_margin"]["mean_effect"] > 0
    )
    if within_profile and cross_profile and hard_boundary:
        decision = "GO_HARD_ADAPTIVE_SEGMENTATION"
    elif within_profile and cross_profile:
        decision = "GO_SOFT_ADAPTIVE_COVARIANCE_NOT_HARD_BOUNDARIES"
    elif within_profile:
        decision = "SUBJECT_SPECIFIC_TIMING_ONLY"
    else:
        decision = "STOP_GEOMETRY_TIMING_AS_PRIMARY_DIRECTION"

    summary = {
        "decision": decision,
        "subjects": 9,
        "classes": list(classes),
        "trials_per_session": 2592,
        "sessions": list(SESSIONS),
        "probe_window_s": 0.5,
        "probe_stride_s": 0.1,
        "dp_segments": 5,
        "mean_same_profile_cos": float(subject_df["same_profile_cos"].mean()),
        "mean_wrongclass_profile_cos": float(subject_df["wrongclass_profile_cos"].mean()),
        "mean_wrongsubject_profile_cos": float(subject_df["wrongsubject_profile_cos"].mean()),
        "mean_same_boundary_mae_s": float(subject_df["same_boundary_mae_s"].mean()),
        "median_cell_boundary_jitter_s": float(np.median(jitter_values)),
        "mean_cell_boundary_jitter_s": float(np.mean(jitter_values)),
        "fixed5_proximity_fraction_100ms": fixed10,
        "fixed5_proximity_fraction_200ms": fixed20,
        "cross_subject_profile_nearest_class_accuracy": float(class_match_correct / class_match_total),
        "cross_subject_profile_nearest_class_correct": int(class_match_correct),
        "cross_subject_profile_nearest_class_total": int(class_match_total),
        "primary_tests": tests,
        "holm_family_size": 6,
        "elapsed_seconds": float(time.time() - t0),
        "interpretation": {
            "within_profile": bool(within_profile),
            "cross_subject_profile": bool(cross_profile),
            "hard_boundary": bool(hard_boundary),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    np.savez_compressed(
        OUT / "geometry_arrays.npz",
        train_profiles=trial_profiles["0train"],
        test_profiles=trial_profiles["1test"],
        train_boundaries_s=trial_bounds["0train"],
        test_boundaries_s=trial_bounds["1test"],
    )

    lines = [
        "# Geometry Transition Reliability V1",
        "",
        "**Decision: {}**".format(decision),
        "",
        "This diagnostic reuses the frozen adaptive-segmentation covariance probe but tests both BNCI2014_001 sessions. No classifier, target adaptation, or target-fitted statistic is used.",
        "",
        "## Main descriptive results",
        "",
        "- Same subject/class T-E change-profile cosine: {:.4f}".format(summary["mean_same_profile_cos"]),
        "- Wrong-class (same subject) profile cosine: {:.4f}".format(summary["mean_wrongclass_profile_cos"]),
        "- Wrong-subject (same class) profile cosine: {:.4f}".format(summary["mean_wrongsubject_profile_cos"]),
        "- Same-cell mean DP-boundary T-E MAE: {:.3f} s".format(summary["mean_same_boundary_mae_s"]),
        "- Median trial-boundary jitter within subject/class/session: {:.3f} s".format(summary["median_cell_boundary_jitter_s"]),
        "- DP boundaries within 100 ms / 200 ms of equal FIXED5 boundaries: {:.1%} / {:.1%}".format(fixed10, fixed20),
        "- Cross-subject nearest-class match from timing profile: {}/{} = {:.1%}".format(class_match_correct, class_match_total, class_match_correct / class_match_total),
        "",
        "## Primary exact subject-level tests (Holm over 6)",
        "",
        "| test | mean effect | raw p | Holm p |",
        "|---|---:|---:|---:|",
    ]
    for r in tests:
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(r["test"], r["mean_effect"], r["p_one_sided"], r["p_holm_6"]))
    lines += [
        "",
        "Decision rule: hard adaptive segmentation requires class-specific profile repeatability across sessions, cross-subject class-specific timing, and reproducible hard boundaries. If timing profiles generalize but hard boundaries do not, prefer soft adaptive covariance construction. If only within-subject timing repeats, treat timing as subject-specific rather than a cross-subject segmentation mechanism.",
        "",
        "The test concerns transition timing, not whether local covariance magnitude or a downstream temporal model can help classification.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((OUT / "REPORT.md").read_text(), flush=True)
    print("SUMMARY_JSON_BEGIN", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("SUMMARY_JSON_END", flush=True)


if __name__ == "__main__":
    main()
