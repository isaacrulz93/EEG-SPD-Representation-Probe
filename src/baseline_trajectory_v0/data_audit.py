"""Raw-timing audit and immutable covariance cache for baseline trajectory V0.

Audited lineage, not merged:
- source branch: pilot/trial-movement-incremental-utility-v0
- source commit: 4a0e7966f676a02838e4300114667ff5e40cd9ae
- source paths: src/data.py, src/covariance.py,
  src/trajectory_within_subject_data_v1.py

The implementation deliberately loads BNCI2014_001 run-level Raw objects.
The stim event is the trial start and the raw annotation is cue onset exactly
two seconds later. MOABB's declared interval [2, 6] reproduces the same cue
anchor when epoching from the stim event. Filtering matches
the audited lineage: MNE Raw.filter(8, 32, method="iir", picks="data") on the
continuous run before any event-relative slicing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances


SUBJECTS = tuple(range(1, 10))
SESSIONS = ("0train", "1test")
CLASSES = ("left_hand", "right_hand", "feet", "tongue")
CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
)
SFREQ = 250.0
CUE_OFFSET_SAMPLES = 500
BASELINE_SAMPLES = 250
POST_SAMPLES = 1000
LOCAL_WINDOWS = 5
LOCAL_SAMPLES = 200
EXPECTED_TRIALS = 5184
REPRO_ATOL = 1e-12
REPRO_RTOL = 1e-12


class DataTimingAuditError(RuntimeError):
    """A fail-closed raw timing, numerical, or lineage reproduction error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False
    ) as handle:
        partial = Path(handle.name)
    try:
        np.savez_compressed(partial, **arrays)
        generated = partial.with_suffix(partial.suffix + ".npz")
        if generated.exists():
            partial.unlink(missing_ok=True)
            partial = generated
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _hardware() -> dict[str, Any]:
    memory_bytes = _command("sysctl", "-n", "hw.memsize")
    return {
        "cpu_model": _command("sysctl", "-n", "machdep.cpu.brand_string"),
        "logical_cores": os.cpu_count(),
        "ram_bytes": int(memory_bytes) if memory_bytes.isdigit() else None,
        "gpu": _command("system_profiler", "SPDisplaysDataType"),
        "disk": shutil.disk_usage(Path.cwd())._asdict(),
    }


def _raw_inventory(raw_data_dir: Path) -> list[dict[str, Any]]:
    source_dir = raw_data_dir / "MNE-bnci-data" / "~bci" / "database" / "001-2014"
    files = sorted(source_dir.glob("A??[TE].mat"))
    if len(files) != 18:
        raise DataTimingAuditError(
            f"RAW_EVENT_TIMING_MISMATCH: expected 18 MAT files, found {len(files)}"
        )
    return [
        {
            "path": str(path),
            "subject": int(path.name[1:3]),
            "session": "0train" if path.stem.endswith("T") else "1test",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def _metadata_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records).sort_values(
        ["subject", "session", "run", "event_sample"], kind="stable"
    ).reset_index(drop=True)
    frame["trial_id"] = (
        frame.groupby(["subject", "session"], sort=False).cumcount() + 1
    ).astype(np.int64)
    frame["run_trial_id"] = (
        frame.groupby(["subject", "session", "run"], sort=False).cumcount() + 1
    ).astype(np.int64)
    frame["trial_uid"] = [
        f"S{subject:02d}_{session}_T{trial_id:03d}"
        for subject, session, trial_id in frame[
            ["subject", "session", "trial_id"]
        ].itertuples(index=False, name=None)
    ]
    frame.insert(0, "sample_index", np.arange(len(frame), dtype=np.int64))
    return frame


def _validate_counts(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    if len(metadata) != EXPECTED_TRIALS:
        raise DataTimingAuditError(
            f"RAW_EVENT_TIMING_MISMATCH: expected {EXPECTED_TRIALS} trials, got {len(metadata)}"
        )
    if metadata["trial_uid"].duplicated().any():
        raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: duplicate trial_uid")
    counts = (
        metadata.groupby(["subject", "session", "run", "class_label"], sort=True)
        .size()
        .rename("n_trials")
        .reset_index()
    )
    if len(counts) != 9 * 2 * 6 * 4 or not (counts["n_trials"] == 12).all():
        raise DataTimingAuditError(
            "RAW_EVENT_TIMING_MISMATCH: expected 12 trials per subject/session/run/class"
        )
    return counts.to_dict(orient="records")


def _covariance_batches(
    raw_by_key: dict[tuple[int, str, int], Any], metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    estimator = Covariances(estimator="oas")
    c0 = np.empty((len(metadata), 22, 22), dtype=np.float64)
    local = np.empty((len(metadata), 5, 22, 22), dtype=np.float64)
    full = np.empty((len(metadata), 22, 22), dtype=np.float64)
    filter_rows: list[dict[str, Any]] = []
    for key, rows in metadata.groupby(["subject", "session", "run"], sort=False):
        raw = raw_by_key[key].copy().load_data()
        raw.filter(
            l_freq=8.0,
            h_freq=32.0,
            method="iir",
            picks="data",
            verbose=False,
        )
        if tuple(raw.ch_names[:22]) != CHANNELS:
            raise DataTimingAuditError("FILTER_IMPLEMENTATION_MISMATCH: EEG channel order")
        positions = rows.index.to_numpy(dtype=np.int64)
        baseline_batch = np.stack(
            [
                raw.get_data(
                    picks=list(CHANNELS),
                    start=int(row.baseline_start_sample),
                    stop=int(row.baseline_stop_sample),
                )
                for row in rows.itertuples()
            ]
        ).astype(np.float64, copy=False)
        post_batch = np.stack(
            [
                raw.get_data(
                    picks=list(CHANNELS),
                    start=int(row.post_start_sample),
                    stop=int(row.post_stop_sample),
                )
                for row in rows.itertuples()
            ]
        ).astype(np.float64, copy=False)
        if baseline_batch.shape != (48, 22, 250) or post_batch.shape != (48, 22, 1000):
            raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: extracted epoch shape")
        local_batch = (
            post_batch.reshape(48, 22, 5, 200)
            .transpose(0, 2, 1, 3)
            .reshape(48 * 5, 22, 200)
        )
        c0[positions] = estimator.transform(baseline_batch)
        full[positions] = estimator.transform(post_batch)
        local[positions] = estimator.transform(local_batch).reshape(48, 5, 22, 22)
        filter_rows.append(
            {
                "subject": int(key[0]),
                "session": str(key[1]),
                "run": int(key[2]),
                "n_times": int(raw.n_times),
                "sfreq": float(raw.info["sfreq"]),
                "method": "iir",
                "iir_default": "Butterworth order 4 SOS, zero-phase forward-backward",
                "picks": "data (EEG+EOG; stim excluded), EEG selected after filtering",
                "pad": "reflect_limited at continuous-run edges only",
                "event_boundary_padding": False,
                "cue_boundary_leakage": "possible by design: zero-phase continuous filtering uses future and past samples",
            }
        )
    for array in (c0, local, full):
        array[...] = 0.5 * (array + np.swapaxes(array, -1, -2))
    return c0, local, full, filter_rows


def _spd_summary(name: str, matrices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrices, dtype=np.float64).reshape(-1, 22, 22)
    if not np.isfinite(values).all():
        raise DataTimingAuditError(f"NUMERICAL_MISMATCH: {name} contains nonfinite values")
    symmetry = np.linalg.norm(values - values.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
        np.linalg.norm(values, axis=(1, 2)), np.finfo(float).tiny
    )
    eigenvalues = np.linalg.eigvalsh(values)
    if float(eigenvalues[:, 0].min()) <= 0.0:
        raise DataTimingAuditError(f"NUMERICAL_MISMATCH: {name} is not SPD")
    return {
        "name": name,
        "count": len(values),
        "finite": True,
        "min_eigenvalue": float(eigenvalues[:, 0].min()),
        "max_eigenvalue": float(eigenvalues[:, -1].max()),
        "max_symmetry_relative_error": float(symmetry.max()),
        "spd": True,
    }


def _reproduce_lineage(
    lineage_cache: Path,
    metadata: pd.DataFrame,
    local: np.ndarray,
) -> dict[str, Any]:
    covariance_path = lineage_cache / "covariances.npz"
    metadata_path = lineage_cache / "prepared_metadata.csv"
    if not covariance_path.is_file() or not metadata_path.is_file():
        raise DataTimingAuditError("VERSION_MISMATCH: frozen lineage cache unavailable")
    old_meta = pd.read_csv(metadata_path, dtype={"session": str, "run": str})
    new_meta = metadata[metadata["session"] == "0train"].reset_index(drop=True)
    identity = [
        "subject", "session", "run", "trial_id", "run_trial_id", "trial_uid", "class_label"
    ]
    if len(old_meta) != len(new_meta) or any(
        not np.array_equal(old_meta[column].astype(str), new_meta[column].astype(str))
        for column in identity
    ):
        raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: old/new trial identity mismatch")
    with np.load(covariance_path, allow_pickle=False) as archive:
        old_window = np.asarray(archive["window5"], dtype=np.float64)
    new_window = local[metadata["session"].to_numpy() == "0train"].reshape(-1, 22, 22)
    if old_window.shape != new_window.shape:
        raise DataTimingAuditError("VERSION_MISMATCH: old/new WINDOW5 shape mismatch")
    difference = np.abs(old_window - new_window)
    max_abs = float(difference.max())
    denominator = np.maximum(np.abs(old_window), np.finfo(float).tiny)
    max_rel = float((difference / denominator).max())
    passed = bool(np.allclose(old_window, new_window, atol=REPRO_ATOL, rtol=REPRO_RTOL))
    if not passed:
        raise DataTimingAuditError(
            f"FILTER_IMPLEMENTATION_MISMATCH: WINDOW5 max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        )
    return {
        "status": "PASS",
        "old_covariance_path": str(covariance_path),
        "old_covariance_sha256": _sha256(covariance_path),
        "old_metadata_path": str(metadata_path),
        "old_metadata_sha256": _sha256(metadata_path),
        "shape": list(old_window.shape),
        "atol": REPRO_ATOL,
        "rtol": REPRO_RTOL,
        "max_absolute_difference": max_abs,
        "max_elementwise_relative_difference": max_rel,
    }


def _report(payload: dict[str, Any]) -> str:
    reproduction = payload["old_window5_reproduction"]
    spd = payload["covariance_sanity"]
    return "\n".join(
        [
            "# BNCI2014_001 Data Timing Audit",
            "",
            f"DATA gate: **{payload['data_gate']}**",
            "",
            "- Dataset: `moabb.datasets.BNCI2014_001` under MOABB " + payload["versions"]["moabb"],
            "- Raw inventory: 18 MAT files, 9 subjects, 2 sessions, 6 runs/session.",
            "- Stim event: trial start; raw annotation: cue onset exactly +2.0 s / +500 samples later.",
            "- Baseline C0: `[cue-250, cue)`, exactly 250 samples.",
            "- Post-cue: `[cue, cue+1000)`, exactly 1000 samples; five non-overlapping 200-sample windows.",
            "- Channels: 22 ordered EEG channels; EOG1-3 and STI excluded from covariance.",
            "- Filtering: continuous run, MNE IIR 8-32 Hz, default order-4 Butterworth SOS, zero-phase forward-backward.",
            "- Leakage note: zero-phase filtering intentionally crosses the cue boundary. Padding occurs only at continuous-run edges, never at trial/window boundaries.",
            "- Covariance: pyRiemann OAS, float64, numerical symmetrization, no trace normalization, no added jitter.",
            f"- Trial balance: {payload['n_trials']} total; 12 per subject/session/run/class.",
            f"- Old WINDOW5 reproduction: PASS, max absolute difference `{reproduction['max_absolute_difference']:.3e}`.",
            "",
            "## SPD numerical gate",
            "",
        ]
        + [
            f"- {row['name']}: n={row['count']}, min eigenvalue={row['min_eigenvalue']:.6e}, max symmetry error={row['max_symmetry_relative_error']:.3e}."
            for row in spd
        ]
        + [
            "",
            "The full event-level timing table is `data_audit/event_timing.csv`; run/filter details are in `data_audit/filter_contract.csv`. No label enters filtering, slicing, or covariance estimation.",
            "",
        ]
    )


def run_data_timing_audit(
    project_root: str | Path,
    raw_data_dir: str | Path,
    lineage_cache: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Audit raw timing and atomically create C0/C1..C5/Cfull cache."""

    root = Path(project_root).resolve()
    output = root / "outputs/bnci2014_001_baseline_trajectory_identifiability_v0"
    cache = root / "cache/bnci2014_001_baseline_trajectory_identifiability_v0"
    audit_json = output / "data_timing_audit.json"
    cache_path = cache / "baseline_trajectory_covariances.npz"
    if resume and audit_json.is_file() and cache_path.is_file():
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        if payload.get("data_gate") == "PASS" and payload.get("cache_sha256") == _sha256(cache_path):
            return payload

    raw_dir = Path(raw_data_dir).resolve()
    lineage = Path(lineage_cache).resolve()
    os.environ["MNE_DATA"] = str(raw_dir)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(raw_dir)
    import importlib.metadata
    import mne
    from moabb.datasets import BNCI2014_001

    mne.set_log_level("ERROR")
    inventory = _raw_inventory(raw_dir)
    dataset = BNCI2014_001(subjects=list(SUBJECTS))
    if list(dataset.interval) != [2, 6] or dataset.event_id != {
        "left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4
    }:
        raise DataTimingAuditError("VERSION_MISMATCH: BNCI dataset interval/event map")
    nested = dataset.get_data(subjects=list(SUBJECTS))
    raw_by_key: dict[tuple[int, str, int], Any] = {}
    records: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        if tuple(nested[subject]) != SESSIONS:
            raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: session inventory")
        for session in SESSIONS:
            runs = nested[subject][session]
            if tuple(runs) != tuple(str(value) for value in range(6)):
                raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: run inventory")
            for run_string, raw in runs.items():
                run = int(run_string)
                key = (subject, session, run)
                raw_by_key[key] = raw
                if float(raw.info["sfreq"]) != SFREQ or tuple(raw.ch_names[:22]) != CHANNELS:
                    raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: sfreq/channel order")
                types = tuple(raw.get_channel_types())
                if types[:22] != ("eeg",) * 22 or types[22:25] != ("eog",) * 3:
                    raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: channel types/EOG")
                selected = [
                    annotation
                    for annotation in zip(
                        raw.annotations.onset,
                        raw.annotations.duration,
                        raw.annotations.description,
                        strict=True,
                    )
                    if str(annotation[2]) in CLASSES
                ]
                if len(selected) != 48:
                    raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: 48 events/run")
                stim_events = {
                    int(event[0]): int(event[2])
                    for event in mne.find_events(raw, shortest_event=0, verbose=False)
                    if int(event[2]) in dataset.event_id.values()
                }
                for onset, duration, description in selected:
                    cue_sample = int(raw.time_as_index(float(onset), use_rounding=True)[0])
                    event_sample = cue_sample - CUE_OFFSET_SAMPLES
                    expected_code = int(dataset.event_id[str(description)])
                    if stim_events.get(event_sample) != expected_code:
                        raise DataTimingAuditError(
                            "RAW_EVENT_TIMING_MISMATCH: cue annotation/stim event disagreement"
                        )
                    baseline_start = cue_sample - BASELINE_SAMPLES
                    baseline_stop = cue_sample
                    post_start = cue_sample
                    post_stop = cue_sample + POST_SAMPLES
                    if baseline_start < 0 or post_stop > raw.n_times:
                        raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: trial crosses run boundary")
                    if baseline_stop != post_start:
                        raise DataTimingAuditError("RAW_EVENT_TIMING_MISMATCH: baseline/post overlap")
                    records.append(
                        {
                            "subject": subject,
                            "session": session,
                            "run": run,
                            "class_label": str(description),
                            "event_code": expected_code,
                            "event_sample": event_sample,
                            "trial_start_sample": event_sample,
                            "trial_start_time_seconds": event_sample / SFREQ,
                            "cue_sample": cue_sample,
                            "cue_time_seconds": cue_sample / SFREQ,
                            "annotation_duration_seconds": float(duration),
                            "available_precue_samples_from_trial_start": cue_sample - event_sample,
                            "available_postcue_samples_to_run_end": int(raw.n_times - cue_sample),
                            "baseline_start_sample": baseline_start,
                            "baseline_stop_sample": baseline_stop,
                            "post_start_sample": post_start,
                            "post_stop_sample": post_stop,
                            "baseline_samples": baseline_stop - baseline_start,
                            "post_samples": post_stop - post_start,
                            "local_samples": LOCAL_SAMPLES,
                            "sfreq": SFREQ,
                            "run_n_times": int(raw.n_times),
                            "run_first_samp": int(raw.first_samp),
                        }
                    )

    metadata = _metadata_frame(records)
    count_rows = _validate_counts(metadata)
    c0, local, full, filter_rows = _covariance_batches(raw_by_key, metadata)
    covariance_sanity = [
        _spd_summary("C0", c0),
        _spd_summary("C1..C5", local),
        _spd_summary("Cfull", full),
    ]
    reproduction = _reproduce_lineage(lineage, metadata, local)
    _atomic_npz(
        cache_path,
        c0=c0,
        local=local,
        full=full,
        sample_index=metadata["sample_index"].to_numpy(np.int64),
        subject=metadata["subject"].to_numpy(np.int64),
        session=metadata["session"].to_numpy(str),
        run=metadata["run"].to_numpy(np.int64),
        trial_id=metadata["trial_id"].to_numpy(np.int64),
        run_trial_id=metadata["run_trial_id"].to_numpy(np.int64),
        trial_uid=metadata["trial_uid"].to_numpy(str),
        class_label=metadata["class_label"].to_numpy(str),
        channel_names=np.asarray(CHANNELS),
        sfreq=np.asarray(SFREQ),
    )
    payload = {
        "schema": "baseline-trajectory-data-timing-audit-v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_gate": "PASS",
        "dataset": "BNCI2014_001",
        "dataset_class": "moabb.datasets.BNCI2014_001",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mne", "moabb", "pyriemann", "scikit-learn", "numpy", "scipy")
        },
        "raw_inventory": inventory,
        "sessions": list(SESSIONS),
        "runs_per_session": 6,
        "event_codes": dataset.event_id,
        "event_definition": "stim event is trial start; raw annotation is cue onset at stim + dataset.interval[0] = +2.0 s",
        "sfreq": SFREQ,
        "channel_names": list(CHANNELS),
        "channel_types": ["eeg"] * 22,
        "excluded_channels": ["EOG1", "EOG2", "EOG3", "STI"],
        "event_relative_epoch_boundaries_seconds": {
            "baseline_relative_to_cue": [-1.0, 0.0],
            "post_relative_to_cue": [0.0, 4.0],
            "local_windows_relative_to_cue": [[0.8 * i, 0.8 * (i + 1)] for i in range(5)],
        },
        "sample_counts": {"baseline": 250, "post": 1000, "local_windows": [200] * 5},
        "filter_contract": filter_rows[0],
        "filter_application_point": "continuous raw/run before event slicing",
        "filter_edge_padding": "reflect_limited at run edges; no per-window filtering",
        "filter_leakage": "zero-phase IIR crosses the cue boundary; explicitly retained to reproduce frozen lineage",
        "covariance_contract": {
            "estimator": "pyriemann Covariances(estimator='oas') delegating to sklearn OAS",
            "dtype": "float64",
            "symmetrization": True,
            "trace_normalization": False,
            "added_jitter": False,
        },
        "n_trials": len(metadata),
        "trial_count_rows": count_rows,
        "trial_ids_unique": True,
        "run_boundary_safe": True,
        "baseline_post_overlap": False,
        "label_dependent_preprocessing": False,
        "covariance_sanity": covariance_sanity,
        "old_window5_reproduction": reproduction,
        "event_timing_csv": "data_audit/event_timing.csv",
        "cache_path": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "hardware": _hardware(),
        "platform": platform.platform(),
    }
    _atomic_csv(output / "data_audit/event_timing.csv", metadata)
    _atomic_csv(output / "data_audit/filter_contract.csv", pd.DataFrame(filter_rows))
    _atomic_csv(output / "data_audit/trial_counts.csv", pd.DataFrame(count_rows))
    _atomic_json(audit_json, payload)
    report = _report(payload)
    _atomic_text(output / "DATA_TIMING_AUDIT.md", report)
    _atomic_text(output / "data_audit/DATA_TIMING_AUDIT.md", report)
    return payload


__all__ = ["DataTimingAuditError", "run_data_timing_audit"]
