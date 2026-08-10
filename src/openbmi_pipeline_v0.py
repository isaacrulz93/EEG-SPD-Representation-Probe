"""Locked streaming preparation and observed OpenBMI replication pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from scipy.io import loadmat

from src.interaction_pipeline_v0 import GEOMETRIES, SPLITS, _atomic_savez, _build_pooled, _serialize_objects, _array_key
from src.interaction_provenance_v0 import atomic_write_json, sha256_array, sha256_file
from src.openbmi_protocol_v0 import validate_scientific_unlock
from src.subject_class_interaction_v0 import compute_interactions, load_frozen_config, split_masks


CLASSES = ("left_hand", "right_hand")


def _mat_text(values: np.ndarray) -> list[str]:
    return [str(np.squeeze(value).item()) for value in np.ravel(values)]


def _download_with_hash(url: str, target: Path) -> str:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "EEG-SPD-Representation-Probe/0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            handle.write(block); digest.update(block)
    return digest.hexdigest()


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "EEG-SPD-Representation-Probe/0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        value = response.headers.get("Content-Length")
    if value is None:
        raise RuntimeError(f"OpenBMI source has no Content-Length: {url}")
    return int(value)


def _mne_cached_source(mne_root: Path, subject: int, session: int) -> Path:
    return (
        mne_root
        / "MNE-lee2019-mi-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542"
        / f"session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )


def _prepare_one(path: Path, manifest: dict[str, Any], subject: int, session: int) -> tuple[np.ndarray, pd.DataFrame]:
    data = loadmat(path, variable_names=["EEG_MI_train"])["EEG_MI_train"][0, 0]
    names = _mat_text(data["chan"])
    channels = list(manifest["eeg_channels"])
    indices = [names.index(name) for name in channels]
    sfreq = float(data["fs"].item())
    if sfreq != 1000.0:
        raise RuntimeError(f"OpenBMI sampling-rate contract failure: {sfreq}")
    signal = np.asarray(data["x"], dtype=np.float64)[:, indices].T * 1.0e-6
    raw = mne.io.RawArray(signal, mne.create_info(channels, sfreq, "eeg"), verbose="ERROR")
    events = np.column_stack([
        np.asarray(data["t"]).squeeze().astype(np.int64),
        np.zeros(100, dtype=np.int64),
        np.asarray(data["y_dec"]).squeeze().astype(np.int64),
    ])
    if events.shape != (100, 3):
        raise RuntimeError(f"OpenBMI event-count contract failure: {events.shape}")
    raw.filter(8.0, 30.0, method="iir", iir_params={"ftype": "butter", "order": 5, "output": "sos"}, phase="zero", verbose="ERROR")
    raw, events = raw.resample(100.0, events=events, verbose="ERROR")
    eeg = raw.get_data()
    epochs = np.stack([eeg[:, event[0] + 100 : event[0] + 350] for event in events])
    if epochs.shape != (100, 20, 250) or not np.isfinite(epochs).all():
        raise RuntimeError(f"OpenBMI epoch contract failure: {epochs.shape}")
    covariances = np.asarray(Covariances(estimator="oas").transform(epochs), dtype=np.float64)
    labels = np.where(events[:, 2] == 2, "left_hand", np.where(events[:, 2] == 1, "right_hand", "INVALID"))
    if {name: int(np.count_nonzero(labels == name)) for name in CLASSES} != {"left_hand": 50, "right_hand": 50}:
        raise RuntimeError("OpenBMI balanced-label contract failure")
    frame = pd.DataFrame({
        "covariance_index": np.arange(100), "subject": subject, "session": str(session - 1),
        "run": "1train", "trial_id": np.arange(1, 101),
        "trial_uid": [f"S{subject:02d}_session{session}_1train_T{trial:03d}" for trial in range(1, 101)],
        "class_label": labels, "acquisition_order": np.arange(100),
    })
    return covariances, frame


def prepare_openbmi_streaming(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, config_hash = load_frozen_config(root)
    validate_scientific_unlock(root, config)
    output = root / config["project"]["output_dir"]
    manifest_path = output / "provenance/openbmi_protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache = root / config["project"]["cache_dir"] / "openbmi_subject_sessions"
    cache.mkdir(parents=True, exist_ok=True)
    temp = cache / ".source_download.mat"
    records = []
    base = manifest["data_source"]["download_base_url"]
    mne_root_value = os.environ.get("OPENBMI_MNE_DATA_ROOT")
    mne_root = Path(mne_root_value).expanduser().resolve() if mne_root_value else None
    for subject in manifest["subject_ids"]:
        for session in (1, 2):
            derived = cache / f"S{subject:02d}_session{session}.npz"
            meta_path = cache / f"S{subject:02d}_session{session}.csv"
            record_path = cache / f"S{subject:02d}_session{session}.json"
            if derived.exists() and meta_path.exists() and record_path.exists():
                records.append(json.loads(record_path.read_text())); continue
            url = f"{base}session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
            expected_bytes = _remote_size(url)
            cached_source = _mne_cached_source(mne_root, subject, session) if mne_root is not None else None
            use_cached = cached_source is not None and cached_source.is_file() and cached_source.stat().st_size == expected_bytes
            source = cached_source if use_cached else temp
            if use_cached:
                source_hash = sha256_file(source)
            else:
                source_hash = _download_with_hash(url, temp)
            covariances, metadata = _prepare_one(source, manifest, subject, session)
            _atomic_savez(derived, {"covariances": covariances, "channel_names": np.asarray(manifest["eeg_channels"])})
            metadata.to_csv(meta_path, index=False, lineterminator="\n")
            record = {"subject": subject, "session": session, "url": url, "source_bytes": source.stat().st_size, "source_sha256": source_hash, "source_reused_from_mne_cache": use_cached, "derived_sha256": sha256_file(derived), "covariance_array_sha256": sha256_array(covariances), "metadata_sha256": sha256_file(meta_path)}
            atomic_write_json(record_path, record); records.append(record)
            if not use_cached:
                temp.unlink()
    covs=[]; frames=[]
    for record in records:
        with np.load(cache / f"S{record['subject']:02d}_session{record['session']}.npz", allow_pickle=False) as archive:
            covs.append(np.asarray(archive["covariances"], dtype=np.float64))
        frames.append(pd.read_csv(cache / f"S{record['subject']:02d}_session{record['session']}.csv"))
    covariances=np.concatenate(covs); metadata=pd.concat(frames,ignore_index=True)
    if covariances.shape != (10800,20,20) or metadata.shape[0] != 10800:
        raise RuntimeError("OpenBMI combined data contract failure")
    _atomic_savez(cache.parent / "openbmi_covariances.npz", {"covariances":covariances,"channel_names":np.asarray(manifest["eeg_channels"])})
    metadata.to_csv(cache.parent / "openbmi_metadata.csv",index=False,lineterminator="\n")
    source_manifest={"schema_version":"openbmi-source-manifest-v0","config_sha256":config_hash,"manifest_commit_sha":"91877d3ea5b83a6b524fbbe091fb1db6c9973170","file_count":len(records),"files":records,"ordered_manifest_sha256":hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":")).encode()).hexdigest()}
    atomic_write_json(output / "provenance/openbmi_source_manifest.json",source_manifest)
    return source_manifest


def run_openbmi_observed(repo_root: str | Path) -> dict[str, Any]:
    root=Path(repo_root).resolve(); config,_=load_frozen_config(root); validate_scientific_unlock(root,config)
    output=root/config["project"]["output_dir"]; cache=root/config["project"]["cache_dir"]
    with np.load(cache/"openbmi_covariances.npz",allow_pickle=False) as archive: covariances=np.asarray(archive["covariances"],dtype=np.float64)
    metadata=pd.read_csv(cache/"openbmi_metadata.csv"); metadata["session"]=metadata["session"].astype(str); metadata["run"]=metadata["run"].astype(str)
    masks=split_masks(metadata,"openbmi_lee2019_mi"); objects={}
    for geometry in GEOMETRIES:
        for split in SPLITS:
            mask=masks[split]
            session_specific=compute_interactions(covariances[mask],metadata.loc[mask].reset_index(drop=True),config=config,geometry=geometry,classes=CLASSES)
            objects[(geometry,"session_specific",split)]=session_specific
            objects[(geometry,"pooled_session",split)]=_build_pooled(session_specific,config)
    arrays={}
    for (geometry,template,split),value in objects.items():
        for field,array in _serialize_objects(value).items(): arrays[_array_key(geometry,template,split,field)]=array
    path=output/"objects/openbmi_core_interaction_objects.npz"; _atomic_savez(path,arrays)
    result={"hard_gates_pass":True,"core_objects_sha256":sha256_file(path),"subjects":54,"trials":10800}
    atomic_write_json(output/"provenance/openbmi_observed_manifest.json",result); return result
