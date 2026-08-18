"""Direct MAT streaming and frozen preprocessing for Stieger2021 V0.

No function in this module computes a population statistic. Outcome fields are
serialized into a separate sealed record and are never consumed by the compact
covariance builder or eligibility code.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn.covariance import OAS


FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/13123148"
FILE_RE = re.compile(r"^S(?P<subject>\d+)_Session_(?P<session>\d+)\.mat$")
SEALED_FIELDS = ("result", "forcedresult", "targethitnumber", "performance")
REQUIRED_TRIAL_FIELDS = ("tasknumber", "runnumber", "trialnumber", "targetnumber", "artifact")


class StiegerDataContractError(RuntimeError):
    """Fail-closed source, schema, numerical, or provenance error."""


@dataclass(frozen=True)
class SourceFile:
    subject: int
    session: int
    figshare_file_id: int
    filename: str
    url: str
    reported_size: int
    reported_md5: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "session": self.session,
            "figshare_file_id": self.figshare_file_id,
            "filename": self.filename,
            "url": self.url,
            "reported_size": self.reported_size,
            "reported_md5": self.reported_md5,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.md5()  # nosec B324 - required official integrity checksum, not security
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_figshare_article(api_url: str = FIGSHARE_ARTICLE_API) -> dict[str, Any]:
    request = urllib.request.Request(api_url, headers={"User-Agent": "EEG-SPD-Representation-Probe/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310 - frozen HTTPS URL
        return json.loads(response.read().decode("utf-8"))


def select_source_files(
    article: Mapping[str, Any], subjects: Iterable[int] = range(1, 63), sessions: Sequence[int] = (2, 3)
) -> list[SourceFile]:
    wanted = {(int(s), int(q)) for s in subjects for q in sessions}
    selected: dict[tuple[int, int], SourceFile] = {}
    for entry in article.get("files", []):
        match = FILE_RE.match(str(entry.get("name", "")))
        if not match:
            continue
        key = (int(match.group("subject")), int(match.group("session")))
        if key not in wanted:
            continue
        if key in selected:
            raise StiegerDataContractError(f"duplicate official source pair {key}")
        checksum = str(entry.get("computed_md5") or entry.get("supplied_md5") or "").lower()
        if len(checksum) != 32:
            raise StiegerDataContractError(f"missing official MD5 for {entry.get('name')}")
        selected[key] = SourceFile(
            subject=key[0],
            session=key[1],
            figshare_file_id=int(entry["id"]),
            filename=str(entry["name"]),
            url=str(entry["download_url"]),
            reported_size=int(entry["size"]),
            reported_md5=checksum,
        )
    missing = sorted(wanted - set(selected))
    if missing:
        raise StiegerDataContractError(f"official manifest missing {len(missing)} pairs: {missing[:8]}")
    return [selected[key] for key in sorted(selected)]


def canonical_source_manifest(files: Sequence[SourceFile], article: Mapping[str, Any]) -> dict[str, Any]:
    rows = [item.as_dict() for item in files]
    payload = {
        "article_id": int(article["id"]),
        "article_url": str(article.get("url_public_api", FIGSHARE_ARTICLE_API)),
        "doi": str(article.get("doi", "")),
        "license": article.get("license"),
        "selected_count": len(rows),
        "selected_total_bytes": sum(row["reported_size"] for row in rows),
        "files": rows,
    }
    payload["canonical_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def stream_download(source: SourceFile, destination: Path, chunk_bytes: int = 8 << 20) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    md5 = hashlib.md5()  # nosec B324 - official Figshare checksum
    total = destination.stat().st_size if destination.exists() else 0
    if total > source.reported_size:
        raise StiegerDataContractError(f"partial file exceeds official size for {source.filename}")
    if total:
        with destination.open("rb") as existing:
            while chunk := existing.read(chunk_bytes):
                sha.update(chunk)
                md5.update(chunk)
    if total == source.reported_size:
        result = {"bytes": total, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}
        if result["md5"].lower() != source.reported_md5.lower():
            raise StiegerDataContractError(f"MD5 mismatch for completed retained {source.filename}")
        return result
    aria2 = shutil.which("aria2c")
    if aria2:
        command = [
            aria2,
            "--continue=true",
            "--max-connection-per-server=4",
            "--split=4",
            "--min-split-size=32M",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--summary-interval=0",
            f"--dir={destination.parent}",
            f"--out={destination.name}",
            source.url,
        ]
        stalled_attempts = 0
        last_error = ""
        for _ in range(64):
            before = destination.stat().st_size if destination.exists() else 0
            completed = subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            after = destination.stat().st_size if destination.exists() else 0
            if completed.returncode == 0 and after == source.reported_size:
                break
            last_error = completed.stderr[-500:]
            stalled_attempts = stalled_attempts + 1 if after <= before else 0
            if stalled_attempts >= 3:
                raise StiegerDataContractError(
                    f"bounded aria2 transfer stalled for {source.filename}; partial retained: {last_error}"
                )
        else:
            raise StiegerDataContractError(
                f"bounded aria2 transfer exceeded 64 fresh-redirect attempts for {source.filename}; partial retained"
            )
        total = destination.stat().st_size
        result = {"bytes": total, "sha256": sha256_file(destination), "md5": md5_file(destination)}
        if total != source.reported_size:
            raise StiegerDataContractError(f"size mismatch for {source.filename}: {total} != {source.reported_size}")
        if result["md5"].lower() != source.reported_md5.lower():
            raise StiegerDataContractError(f"MD5 mismatch for {source.filename}")
        return result
    headers = {"User-Agent": "EEG-SPD-Representation-Probe/1.0"}
    if 0 < total < source.reported_size:
        headers["Range"] = f"bytes={total}-"
    request = urllib.request.Request(source.url, headers=headers)
    mode = "ab" if total else "wb"
    with urllib.request.urlopen(request, timeout=300) as response, destination.open(mode) as output:  # nosec B310
        if total and total < source.reported_size:
            status = getattr(response, "status", response.getcode())
            content_range = str(response.headers.get("Content-Range", ""))
            if int(status) != 206 or not content_range.startswith(f"bytes {total}-"):
                raise StiegerDataContractError(
                    f"server did not honor safe resume for {source.filename}; partial is retained"
                )
        while chunk := response.read(chunk_bytes):
            output.write(chunk)
            sha.update(chunk)
            md5.update(chunk)
            total += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    result = {"bytes": total, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}
    if total != source.reported_size:
        raise StiegerDataContractError(f"size mismatch for {source.filename}: {total} != {source.reported_size}")
    if result["md5"].lower() != source.reported_md5.lower():
        raise StiegerDataContractError(f"MD5 mismatch for {source.filename}")
    return result


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return getattr(obj, name, default)


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value).squeeze()
    if array.size != 1:
        raise StiegerDataContractError(f"{name} is not scalar: shape {array.shape}")
    return float(array)


def _int_field(record: Any, name: str, default: int | None = None) -> int:
    value = _field(record, name, None)
    if value is None:
        if default is None:
            raise StiegerDataContractError(f"missing TrialData.{name}")
        return int(default)
    return int(round(_scalar(value, f"TrialData.{name}")))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    array = np.asarray(value).squeeze()
    if array.size == 1:
        try:
            return str(array.item())
        except ValueError:
            pass
    return json.dumps(np.asarray(value).tolist(), sort_keys=True, default=str)


def normalize_channel_name(name: Any) -> str:
    text = str(name).strip().replace(" ", "")
    upper = text.upper()
    replacements = {"FPZ": "Fpz", "FP1": "Fp1", "FP2": "Fp2", "AFZ": "AFz", "FCZ": "FCz", "CPZ": "CPz", "POZ": "POz", "OZ": "Oz", "CZ": "Cz", "PZ": "Pz", "FZ": "Fz"}
    return replacements.get(upper, upper)


def _channel_labels(bci: Any) -> list[str]:
    chaninfo = _field(bci, "chaninfo")
    labels = _field(chaninfo, "label", _field(chaninfo, "labels"))
    if labels is None:
        raise StiegerDataContractError("BCI.chaninfo.label missing")
    return [normalize_channel_name(item) for item in _as_sequence(labels)]


def _noise_indices(bci: Any, n_channels: int) -> list[int]:
    chaninfo = _field(bci, "chaninfo")
    raw = _field(bci, "noisechan", _field(chaninfo, "noisechan", []))
    if raw is None or np.asarray(raw).size == 0:
        return []
    values = sorted({int(round(float(v))) - 1 for v in np.asarray(raw).reshape(-1)})
    if any(v < 0 or v >= n_channels for v in values):
        raise StiegerDataContractError(f"noisechan outside one-based montage range: {values}")
    return values


def _extract_recorded_positions(bci: Any, labels: Sequence[str]) -> dict[str, np.ndarray] | None:
    chaninfo = _field(bci, "chaninfo")
    candidates = (
        _field(chaninfo, "coordinates"),
        _field(chaninfo, "positions"),
        _field(chaninfo, "electrodes"),
        _field(bci, "electrode_coordinates"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_array = np.asarray(candidate).squeeze()
        if candidate_array.dtype == object:
            parsed: dict[str, np.ndarray] = {}
            for record in candidate_array.reshape(-1):
                label = _field(record, "label")
                coordinates = [_field(record, axis) for axis in ("X", "Y", "Z")]
                if label is None or any(value is None for value in coordinates):
                    continue
                parsed[normalize_channel_name(label)] = np.asarray(coordinates, dtype=np.float64)
            if all(label in parsed for label in labels):
                return {label: parsed[label] for label in labels}
            continue
        try:
            array = np.asarray(candidate, dtype=float).squeeze()
        except (TypeError, ValueError):
            continue
        if array.ndim != 2:
            continue
        if array.shape == (3, len(labels)):
            array = array.T
        if array.shape != (len(labels), 3) or not np.all(np.isfinite(array)):
            continue
        norms = np.linalg.norm(array, axis=1)
        if np.any(norms <= 0):
            continue
        scale = np.median(norms)
        if scale > 1.0:
            array = array / 1000.0 if scale > 10.0 else array / 100.0
        return {label: array[i] for i, label in enumerate(labels)}
    return None


def interpolation_matrix(
    labels: Sequence[str], bad_indices: Sequence[int], recorded_positions: Mapping[str, np.ndarray] | None = None
) -> np.ndarray:
    """Return deterministic complete-montage interpolation as a linear map."""
    n_channels = len(labels)
    if not bad_indices:
        return np.eye(n_channels, dtype=np.float64)
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - environment gate
        raise StiegerDataContractError("MNE required for frozen bad-channel interpolation") from exc
    info = mne.create_info(list(labels), sfreq=100.0, ch_types="eeg", verbose=False)
    if recorded_positions:
        montage = mne.channels.make_dig_montage(ch_pos=dict(recorded_positions), coord_frame="head")
    else:
        montage = mne.channels.make_standard_montage("standard_1005")
    try:
        info.set_montage(montage, match_case=False, on_missing="raise", verbose=False)
        raw = mne.io.RawArray(np.eye(n_channels, dtype=np.float64), info, verbose=False)
        raw.info["bads"] = [labels[i] for i in bad_indices]
        raw.interpolate_bads(reset_bads=False, mode="accurate", verbose=False)
    except Exception as exc:
        raise StiegerDataContractError(f"deterministic interpolation failed: {exc}") from exc
    matrix = np.asarray(raw.get_data(), dtype=np.float64)
    if matrix.shape != (n_channels, n_channels) or not np.all(np.isfinite(matrix)):
        raise StiegerDataContractError("invalid interpolation matrix")
    return matrix


def _apply_interpolation(trial: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(trial, dtype=np.float64)
    if values.ndim != 2:
        raise StiegerDataContractError(f"trial data must be 2D, got {values.shape}")
    n_channels = matrix.shape[0]
    if values.shape[1] == n_channels:
        values = values.T
    elif values.shape[0] != n_channels:
        raise StiegerDataContractError(f"trial channel dimension does not match montage: {values.shape}")
    return matrix @ values


def filter_resample_trial(eeg_channels_by_samples: np.ndarray, raw_sfreq: float, target_sfreq: float = 100.0) -> np.ndarray:
    values = np.asarray(eeg_channels_by_samples, dtype=np.float64) * 1.0e-6
    if values.ndim != 2 or values.shape[1] < int(raw_sfreq):
        raise StiegerDataContractError("trial is too short for frozen zero-phase filtering")
    if not np.all(np.isfinite(values)):
        raise StiegerDataContractError("nonfinite raw EEG")
    sos = signal.butter(5, [8.0, 30.0], btype="bandpass", fs=float(raw_sfreq), output="sos")
    filtered = signal.sosfiltfilt(sos, values, axis=1)
    up = int(round(target_sfreq))
    down = int(round(raw_sfreq))
    if abs(raw_sfreq - down) > 1.0e-9 or abs(target_sfreq - up) > 1.0e-9:
        raise StiegerDataContractError("frozen resampling requires integer source and target rates")
    resampled = signal.resample_poly(filtered, up=up, down=down, axis=1)
    if not np.all(np.isfinite(resampled)):
        raise StiegerDataContractError("nonfinite filtered/resampled EEG")
    return np.asarray(resampled, dtype=np.float64)


def crop_epoch(resampled: np.ndarray, source_time_start: float, window: Sequence[float], sfreq: float = 100.0) -> np.ndarray:
    times = float(source_time_start) + np.arange(resampled.shape[1], dtype=np.float64) / float(sfreq)
    mask = (times >= float(window[0]) - 1e-12) & (times < float(window[1]) - 1e-12)
    expected = int(round((float(window[1]) - float(window[0])) * sfreq))
    if int(mask.sum()) != expected:
        raise StiegerDataContractError(f"window {tuple(window)} unavailable exactly: {int(mask.sum())} != {expected}")
    return resampled[:, mask]


def time_vector_seconds(exact_time: np.ndarray, raw_sfreq: float) -> tuple[np.ndarray, str]:
    """Convert the official exact time vector to seconds without changing it in provenance."""
    values = np.asarray(exact_time, dtype=np.float64).reshape(-1)
    step = float(np.median(np.diff(values)))
    seconds_step = 1.0 / float(raw_sfreq)
    if math.isclose(step, seconds_step, rel_tol=1e-8, abs_tol=1e-12):
        return values.copy(), "seconds"
    milliseconds_step = 1000.0 / float(raw_sfreq)
    if math.isclose(step, milliseconds_step, rel_tol=1e-8, abs_tol=1e-9):
        return values / 1000.0, "milliseconds"
    raise StiegerDataContractError(
        f"unrecognized time-vector unit: step={step}, sampling_rate={raw_sfreq}"
    )


def oas_covariance(epoch: np.ndarray) -> np.ndarray:
    values = np.asarray(epoch, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise StiegerDataContractError("invalid epoch for covariance")
    covariance = OAS(store_precision=False, assume_centered=False).fit(values.T).covariance_
    covariance = np.asarray((covariance + covariance.T) / 2.0, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= 0.0:
        raise StiegerDataContractError("OAS covariance failed finite/SPD gate")
    return covariance


def _identity_from_bci(bci: Any, source: SourceFile) -> tuple[int, int]:
    metadata = _field(bci, "metadata")
    subject_values = (_field(bci, "subject"), _field(metadata, "subject"), _field(metadata, "subjectnumber"))
    session_values = (_field(bci, "session"), _field(metadata, "session"), _field(metadata, "sessionnumber"))
    parsed_subject = next((int(round(_scalar(v, "subject identity"))) for v in subject_values if v is not None), source.subject)
    parsed_session = next((int(round(_scalar(v, "session identity"))) for v in session_values if v is not None), source.session)
    if (parsed_subject, parsed_session) != (source.subject, source.session):
        raise StiegerDataContractError(
            f"parsed identity {(parsed_subject, parsed_session)} != manifest {(source.subject, source.session)}"
        )
    return parsed_subject, parsed_session


def parse_and_preprocess_mat(
    mat_path: Path, source: SourceFile, config: Mapping[str, Any], compact_path: Path, sealed_path: Path
) -> dict[str, Any]:
    """Parse a session directly, create compact task-3 covariances, and seal outcomes."""
    loaded = loadmat(mat_path, squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=True)
    if "BCI" not in loaded:
        raise StiegerDataContractError("MAT does not contain BCI")
    bci = loaded["BCI"]
    subject, session_id = _identity_from_bci(bci, source)
    raw_sfreq = _scalar(_field(bci, "SRATE"), "BCI.SRATE")
    labels = _channel_labels(bci)
    if len(labels) != len(set(labels)):
        raise StiegerDataContractError("duplicate normalized channel labels")
    primary_order = list(config["channels"]["primary_order"])
    label_index = {name: i for i, name in enumerate(labels)}
    missing = [name for name in primary_order if name not in label_index]
    if missing:
        raise StiegerDataContractError(f"missing primary channels: {missing}")
    noise_indices = _noise_indices(bci, len(labels))
    primary_bad = sorted(set(noise_indices) & {label_index[name] for name in primary_order})
    if len(primary_bad) > int(config["channels"]["maximum_bad_primary_channels"]):
        raise StiegerDataContractError(f"too many bad primary channels: {len(primary_bad)}")
    chaninfo = _field(bci, "chaninfo")
    positions_value = _field(bci, "positionsrecorded", _field(chaninfo, "positionsrecorded", 0))
    positions_recorded = bool(_int_field({"value": positions_value}, "value", default=0))
    recorded_positions = _extract_recorded_positions(bci, labels)
    if positions_recorded and recorded_positions is None:
        raise StiegerDataContractError("positionsrecorded is true but coordinates could not be parsed")
    interpolation = interpolation_matrix(labels, noise_indices, recorded_positions)
    primary_indices = np.asarray([label_index[name] for name in primary_order], dtype=np.int64)

    data_trials = _as_sequence(_field(bci, "data"))
    time_trials = _as_sequence(_field(bci, "time"))
    trial_records = _as_sequence(_field(bci, "TrialData"))
    if not (len(data_trials) == len(time_trials) == len(trial_records)):
        raise StiegerDataContractError("BCI data/time/TrialData lengths differ")

    primary_cov: list[np.ndarray] = []
    pretarget_cov: list[np.ndarray] = []
    feedback_cov: list[np.ndarray] = []
    feedback_indices: list[int] = []
    targetnumber: list[int] = []
    runnumber: list[int] = []
    trialnumber: list[int] = []
    acquisition_index: list[int] = []
    triallength: list[float] = []
    time_start: list[float] = []
    time_stop: list[float] = []
    all_task3_count = {str(c): 0 for c in range(1, 5)}
    artifact_free_count = {str(c): 0 for c in range(1, 5)}
    artifact_count = {str(c): 0 for c in range(1, 5)}
    sealed_rows: list[dict[str, Any]] = []
    all_tasknumber: list[int] = []
    all_targetnumber: list[int] = []
    all_runnumber: list[int] = []
    all_trialnumber: list[int] = []
    all_triallength: list[float] = []
    all_artifact: list[int] = []
    unique_time_vectors: list[np.ndarray] = []
    unique_time_lookup: dict[str, int] = {}
    all_time_vector_index: list[int] = []
    observed_time_units: set[str] = set()

    windows = config["preprocessing"]
    for index, (trial, trial_time, record) in enumerate(zip(data_trials, time_trials, trial_records, strict=True)):
        for required in REQUIRED_TRIAL_FIELDS:
            if _field(record, required) is None:
                raise StiegerDataContractError(f"missing TrialData.{required} at acquisition index {index}")
        task = _int_field(record, "tasknumber")
        target = _int_field(record, "targetnumber")
        artifact = _int_field(record, "artifact")
        exact_times = np.asarray(trial_time, dtype=np.float64).reshape(-1)
        if exact_times.size < 2 or not np.all(np.isfinite(exact_times)) or not np.all(np.diff(exact_times) > 0):
            raise StiegerDataContractError(f"invalid exact time vector at acquisition index {index}")
        times_seconds, time_unit = time_vector_seconds(exact_times, raw_sfreq)
        observed_time_units.add(time_unit)
        time_digest = hashlib.sha256(exact_times.tobytes(order="C")).hexdigest()
        if time_digest not in unique_time_lookup:
            unique_time_lookup[time_digest] = len(unique_time_vectors)
            unique_time_vectors.append(exact_times.copy())
        all_time_vector_index.append(unique_time_lookup[time_digest])
        all_tasknumber.append(task)
        all_targetnumber.append(target)
        all_runnumber.append(_int_field(record, "runnumber"))
        all_trialnumber.append(_int_field(record, "trialnumber"))
        all_triallength.append(_scalar(_field(record, "triallength", np.nan), "TrialData.triallength"))
        all_artifact.append(artifact)
        sealed_rows.append(
            {
                "acquisition_index": index,
                "tasknumber": task,
                "targetnumber": target,
                **{field: _safe_text(_field(record, field)) for field in SEALED_FIELDS},
            }
        )
        if task != int(config["dataset"]["primary_tasknumber"]):
            continue
        if target not in (1, 2, 3, 4):
            raise StiegerDataContractError(f"invalid task-3 targetnumber {target}")
        all_task3_count[str(target)] += 1
        if artifact != int(windows["include_if_artifact_equals"]):
            artifact_count[str(target)] += 1
            continue
        artifact_free_count[str(target)] += 1
        times = times_seconds
        interpolated = _apply_interpolation(np.asarray(trial), interpolation)[primary_indices]
        resampled = filter_resample_trial(interpolated, raw_sfreq, float(windows["resample_hz"]))
        primary = crop_epoch(resampled, float(times[0]), windows["epoch_primary_seconds"])
        pretarget = crop_epoch(resampled, float(times[0]), windows["epoch_pretarget_seconds"])
        primary_cov.append(oas_covariance(primary))
        pretarget_cov.append(oas_covariance(pretarget))
        compact_index = len(primary_cov) - 1
        try:
            feedback = crop_epoch(resampled, float(times[0]), windows["epoch_feedback_sensitivity_seconds"])
        except StiegerDataContractError:
            feedback = None
        if feedback is not None:
            feedback_cov.append(oas_covariance(feedback))
            feedback_indices.append(compact_index)
        targetnumber.append(target)
        runnumber.append(_int_field(record, "runnumber"))
        trialnumber.append(_int_field(record, "trialnumber"))
        acquisition_index.append(index)
        triallength.append(_scalar(_field(record, "triallength", np.nan), "TrialData.triallength"))
        time_start.append(float(times[0]))
        time_stop.append(float(times[-1]))

    n_primary = len(primary_cov)
    if n_primary == 0:
        raise StiegerDataContractError("no artifact-free task-3 trials")
    time_offsets = np.zeros(len(unique_time_vectors) + 1, dtype=np.int64)
    if unique_time_vectors:
        time_offsets[1:] = np.cumsum([len(value) for value in unique_time_vectors])
        time_values = np.concatenate(unique_time_vectors).astype(np.float64, copy=False)
    else:  # pragma: no cover - a valid BCI session always has trials
        time_values = np.empty(0, dtype=np.float64)
    metadata = _field(bci, "metadata")
    mbsr_value = _field(bci, "MBSRsubject", _field(metadata, "MBSRsubject", -1))
    try:
        mbsr_subject = _int_field({"value": mbsr_value}, "value", default=-1)
    except (StiegerDataContractError, TypeError, ValueError):
        mbsr_subject = -1
    arrays = {
        "primary_covariances": np.stack(primary_cov).astype(np.float64),
        "pretarget_covariances": np.stack(pretarget_cov).astype(np.float64),
        "feedback_covariances": np.stack(feedback_cov).astype(np.float64) if feedback_cov else np.empty((0, 20, 20)),
        "feedback_compact_indices": np.asarray(feedback_indices, dtype=np.int64),
        "targetnumber": np.asarray(targetnumber, dtype=np.int8),
        "runnumber": np.asarray(runnumber, dtype=np.int16),
        "trialnumber": np.asarray(trialnumber, dtype=np.int32),
        "acquisition_index": np.asarray(acquisition_index, dtype=np.int32),
        "triallength": np.asarray(triallength, dtype=np.float64),
        "time_start": np.asarray(time_start, dtype=np.float64),
        "time_stop": np.asarray(time_stop, dtype=np.float64),
        "subject": np.asarray(subject, dtype=np.int16),
        "session": np.asarray(session_id, dtype=np.int8),
        "raw_sfreq": np.asarray(raw_sfreq, dtype=np.float64),
        "primary_channel_order": np.asarray(primary_order, dtype="U8"),
        "recorded_channel_order": np.asarray(labels, dtype="U8"),
        "noise_channel_indices_zero_based": np.asarray(noise_indices, dtype=np.int16),
        "positionsrecorded": np.asarray(positions_recorded, dtype=np.bool_),
        "recorded_positions_used": np.asarray(recorded_positions is not None, dtype=np.bool_),
        "time_vector_unit": np.asarray(sorted(observed_time_units), dtype="U16"),
        "MBSRsubject": np.asarray(mbsr_subject, dtype=np.int8),
        "all_tasknumber": np.asarray(all_tasknumber, dtype=np.int8),
        "all_targetnumber": np.asarray(all_targetnumber, dtype=np.int8),
        "all_runnumber": np.asarray(all_runnumber, dtype=np.int16),
        "all_trialnumber": np.asarray(all_trialnumber, dtype=np.int32),
        "all_triallength": np.asarray(all_triallength, dtype=np.float64),
        "all_artifact": np.asarray(all_artifact, dtype=np.int8),
        "all_time_vector_index": np.asarray(all_time_vector_index, dtype=np.int16),
        "unique_time_vector_offsets": time_offsets,
        "unique_time_vector_values": time_values,
    }
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(compact_path, **arrays)
    validate_compact_object(compact_path, source, config)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_payload = {
        "contract": "EVALUATION_AND_OUTCOME_FIELDS_SEALED_NOT_SCIENTIFIC_INPUT",
        "subject": subject,
        "session": session_id,
        "fields": list(SEALED_FIELDS),
        "rows": sealed_rows,
    }
    sealed_path.write_bytes(_canonical_json(sealed_payload) + b"\n")
    summary = {
        "subject": subject,
        "session": session_id,
        "raw_sfreq": raw_sfreq,
        "recorded_channels": len(labels),
        "noise_channel_indices_zero_based": noise_indices,
        "primary_bad_count": len(primary_bad),
        "positionsrecorded": positions_recorded,
        "recorded_positions_used": recorded_positions is not None,
        "time_vector_units": sorted(observed_time_units),
        "MBSRsubject": mbsr_subject,
        "all_trials": len(all_tasknumber),
        "unique_exact_time_vectors": len(unique_time_vectors),
        "task3_total_by_class": all_task3_count,
        "task3_artifact_by_class": artifact_count,
        "task3_primary_by_class": artifact_free_count,
        "primary_trials": n_primary,
        "feedback_trials": len(feedback_cov),
        "compact_sha256": sha256_file(compact_path),
        "compact_bytes": compact_path.stat().st_size,
        "sealed_sha256": sha256_file(sealed_path),
    }
    return summary


def validate_compact_object(path: Path, source: SourceFile, config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "primary_covariances",
        "pretarget_covariances",
        "targetnumber",
        "acquisition_index",
        "subject",
        "session",
        "primary_channel_order",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise StiegerDataContractError(f"compact object missing keys {sorted(missing)}")
        subject, session_id = int(data["subject"]), int(data["session"])
        if (subject, session_id) != (source.subject, source.session):
            raise StiegerDataContractError("compact object identity mismatch")
        covariance = np.asarray(data["primary_covariances"], dtype=np.float64)
        pretarget = np.asarray(data["pretarget_covariances"], dtype=np.float64)
        labels = np.asarray(data["targetnumber"], dtype=np.int64)
        if covariance.ndim != 3 or covariance.shape[1:] != (20, 20) or pretarget.shape != covariance.shape:
            raise StiegerDataContractError(f"invalid compact covariance shape {covariance.shape}/{pretarget.shape}")
        if len(labels) != len(covariance) or not set(np.unique(labels)).issubset({1, 2, 3, 4}):
            raise StiegerDataContractError("invalid compact labels")
        if list(data["primary_channel_order"].astype(str)) != list(config["channels"]["primary_order"]):
            raise StiegerDataContractError("compact primary channel order mismatch")
        for stack in (covariance, pretarget):
            if not np.all(np.isfinite(stack)) or not np.allclose(stack, stack.transpose(0, 2, 1), atol=0.0, rtol=0.0):
                raise StiegerDataContractError("compact covariance finite/symmetry failure")
            if np.any(np.linalg.eigvalsh(stack)[:, 0] <= 0.0):
                raise StiegerDataContractError("compact covariance SPD failure")
    return {"subject": subject, "session": session_id, "trials": len(labels), "sha256": sha256_file(path)}


def process_source_file(source: SourceFile, config: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    """Download, compact, reread-validate, record, then safely delete one MAT."""
    raw_dir = cache_dir / "raw_inflight"
    compact_dir = cache_dir / "sessions"
    sealed_dir = cache_dir / "sealed_outcomes"
    record_dir = cache_dir / "records"
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    compact_path = compact_dir / f"S{source.subject:02d}_session{source.session}.npz"
    sealed_path = sealed_dir / f"S{source.subject:02d}_session{source.session}.json"
    record_path = record_dir / f"S{source.subject:02d}_session{source.session}.json"
    if record_path.exists() and compact_path.exists() and sealed_path.exists():
        record = json.loads(record_path.read_text())
        validate_compact_object(compact_path, source, config)
        if record.get("compact_sha256") != sha256_file(compact_path):
            raise StiegerDataContractError("resume compact hash mismatch")
        return record

    raw_path = raw_dir / source.filename
    download = stream_download(source, raw_path)
    raw_deleted = False
    try:
        summary = parse_and_preprocess_mat(raw_path, source, config, compact_path, sealed_path)
        validation = validate_compact_object(compact_path, source, config)
        if validation["sha256"] != summary["compact_sha256"]:
            raise StiegerDataContractError("compact reread hash mismatch")
        record = {
            **source.as_dict(),
            "local_source_sha256": download["sha256"],
            "local_source_md5": download["md5"],
            **summary,
            "compact_reread_validated": True,
            "raw_deletion_authorized": True,
        }
        temporary_record = record_path.with_suffix(".json.tmp")
        temporary_record.write_bytes(_canonical_json(record) + b"\n")
        os.replace(temporary_record, record_path)
        raw_path.unlink()
        raw_deleted = True
        record["raw_deleted_after_validation"] = True
        record_path.write_bytes(_canonical_json(record) + b"\n")
        return record
    finally:
        if not raw_deleted and raw_path.exists():
            # Intentionally retain a failed source file for diagnosis; never delete early.
            pass


def validate_streamed_records(files: Sequence[SourceFile], cache_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for source in files:
        record_path = cache_dir / "records" / f"S{source.subject:02d}_session{source.session}.json"
        compact_path = cache_dir / "sessions" / f"S{source.subject:02d}_session{source.session}.npz"
        sealed_path = cache_dir / "sealed_outcomes" / f"S{source.subject:02d}_session{source.session}.json"
        if not (record_path.exists() and compact_path.exists() and sealed_path.exists()):
            raise StiegerDataContractError(f"incomplete streamed pair {source.subject}/{source.session}")
        record = json.loads(record_path.read_text())
        validation = validate_compact_object(compact_path, source, config)
        if record.get("compact_sha256") != validation["sha256"] or record.get("sealed_sha256") != sha256_file(sealed_path):
            raise StiegerDataContractError(f"streamed hash mismatch {source.subject}/{source.session}")
        if not record.get("raw_deleted_after_validation"):
            raise StiegerDataContractError(f"raw deletion state missing {source.subject}/{source.session}")
        records.append(record)
    canonical = {"count": len(records), "records": records}
    canonical["canonical_sha256"] = hashlib.sha256(_canonical_json(canonical)).hexdigest()
    return canonical


def safe_copy_then_validate_delete(source_path: Path, compact_path: Path, validator: Any) -> None:
    """Synthetic-test helper demonstrating that deletion follows validation."""
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, compact_path)
    validator(compact_path)
    source_path.unlink()
