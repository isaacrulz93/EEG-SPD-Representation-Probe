"""Leakage-auditable sample identity partitions for the WHOLE-SPD V2 audit.

This module deliberately contains no covariance, geometry, feature, or classifier
code.  It owns only canonical metadata ordering, trial-UID hashing, LOSO/T1/T2
sample partitions, and machine-readable identity audit rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


FROZEN_SUBJECTS = tuple(range(1, 10))
FROZEN_SESSION = "0train"
FROZEN_RUNS = tuple(range(6))
FROZEN_CLASSES = ("left_hand", "right_hand", "feet", "tongue")
EXPECTED_TRIALS_PER_SUBJECT = 288
EXPECTED_TRIALS_PER_RUN = 48
EXPECTED_TRIALS_PER_RUN_CLASS = 12
EXPECTED_TRIALS_PER_HALF = 144
EXPECTED_TRIALS_PER_HALF_CLASS = 36

IDENTITY_COLUMNS = ("subject", "session", "run", "trial_id", "trial_uid")
LABEL_FREE_COLUMNS = IDENTITY_COLUMNS
_ROW_POSITION = "__row_position__"


class IdentityAuditError(AssertionError):
    """Raised when a predeclared sample-identity assertion fails."""


@dataclass(frozen=True)
class LabelFreeMetadataView:
    """Immutable center-fit metadata with no label-bearing field."""

    row_positions: tuple[int, ...]
    subject: tuple[int, ...]
    session: tuple[str, ...]
    run: tuple[int, ...]
    trial_id: tuple[int, ...]
    trial_uid: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return LABEL_FREE_COLUMNS

    def __len__(self) -> int:
        return len(self.trial_uid)

    def to_frame(self) -> pd.DataFrame:
        """Return a new label-free DataFrame in canonical order."""

        return pd.DataFrame(
            {
                "subject": self.subject,
                "session": self.session,
                "run": self.run,
                "trial_id": self.trial_id,
                "trial_uid": self.trial_uid,
            }
        )


@dataclass(frozen=True)
class LosoPartition:
    """One frozen nine-subject LOSO partition, expressed in input row positions."""

    target_subject: int
    source_subjects: tuple[int, ...]
    source_row_positions: tuple[int, ...]
    target_row_positions: tuple[int, ...]
    source_trial_uids: tuple[str, ...]
    target_trial_uids: tuple[str, ...]
    source_trial_uid_sha256: str
    target_trial_uid_sha256: str

    @property
    def n_source_trials(self) -> int:
        return len(self.source_trial_uids)

    @property
    def n_target_trials(self) -> int:
        return len(self.target_trial_uids)


@dataclass(frozen=True)
class CalibrationSplit:
    """One target-subject calibration/evaluation half split."""

    name: str
    target_subject: int
    calibration_runs: tuple[int, ...]
    evaluation_runs: tuple[int, ...]
    calibration_row_positions: tuple[int, ...]
    evaluation_row_positions: tuple[int, ...]
    calibration_trial_uids: tuple[str, ...]
    evaluation_trial_uids: tuple[str, ...]
    calibration_trial_uid_sha256: str
    evaluation_trial_uid_sha256: str
    transductive_overlap: bool = False

    @property
    def n_calibration_trials(self) -> int:
        return len(self.calibration_trial_uids)

    @property
    def n_evaluation_trials(self) -> int:
        return len(self.evaluation_trial_uids)


def _integer_series(series: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain non-null integer values")
    return numeric.astype(np.int64)


def _integer_scalar(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _normalized_metadata(
    metadata: pd.DataFrame, *, require_class_label: bool
) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if metadata.empty:
        raise ValueError("metadata cannot be empty")
    if _ROW_POSITION in metadata.columns:
        raise ValueError(f"reserved metadata column is present: {_ROW_POSITION}")
    required = set(IDENTITY_COLUMNS)
    if require_class_label:
        required.add("class_label")
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")

    frame = metadata.copy(deep=True).reset_index(drop=True)
    frame.insert(0, _ROW_POSITION, np.arange(len(frame), dtype=np.int64))
    if frame[list(required)].isna().any(axis=None):
        raise ValueError("required identity/label metadata cannot contain null values")
    frame["subject"] = _integer_series(frame["subject"], name="subject")
    frame["run"] = _integer_series(frame["run"], name="run")
    frame["trial_id"] = _integer_series(frame["trial_id"], name="trial_id")
    frame["session"] = frame["session"].astype(str)
    frame["trial_uid"] = frame["trial_uid"].astype(str)
    if (frame["trial_uid"].str.len() == 0).any():
        raise ValueError("trial_uid cannot be empty")
    if require_class_label:
        frame["class_label"] = frame["class_label"].astype(str)

    duplicate_uids = frame["trial_uid"].duplicated(keep=False)
    if duplicate_uids.any():
        examples = sorted(frame.loc[duplicate_uids, "trial_uid"].unique())[:5]
        raise ValueError(f"trial_uid must be globally unique; duplicates: {examples}")
    duplicate_identity = frame.duplicated(
        ["subject", "session", "trial_id"], keep=False
    )
    if duplicate_identity.any():
        examples = frame.loc[
            duplicate_identity, ["subject", "session", "trial_id"]
        ].head(5)
        raise ValueError(
            "(subject, session, trial_id) must identify one WHOLE trial; examples: "
            f"{examples.to_dict(orient='records')}"
        )
    return frame


def _identity_only_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Copy only the strict identity whitelist, excluding every label field."""

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    missing = set(IDENTITY_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    return metadata.loc[:, list(IDENTITY_COLUMNS)].copy(deep=True)


def _canonical_internal(
    metadata: pd.DataFrame, *, require_class_label: bool
) -> pd.DataFrame:
    frame = _normalized_metadata(
        metadata, require_class_label=require_class_label
    )
    return frame.sort_values(
        ["subject", "session", "run", "trial_id", "trial_uid"],
        kind="mergesort",
    ).reset_index(drop=True)


def canonical_stable_sort(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return metadata in canonical subject/session/run/trial/UID order.

    Input columns are preserved, integer identity columns are normalized, and the
    returned index is reset.  The operation is deterministic and independent of
    input row order.
    """

    original_columns = list(metadata.columns) if isinstance(metadata, pd.DataFrame) else []
    frame = _canonical_internal(
        metadata, require_class_label="class_label" in original_columns
    )
    return frame.drop(columns=[_ROW_POSITION])[original_columns].reset_index(drop=True)


def trial_uid_sha256(trial_uids: Iterable[str]) -> str:
    """Hash one unique trial-UID set after deterministic lexical sorting."""

    values = [str(value) for value in trial_uids]
    if not values:
        raise ValueError("cannot hash an empty trial_uid collection")
    if any(not value for value in values):
        raise ValueError("trial_uid cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError("trial_uid hash input contains duplicates")
    payload = json.dumps(
        {"trial_uids": sorted(values)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_row_positions(
    row_positions: Sequence[int] | np.ndarray | None, n_rows: int
) -> np.ndarray:
    if row_positions is None:
        return np.arange(n_rows, dtype=np.int64)
    values = np.asarray(row_positions)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("row_positions must be a one-dimensional integer sequence")
    values = values.astype(np.int64, copy=False)
    if len(values) == 0:
        raise ValueError("row_positions cannot be empty")
    if len(values) != len(np.unique(values)):
        raise ValueError("row_positions cannot contain duplicates")
    if values.min() < 0 or values.max() >= n_rows:
        raise ValueError("row_positions contain an out-of-range value")
    return values


def label_free_metadata_view(
    metadata: pd.DataFrame,
    row_positions: Sequence[int] | np.ndarray | None = None,
) -> LabelFreeMetadataView:
    """Create an immutable, canonical metadata view safe for center fitting.

    The returned type exposes only identity fields.  In particular it has no
    ``class_label``, ``label``, ``target``, or generic passthrough column.
    ``row_positions`` always refer to positional rows in the supplied DataFrame.
    """

    frame = _normalized_metadata(
        _identity_only_metadata(metadata), require_class_label=False
    )
    selected_positions = _validated_row_positions(row_positions, len(frame))
    selected = frame[frame[_ROW_POSITION].isin(selected_positions)].sort_values(
        ["subject", "session", "run", "trial_id", "trial_uid"],
        kind="mergesort",
    )
    if len(selected) != len(selected_positions):
        raise IdentityAuditError("label-free view lost a requested row position")
    return LabelFreeMetadataView(
        row_positions=tuple(int(value) for value in selected[_ROW_POSITION]),
        subject=tuple(int(value) for value in selected["subject"]),
        session=tuple(str(value) for value in selected["session"]),
        run=tuple(int(value) for value in selected["run"]),
        trial_id=tuple(int(value) for value in selected["trial_id"]),
        trial_uid=tuple(str(value) for value in selected["trial_uid"]),
    )


def _validate_frozen_full_metadata(frame: pd.DataFrame) -> None:
    """Validate the frozen identity/count contract without reading labels."""

    subjects = tuple(sorted(int(value) for value in frame["subject"].unique()))
    if subjects != FROZEN_SUBJECTS:
        raise ValueError(
            f"frozen LOSO requires subjects {FROZEN_SUBJECTS}, observed {subjects}"
        )
    sessions = tuple(sorted(frame["session"].unique()))
    if sessions != (FROZEN_SESSION,):
        raise ValueError(
            f"frozen LOSO requires only session {FROZEN_SESSION!r}, observed {sessions}"
        )
    subject_counts = frame.groupby("subject", observed=True).size()
    if not (subject_counts == EXPECTED_TRIALS_PER_SUBJECT).all():
        raise ValueError(
            "every subject must contain 288 WHOLE trials; observed "
            f"{subject_counts.to_dict()}"
        )
    subject_runs = frame.groupby("subject", observed=True)["run"].agg(
        lambda values: tuple(sorted(int(value) for value in values.unique()))
    )
    if not all(value == FROZEN_RUNS for value in subject_runs):
        raise ValueError("every subject must contain runs exactly 0..5")
    run_counts = frame.groupby(["subject", "run"], observed=True).size()
    if not (run_counts == EXPECTED_TRIALS_PER_RUN).all():
        raise ValueError("every subject/run cell must contain 48 trials")


def make_loso_partition(
    metadata: pd.DataFrame, target_subject: int
) -> LosoPartition:
    """Build one frozen LOSO partition with exactly eight source subjects."""

    frame = _canonical_internal(
        _identity_only_metadata(metadata), require_class_label=False
    )
    _validate_frozen_full_metadata(frame)
    target = _integer_scalar(target_subject, name="target_subject")
    if target not in FROZEN_SUBJECTS:
        raise ValueError(f"target_subject must be one of {FROZEN_SUBJECTS}")

    target_frame = frame[frame["subject"] == target]
    source_frame = frame[frame["subject"] != target]
    source_subjects = tuple(
        sorted(int(value) for value in source_frame["subject"].unique())
    )
    expected_sources = tuple(value for value in FROZEN_SUBJECTS if value != target)
    if source_subjects != expected_sources or len(source_subjects) != 8:
        raise IdentityAuditError("LOSO source subject set is not exactly the other eight")

    source_uids = tuple(str(value) for value in source_frame["trial_uid"])
    target_uids = tuple(str(value) for value in target_frame["trial_uid"])
    if set(source_uids) & set(target_uids):
        raise IdentityAuditError("LOSO source and target trial UIDs overlap")
    if len(source_frame) != 8 * EXPECTED_TRIALS_PER_SUBJECT:
        raise IdentityAuditError("LOSO source trial count is not 2304")
    if len(target_frame) != EXPECTED_TRIALS_PER_SUBJECT:
        raise IdentityAuditError("LOSO target trial count is not 288")

    return LosoPartition(
        target_subject=target,
        source_subjects=source_subjects,
        source_row_positions=tuple(int(value) for value in source_frame[_ROW_POSITION]),
        target_row_positions=tuple(int(value) for value in target_frame[_ROW_POSITION]),
        source_trial_uids=source_uids,
        target_trial_uids=target_uids,
        source_trial_uid_sha256=trial_uid_sha256(source_uids),
        target_trial_uid_sha256=trial_uid_sha256(target_uids),
    )


def _validate_target_runs(target_frame: pd.DataFrame) -> None:
    if len(target_frame) != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"target subject must contain 288 trials, observed {len(target_frame)}"
        )
    sessions = tuple(sorted(target_frame["session"].unique()))
    if sessions != (FROZEN_SESSION,):
        raise ValueError(f"target must contain only session {FROZEN_SESSION!r}")
    runs = tuple(sorted(int(value) for value in target_frame["run"].unique()))
    if runs != FROZEN_RUNS:
        raise ValueError(f"target runs must be exactly {FROZEN_RUNS}, observed {runs}")
    run_counts = target_frame.groupby("run", observed=True).size()
    if not (run_counts == EXPECTED_TRIALS_PER_RUN).all():
        raise ValueError(
            "every target run must contain 48 trials; observed "
            f"{run_counts.to_dict()}"
        )


def assert_t1_overlap(
    target_center_fit_trial_uids: Iterable[str],
    evaluation_trial_uids: Iterable[str],
) -> None:
    """Assert the preregistered T1 exact covariate overlap."""

    fitted = tuple(str(value) for value in target_center_fit_trial_uids)
    evaluated = tuple(str(value) for value in evaluation_trial_uids)
    if not fitted or not evaluated:
        raise IdentityAuditError("T1 target-center/evaluation UID sets cannot be empty")
    if len(fitted) != len(set(fitted)) or len(evaluated) != len(set(evaluated)):
        raise IdentityAuditError("T1 UID sets must be internally unique")
    if set(fitted) != set(evaluated):
        raise IdentityAuditError(
            "T1 requires target center-fit and evaluation trial UIDs to be equal"
        )


def assert_t2_disjoint(
    calibration_trial_uids: Iterable[str],
    evaluation_trial_uids: Iterable[str],
) -> None:
    """Assert non-empty, internally unique, disjoint T2 UID sets."""

    calibration = tuple(str(value) for value in calibration_trial_uids)
    evaluation = tuple(str(value) for value in evaluation_trial_uids)
    if not calibration or not evaluation:
        raise IdentityAuditError("T2 calibration/evaluation UID sets cannot be empty")
    if len(calibration) != len(set(calibration)) or len(evaluation) != len(set(evaluation)):
        raise IdentityAuditError("T2 UID sets must be internally unique")
    overlap = set(calibration) & set(evaluation)
    if overlap:
        raise IdentityAuditError(
            f"T2 calibration/evaluation trial UIDs overlap: {sorted(overlap)[:5]}"
        )


def make_calibration_splits(
    metadata: pd.DataFrame, target_subject: int | None = None
) -> tuple[CalibrationSplit, CalibrationSplit]:
    """Build deterministic A/B run halves for one target subject.

    This identity-only function neither requires nor reads class labels.  The
    dataset loader owns the frozen class/count contract before this boundary.
    """

    frame = _canonical_internal(
        _identity_only_metadata(metadata), require_class_label=False
    )
    observed_subjects = tuple(sorted(int(value) for value in frame["subject"].unique()))
    if target_subject is None:
        if len(observed_subjects) != 1:
            raise ValueError(
                "target_subject is required when metadata contains multiple subjects"
            )
        target = observed_subjects[0]
    else:
        target = _integer_scalar(target_subject, name="target_subject")
        if target not in observed_subjects:
            raise ValueError(f"target subject {target} is absent from metadata")
        frame = frame[frame["subject"] == target].reset_index(drop=True)
    if tuple(sorted(int(value) for value in frame["subject"].unique())) != (target,):
        raise ValueError("calibration split input did not resolve to one target subject")
    _validate_target_runs(frame)

    specifications = (
        ("A", (0, 1, 2), (3, 4, 5)),
        ("B", (3, 4, 5), (0, 1, 2)),
    )
    splits: list[CalibrationSplit] = []
    target_uid_set = set(frame["trial_uid"].astype(str))
    for name, calibration_runs, evaluation_runs in specifications:
        calibration = frame[frame["run"].isin(calibration_runs)]
        evaluation = frame[frame["run"].isin(evaluation_runs)]
        calibration_uids = tuple(str(value) for value in calibration["trial_uid"])
        evaluation_uids = tuple(str(value) for value in evaluation["trial_uid"])
        assert_t2_disjoint(calibration_uids, evaluation_uids)
        if len(calibration) != EXPECTED_TRIALS_PER_HALF:
            raise IdentityAuditError(f"split {name} calibration count is not 144")
        if len(evaluation) != EXPECTED_TRIALS_PER_HALF:
            raise IdentityAuditError(f"split {name} evaluation count is not 144")
        if set(calibration_uids) | set(evaluation_uids) != target_uid_set:
            raise IdentityAuditError(f"split {name} halves do not cover every target trial")
        splits.append(
            CalibrationSplit(
                name=name,
                target_subject=target,
                calibration_runs=calibration_runs,
                evaluation_runs=evaluation_runs,
                calibration_row_positions=tuple(
                    int(value) for value in calibration[_ROW_POSITION]
                ),
                evaluation_row_positions=tuple(
                    int(value) for value in evaluation[_ROW_POSITION]
                ),
                calibration_trial_uids=calibration_uids,
                evaluation_trial_uids=evaluation_uids,
                calibration_trial_uid_sha256=trial_uid_sha256(calibration_uids),
                evaluation_trial_uid_sha256=trial_uid_sha256(evaluation_uids),
            )
        )
    if set(splits[0].calibration_trial_uids) != set(splits[1].evaluation_trial_uids):
        raise IdentityAuditError("split A calibration must equal split B evaluation")
    if set(splits[0].evaluation_trial_uids) != set(splits[1].calibration_trial_uids):
        raise IdentityAuditError("split A evaluation must equal split B calibration")
    return splits[0], splits[1]


def _json_values(values: Iterable[object]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _positions_frame(frame: pd.DataFrame, positions: Sequence[int]) -> pd.DataFrame:
    selected = frame[frame[_ROW_POSITION].isin(positions)].sort_values(
        ["subject", "session", "run", "trial_id", "trial_uid"], kind="mergesort"
    )
    if len(selected) != len(positions):
        raise IdentityAuditError("audit selection lost or duplicated row positions")
    return selected


def make_sample_id_audit_rows(
    metadata: pd.DataFrame,
    loso: LosoPartition,
    *,
    protocol: str,
    calibration_split: CalibrationSplit | None = None,
) -> pd.DataFrame:
    """Build role-level, label-policy-aware identity audit rows for T1 or T2.

    There is one ``source_center_fit`` row per source subject, followed by one
    aggregate ``classifier_train`` row, one ``target_center_fit`` row, and one
    ``evaluation`` row.  No label values are copied into the audit output.
    """

    protocol_name = str(protocol).upper()
    if protocol_name not in {"T1", "T2"}:
        raise ValueError("protocol must be 'T1' or 'T2'")
    if protocol_name == "T1" and calibration_split is not None:
        raise ValueError("T1 audit does not accept a calibration split")
    if protocol_name == "T2" and calibration_split is None:
        raise ValueError("T2 audit requires a calibration split")
    if calibration_split is not None and (
        calibration_split.target_subject != loso.target_subject
    ):
        raise ValueError("calibration split and LOSO target subjects differ")

    frame = _normalized_metadata(
        _identity_only_metadata(metadata), require_class_label=False
    )
    source = _positions_frame(frame, loso.source_row_positions)
    target = _positions_frame(frame, loso.target_row_positions)
    if tuple(source.sort_values(["subject", "run", "trial_id"])["trial_uid"]) != tuple(
        loso.source_trial_uids
    ):
        raise IdentityAuditError("LOSO source identity does not match supplied metadata")
    if tuple(target.sort_values(["subject", "run", "trial_id"])["trial_uid"]) != tuple(
        loso.target_trial_uids
    ):
        raise IdentityAuditError("LOSO target identity does not match supplied metadata")
    if set(source["trial_uid"]) & set(target["trial_uid"]):
        raise IdentityAuditError("source and target overlap during audit construction")

    if protocol_name == "T1":
        center_positions = loso.target_row_positions
        evaluation_positions = loso.target_row_positions
        assert_t1_overlap(loso.target_trial_uids, loso.target_trial_uids)
        split_name = "ALL"
        transductive = True
    else:
        assert calibration_split is not None
        center_positions = calibration_split.calibration_row_positions
        evaluation_positions = calibration_split.evaluation_row_positions
        assert_t2_disjoint(
            calibration_split.calibration_trial_uids,
            calibration_split.evaluation_trial_uids,
        )
        split_name = calibration_split.name
        transductive = False

    target_center = _positions_frame(frame, center_positions)
    evaluation = _positions_frame(frame, evaluation_positions)
    if calibration_split is not None:
        selected_center_uids = tuple(str(value) for value in target_center["trial_uid"])
        selected_evaluation_uids = tuple(str(value) for value in evaluation["trial_uid"])
        if set(selected_center_uids) != set(calibration_split.calibration_trial_uids):
            raise IdentityAuditError(
                "calibration split center identities do not match supplied metadata"
            )
        if set(selected_evaluation_uids) != set(
            calibration_split.evaluation_trial_uids
        ):
            raise IdentityAuditError(
                "calibration split evaluation identities do not match supplied metadata"
            )
        if trial_uid_sha256(selected_center_uids) != (
            calibration_split.calibration_trial_uid_sha256
        ):
            raise IdentityAuditError("calibration split center hash is inconsistent")
        if trial_uid_sha256(selected_evaluation_uids) != (
            calibration_split.evaluation_trial_uid_sha256
        ):
            raise IdentityAuditError("calibration split evaluation hash is inconsistent")
    evaluation_uid_set = set(evaluation["trial_uid"].astype(str))
    # Construct these views as an executable contract: center-fit callers need
    # only row positions plus this immutable, label-free object.
    target_center_view = label_free_metadata_view(metadata, center_positions)

    rows: list[dict[str, object]] = []

    def append_row(
        role: str,
        selected: pd.DataFrame,
        *,
        fit_subject: int | None,
        label_access: str,
        label_free: bool,
        expected_relation: str,
    ) -> None:
        uids = tuple(str(value) for value in selected["trial_uid"])
        overlap_n = len(set(uids) & evaluation_uid_set)
        if expected_relation == "disjoint":
            relation_pass = overlap_n == 0
        elif expected_relation in {"equal", "self"}:
            relation_pass = set(uids) == evaluation_uid_set
        else:
            raise ValueError(f"unknown audit relation: {expected_relation}")
        if not relation_pass:
            raise IdentityAuditError(
                f"{protocol_name}/{split_name}/{role} failed {expected_relation} "
                "relation to evaluation"
            )
        rows.append(
            {
                "protocol": protocol_name,
                "split": split_name,
                "target_subject": loso.target_subject,
                "source_subjects": _json_values(loso.source_subjects),
                "role": role,
                "fit_subject": fit_subject if fit_subject is not None else pd.NA,
                "sample_subjects": _json_values(
                    sorted(int(value) for value in selected["subject"].unique())
                ),
                "runs": _json_values(
                    sorted(int(value) for value in selected["run"].unique())
                ),
                "session": FROZEN_SESSION,
                "n_trials": len(uids),
                "trial_uids_json": _json_values(uids),
                "trial_uid_sha256": trial_uid_sha256(uids),
                "overlap_with_evaluation_n": overlap_n,
                "expected_relation_to_evaluation": expected_relation,
                "relation_assertion_pass": relation_pass,
                "transductive_overlap": transductive,
                "label_access": label_access,
                "label_free_metadata": label_free,
                "metadata_columns": _json_values(
                    LABEL_FREE_COLUMNS
                    if label_free
                    else (*IDENTITY_COLUMNS, "class_label")
                ),
            }
        )

    for source_subject in loso.source_subjects:
        selected = source[source["subject"] == source_subject]
        # Fail immediately if a future refactor cannot construct the promised
        # label-free center-fit view for this exact selection.
        label_free_metadata_view(
            metadata, tuple(int(value) for value in selected[_ROW_POSITION])
        )
        append_row(
            "source_center_fit",
            selected,
            fit_subject=source_subject,
            label_access="forbidden",
            label_free=True,
            expected_relation="disjoint",
        )
    append_row(
        "classifier_train",
        source,
        fit_subject=None,
        label_access="source_labels_only",
        label_free=False,
        expected_relation="disjoint",
    )
    if tuple(target_center_view.trial_uid) != tuple(target_center["trial_uid"]):
        raise IdentityAuditError("target label-free view changed canonical trial identity")
    append_row(
        "target_center_fit",
        target_center,
        fit_subject=loso.target_subject,
        label_access="forbidden",
        label_free=True,
        expected_relation="equal" if protocol_name == "T1" else "disjoint",
    )
    append_row(
        "evaluation",
        evaluation,
        fit_subject=None,
        label_access="target_labels_only",
        label_free=False,
        expected_relation="self",
    )
    result = pd.DataFrame.from_records(rows)
    result["fit_subject"] = result["fit_subject"].astype("Int64")
    return result
