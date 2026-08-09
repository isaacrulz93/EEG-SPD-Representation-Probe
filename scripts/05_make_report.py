#!/usr/bin/env python3
"""Render the frozen BNCI2014_001 representation-probe report.

This stage does not recompute EEG features or tune any decision rule.  It reads
the frozen protocol and the audit tables written by stages 01--04, applies the
thresholds declared in that protocol, and writes one deterministic Markdown
report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = (
    "whole_raw",
    "window5_raw",
    "whole_centered",
    "window5_centered",
)
REPRESENTATION_LABELS = {
    "whole_raw": "WHOLE RAW",
    "window5_raw": "WINDOW5 RAW",
    "whole_centered": "WHOLE CENTERED",
    "window5_centered": "WINDOW5 CENTERED",
}
VERDICTS = {"SUPPORTED", "MIXED", "NOT SUPPORTED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise threshold logic with synthetic values; do not write the report",
    )
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Required report input is empty: {path}")
    return frame


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {_token(column): str(column) for column in frame.columns}
    for name in names:
        if _token(name) in lookup:
            return lookup[_token(name)]
    return None


def _finite_float(value: object, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a number for {context}, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite number for {context}, got {result}")
    return result


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _window_index(row: Mapping[str, object], columns: Sequence[str]) -> int | None:
    lookup = {_token(key): key for key in columns}
    for candidate in ("window_index", "window", "temporal_window"):
        key = lookup.get(candidate)
        if key is not None and not pd.isna(row[key]):
            text = _token(row[key])
            match = re.search(r"([1-5])$", text)
            if match:
                return int(match.group(1))
            try:
                value = int(float(row[key]))
            except (TypeError, ValueError):
                continue
            if 1 <= value <= 5:
                return value
    for candidate in ("scope", "subset", "metric"):
        key = lookup.get(candidate)
        if key is not None:
            match = re.search(r"window_?([1-5])", _token(row[key]))
            if match:
                return int(match.group(1))
    return None


def canonical_representation(row: Mapping[str, object], columns: Sequence[str]) -> str:
    lookup = {_token(key): key for key in columns}
    rep_parts: list[str] = []
    for candidate in ("representation", "representation_name", "feature_set"):
        key = lookup.get(candidate)
        if key is not None and not pd.isna(row[key]):
            rep_parts.append(_token(row[key]))
            break
    for candidate in ("state", "centering", "coordinate_state"):
        key = lookup.get(candidate)
        if key is not None and not pd.isna(row[key]):
            rep_parts.append(_token(row[key]))
            break
    joined = "_".join(rep_parts)
    if not joined:
        raise ValueError("Diagnostic row has no representation column")
    if "window" in joined:
        base = "window5"
    elif "whole" in joined:
        base = "whole"
    else:
        raise ValueError(f"Unrecognized representation: {joined!r}")
    if "center" in joined or joined.endswith("_true"):
        state = "centered"
    elif "raw" in joined or joined.endswith("_false"):
        state = "raw"
    else:
        raise ValueError(f"Representation does not identify RAW/CENTERED: {joined!r}")
    return f"{base}_{state}"


def canonical_metric(value: object) -> str | None:
    name = _token(value)
    if "silhouette" in name:
        if "subject" in name:
            return "subject_silhouette"
        if "class" in name:
            return "class_silhouette"
        if "window" in name:
            return "window_silhouette"
    if "ratio" in name:
        if "subject" in name:
            return "subject_separation_ratio"
        if "class" in name:
            return "class_separation_ratio"
    aliases = {
        "subject_silhouette": "subject_silhouette",
        "class_silhouette": "class_silhouette",
        "window_silhouette": "window_silhouette",
        "subject_separation_ratio": "subject_separation_ratio",
        "class_separation_ratio": "class_separation_ratio",
    }
    return aliases.get(name)


@dataclass(frozen=True)
class Separability:
    values: Mapping[tuple[str, int | None, str], float]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Separability":
        values: dict[tuple[str, int | None, str], float] = {}
        metric_col = _column(frame, "metric", "metric_name")
        value_col = _column(frame, "value", "score", "metric_value")
        for _, series in frame.iterrows():
            row = series.to_dict()
            rep = canonical_representation(row, list(frame.columns))
            window = _window_index(row, list(frame.columns))
            candidates: list[tuple[str, object]] = []
            if metric_col is not None and value_col is not None:
                metric = canonical_metric(row[metric_col])
                if metric is not None:
                    candidates.append((metric, row[value_col]))
            else:
                for column in frame.columns:
                    metric = canonical_metric(column)
                    if metric is not None:
                        candidates.append((metric, row[column]))
            for metric, raw_value in candidates:
                value = _optional_float(raw_value)
                if value is None:
                    continue
                key = (rep, window, metric)
                if key in values and not np.isclose(values[key], value, atol=1e-12):
                    raise ValueError(f"Conflicting separability values for {key}")
                values[key] = value
        return cls(values)

    def get(
        self,
        representation: str,
        metric: str,
        window: int | None = None,
        *,
        required: bool = True,
    ) -> float | None:
        key = (representation, window, metric)
        if key not in self.values:
            if required:
                raise ValueError(f"Missing separability metric {key}")
            return None
        return self.values[key]

    def window_scores(self, representation: str) -> list[float]:
        return [
            float(self.get(representation, "class_silhouette", window))
            for window in range(1, 6)
        ]


def canonical_target(value: object) -> str:
    name = _token(value)
    if "class" in name:
        return "class"
    if "subject" in name:
        return "subject"
    raise ValueError(f"Unrecognized linear-probe target: {value!r}")


@dataclass(frozen=True)
class ProbeResult:
    accuracy: float
    std: float | None
    chance: float
    folds: int | None
    aggregation: str | None = None
    point_accuracy: float | None = None
    convergence_warning: bool = False


@dataclass(frozen=True)
class Probes:
    values: Mapping[tuple[str, str], ProbeResult]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Probes":
        target_col = _column(frame, "target", "probe_target", "prediction_target")
        if target_col is None:
            raise ValueError("linear_probe_metrics.csv has no target column")
        accuracy_col = _column(
            frame,
            "accuracy_mean",
            "mean_accuracy",
            "accuracy",
            "diagnostic_accuracy",
        )
        chance_col = _column(frame, "chance_accuracy", "chance_level", "chance")
        std_col = _column(frame, "accuracy_std", "std_accuracy", "std")
        folds_col = _column(frame, "n_folds", "folds")
        fold_col = _column(frame, "fold", "fold_index")
        aggregation_col = _column(frame, "aggregation")
        point_col = _column(frame, "pooled_oof_point_accuracy", "point_accuracy")
        warning_col = _column(frame, "any_convergence_warning", "convergence_warning")
        if accuracy_col is None:
            # Also accept a long metric/value layout.
            metric_col = _column(frame, "metric", "metric_name")
            value_col = _column(frame, "value", "metric_value")
            if metric_col is None or value_col is None:
                raise ValueError("linear_probe_metrics.csv has no accuracy values")
            pivot_keys = [column for column in frame.columns if column not in {metric_col, value_col}]
            frame = (
                frame.assign(_metric=frame[metric_col].map(_token))
                .pivot_table(index=pivot_keys, columns="_metric", values=value_col, aggfunc="first")
                .reset_index()
            )
            target_col = _column(frame, "target", "probe_target", "prediction_target")
            accuracy_col = _column(frame, "accuracy_mean", "mean_accuracy", "accuracy")
            chance_col = _column(frame, "chance_accuracy", "chance_level", "chance")
            std_col = _column(frame, "accuracy_std", "std_accuracy", "std")
            folds_col = _column(frame, "n_folds", "folds")
            fold_col = _column(frame, "fold", "fold_index")
            aggregation_col = _column(frame, "aggregation")
            point_col = _column(frame, "pooled_oof_point_accuracy", "point_accuracy")
            warning_col = _column(frame, "any_convergence_warning", "convergence_warning")
        if target_col is None or accuracy_col is None:
            raise ValueError("Could not normalize linear-probe table")

        grouped: dict[
            tuple[str, str],
            list[
                tuple[
                    float,
                    float | None,
                    float | None,
                    int | None,
                    str | None,
                    float | None,
                    bool,
                ]
            ],
        ] = {}
        for _, series in frame.iterrows():
            row = series.to_dict()
            accuracy = _optional_float(row[accuracy_col])
            if accuracy is None:
                continue
            rep = canonical_representation(row, list(frame.columns))
            target = canonical_target(row[target_col])
            std = _optional_float(row[std_col]) if std_col else None
            chance = _optional_float(row[chance_col]) if chance_col else None
            folds = None
            if folds_col and not pd.isna(row[folds_col]):
                folds = int(row[folds_col])
            elif fold_col:
                folds = 1
            aggregation = (
                str(row[aggregation_col])
                if aggregation_col and not pd.isna(row[aggregation_col])
                else None
            )
            point_accuracy = _optional_float(row[point_col]) if point_col else None
            warning = (
                _token(row[warning_col]) in {"true", "1"}
                if warning_col and not pd.isna(row[warning_col])
                else False
            )
            grouped.setdefault((rep, target), []).append(
                (accuracy, std, chance, folds, aggregation, point_accuracy, warning)
            )

        values: dict[tuple[str, str], ProbeResult] = {}
        for key, records in grouped.items():
            accuracies = np.asarray([record[0] for record in records], dtype=float)
            explicit_std = [record[1] for record in records if record[1] is not None]
            std = explicit_std[0] if explicit_std else (
                float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None
            )
            chances = [record[2] for record in records if record[2] is not None]
            target = key[1]
            chance = chances[0] if chances else (0.25 if target == "class" else 1.0 / 9.0)
            if any(not np.isclose(chance, other, atol=1e-12) for other in chances):
                raise ValueError(f"Conflicting chance levels for probe {key}")
            folds_values = [record[3] for record in records if record[3] is not None]
            folds = max(folds_values) if len(records) == 1 and folds_values else len(records)
            aggregations = [record[4] for record in records if record[4] is not None]
            point_values = [record[5] for record in records if record[5] is not None]
            values[key] = ProbeResult(
                float(np.mean(accuracies)),
                std,
                float(chance),
                folds,
                aggregations[0] if aggregations else None,
                float(np.mean(point_values)) if point_values else None,
                any(record[6] for record in records),
            )
        return cls(values)

    def get(self, representation: str, target: str) -> ProbeResult:
        key = (representation, target)
        if key not in self.values:
            raise ValueError(f"Missing linear probe result {key}")
        return self.values[key]


@dataclass(frozen=True)
class TransitionDiagnostics:
    class_eta_squared: float
    class_mean_range_fraction: float
    overall_mean: float | None
    centered_invariance_max_abs: float | None
    class_means: Mapping[str, float]
    subject_means: Mapping[str, float]
    pair_means: Mapping[str, float]
    cosine_min: float | None
    cosine_max: float | None

    @classmethod
    def from_tables(cls, tables_dir: Path) -> "TransitionDiagnostics":
        candidates = sorted(tables_dir.glob("transition*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No transition diagnostic CSV was found under {tables_dir}"
            )
        frames = [(path.name, read_csv(path)) for path in candidates]

        def named_scalar(names: Iterable[str]) -> float | None:
            wanted = {_token(name) for name in names}
            for _, frame in frames:
                for column in frame.columns:
                    if _token(column) in wanted:
                        finite = [
                            value
                            for value in (_optional_float(item) for item in frame[column])
                            if value is not None
                        ]
                        if finite:
                            return float(finite[0])
                metric_col = _column(frame, "metric", "metric_name", "statistic")
                value_col = _column(frame, "value", "metric_value", "score")
                if metric_col and value_col:
                    for _, row in frame.iterrows():
                        if _token(row[metric_col]) in wanted:
                            value = _optional_float(row[value_col])
                            if value is not None:
                                return value
            return None

        eta = named_scalar(
            (
                "class_eta_squared",
                "eta_squared_class",
                "class_effect_eta_squared",
                "transition_class_eta_squared",
            )
        )
        range_fraction = named_scalar(
            (
                "class_mean_range_fraction",
                "transition_class_mean_range_fraction",
                "class_range_fraction",
            )
        )
        overall_mean = named_scalar(("overall_mean", "overall_transition_mean"))
        centered_invariance = named_scalar(
            ("centered_invariance_max_abs", "centering_invariance_max_abs")
        )

        class_means: dict[str, float] = {}
        subject_means: dict[str, float] = {}
        pair_means: dict[str, float] = {}
        cosine_values: list[float] = []
        for _, frame in frames:
            mean_col = _column(
                frame,
                "mean_transition_magnitude",
                "transition_magnitude_mean",
                "mean_magnitude",
            )
            class_col = _column(frame, "class_label", "class")
            pair_col = _column(frame, "window_pair", "transition", "transition_pair")
            subject_col = _column(frame, "subject", "subject_id")
            if mean_col:
                for _, row in frame.iterrows():
                    value = _optional_float(row[mean_col])
                    if value is None:
                        continue
                    if class_col and not pd.isna(row[class_col]) and not subject_col:
                        label = str(row[class_col])
                        if pair_col is None or pd.isna(row[pair_col]) or _token(row[pair_col]) in {"all", "overall"}:
                            class_means.setdefault(label, value)
                    if subject_col and not pd.isna(row[subject_col]) and not class_col:
                        label = str(row[subject_col])
                        if pair_col is None or pd.isna(row[pair_col]) or _token(row[pair_col]) in {"all", "overall"}:
                            subject_means.setdefault(label, value)
                    if pair_col and not pd.isna(row[pair_col]) and not class_col and not subject_col:
                        pair_means.setdefault(str(row[pair_col]), value)
            cosine_col = _column(frame, "cosine_similarity", "mean_vector_cosine_similarity")
            if cosine_col:
                cosine_values.extend(
                    value
                    for value in (_optional_float(item) for item in frame[cosine_col])
                    if value is not None
                )

        if range_fraction is None and len(class_means) >= 2:
            means = np.asarray(list(class_means.values()), dtype=float)
            denominator = max(abs(float(np.mean(means))), np.finfo(float).eps)
            range_fraction = float((means.max() - means.min()) / denominator)
        if eta is None:
            raise ValueError("Transition tables do not contain class eta-squared")
        if range_fraction is None:
            raise ValueError("Transition tables do not contain a class mean-range fraction")
        return cls(
            class_eta_squared=eta,
            class_mean_range_fraction=range_fraction,
            overall_mean=overall_mean,
            centered_invariance_max_abs=centered_invariance,
            class_means=class_means,
            subject_means=subject_means,
            pair_means=pair_means,
            cosine_min=min(cosine_values) if cosine_values else None,
            cosine_max=max(cosine_values) if cosine_values else None,
        )


@dataclass(frozen=True)
class Evidence:
    label: str
    value: float
    threshold: float
    direction: str
    vote: int


def delta_evidence(label: str, value: float, threshold: float) -> Evidence:
    vote = 1 if value >= threshold else (-1 if value <= -threshold else 0)
    return Evidence(label, value, threshold, "increase", vote)


def pass_evidence(label: str, value: float, threshold: float) -> Evidence:
    return Evidence(label, value, threshold, "at least", int(value >= threshold))


def improvement_verdict(evidence: Sequence[Evidence], minimum_positive: int = 2) -> str:
    positives = sum(item.vote > 0 for item in evidence)
    negatives = sum(item.vote < 0 for item in evidence)
    if positives >= minimum_positive and negatives == 0:
        return "SUPPORTED"
    if positives or negatives:
        return "MIXED"
    return "NOT SUPPORTED"


def binary_verdict(evidence: Sequence[Evidence], required: int | None = None) -> str:
    passes = sum(item.vote > 0 for item in evidence)
    required = len(evidence) if required is None else required
    if passes >= required:
        return "SUPPORTED"
    if passes:
        return "MIXED"
    return "NOT SUPPORTED"


@dataclass(frozen=True)
class VerdictBundle:
    verdicts: Mapping[str, str]
    evidence: Mapping[str, Sequence[Evidence]]
    retained_windows: Sequence[int]
    distinctions: Mapping[str, str]


def compute_verdicts(
    separability: Separability,
    probes: Probes,
    transitions: TransitionDiagnostics,
    thresholds: Mapping[str, object],
) -> VerdictBundle:
    sil_t = float(thresholds["metric_silhouette_delta"])
    probe_t = float(thresholds["probe_accuracy_delta"])
    chance_t = float(thresholds["above_chance_margin"])
    window_range_t = float(thresholds["window_silhouette_range"])
    ratio_t = float(thresholds["distance_ratio_relative_delta"])
    eta_t = float(thresholds["transition_eta_squared"])
    transition_range_t = float(thresholds["transition_mean_range_fraction"])

    h1 = [
        delta_evidence(
            "RAW class silhouette: WINDOW5 - WHOLE",
            float(separability.get("window5_raw", "class_silhouette"))
            - float(separability.get("whole_raw", "class_silhouette")),
            sil_t,
        ),
        delta_evidence(
            "RAW class-probe accuracy: WINDOW5 - WHOLE",
            probes.get("window5_raw", "class").accuracy
            - probes.get("whole_raw", "class").accuracy,
            probe_t,
        ),
    ]
    whole_ratio = separability.get(
        "whole_raw", "class_separation_ratio", required=False
    )
    window_ratio = separability.get(
        "window5_raw", "class_separation_ratio", required=False
    )
    if whole_ratio is not None and window_ratio is not None:
        h1.append(
            delta_evidence(
                "RAW class distance ratio: relative WINDOW5 - WHOLE",
                (window_ratio - whole_ratio) / max(abs(whole_ratio), np.finfo(float).eps),
                ratio_t,
            )
        )

    h2: list[Evidence] = []
    h2_counts: dict[str, int] = {}
    for rep in ("whole_raw", "window5_raw"):
        local = [
            pass_evidence(
                f"{REPRESENTATION_LABELS[rep]} subject silhouette",
                float(separability.get(rep, "subject_silhouette")),
                sil_t,
            ),
            pass_evidence(
                f"{REPRESENTATION_LABELS[rep]} subject-probe margin over chance",
                probes.get(rep, "subject").accuracy - probes.get(rep, "subject").chance,
                chance_t,
            ),
        ]
        ratio = separability.get(rep, "subject_separation_ratio", required=False)
        if ratio is not None:
            local.append(
                pass_evidence(
                    f"{REPRESENTATION_LABELS[rep]} subject distance-ratio excess over 1",
                    ratio - 1.0,
                    ratio_t,
                )
            )
        h2.extend(local)
        h2_counts[rep] = sum(item.vote > 0 for item in local)
    h2_verdict = (
        "SUPPORTED"
        if all(h2_counts[rep] >= 2 for rep in ("whole_raw", "window5_raw"))
        else "MIXED"
        if any(h2_counts.values())
        else "NOT SUPPORTED"
    )

    h3: list[Evidence] = []
    h3_counts: dict[str, int] = {}
    h3_negative = False
    for base in ("whole", "window5"):
        raw = f"{base}_raw"
        centered = f"{base}_centered"
        local = [
            delta_evidence(
                f"{base.upper()} subject silhouette reduction",
                float(separability.get(raw, "subject_silhouette"))
                - float(separability.get(centered, "subject_silhouette")),
                sil_t,
            ),
            delta_evidence(
                f"{base.upper()} subject-probe accuracy reduction",
                probes.get(raw, "subject").accuracy
                - probes.get(centered, "subject").accuracy,
                probe_t,
            ),
        ]
        raw_ratio = separability.get(raw, "subject_separation_ratio", required=False)
        centered_ratio = separability.get(
            centered, "subject_separation_ratio", required=False
        )
        if raw_ratio is not None and centered_ratio is not None:
            local.append(
                delta_evidence(
                    f"{base.upper()} subject distance-ratio relative reduction",
                    (raw_ratio - centered_ratio)
                    / max(abs(raw_ratio), np.finfo(float).eps),
                    ratio_t,
                )
            )
        h3.extend(local)
        h3_counts[base] = sum(item.vote > 0 for item in local)
        h3_negative = h3_negative or any(item.vote < 0 for item in local)
    h3_verdict = (
        "SUPPORTED"
        if all(h3_counts[base] >= 2 for base in ("whole", "window5"))
        and not h3_negative
        else "MIXED"
        if any(h3_counts.values()) or h3_negative
        else "NOT SUPPORTED"
    )

    raw_windows = separability.window_scores("window5_raw")
    centered_windows = separability.window_scores("window5_centered")
    retained = [
        index
        for index, (raw, centered) in enumerate(zip(raw_windows, centered_windows), start=1)
        if centered >= sil_t and centered >= raw - sil_t
    ]
    centered_class_probe = probes.get("window5_centered", "class")
    h4 = [
        pass_evidence("Number of retained positive temporal windows", float(len(retained)), 1.0),
        pass_evidence(
            "CENTERED WINDOW5 class-probe margin over chance",
            centered_class_probe.accuracy - centered_class_probe.chance,
            chance_t,
        ),
    ]

    raw_range = max(raw_windows) - min(raw_windows)
    centered_range = max(centered_windows) - min(centered_windows)
    h5 = [
        pass_evidence("RAW per-window class-silhouette range", raw_range, window_range_t),
        pass_evidence(
            "CENTERED per-window class-silhouette range", centered_range, window_range_t
        ),
    ]

    h6 = [
        pass_evidence(
            "Class effect on transition magnitude (eta-squared)",
            transitions.class_eta_squared,
            eta_t,
        ),
        pass_evidence(
            "Range of class mean transition magnitudes / grand mean",
            transitions.class_mean_range_fraction,
            transition_range_t,
        ),
    ]

    verdicts = {
        "H1": improvement_verdict(h1, minimum_positive=2),
        "H2": h2_verdict,
        "H3": h3_verdict,
        "H4": binary_verdict(h4),
        "H5": binary_verdict(h5),
        "H6": binary_verdict(h6),
    }
    if not all(value in VERDICTS for value in verdicts.values()):
        raise AssertionError("Internal error: invalid verdict label")

    a_status = (
        "SUPPORTED"
        if verdicts["H1"] == "SUPPORTED"
        else "MIXED"
        if verdicts["H1"] == "MIXED" or verdicts["H4"] in {"SUPPORTED", "MIXED"}
        else "NOT SUPPORTED"
    )
    b_status = verdicts["H6"]
    c_status = (
        "SUPPORTED"
        if a_status == "NOT SUPPORTED" and b_status == "NOT SUPPORTED"
        else "MIXED"
        if a_status != "SUPPORTED" and b_status != "SUPPORTED"
        else "NOT SUPPORTED"
    )
    distinctions = {
        "A": a_status,
        "B": b_status,
        "C": c_status,
    }
    return VerdictBundle(verdicts, {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6}, retained, distinctions)


def fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_sci(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3e}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(cell(item) for item in row) + " |" for row in rows)
    return "\n".join(output)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).map(_token)
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"Could not parse boolean column {series.name}")
    return normalized.isin({"true", "1"})


def figure_links(names: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"[{label}](../figures/{filename})" for label, filename in names)


def render_report(
    *,
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    environment: Mapping[str, Any],
    covariance_summary: Mapping[str, Any],
    sanity: pd.DataFrame,
    embedding: Mapping[str, Any],
    diagnostics_summary: Mapping[str, Any],
    separability: Separability,
    probes: Probes,
    window_probe_metrics: pd.DataFrame,
    transitions: TransitionDiagnostics,
    bundle: VerdictBundle,
) -> str:
    prep = config["preprocessing"]
    representation = config["representation"]
    embed_config = config["embedding"]
    probe_config = config["diagnostics"]["linear_probe"]
    thresholds = config["diagnostics"]["verdict_thresholds"]

    sanity_rows = []
    for rep in ("WHOLE", "WINDOW5"):
        subset = sanity[sanity["representation"].astype(str).str.upper() == rep]
        if subset.empty:
            raise ValueError(f"covariance_sanity.csv has no {rep} rows")
        spd = bool_series(subset["is_spd"])
        nan = bool_series(subset["has_nan"])
        inf = bool_series(subset["has_inf"])
        sanity_rows.append(
            (
                rep,
                len(subset),
                int(spd.sum()),
                int((~spd).sum()),
                int(nan.sum()),
                int(inf.sum()),
                f"{float(subset['min_eig'].min()):.3e}",
                f"{float(subset['max_eig'].max()):.3e}",
                f"{float(subset['condition_number'].max()):.3e}",
                f"{float(subset['symmetry_error'].max()):.3e}",
            )
        )

    overall_sep_rows = []
    for rep in REPRESENTATIONS:
        overall_sep_rows.append(
            (
                REPRESENTATION_LABELS[rep],
                fmt(separability.get(rep, "class_silhouette")),
                fmt(separability.get(rep, "subject_silhouette")),
                fmt(separability.get(rep, "window_silhouette", required=False)),
                fmt(separability.get(rep, "class_separation_ratio", required=False)),
                fmt(separability.get(rep, "subject_separation_ratio", required=False)),
            )
        )
    probe_rows = []
    for rep in REPRESENTATIONS:
        for target in ("class", "subject"):
            result = probes.get(rep, target)
            probe_rows.append(
                (
                    REPRESENTATION_LABELS[rep],
                    target,
                    fmt(result.accuracy),
                    fmt(result.std),
                    fmt(result.chance),
                    result.folds if result.folds is not None else "N/A",
                    result.aggregation or "N/A",
                    fmt(result.point_accuracy),
                    "yes" if result.convergence_warning else "no",
                )
            )
    warning_probes = [
        f"{REPRESENTATION_LABELS[rep]}→{target}"
        for rep in REPRESENTATIONS
        for target in ("class", "subject")
        if probes.get(rep, target).convergence_warning
    ]
    warning_text = ", ".join(warning_probes) if warning_probes else "none"
    window_rows = [
        (
            index,
            fmt(separability.get("window5_raw", "class_silhouette", index)),
            fmt(separability.get("window5_centered", "class_silhouette", index)),
            fmt(
                float(separability.get("window5_centered", "class_silhouette", index))
                - float(separability.get("window5_raw", "class_silhouette", index))
            ),
        )
        for index in range(1, 6)
    ]
    window_probe_rows = []
    required_window_probe_columns = {
        "representation",
        "target",
        "scope",
        "window_index",
        "accuracy",
    }
    missing_window_probe_columns = required_window_probe_columns.difference(
        window_probe_metrics.columns
    )
    if missing_window_probe_columns:
        raise ValueError(
            "linear_probe_window_metrics.csv is missing columns: "
            f"{sorted(missing_window_probe_columns)}"
        )
    for index in range(1, 6):
        accuracies: dict[str, float] = {}
        for rep in ("window5_raw", "window5_centered"):
            selected = window_probe_metrics[
                (window_probe_metrics["representation"].astype(str) == rep)
                & (window_probe_metrics["target"].astype(str) == "class")
                & (window_probe_metrics["scope"].astype(str) == "pooled_oof")
                & (pd.to_numeric(window_probe_metrics["window_index"]) == index)
            ]
            if len(selected) != 1:
                raise ValueError(
                    "Expected one pooled class point-accuracy row for "
                    f"{rep}, window {index}; observed {len(selected)}"
                )
            accuracies[rep] = _finite_float(
                selected.iloc[0]["accuracy"],
                context=f"{rep} window {index} class point accuracy",
            )
        window_probe_rows.append(
            (
                index,
                fmt(accuracies["window5_raw"]),
                fmt(accuracies["window5_centered"]),
                fmt(accuracies["window5_centered"] - accuracies["window5_raw"]),
            )
        )
    raw_window_scores = separability.window_scores("window5_raw")
    centered_window_scores = separability.window_scores("window5_centered")

    subject_effect_rows = []
    for base in ("whole", "window5"):
        raw, centered = f"{base}_raw", f"{base}_centered"
        subject_effect_rows.append(
            (
                base.upper(),
                fmt(separability.get(raw, "subject_silhouette")),
                fmt(separability.get(centered, "subject_silhouette")),
                fmt(
                    float(separability.get(raw, "subject_silhouette"))
                    - float(separability.get(centered, "subject_silhouette"))
                ),
                fmt(probes.get(raw, "subject").accuracy),
                fmt(probes.get(centered, "subject").accuracy),
                fmt(probes.get(raw, "subject").accuracy - probes.get(centered, "subject").accuracy),
            )
        )

    verdict_descriptions = {
        "H1": "WHOLE보다 WINDOW5에서 class structure가 더 명확하다.",
        "H2": "RAW representation에서 subject structure가 강하다.",
        "H3": "subject centering 후 subject structure가 감소한다.",
        "H4": "centering 후에도 특정 WINDOW5 구간의 class structure가 유지/강화된다.",
        "H5": "class information은 window에 따라 비균일하다.",
        "H6": "transition statistics는 class별로 체계적으로 다르다.",
    }
    verdict_rows = [
        (key, bundle.verdicts[key], verdict_descriptions[key]) for key in verdict_descriptions
    ]
    evidence_rows = []
    for hypothesis, items in bundle.evidence.items():
        for item in items:
            comparator = (
                f">= {item.threshold:.4f}"
                if item.direction == "at least"
                else f">= +{item.threshold:.4f} support; <= -{item.threshold:.4f} contradict"
            )
            evidence_rows.append(
                (
                    hypothesis,
                    item.label,
                    fmt(item.value),
                    comparator,
                    "+" if item.vote > 0 else "-" if item.vote < 0 else "0",
                )
            )

    distinction_text = {
        "A": "Windowed covariance states add class information beyond WHOLE.",
        "B": "Temporal order/transition may carry class-dependent information.",
        "C": "No major difference from a single covariance was detected by these diagnostics.",
    }
    distinction_rows = [
        (key, bundle.distinctions[key], distinction_text[key]) for key in ("A", "B", "C")
    ]

    classes = ", ".join(str(item) for item in dataset["classes_observed"])
    sessions = ", ".join(str(item) for item in dataset["sessions_observed"])
    sessions_available = ", ".join(
        str(item) for item in dataset.get("sessions_available", [])
    )
    epoch_relative = dataset["epoch_time_interval_relative_to_mi_cue_seconds"]
    epoch_observed_source = dataset["epoch_time_interval_observed_source_seconds"]
    shape = " × ".join(str(item) for item in dataset["array_shape"])
    class_counts = ", ".join(
        f"{label}={count}" for label, count in sorted(dataset["trials_per_class"].items())
    )
    subject_counts = sorted(int(value) for value in dataset["trials_per_subject"].values())
    trials_subject_text = (
        str(subject_counts[0])
        if len(set(subject_counts)) == 1
        else f"{min(subject_counts)}–{max(subject_counts)}"
    )
    subject_class_counts = [
        int(count)
        for per_class in dataset["trials_per_subject_class"].values()
        for count in per_class.values()
    ]
    trials_subject_class_text = (
        str(subject_class_counts[0])
        if len(set(subject_class_counts)) == 1
        else f"{min(subject_class_counts)}–{max(subject_class_counts)}"
    )
    package_versions = environment.get("packages", {})
    runtime = (
        f"{environment.get('os', {}).get('platform', 'unknown OS')}; Python "
        f"{environment.get('python', {}).get('version', 'unknown')}; NumPy "
        f"{package_versions.get('numpy', 'unknown')}; scikit-learn "
        f"{package_versions.get('scikit_learn', 'unknown')}; MOABB "
        f"{package_versions.get('moabb', 'unknown')}; pyRiemann "
        f"{package_versions.get('pyriemann', 'unknown')}."
    )
    resampling_text = (
        "no resampling"
        if prep["resample_hz"] is None
        else f"resampling to {prep['resample_hz']} Hz"
    )

    transition_class_rows = [
        (label, fmt(value)) for label, value in sorted(transitions.class_means.items())
    ] or [("Not separately exported", "N/A")]
    transition_pair_rows = [
        (pair, fmt(value)) for pair, value in sorted(transitions.pair_means.items())
    ] or [("Not separately exported", "N/A")]
    transition_subject_rows = [
        (subject, fmt(value))
        for subject, value in sorted(
            transitions.subject_means.items(),
            key=lambda item: (
                (0, float(item[0]))
                if str(item[0]).replace(".", "", 1).isdigit()
                else (1, str(item[0]))
            ),
        )
    ] or [("Not separately exported", "N/A")]
    cosine_text = (
        "not exported"
        if transitions.cosine_min is None
        else f"{transitions.cosine_min:.4f} to {transitions.cosine_max:.4f}"
    )
    centering_residuals = embedding.get("subject_centering_max_abs_mean", {})
    whole_centering_residual = _optional_float(centering_residuals.get("whole"))
    window_centering_residual = _optional_float(centering_residuals.get("window5"))
    silhouette_audit = diagnostics_summary.get("silhouette", {})

    h1_class_sil_delta = (
        float(separability.get("window5_raw", "class_silhouette"))
        - float(separability.get("whole_raw", "class_silhouette"))
    )
    h1_probe_delta = (
        probes.get("window5_raw", "class").accuracy
        - probes.get("whole_raw", "class").accuracy
    )
    retained_text = ", ".join(map(str, bundle.retained_windows)) or "none"
    lines = [
        "# BNCI2014_001 SPD Representation Probe",
        "",
        "## 1. Question",
        "",
        "This frozen pilot asks what is hidden when one motor-imagery trial is compressed into one 22 × 22 covariance matrix. It compares that WHOLE state with five ordered, non-overlapping local covariance states from exactly the same preprocessed trial interval. The two questions are whether local class/subject/temporal structure becomes visible and whether class-related temporal structure remains after label-free subject marginal centering. It does not propose or evaluate a new model.",
        "",
        "## 2. Frozen protocol",
        "",
        f"The primary analysis loaded only session `{sessions}` from the {dataset.get('n_sessions_available', 'N/A')} available `{dataset['dataset']}` sessions ({sessions_available or 'keys not exported'}), using all {dataset['n_subjects']} subjects, four classes ({classes}), and {dataset['n_eeg_channels']} EEG channels. The output contained {dataset.get('n_eog_channels_output', 'N/A')} EOG channels. The configured epoch was {epoch_relative[0]:.3f}–{epoch_relative[1]:.3f} s relative to the motor-imagery cue; its observed source-time interval was {epoch_observed_source[0]:.3f}–{epoch_observed_source[1]:.3f} s within the dataset event interval {dataset['source_event_interval_seconds'][0]:.3f}–{dataset['source_event_interval_seconds'][1]:.3f} s ({dataset['samples_per_trial']} samples at {dataset['sampling_frequency_hz']:.1f} Hz). Cached EEG amplitudes were in {dataset.get('signal_units', 'unreported units')} (source MAT units: {dataset.get('source_mat_signal_units', 'unreported')}).",
        "",
        f"A single pipeline applied an {prep['bandpass_hz'][0]:g}–{prep['bandpass_hz'][1]:g} Hz band-pass, no baseline correction, and {resampling_text}. WHOLE used all {dataset['samples_per_trial']} samples. WINDOW5 divided those samples into five consecutive {representation['expected_window_samples']}-sample blocks, with no overlap and `{representation['remainder_policy']}` remainder handling. Both used {str(representation['covariance_estimator']).upper()} covariance ({covariance_summary.get('estimator_implementation', 'implementation not exported')}), deterministic symmetrization, and no extra regularization.",
        "",
        f"Each SPD matrix was mapped by a symmetric eigendecomposition to `log(C)` and Frobenius-isometric `svec` coordinates ({dataset['n_eeg_channels']} × {dataset['n_eeg_channels'] + 1} / 2 = 253 dimensions; diagonal unchanged, off-diagonal multiplied by sqrt(2)). No StandardScaler was used. CENTERED coordinates subtract one mean per subject: all trials for WHOLE and all trial × window samples together for WINDOW5. Neither class labels nor window indices enter centering.",
        "",
        f"Visualization used one fixed PCA({embed_config['pca_components']}) + t-SNE fit for each of the four representation/state combinations (seed {config['project']['seed']}, perplexity {embed_config['perplexity']}); every WINDOW5 panel reuses its representation's global coordinates. Quantitative diagnostics use the original 253-D coordinates, not t-SNE. The linear information probe was fixed multinomial logistic regression (`C={probe_config['c']}`, {probe_config['folds']} stratified group folds, no scaling or tuning), with every trial's five windows kept in one fold.",
        "",
        f"Runtime record: {runtime}",
        "",
        "## 3. Data sanity",
        "",
        f"Observed epoch array shape was **{shape}** (trials × EEG channels × time). There were {dataset['n_trials']} trials: {trials_subject_text} per subject, {trials_subject_class_text} per subject/class, and {class_counts}. WHOLE produced {covariance_summary['whole_shape'][0]} matrices of shape {covariance_summary['whole_shape'][1]} × {covariance_summary['whole_shape'][2]}; WINDOW5 produced {covariance_summary['window5_shape'][0]} matrices of the same shape.",
        "",
        markdown_table(
            ["Representation", "count", "SPD", "non-SPD", "NaN", "Inf", "min eig", "max eig", "max condition", "max symmetry error"],
            sanity_rows,
        ),
        "",
        f"No covariance was silently removed. `is_spd` requires finite entries, positive minimum eigenvalue, and symmetry within the pipeline tolerance. After centering, the maximum absolute subject-mean coordinate was {fmt_sci(whole_centering_residual)} for WHOLE and {fmt_sci(window_centering_residual)} for WINDOW5. The saved coordinate dimension was {embedding.get('coordinate_dimension', 'N/A')}, and StandardScaler applied was `{embedding.get('standard_scaler_applied', 'N/A')}`. Detailed row-level checks are in [`covariance_sanity.csv`](../tables/covariance_sanity.csv).",
        "",
        "## 4. Raw representation",
        "",
        f"In original Log-Euclidean space, the RAW WINDOW5-minus-WHOLE class-silhouette difference was {h1_class_sil_delta:+.4f}, and the grouped class-probe accuracy difference was {h1_probe_delta:+.4f}. These are descriptive representation diagnostics, not claims of classifier performance. H1 is **{bundle.verdicts['H1']}** under the frozen rules in Section 10.",
        "",
        "WHOLE RAW figures: " + figure_links((("1A class", "figure_1a_whole_raw_class.png"), ("1B subject", "figure_1b_whole_raw_subject.png"))) + ".",
        "",
        "WINDOW5 RAW figures: " + figure_links((("2A class", "figure_2a_window5_raw_class.png"), ("2B subject", "figure_2b_window5_raw_subject.png"), ("2C window", "figure_2c_window5_raw_window.png"), ("2D class by window", "figure_2d_window5_raw_class_panels.png"))) + ".",
        "",
        "The plots are visualization only. Cluster appearance in t-SNE is not counted as evidence for a verdict.",
        "",
        "## 5. Effect of subject centering",
        "",
        markdown_table(
            ["Representation", "subject sil. RAW", "subject sil. CENTERED", "reduction", "subject probe RAW", "subject probe CENTERED", "reduction"],
            subject_effect_rows,
        ),
        "",
        f"H2 (strong RAW subject structure) is **{bundle.verdicts['H2']}** and H3 (reduction after subject centering) is **{bundle.verdicts['H3']}**. Centering is a diagnostic transform estimated from every sample of each subject in the primary session; therefore these numbers must not be read as held-out domain-adaptation performance.",
        "",
        "Centered figures: " + figure_links((("3A WHOLE class", "figure_3a_whole_centered_class.png"), ("3B WHOLE subject", "figure_3b_whole_centered_subject.png"), ("4A WINDOW5 class", "figure_4a_window5_centered_class.png"), ("4B WINDOW5 subject", "figure_4b_window5_centered_subject.png"), ("4C WINDOW5 window", "figure_4c_window5_centered_window.png"), ("4D class by window", "figure_4d_window5_centered_class_panels.png"))) + ".",
        "",
        "## 6. Temporal-window diagnostics",
        "",
        markdown_table(["Window", "RAW class silhouette", "CENTERED class silhouette", "CENTERED - RAW"], window_rows),
        "",
        "Secondary pooled OOF class point accuracy by window:",
        "",
        markdown_table(["Window", "RAW point accuracy", "CENTERED point accuracy", "CENTERED - RAW"], window_probe_rows),
        "",
        "These per-window point accuracies disclose where the fixed linear probe could access class information. They are secondary diagnostics and are **not** part of the frozen H5 verdict, which uses the predeclared per-window class-silhouette range threshold.",
        "",
        f"The RAW per-window class-silhouette range was {max(raw_window_scores) - min(raw_window_scores):.4f}; the CENTERED range was {max(centered_window_scores) - min(centered_window_scores):.4f}. Windows satisfying both a positive centered silhouette of at least {float(thresholds['metric_silhouette_delta']):.4f} and retention within that tolerance of their RAW value: **{retained_text}**. Thus H4 is **{bundle.verdicts['H4']}** and H5 is **{bundle.verdicts['H5']}**.",
        "",
        "The panel comparison uses one global embedding per state and is not an independent fit per window. The silhouette table above, not visual panel separation, supplies the quantitative evidence.",
        "",
        "## 7. Transition diagnostics",
        "",
        f"Consecutive displacement magnitudes were computed as `||z_(w+1) - z_w||₂` for pairs 1→2 through 4→5. Subject centering subtracts a constant within each subject, so it cannot change these within-trial displacement vectors except for floating-point noise; the measured maximum coordinate-wise RAW-versus-CENTERED displacement difference was {fmt_sci(transitions.centered_invariance_max_abs)}. The exported overall mean was {fmt(transitions.overall_mean)}. The class eta-squared was {transitions.class_eta_squared:.4f}; the class mean-range/grand-mean fraction was {transitions.class_mean_range_fraction:.4f}. Pairwise class mean-vector cosine similarities ranged from {cosine_text}.",
        "",
        "Class mean transition magnitudes:",
        "",
        markdown_table(["Class", "Mean magnitude"], transition_class_rows),
        "",
        "Mean magnitude by window pair:",
        "",
        markdown_table(["Window pair", "Mean magnitude"], transition_pair_rows),
        "",
        "Subject mean transition magnitudes:",
        "",
        markdown_table(["Subject", "Mean magnitude"], transition_subject_rows),
        "",
        "The complete descriptive outputs are [`transition_class_summary.csv`](../tables/transition_class_summary.csv), [`transition_subject_summary.csv`](../tables/transition_subject_summary.csv), [`transition_pair_summary.csv`](../tables/transition_pair_summary.csv), [`transition_cosine_similarity.csv`](../tables/transition_cosine_similarity.csv), and [`transition_effects.csv`](../tables/transition_effects.csv).",
        "",
        f"H6 is **{bundle.verdicts['H6']}**. This is a low-cost descriptive effect-size diagnostic; it is neither a trajectory classifier nor proof that order is causally necessary.",
        "",
        "Trajectory figures use the predeclared smallest-trial-ID rule for subjects " + ", ".join(map(str, config['diagnostics']['trajectory_subjects'])) + ": " + figure_links((("5A RAW", "figure_5a_window5_raw_example_trajectories.png"), ("5B CENTERED", "figure_5b_window5_centered_example_trajectories.png"))) + ".",
        "",
        "## 8. Quantitative separability",
        "",
        markdown_table(["Representation", "class silhouette", "subject silhouette", "window silhouette", "class distance ratio", "subject distance ratio"], overall_sep_rows),
        "",
        f"Silhouettes and between/within distance ratios were computed in the original 253-D log-svec space. WHOLE global silhouettes used {silhouette_audit.get('whole_points', 'N/A')} trial points; WINDOW5 global silhouettes used {silhouette_audit.get('window5_points', 'N/A')} points from {silhouette_audit.get('window5_trials', 'N/A')} deterministically subject/class-balanced whole trials. Per-window class silhouettes used all trials at each window. A distance ratio above 1 means average between-group distance exceeds average within-group distance. The full deterministic table is [`separability_metrics.csv`](../tables/separability_metrics.csv). Cross-representation silhouette deltas compare one point per WHOLE trial with repeated local states per WINDOW5 trial and different deterministic sample counts; they are descriptive, do not adjust for repeated measurements, and do not test statistical significance.",
        "",
        "## 9. Linear information probes",
        "",
        markdown_table(["Representation", "Target", "Primary accuracy", "Fold SD", "Chance", "Folds", "Aggregation", "Point accuracy", "Convergence warning"], probe_rows),
        "",
        f"These values measure linearly accessible information only. Trial-grouped folds prevent the five windows from one trial crossing train/test boundaries. WHOLE accuracy is point/trial accuracy; WINDOW5 primary accuracy averages held-out probabilities across the five windows before assigning one prediction per trial, while its point accuracy is shown separately. Probes with a convergence warning: **{warning_text}**. This within-primary-session diagnostic does not test new sessions or unseen subjects and was not tuned. Full fold/summary output is in [`linear_probe_metrics.csv`](../tables/linear_probe_metrics.csv).",
        "",
        "## 10. Hypothesis verdicts",
        "",
        markdown_table(["Hypothesis", "Verdict", "Claim"], verdict_rows),
        "",
        "The decision rules were fixed from the YAML thresholds before reading results: H1 requires at least two of RAW class-silhouette, class-probe, and (when exported) relative class-distance-ratio deltas to reach their thresholds, with no threshold-sized contradiction. H2 requires at least two of subject silhouette, above-chance subject probe, and (when exported) subject distance ratio to pass in both RAW representations. H3 requires at least two threshold-sized reductions in both representations, with no threshold-sized increase. H4 requires at least one retained positive window and a CENTERED WINDOW5 class-probe margin over chance. H5 requires the per-window silhouette range threshold in both RAW and CENTERED. H6 requires both transition effect thresholds. Partial or conflicting evidence is MIXED; zero passing evidence is NOT SUPPORTED.",
        "",
        markdown_table(["Hypothesis", "Evidence", "Measured value", "Frozen rule", "Vote"], evidence_rows),
        "",
        "Thresholds: `metric_silhouette_delta={metric_silhouette_delta}`, `probe_accuracy_delta={probe_accuracy_delta}`, `above_chance_margin={above_chance_margin}`, `window_silhouette_range={window_silhouette_range}`, `distance_ratio_relative_delta={distance_ratio_relative_delta}`, `transition_eta_squared={transition_eta_squared}`, `transition_mean_range_fraction={transition_mean_range_fraction}`.".format(**thresholds),
        "",
        "## 11. What we learned",
        "",
        markdown_table(["Possibility", "Assessment", "Meaning in this pilot"], distinction_rows),
        "",
        f"1. WINDOW5 changed RAW class silhouette by {h1_class_sil_delta:+.4f} and class-probe accuracy by {h1_probe_delta:+.4f} relative to WHOLE; the predeclared H1 verdict is {bundle.verdicts['H1']}.",
        "",
        f"2. Subject centering changed subject-probe accuracy from {probes.get('whole_raw', 'subject').accuracy:.4f} to {probes.get('whole_centered', 'subject').accuracy:.4f} for WHOLE and from {probes.get('window5_raw', 'subject').accuracy:.4f} to {probes.get('window5_centered', 'subject').accuracy:.4f} for WINDOW5; H3 is {bundle.verdicts['H3']}.",
        "",
        f"3. Window nonuniformity is {bundle.verdicts['H5']} (silhouette ranges {max(raw_window_scores) - min(raw_window_scores):.4f} RAW, {max(centered_window_scores) - min(centered_window_scores):.4f} CENTERED), while class-dependent transition evidence is {bundle.verdicts['H6']} (eta-squared {transitions.class_eta_squared:.4f}, range fraction {transitions.class_mean_range_fraction:.4f}).",
        "",
        "These alternatives are kept separate: evidence for local states (A) does not by itself establish that order matters (B), and absence of either under these fixed diagnostics supports the no-major-difference reading (C).",
        "",
        "## 12. What we should NOT conclude yet",
        "",
        "- This pilot does not establish that a distribution classifier is needed.",
        "",
        "- It does not establish that a trajectory or sequence model is needed.",
        "",
        "- It does not show that domain adaptation will improve, or evaluate cross-session/cross-subject adaptation.",
        "",
        "- It does not support a claim that pseudo-label-free conditional alignment has been solved or is feasible.",
        "",
        "- It contains no neural model, SOTA comparison, hyperparameter sweep, inferential test, or correction for multiple diagnostics. Negative and mixed verdicts are retained as such.",
        "",
        "## 13. Recommended next experiment",
        "",
        "Run exactly one locked replication on BNCI2014_001 session 2: change only the session selector and output/cache namespace, then execute the identical 22-channel, 8–32 Hz, 0–3.996 s, OAS WHOLE/WINDOW5, log-svec, subject-centering, embedding, diagnostic, and verdict pipeline with the same seed and thresholds. Do not use session-1 results to alter preprocessing, windows, probes, or verdict rules. The purpose is solely to test whether the session-1 conclusions reproduce.",
        "",
    ]
    headings = [line for line in lines if line.startswith("## ")]
    expected = [
        "## 1. Question",
        "## 2. Frozen protocol",
        "## 3. Data sanity",
        "## 4. Raw representation",
        "## 5. Effect of subject centering",
        "## 6. Temporal-window diagnostics",
        "## 7. Transition diagnostics",
        "## 8. Quantitative separability",
        "## 9. Linear information probes",
        "## 10. Hypothesis verdicts",
        "## 11. What we learned",
        "## 12. What we should NOT conclude yet",
        "## 13. Recommended next experiment",
    ]
    if headings != expected:
        raise AssertionError(f"Report section contract changed: {headings}")
    return "\n".join(lines)


def load_frozen_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    supplied_path = config_path.expanduser().resolve()
    if not supplied_path.is_file():
        raise FileNotFoundError(supplied_path)
    supplied = yaml.safe_load(supplied_path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError(f"Expected a YAML mapping in {supplied_path}")
    output_dir = resolve(ROOT, supplied["project"]["output_dir"])
    frozen_path = output_dir / "frozen_config.yaml"
    if not frozen_path.is_file():
        raise FileNotFoundError(
            f"Frozen config is missing: {frozen_path}; run stage 01 first"
        )
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if supplied != frozen:
        raise RuntimeError(
            f"Supplied config differs from the protocol frozen at {frozen_path}"
        )
    return frozen, output_dir


def validate_figure_inputs(figures_dir: Path) -> None:
    expected = [
        "figure_1a_whole_raw_class.png",
        "figure_1b_whole_raw_subject.png",
        "figure_2a_window5_raw_class.png",
        "figure_2b_window5_raw_subject.png",
        "figure_2c_window5_raw_window.png",
        "figure_2d_window5_raw_class_panels.png",
        "figure_3a_whole_centered_class.png",
        "figure_3b_whole_centered_subject.png",
        "figure_4a_window5_centered_class.png",
        "figure_4b_window5_centered_subject.png",
        "figure_4c_window5_centered_window.png",
        "figure_4d_window5_centered_class_panels.png",
        "figure_5a_window5_raw_example_trajectories.png",
        "figure_5b_window5_centered_example_trajectories.png",
    ]
    missing = [name for name in expected if not (figures_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required report figures are missing: {missing}")


def synthetic_dry_run(config_path: Path) -> None:
    supplied = yaml.safe_load(config_path.expanduser().resolve().read_text(encoding="utf-8"))
    thresholds = supplied["diagnostics"]["verdict_thresholds"]
    sep_values: dict[tuple[str, int | None, str], float] = {}
    for rep in REPRESENTATIONS:
        sep_values[(rep, None, "class_silhouette")] = 0.01
        sep_values[(rep, None, "subject_silhouette")] = 0.10 if rep.endswith("raw") else 0.01
        sep_values[(rep, None, "class_separation_ratio")] = 1.02
        sep_values[(rep, None, "subject_separation_ratio")] = 1.20 if rep.endswith("raw") else 1.01
        if rep.startswith("window5"):
            sep_values[(rep, None, "window_silhouette")] = 0.03
            for window in range(1, 6):
                sep_values[(rep, window, "class_silhouette")] = 0.01 * window
    sep_values[("window5_raw", None, "class_silhouette")] = 0.05
    sep_values[("window5_centered", None, "class_silhouette")] = 0.04
    probes = Probes(
        {
            (rep, target): ProbeResult(
                accuracy=(0.40 if target == "class" else (0.80 if rep.endswith("raw") else 0.18)),
                std=0.01,
                chance=(0.25 if target == "class" else 1 / 9),
                folds=5,
            )
            for rep in REPRESENTATIONS
            for target in ("class", "subject")
        }
    )
    bundle = compute_verdicts(
        Separability(sep_values),
        probes,
        TransitionDiagnostics(0.02, 0.08, 1.0, 0.0, {}, {}, {}, None, None),
        thresholds,
    )
    if set(bundle.verdicts) != {"H1", "H2", "H3", "H4", "H5", "H6"}:
        raise AssertionError("Dry-run verdict keys are incomplete")
    print(json.dumps({"dry_run": "ok", "verdicts": bundle.verdicts}, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        synthetic_dry_run(args.config)
        return
    config, output_dir = load_frozen_config(args.config)
    tables_dir = output_dir / "tables"
    report_dir = output_dir / "report"
    validate_figure_inputs(output_dir / "figures")
    dataset = read_json(tables_dir / "dataset_metadata.json")
    environment = read_json(tables_dir / "environment.json")
    covariance_summary = read_json(tables_dir / "covariance_summary.json")
    embedding = read_json(tables_dir / "embedding_metadata.json")
    diagnostics_summary = read_json(tables_dir / "diagnostics_summary.json")
    sanity = read_csv(tables_dir / "covariance_sanity.csv")
    separability = Separability.from_frame(read_csv(tables_dir / "separability_metrics.csv"))
    probes = Probes.from_frame(read_csv(tables_dir / "linear_probe_metrics.csv"))
    window_probe_metrics = read_csv(tables_dir / "linear_probe_window_metrics.csv")
    transitions = TransitionDiagnostics.from_tables(tables_dir)
    bundle = compute_verdicts(
        separability,
        probes,
        transitions,
        config["diagnostics"]["verdict_thresholds"],
    )
    report = render_report(
        config=config,
        dataset=dataset,
        environment=environment,
        covariance_summary=covariance_summary,
        sanity=sanity,
        embedding=embedding,
        diagnostics_summary=diagnostics_summary,
        separability=separability,
        probes=probes,
        window_probe_metrics=window_probe_metrics,
        transitions=transitions,
        bundle=bundle,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "representation_probe_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "verdicts": bundle.verdicts,
                "distinctions": bundle.distinctions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
