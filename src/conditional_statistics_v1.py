"""Subject-level statistics for Conditional-Geometry Anatomy v1.

The functions in this module operate only on cached numeric arrays.  They do
not know how to load BNCI2014_001 and cannot cross the discovery/confirmatory
session boundary.  In particular, A/B halves are never promoted to independent
group replicates: every group statistic is a median over subjects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


__all__ = [
    "DEGENERACY_MULTIPLIER",
    "RANK_TOLERANCE",
    "NullSummary",
    "BootstrapSummary",
    "OracleSubjectSummary",
    "StageEvidence",
    "StageDecision",
    "normalize_shape_vectors",
    "cosine_rows",
    "reliability_subject_scores",
    "loso_templates",
    "discovery_shared_subject_scores",
    "confirmatory_shared_subject_scores",
    "discovery_oracle_score_sets",
    "confirmatory_oracle_score_sets",
    "conservative_candidate_ranks",
    "summarize_oracle_scores",
    "plus_one_null_summary",
    "subject_null_percentiles",
    "leave_one_subject_out_influence",
    "subject_bootstrap_median",
    "subject_bootstrap_paired_median_delta",
    "evaluate_fixed_sequence",
    "terminal_airm_decision",
    "le_robustness_label",
]


DEGENERACY_MULTIPLIER = 100.0
RANK_TOLERANCE = 1.0e-12
STAGE_ORDER = ("R", "S", "P")


def _finite_float_array(value: np.ndarray, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _degeneracy_threshold(vectors: np.ndarray) -> np.ndarray:
    maximum = np.max(np.abs(vectors), axis=-1)
    return DEGENERACY_MULTIPLIER * np.finfo(np.float64).eps * np.maximum(1.0, maximum)


def normalize_shape_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors with the protocol's exact machine-scale hard gate."""

    array = _finite_float_array(vectors, name="vectors")
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError("vectors must have a non-empty final feature axis")
    norms = np.linalg.norm(array, axis=-1)
    thresholds = _degeneracy_threshold(array)
    if np.any(norms <= thresholds):
        index = tuple(np.argwhere(norms <= thresholds)[0])
        raise ValueError(
            "DEGENERATE_CLASS_GEOMETRY: shape norm is at or below the frozen "
            f"machine-scale threshold at index {index}"
        )
    normalized = array / norms[..., None]
    if not np.isfinite(normalized).all():
        raise ValueError("DEGENERATE_CLASS_GEOMETRY: normalization is non-finite")
    return normalized


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise cosine after applying the same frozen shape normalization."""

    a = normalize_shape_vectors(left)
    b = normalize_shape_vectors(right)
    if a.shape != b.shape:
        raise ValueError(f"cosine operands must have equal shape, got {a.shape} and {b.shape}")
    return np.sum(a * b, axis=-1, dtype=np.float64)


def _subject_split_shapes(shapes: np.ndarray, *, name: str) -> np.ndarray:
    values = _finite_float_array(shapes, name=name, ndim=3)
    if values.shape[0] < 2 or values.shape[1] != 3 or values.shape[2] < 1:
        raise ValueError(f"{name} must have shape (subjects>=2, 3[A,B,F], features)")
    return normalize_shape_vectors(values)


def reliability_subject_scores(shapes: np.ndarray) -> np.ndarray:
    """Return one A-vs-B cosine per subject from ``(subject,A/B/F,feature)``."""

    values = _subject_split_shapes(shapes, name="shapes")
    return np.sum(values[:, 0] * values[:, 1], axis=1, dtype=np.float64)


def loso_templates(subject_shapes: np.ndarray) -> np.ndarray:
    """Normalize the sum of the other subjects' already-unit shape vectors."""

    values = _finite_float_array(subject_shapes, name="subject_shapes", ndim=2)
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("subject_shapes must have at least two subjects and one feature")
    unit = normalize_shape_vectors(values)
    # Sum only the actual sources.  Avoid total-minus-target so even floating
    # roundoff cannot make target s contribute to its own template.
    source_sums = np.stack(
        [
            np.sum(np.delete(unit, target, axis=0), axis=0, dtype=np.float64)
            for target in range(len(unit))
        ],
        axis=0,
    )
    return normalize_shape_vectors(source_sums)


def discovery_shared_subject_scores(discovery_shapes: np.ndarray) -> np.ndarray:
    """Cross-half discovery Stage-S score for each LOSO target subject."""

    shapes = _subject_split_shapes(discovery_shapes, name="discovery_shapes")
    template_a = loso_templates(shapes[:, 0])
    template_b = loso_templates(shapes[:, 1])
    a_to_b = np.sum(template_a * shapes[:, 1], axis=1, dtype=np.float64)
    b_to_a = np.sum(template_b * shapes[:, 0], axis=1, dtype=np.float64)
    return 0.5 * (a_to_b + b_to_a)


def confirmatory_shared_subject_scores(
    discovery_shapes: np.ndarray,
    confirmatory_shapes: np.ndarray,
) -> np.ndarray:
    """Frozen discovery-F template versus confirmatory A/B for each subject."""

    discovery = _subject_split_shapes(discovery_shapes, name="discovery_shapes")
    confirmatory = _subject_split_shapes(confirmatory_shapes, name="confirmatory_shapes")
    if discovery.shape != confirmatory.shape:
        raise ValueError("discovery and confirmatory shapes must have equal shape")
    template_f = loso_templates(discovery[:, 2])
    scores_a = np.sum(template_f * confirmatory[:, 0], axis=1, dtype=np.float64)
    scores_b = np.sum(template_f * confirmatory[:, 1], axis=1, dtype=np.float64)
    return 0.5 * (scores_a + scores_b)


def _candidate_shape_bank(candidate_shapes: np.ndarray, n_subjects: int, n_features: int) -> np.ndarray:
    bank = _finite_float_array(candidate_shapes, name="candidate_shapes", ndim=4)
    expected_prefix = (n_subjects, 3, 24)
    if bank.shape[:3] != expected_prefix or bank.shape[-1] != n_features:
        raise ValueError(
            "candidate_shapes must have shape "
            f"({n_subjects},3,24,{n_features}), got {bank.shape}"
        )
    return normalize_shape_vectors(bank)


def discovery_oracle_score_sets(
    discovery_shapes: np.ndarray,
    candidate_shapes: np.ndarray,
) -> np.ndarray:
    """Return discovery cross-half scores for all 24 target permutations.

    ``candidate_shapes[s,h,p]`` is the normalized shape reconstructed from the
    target object's p-th S4 action.  Candidate scores are averaged before rank.
    """

    shapes = _subject_split_shapes(discovery_shapes, name="discovery_shapes")
    candidates = _candidate_shape_bank(candidate_shapes, shapes.shape[0], shapes.shape[2])
    template_a = loso_templates(shapes[:, 0])
    template_b = loso_templates(shapes[:, 1])
    a_to_b = np.einsum("sf,spf->sp", template_a, candidates[:, 1], optimize=True)
    b_to_a = np.einsum("sf,spf->sp", template_b, candidates[:, 0], optimize=True)
    return 0.5 * (a_to_b + b_to_a)


def confirmatory_oracle_score_sets(
    discovery_shapes: np.ndarray,
    confirmatory_candidate_shapes: np.ndarray,
) -> np.ndarray:
    """Return frozen discovery-F versus confirmatory-F scores for all S4."""

    discovery = _subject_split_shapes(discovery_shapes, name="discovery_shapes")
    candidates = _candidate_shape_bank(
        confirmatory_candidate_shapes, discovery.shape[0], discovery.shape[2]
    )
    template_f = loso_templates(discovery[:, 2])
    return np.einsum("sf,spf->sp", template_f, candidates[:, 2], optimize=True)


def conservative_candidate_ranks(
    score_sets: np.ndarray,
    *,
    tolerance: float = RANK_TOLERANCE,
) -> np.ndarray:
    """Rank every candidate worst-within-tolerance, independently by subject."""

    scores = _finite_float_array(score_sets, name="score_sets", ndim=2)
    if scores.shape[1] != 24:
        raise ValueError("score_sets must contain exactly 24 candidate scores")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    # [subject, candidate-being-ranked, candidate-being-compared]
    ranks = np.sum(
        scores[:, None, :] >= scores[:, :, None] - float(tolerance),
        axis=2,
        dtype=np.int64,
    )
    if np.any((ranks < 1) | (ranks > 24)):
        raise RuntimeError("internal candidate rank is outside 1..24")
    return ranks


@dataclass(frozen=True)
class OracleSubjectSummary:
    identity_scores: np.ndarray
    identity_ranks: np.ndarray
    normalized_ranks: np.ndarray
    top1_exact: np.ndarray
    margins: np.ndarray
    best_indices: np.ndarray
    second_best_indices: np.ndarray


def summarize_oracle_scores(
    score_sets: np.ndarray,
    permutations: np.ndarray,
    *,
    identity_index: int = 0,
    tolerance: float = RANK_TOLERANCE,
) -> OracleSubjectSummary:
    """Summarize identity rank and deterministic best/second reporting order."""

    scores = _finite_float_array(score_sets, name="score_sets", ndim=2)
    mappings = np.asarray(permutations, dtype=np.int64)
    if scores.shape[1] != 24 or mappings.shape != (24, 4):
        raise ValueError("expected score_sets[:,24] and permutations[24,4]")
    if not np.array_equal(np.sort(mappings, axis=1), np.tile(np.arange(4), (24, 1))):
        raise ValueError("permutations contains a non-S4 row")
    if len({tuple(row) for row in mappings.tolist()}) != 24:
        raise ValueError("permutations must contain 24 unique mappings")
    if not 0 <= int(identity_index) < 24:
        raise ValueError("identity_index is outside 0..23")

    ranks = conservative_candidate_ranks(scores, tolerance=tolerance)
    identity_scores = scores[:, identity_index]
    identity_ranks = ranks[:, identity_index]
    normalized = (24.0 - identity_ranks.astype(np.float64)) / 23.0
    nonidentity = np.delete(scores, identity_index, axis=1)
    margins = identity_scores - np.max(nonidentity, axis=1)

    best = np.empty(scores.shape[0], dtype=np.int64)
    second = np.empty(scores.shape[0], dtype=np.int64)
    tuples = [tuple(int(value) for value in row) for row in mappings]
    for subject_index, row in enumerate(scores):
        order = sorted(range(24), key=lambda index: (-float(row[index]), tuples[index]))
        best[subject_index], second[subject_index] = order[:2]
    return OracleSubjectSummary(
        identity_scores=identity_scores.copy(),
        identity_ranks=identity_ranks.copy(),
        normalized_ranks=normalized,
        top1_exact=identity_ranks == 1,
        margins=margins,
        best_indices=best,
        second_best_indices=second,
    )


@dataclass(frozen=True)
class NullSummary:
    observed: float
    null_median: float
    effect: float
    exceedances: int
    p_value: float
    replicates: int


def plus_one_null_summary(observed: float, null_statistics: np.ndarray) -> NullSummary:
    """Frozen one-sided greater/equal Monte-Carlo test and null-median effect."""

    if not np.isfinite(observed):
        raise ValueError("observed must be finite")
    null = _finite_float_array(null_statistics, name="null_statistics", ndim=1)
    exceedances = int(np.count_nonzero(null >= float(observed)))
    return NullSummary(
        observed=float(observed),
        null_median=float(np.median(null)),
        effect=float(observed - np.median(null)),
        exceedances=exceedances,
        p_value=float((1 + exceedances) / (len(null) + 1)),
        replicates=len(null),
    )


def subject_null_percentiles(
    observed_subject_scores: np.ndarray,
    null_subject_statistics: np.ndarray,
) -> np.ndarray:
    """Plus-one lower-tail percentile of each observed subject score."""

    observed = _finite_float_array(
        observed_subject_scores, name="observed_subject_scores", ndim=1
    )
    null = _finite_float_array(null_subject_statistics, name="null_subject_statistics", ndim=2)
    if null.shape[1] != len(observed):
        raise ValueError("null subject axis does not match observed subjects")
    counts = np.count_nonzero(null <= observed[None, :], axis=0)
    return (1.0 + counts.astype(np.float64)) / (null.shape[0] + 1.0)


def leave_one_subject_out_influence(subject_scores: np.ndarray) -> np.ndarray:
    """Return median(without s) minus the full nine-subject median."""

    scores = _finite_float_array(subject_scores, name="subject_scores", ndim=1)
    if len(scores) < 3:
        raise ValueError("at least three subject scores are required")
    full = float(np.median(scores))
    return np.asarray(
        [float(np.median(np.delete(scores, index))) - full for index in range(len(scores))],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BootstrapSummary:
    statistics: np.ndarray
    ci_low: float
    ci_high: float
    replicates: int


def subject_bootstrap_median(
    subject_scores: np.ndarray,
    *,
    replicates: int = 20_000,
    master_seed: int = 20260809,
    phase_tag: int,
) -> BootstrapSummary:
    """Frozen subject bootstrap using one tagged PCG64DXSM stream per draw."""

    scores = _finite_float_array(subject_scores, name="subject_scores", ndim=1)
    if len(scores) != 9:
        raise ValueError("the frozen bootstrap requires exactly nine subject scores")
    if not isinstance(replicates, (int, np.integer)) or int(replicates) < 1:
        raise ValueError("replicates must be a positive integer")
    # Local import avoids a module import cycle while keeping one RNG authority.
    from src.conditional_nulls_v1 import tagged_replicate_rng

    statistics = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        rng = tagged_replicate_rng(
            family_tag=1401,
            phase_tag=int(phase_tag),
            replicate_index=replicate,
            master_seed=int(master_seed),
        )
        sampled = rng.integers(0, len(scores), size=len(scores), endpoint=False)
        statistics[replicate] = float(np.median(scores[sampled]))
    quantiles = np.quantile(statistics, [0.025, 0.975], method="linear")
    return BootstrapSummary(
        statistics=statistics,
        ci_low=float(quantiles[0]),
        ci_high=float(quantiles[1]),
        replicates=int(replicates),
    )


def subject_bootstrap_paired_median_delta(
    left_subject_scores: np.ndarray,
    right_subject_scores: np.ndarray,
    *,
    replicates: int = 20_000,
    master_seed: int = 20260809,
    phase_tag: int = 2,
) -> BootstrapSummary:
    """Paired-resample ``median(left)-median(right)`` on the same IDs.

    This is the frozen descriptive resampling primitive for discovery versus
    confirmatory and AIRM versus LE stage-effect deltas.  It intentionally does
    not replace the two medians by a median of subjectwise differences.
    """

    left = _finite_float_array(left_subject_scores, name="left_subject_scores", ndim=1)
    right = _finite_float_array(right_subject_scores, name="right_subject_scores", ndim=1)
    if left.shape != (9,) or right.shape != (9,):
        raise ValueError("the frozen paired bootstrap requires two nine-subject vectors")
    if not isinstance(replicates, (int, np.integer)) or int(replicates) < 1:
        raise ValueError("replicates must be a positive integer")
    from src.conditional_nulls_v1 import tagged_replicate_rng

    statistics = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        rng = tagged_replicate_rng(
            family_tag=1401,
            phase_tag=int(phase_tag),
            replicate_index=replicate,
            master_seed=int(master_seed),
        )
        sampled = rng.integers(0, 9, size=9, endpoint=False)
        statistics[replicate] = float(np.median(left[sampled]) - np.median(right[sampled]))
    quantiles = np.quantile(statistics, [0.025, 0.975], method="linear")
    return BootstrapSummary(
        statistics=statistics,
        ci_low=float(quantiles[0]),
        ci_high=float(quantiles[1]),
        replicates=int(replicates),
    )


@dataclass(frozen=True)
class StageEvidence:
    discovery_effect: float
    confirmatory_effect: float
    confirmatory_p: float
    hard_gates_pass: bool

    @property
    def criterion_pass(self) -> bool:
        return bool(
            np.isfinite(self.discovery_effect)
            and np.isfinite(self.confirmatory_effect)
            and np.isfinite(self.confirmatory_p)
            and self.discovery_effect > 0.0
            and self.confirmatory_effect > 0.0
            and 0.0 <= self.confirmatory_p <= 0.025
            and self.hard_gates_pass
        )


@dataclass(frozen=True)
class StageDecision:
    stage: str
    eligible: bool
    criterion_pass: bool
    status: str


def evaluate_fixed_sequence(
    evidence: Mapping[str, StageEvidence],
) -> tuple[dict[str, StageDecision], bool]:
    """Apply R->S->P gatekeeping while retaining all raw criteria."""

    if set(evidence) != set(STAGE_ORDER):
        raise ValueError("evidence must contain exactly R, S and P")
    decisions: dict[str, StageDecision] = {}
    eligible = True
    for stage in STAGE_ORDER:
        criterion = evidence[stage].criterion_pass
        if not eligible:
            status = "DESCRIPTIVE_ONLY"
        elif criterion:
            status = "PASS"
        else:
            status = "FAIL"
        decisions[stage] = StageDecision(
            stage=stage,
            eligible=eligible,
            criterion_pass=criterion,
            status=status,
        )
        eligible = eligible and criterion
    return decisions, all(decisions[stage].status == "PASS" for stage in STAGE_ORDER)


def terminal_airm_decision(
    d_chain_pass: bool,
    g_chain_pass: bool,
    *,
    failure: str | None = None,
) -> str:
    """Return exactly one frozen AIRM terminal label."""

    failure_labels = {
        "data": "UNASSESSED_DATA_CONTRACT_FAILURE",
        "numerical": "UNASSESSED_NUMERICAL_FAILURE",
        "degenerate": "UNASSESSED_DEGENERATE_GEOMETRY",
    }
    if failure is not None:
        if failure not in failure_labels:
            raise ValueError("failure must be one of data/numerical/degenerate")
        return failure_labels[failure]
    pair = (bool(d_chain_pass), bool(g_chain_pass))
    return {
        (True, True): "GO_STRONG",
        (True, False): "GO_METRIC_ONLY",
        (False, True): "STOP_TANGENT_ONLY",
        (False, False): "STOP_NO_SHARED_GEOMETRY",
    }[pair]


def le_robustness_label(
    airm_chain_pass: Sequence[bool],
    le_chain_pass: Sequence[bool],
) -> str:
    """Compare the ordered (D,G) chain-pass pairs using the frozen rule."""

    if len(airm_chain_pass) != 2 or len(le_chain_pass) != 2:
        raise ValueError("AIRM and LE statuses must each be ordered (D,G) pairs")
    airm = tuple(bool(value) for value in airm_chain_pass)
    le = tuple(bool(value) for value in le_chain_pass)
    if airm == le:
        return "AIRM+LE CONSISTENT"
    if not any(airm) and any(le):
        return "LE-ONLY — DOES NOT RESCUE AIRM FAILURE"
    # LE is equal to or a strict subset of AIRM support: no opposite LE-only
    # passing chain exists, so the discrepancy is AIRM-specific.
    if any(airm) and all((not le_i) or airm_i for airm_i, le_i in zip(airm, le, strict=True)):
        return "AIRM-SPECIFIC"
    return "AIRM/LE DISCORDANT"
