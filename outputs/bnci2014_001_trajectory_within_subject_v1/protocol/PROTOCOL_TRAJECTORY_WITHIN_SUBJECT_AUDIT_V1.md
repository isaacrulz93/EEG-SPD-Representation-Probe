# BNCI2014_001 Trajectory Within-Subject Audit v1 — Frozen Protocol

Protocol name: **Trajectory Within-Subject Audit v1**

Protocol version: **1.0**

Freeze date: **2026-08-10** (Asia/Seoul)

Frozen reference branch: **pilot/trajectory-anatomy-v0**

Frozen reference commit: **fcb55ccdccdd4613290b8e8d93be91ea256edd45**

Audit branch: **pilot/trajectory-within-subject-audit-v1**

Label/order master seed: **20260810**

Classifier random state: **20260809**

## 1. Retrospective scope

Trajectory Anatomy v0 asked whether the frozen five-window representation carried a
population-shared class signal under cross-subject LOSO. This audit asks the one missing
interpretive question: whether the same representation carries class information inside an
individual subject and whether that subject-specific information transfers to the same subject's
independent session.

This is not a new trajectory method. It does not add or modify a representation, preprocessing
choice, split, classifier, dataset, threshold, or tuning rule. It does not use WHOLE
subject-by-class interaction features or results. Negative results are terminally valid.

## 2. Immutable references and namespace

The following files at reference commit `fcb55ccdccdd4613290b8e8d93be91ea256edd45`
are immutable inputs:

- `docs/PROTOCOL_TRAJECTORY_ANATOMY_V0.md`
- `configs/bnci2014_001_trajectory_v0.yaml`
- `outputs/bnci2014_001_trajectory_v0/`
- `cache/bnci2014_001_trajectory_v0/trajectory_features_v0.npz`
- the v0 trajectory geometry, feature, evaluation, discovery, and tests

No v0 cache or output is overwritten. New regenerable data live only below
`cache/bnci2014_001_trajectory_within_subject_v1/`. New tracked results live only below
`outputs/bnci2014_001_trajectory_within_subject_v1/`.

## 3. Data and frozen preprocessing

Dataset: MOABB `BNCI2014_001`, subjects 1 through 9, both sessions `0train` and
`1test`, fixed class order `left_hand`, `right_hand`, `feet`, `tongue`.

Each session must contain exactly 2,592 trials: 288 per subject, 72 per
subject/class, 48 per subject/run, and 12 per subject/run/class over runs 0 through 5.
The combined audit contains exactly 5,184 trials. No subject or trial is selected by another
experiment's result.

Session 1 uses the exact v0 preprocessing:

- 22 frozen ordered EEG channels; EOG excluded
- 8–32 Hz band-pass
- cue-relative 0.000–3.996 s
- 250 Hz and exactly 1,000 samples; no resampling
- no baseline
- float32 epoch-cache quantization before covariance construction, matching the frozen V1 cache
- five non-overlapping 200-sample windows in chronological order
- OAS covariance independently per window, output float64
- deterministic symmetrization only
- no diagonal loading, eigenvalue clipping, rejection, imputation, or added regularization
- primary geometry AIRM

The nine session-1 E files are size- and SHA-256-pinned in the frozen config. Loading code may
open only those files. Existing validated caches may be reused only when their complete content
and provenance match the frozen config; otherwise they are regenerated in the new v1 cache.

## 4. Frozen representations

Only the exact v0 AIRM representations are admissible.

1. `PATH_D10`: `[d12,d13,d14,d15,d23,d24,d25,d34,d35,d45]` from the chronological
   five-state AIRM distance matrix.
2. `BAG_CANON_D10`: the exact lexicographically minimal D10 vector over all 120 S5
   relabelings, removing state identity/order.
3. `SCALARS_11`: the frozen v0 eleven-scalar vector in its original order.

No representation, window count, scalar, feature selection, normalization, or learned embedding
is added.

## 5. Pre-result implementation freeze

Before any real within-subject or cross-session class score is computed, this protocol, its YAML
config, and synthetic/unit tests are committed with message
`freeze trajectory within-subject audit protocol`.

After that commit, representation, split, model, C, scaler, null count, statistic, threshold,
seed derivation, and decision logic do not change. A later numerical/implementation repair may
only restore the frozen definition and must be recorded in provenance; it may not adapt to a
scientific result.

## 6. Hard gate 0 — v0 session-0 reproduction

Before scientific evaluation, session-0 data are reconstructed from the hash-pinned v0 WINDOW5
covariances using the frozen v0 AIRM geometry implementation and compared with the frozen v0
feature archive and tables.

Required identities are exact:

- row order and 2,592 trial count
- `sample_index`, subject, session, run, `trial_id`, `trial_uid`, and class label
- fixed class/run/channel order

Required numerical comparisons use `rtol=0` and `atol=1e-12` fixed before results:

- AIRM `PATH_D10`
- AIRM `BAG_CANON_D10`
- AIRM `SCALARS_11`

The frozen v0 geometry gate must be PASS and all reconstructed SPD, distance, barycenter, intrinsic,
and BAG-invariance gates must pass. The tolerance is not widened after inspection. Failure returns
`UNASSESSED_TRAJECTORY_REPRODUCTION_FAILURE`; no Stage W/X/O score is computed.

Session-1 features must independently pass the same v0 numerical geometry gates. A session-1
data/numerical failure also prevents scientific inference and is reported as
`UNASSESSED_TRAJECTORY_REPRODUCTION_FAILURE`. A classifier convergence warning, incomplete grid,
missing replicate, or other technical failure returns `UNASSESSED_TECHNICAL_FAILURE`. Required
fits are never silently dropped or replaced by available cases.

## 7. Frozen classifier and metrics

Every classifier is exactly:

- `StandardScaler(with_mean=True, with_std=True)`, fit on training rows only
- multinomial `LogisticRegression`
- C = 1.0, solver = `lbfgs`, max_iter = 5000, tol = 1e-4
- class_weight = null, no tuning, classifier random_state = 20260809
- BLAS/thread limit one per fit

A convergence warning marks the required fit FAILED. Primary metric is balanced accuracy.
Secondary metrics are accuracy, macro-F1, per-class recall in fixed semantic order, and the fixed
order confusion matrix. Chance is 0.25.

## 8. Stage W — within-subject, within-session class decoding

For every subject and session, Half A is runs 0,1,2 and Half B is runs 3,4,5. Both directions
`A_TO_B` and `B_TO_A` are evaluated. Each fit contains exactly one subject and one session;
training and test trial UIDs are disjoint and exhaust that subject/session.

For subject s:

`W_s = mean BA over both sessions and both directions`.

Primary statistic:

`T_W = median_s W_s`.

PATH is primary. BAG receives the same observed evaluation and label permutations as a mandatory
unordered comparator. SCALARS receives observed descriptive evaluation only and never votes.

## 9. Stage W label-destruction null

Exactly 1,999 deterministic replicates use the label stream in the config. Within every replicate,
labels are independently shuffled inside each subject × session × run group after canonical row
sorting. Every group's original 12-per-class label multiset is preserved. Features and identities
remain fixed. One global label realization is shared by PATH and BAG and is also reused for the
corresponding Stage X replicate.

Every replicate refits every required Stage W classifier and computes the identical `T_W`.

- `E_W = T_W_observed - median(T_W_null)`
- `p_W = (1 + count(null >= observed)) / 2000`

PATH Stage W PASS iff `E_W > 0` and `p_W <= 0.05`. BAG is summarized by the same rule but cannot
rescue or alter the PATH decision.

## 10. Stage X — same-subject cross-session transfer

Stage X observed scores are computed for all representations, but PATH has an inferential vote
only when PATH Stage W passes. For each subject, train on all six runs of one session and test on
all six runs of the other session in both directions. Training contains exactly that one subject;
sessions and trial UIDs are disjoint and no other subject is present.

`X_s = mean(BA_0train_to_1test, BA_1test_to_0train)` and
`T_X = median_s X_s`.

The same 1,999 label realizations and grouping rule from Stage W are used. Each replicate refits
both directions and produces `T_X_null`.

- `E_X = T_X_observed - median(T_X_null)`
- one-sided greater-or-equal plus-one p-value with denominator 2,000

PATH Stage X PASS iff PATH Stage W passes, `E_X > 0`, and `p_X <= 0.05`. BAG is reported by the
same statistical calculation but cannot rescue PATH. SCALARS is descriptive only.

## 11. Stage O — chronological order contribution

Stage O is run only when PATH W and PATH X both pass. Exactly 1,999 deterministic order replicates
are used. For every replicate and every trial, one independent nonidentity permutation is drawn
uniformly from the 119 nonidentity S5 permutations in canonical global trial order.

The same five SPD states and all ten pairwise distances are retained. The distance matrix is
reindexed and `PATH_D10` is rebuilt; only chronological state identity is destroyed.
`BAG_CANON_D10` must remain invariant within absolute tolerance 1e-12 and does not vote in this
null. Each replicate reruns the Stage X PATH pipeline.

- `T_O` is the observed Stage X PATH statistic
- `E_O = T_O - median(T_X_order_null)`
- one-sided greater-or-equal plus-one p-value with denominator 2,000

Stage O PASS iff `E_O > 0` and `p_O <= 0.05`. If W or X fails, O is not run and is labeled
`NOT_RUN_PREREQUISITE`, not PASS or FAIL.

## 12. Deterministic RNG and resumability

Label and order plans use exact NumPy `SeedSequence([20260810, stream_tag]).spawn(1999)` child
order. Each stored child seed is
`int(child.generate_state(1, dtype=np.uint64)[0])`; replicate RNG is
`np.random.default_rng(stored_seed)`. Replicates are numbered 1 through 1,999.

Resumable cache checkpoints record protocol/config/code/input hashes, representation/stage,
replicate identity, stored seed, completion mask, nine subject statistics, and group statistic.
Resume is allowed only on exact identity match. Compact tracked final null NPZ files and seed JSON
must replay exactly. No partial replicate grid receives inference.

## 13. Frozen terminal logic

Decision order is exact:

1. Reproduction/data/numerical hard-gate failure:
   `UNASSESSED_TRAJECTORY_REPRODUCTION_FAILURE`.
2. Any required technical grid failure:
   `UNASSESSED_TECHNICAL_FAILURE`.
3. PATH W FAIL:
   `STOP_WITHIN_SUBJECT_TRAJECTORY_CLASS_POOR`.
4. PATH W PASS and PATH X FAIL:
   `STOP_SESSION_SPECIFIC_TRAJECTORY_ONLY`.
5. PATH W/X PASS and O FAIL:
   `GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY`.
6. PATH W/X/O PASS:
   `GO_STABLE_SUBJECT_SPECIFIC_ORDERED_TRAJECTORY_COMPONENT`.

No absolute BA threshold is used. Full precision values determine outcomes; displays are rounded
only afterward.

## 14. Interpretation and independence restrictions

The strongest admissible successful claim is:

> Under the frozen 5-window AIRM representation, class-discriminative local covariance geometry
> is subject-specific and, if Stage X passes, cross-session reproducible.

The audit does not establish physiology, individual motor strategy, causal dynamics, a cause of
WHOLE-Z or domain-adaptation results, target-unlabeled identifiability, or a performance benefit
from personalization. It does not use WHOLE-Z features, thresholds, correlations, subgroups, or
subject selection. Results from other branches are interpretation-only context after this audit
is complete.

## 15. Required outputs

All new outputs are under `outputs/bnci2014_001_trajectory_within_subject_v1/`:

- protocol copies, environment, git/data provenance
- tables: `data_contract.csv`, `reproduction_gate.csv`, `within_session_scores.csv`,
  `cross_session_scores.csv`, `label_null_summary.csv`, `order_null_summary.csv`,
  `representation_comparison.csv`
- compact deterministic null artifacts and seed manifests
- `decisions/terminal_decision.json` and `decisions/decision_chain.csv`
- six specified PNG/PDF/source-CSV figure triplets
- `report/trajectory_within_subject_audit_v1.md`

The report begins by distinguishing v0's cross-subject question from v1's within-subject and
same-subject cross-session questions. It explains each stage in plain language before numbers,
reports all subjects without best-subject emphasis, and explicitly states what is and is not
supported.

## 16. Version control and stopping policy

The protocol freeze commit and scientific result commits are separate. Raw EEG and large
regenerable caches are never tracked. Final status must be clean, the full pytest suite must pass,
the branch must be pushed, and a draft PR must target `pilot/trajectory-anatomy-v0`. No automatic
merge is permitted. A negative result is reported unchanged and is not rescued by new analysis.
