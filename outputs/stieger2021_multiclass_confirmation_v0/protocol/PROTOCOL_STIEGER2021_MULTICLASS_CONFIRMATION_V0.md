# Stieger2021 Prospective Multiclass Source-Reference Confirmation V0

Status: scientific protocol awaiting immutable freeze. No Stieger EEG sample may be opened before the commit containing this document, its config, implementation, tests, and synthetic-gate report.

## Question and scope

This is the first prospective external Stieger2021 EEG-statistic access in this lineage. It asks whether the montage-registered subject-by-class interaction established retrospectively in OpenBMI generalizes to independent Stieger subjects, repeated post-intervention sessions, four semantic classes in a common task, and the full class-permutation ambiguity. The analyses are structural. They do not train or assess a classifier, domain-adaptation system, TTA method, pseudo-labeler, neural network, loss, ASD model, or intervention effect.

The primary dataset contract is subjects 1–62, sessions 2 and 3, and `TrialData.tasknumber == 3`. The literal class order is `right_hand`, `left_hand`, `both_hand`, `rest`, corresponding to target numbers 1, 2, 3, 4. Task 1 and task 2 are optional binary diagnostics after the primary terminal and never vote. Session 1 and later sessions do not vote.

## Frozen lineage and prospective boundary

The exact parent is PR #18 head `19a3ad1cdc6b57c89526e618779cd24b7db8c99c`. PR #16 established held-out cross-session rank-one OpenBMI population structure, PR #17 established class-permutation non-identifiability and unsigned projected energy recovery, and PR #18 established binary source-reference residualization. Those outputs are immutable and cannot be rescued, changed, or reinterpreted by Stieger outcomes.

Only metadata, file manifests, published schema, and loader source were inspected before this freeze. No Stieger EEG sample, covariance, prototype, interaction, SVD, permutation match, or recovery statistic was accessed. The source/metadata audit records this boundary.

## Source files and streaming

The official Figshare article is 13123148 version 1, DOI `10.6084/m9.figshare.13123148.v1`. The source-file selector parses names matching `S<1..62>_Session_<2|3>.mat` and requires exactly one file per pair. It records Figshare ID, URL, reported bytes, reported MD5, local SHA-256, parse identity, and compact-object SHA-256.

Each file is streamed to a unique temporary path, hashed during transfer, and checked against the official MD5. It is parsed directly with SciPy so `TrialData.tasknumber` is preserved. A raw file is removed only after source hashing, compact serialization, compact reread validation, and identity/metadata recording all pass. Partial or failed files are retained for diagnosis. Raw MAT files and regenerable continuous EEG are never committed.

The parser preserves subject, session, task, run, trial, target, trial length, artifact, result, forced result, target-hit number, performance, channel labels, noise channels, recorded-position metadata, electrode coordinates when present, MBSR subject status, raw sample rate, and exact trial time vectors. Outcome fields are sealed and forbidden from inclusion, feature construction, template estimation, rank selection, matching, or estimator selection.

## Sensors, preprocessing, and eligibility

The immutable sensor order is:

`FC5 FC3 FC1 FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6`.

Noise channels are read as one-based session indices. Marked channels are interpolated deterministically on the complete recorded montage before primary-channel selection. Recorded electrode coordinates are preferred; otherwise the frozen MNE `standard_1005` montage is used. An interpolation error is a data-contract failure. A session with more than four marked primary channels is ineligible.

The complete available trial supplies filtering context. EEG is converted from microvolts to volts, filtered by a deterministic zero-phase fifth-order Butterworth 8–30 Hz bandpass, resampled to 100 Hz, and cropped with left-closed/right-open masks. The primary epoch is `[0.5, 2.0)` seconds; the identically processed pre-target control is `[-1.0, 0.0)`; feedback sensitivity is `[2.0, 3.0)` only when fully sampled. Covariance is float64 OAS, symmetrized exactly, with no baseline correction or extra regularization, and must be finite positive definite. Only task and `artifact == 0` determine primary trial inclusion; outcome, duration, and performance fields cannot do so.

A subject is eligible only when both primary sessions have validated source files, task 3, all four classes, reconstructable sensors, successful interpolation, at least 25 artifact-free primary-window trials per class, valid covariances, and complete class/trial/acquisition metadata. A one-session failure excludes both sessions. Fewer than 40 eligible subjects yields `UNASSESSED_INSUFFICIENT_MULTICLASS_COHORT` before scientific statistics.

For each eligible subject/session/class, FULL uses all eligible trials and A/B uses deterministic alternating acquisition order within class. Observed class proportions are retained; no class is post-hoc subsampled.

## Geometric objects

For class trial covariances, the AIRM Fréchet marginal mean is (M_{s,q}), the class mean is (M_{s,q,c}), and

\[
U_{s,q,c}=\log(M_{s,q}^{-1/2}M_{s,q,c}M_{s,q}^{-1/2}).
\]

For outer-training set (T), training subjects use leave-one-training-subject class templates; held-out subjects use the full training template. After population-class subtraction, the class-proportion-weighted residual is removed:

\[
Z_{s,q,c}=R_{s,q,c}-\sum_k\pi_{s,q,k}R_{s,q,k}.
\]

The literal orthonormal Helmert matrix is the three rows stored in the config. Class matrices are Frobenius-isometric `svec` vectors with square-root-two off-diagonal scaling. `vec(H4 V_Z)` has 630 coordinates and is globally unit normalized for the primary analysis. Coordinate-wise scaling is forbidden.

## Cohort lock and folds

After all source sessions are compacted—but before any population SVD or result—the metadata-only eligibility table, included subjects, exact folds, official/local hashes, compact hashes, and exclusion reasons are committed. The status is `DATA_LOCKED_NO_SCIENTIFIC_RESULT_YET`.

Outer CV has six deterministic paired-subject folds. A namespaced SHA-256 sort is performed within MBSR/control strata when metadata are complete, followed by round-robin assignment and outcome-independent size balancing. Inner CV uses the same deterministic method with five folds inside each outer-training set. Literal memberships are locked at checkpoint two.

## Reliability and population structure

Session 2 and session 3 independently require positive split-half same-subject-versus-other-subject separation, a 1,999-replicate subject-pairing null at (p\le .05), and positive leave-one-subject influence. Failure is `UNASSESSED_STIEGER_MEASUREMENT_RELIABILITY_FAILURE`.

Training-only centered session matrices define (C_{01}=X_0^T X_1/(n-1)). Its two-view SVD is evaluated with ranks `[1,2,3,5,8,13]`, constrained by sample-identifiable rank. Inner paired-subject CV uses the smallest rank within one standard error of the best mean separation. Projected scores use training-only scale. The held-out statistic, both directions, 10,000-subject bootstrap CI, all-rank curve, ranks, full-space baseline, and leave-one-subject influence are stored.

Every voting null reruns the full rank-selection pipeline 1,999 times: fixed-point-free session-3 subject pairing destruction; independent subject/session S4 class permutations sampled uniformly from all 24; and equal-rank Haar random paired subspaces. The identically processed pre-target epoch is analyzed with the same class-semantic gate. If it passes that gate, the primary terminal is `UNASSESSED_PRETARGET_CLASS_OR_SEQUENCE_CONFOUND`.

The structure terminal is `STIEGER_MULTICLASS_STRUCTURE_CONFIRMED_LOW_RANK` only when both reliability gates pass; statistic and directions are positive; all three primary p-values are at most .05; influence sign is stable; full-space signal is stable; median rank is at most 3; at least four of six folds choose rank at most 3; and pre-target does not reproduce the primary semantic gate. Structural success without rank gates is `STIEGER_MULTICLASS_STRUCTURE_CONFIRMED_NOT_LOW_RANK`; reliable failure of structural gates is `STOP_STIEGER_NO_HELDOUT_MULTICLASS_STRUCTURE`.

## Trial projection convention

A selected multiclass population vector has shape 3 by 210 after inverse vectorization. To define a label-free trial coordinate, each selected vector is reduced to its leading right singular sensor direction. The largest-absolute `svec` coordinate is forced positive; multiple rank directions are deterministically modified-Gram-Schmidt orthogonalized in frozen rank order. This is a training-only operational bridge, not a unique decomposition and not source anatomy. It is fixed before Stieger EEG access.

## Source reference and semantic permutation

Total class prototype coordinates (D_{s,q,c}) are projections of (U_{s,q,c}) into that trial-compatible selected space. The training template is (Gamma_{f,q,c}=|T|^{-1}\sum_{r\in T}D_{r,q,c}). The residual coordinate is (D-\Gamma), followed by the same observed-proportion weighted centering used for Z. It must equal projected held-out Z to frozen tolerance, otherwise `UNASSESSED_MULTICLASS_REFERENCE_IDENTITY_FAILURE`.

Source-only mean additive correction is the training mean difference between prototype and mean-trial class coordinates. It is frozen before held-out labels are joined. Source-only affine fitting is non-voting.

For semantic matching, the centered, unit-Frobenius source K-by-r template is compared to an evaluation-only held-out class-centroid matrix after semantic row labels are removed. All 24 S4 permutations are enumerated and squared Frobenius costs saved. No target rotation, reflection, rank selection, or tuning is allowed. The true literal identity mapping succeeds only when it is the unique minimum; a tie within `1e-12` fails. The prospective preservation decision additionally requires the structure terminal, above-chance success in both sessions, pooled bootstrap lower bound above 1/24, exact/permutation p at most .05, leave-one-training-subject source-template stability, and no single-subject control.

## Unlabeled scatter, mixture, and calibration

For label-stripped pooled target trial coordinates, (S_{total}) is sample covariance. The outer-training source within-class scatter (W_{source}) uses source labels only. The primary estimate is the PSD projection of (S_{total}-W_{source}). The evaluation-only oracle is the class-proportion-weighted between-class centroid scatter. Primary score is median Frobenius cosine. Normalized error, trace/eigenvalue errors, rank, session metrics, bootstrap CI, target permutation, matched random direction, and influence are reported. Support requires the structure gate, positive session statistics, positive pooled CI, both p-values at most .05, and influence stability.

Only if semantic permutation and scatter recovery pass is a four-component tied-covariance Gaussian mixture fit. It has fixed K=4, equal-weight initialization, 20 deterministic source-independent multistarts, no component deletion, and no post-result selection. Components remain unordered until all 24 source-template assignments are scored. Target labels join only after component and assignment artifacts are saved.

Only a numerically valid mixture proceeds to budgets `[0,4,8,16,32]`; positive budgets have exactly equal labels per class and 200 deterministic draws. Methods are zero-label source-template assignment, mixture/scatter plus m-label assignment, fair source-referenced direct m-label centroids, source-template-only, and Beta=0. The voting metric is expected per-draw multiclass residual-coordinate MAE. `MULTICLASS_MINIMAL_ANCHOR_EFFICIENT` requires some m in {4,8} to beat the fair direct baseline with paired p at most .05, improvement CI excluding zero, both sessions positive, influence stability, and above-chance assignment. Repeated-draw associations are explicitly labeled `MEAN_OVER_REPEATED_CALIBRATION_DRAWS`.

## Failure and reporting boundaries

All hash, ordering, leakage, finite, SPD, fold, decomposition, and voting-pipeline failures fail closed. No numerical jitter, rank change, channel change, subject replacement, trial threshold change, or preprocessing revision is allowed after freeze. A failure is committed and pushed with a decision JSON and concise report.

Positive results establish only the declared montage-registered structural relationships under the source-template assumptions. They do not establish a full conditional distribution, physiology, anatomy, causality, universal coordinates, classifier improvement, domain adaptation, TTA, pseudo-label validity, ASD biomarkers, or intervention effects.

The next scientific question after a fully positive result is: **Does the frozen multiclass source-reference structure reproduce in a second prospectively locked repeated-session cohort, and can its semantic-template assumption be externally calibrated without target outcome information?**
