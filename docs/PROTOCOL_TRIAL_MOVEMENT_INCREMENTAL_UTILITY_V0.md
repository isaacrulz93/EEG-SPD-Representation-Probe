# Trial-Level Ordered SPD Movement Incremental Utility Audit V0 — Frozen Protocol

Freeze date: `2026-08-17` (Asia/Seoul)
Base commit: `969f80fc29b31993df69980c23f52240ce591ff1`
Branch: `pilot/trial-movement-incremental-utility-v0`
Seed: `20260817`

This is a falsification audit, not a network or domain-adaptation proposal. It asks whether an ordered movement feature computed independently from each trial's five local SPD covariances improves motor-imagery class decoding beyond marginal-centered whole-trial covariance for an unseen subject. No result-dependent change to features, splits, decoder, statistics, terminal rule, or schema is allowed after the protocol-freeze commit.

## Frozen data and identities

The only dataset is BNCI2014_001 session `0train`, subjects 1–9, class order `left_hand`, `right_hand`, `feet`, `tongue`, and the frozen 22-channel order. Epochs are 0.000–3.996 s, 250 Hz, 1000 samples, 8–32 Hz. OAS covariance uses the existing frozen implementation. WHOLE has one 22x22 covariance per trial. WINDOW5 has five consecutive non-overlapping 200-sample covariances per trial.

Required counts are 2592 trials, 2592 WHOLE matrices, 12960 WINDOW5 matrices, 288 trials per subject, 72 trials per subject/class, and windows exactly 1–5 for every trial. The exact V1 config and cache hashes are frozen in the YAML. If the cache is absent it may only be regenerated using the exact V1 scripts and config; otherwise the audit stops as `UNASSESSED_MISSING_FROZEN_INPUT`. A different dataset must never be silently substituted.

WHOLE and WINDOW5 identity must agree exactly on subject, session, run, trial ID, trial UID, and class label. Every trial and all five windows stay together. Labels are retained only as outcome metadata; no class mean, subject-by-class mean, target class label, or label-bearing table is accepted by any feature or centering API.

## Frozen LOSO and centering

The audit calls the V2 identity functions in `src/alignment_v2.py`. For target subject s, sources are the other eight subjects. Each source subject's AIRM Fréchet mean is fit label-free from all 288 of that subject's WHOLE covariances. Static coordinates are `svec(log(M_s^-1/2 C M_s^-1/2))`, dimension 253, using `src.geometry_v2.fit_center`, `FittedCenter.transform`, and the existing log-svec implementation.

Primary T2 uses the V2 complementary directions exactly: A fits the target center on runs 0–2 and evaluates runs 3–5; B fits on runs 3–5 and evaluates runs 0–2. Each direction has 144 calibration and 144 evaluation trials, disjoint trial UIDs, and 36 evaluation trials per class. Direction metrics are averaged within target subject before inference. Secondary T1 fits the target center from all 288 unlabeled target covariates and evaluates those same 288 covariates. T1 cannot alter or rescue the T2 terminal.

## Trial movement features

For each trial, WINDOW5 is reconstructed as `(C1,...,C5)` in exact metadata order. The unmodified `src.local_mean_movement_v0.anti_develop_sequence` is called with `Delta_t=0.8`. Thus each adjacent exact AIRM log displacement is divided by 0.8, transported through its actual ordered reverse prefix to C1, whitened at C1, and stored as `(Z1,...,Z4)`. There is no permutation, DTW, target fitting, class conditioning, or averaging across trials.

All 2592 sequences and 10368 transitions must pass: finite and symmetric Z, successful SPD/log/transport operations, and `||Zi||F = d_AIRM(Ci,Ci+1)/0.8` within 1e-8. Failure gives `UNASSESSED_MOVEMENT_GEOMETRY_FAILURE`.

`MOV_LEN = [||Z1||F,...,||Z4||F]` (4 dimensions).

`MOV_GRAM` is Frobenius-isometric svec of the symmetric 4x4 matrix `Gij=<Zi,Zj>F` (10 dimensions). Its diagonal must equal squared MOV_LEN, its minimum eigenvalue must be at least -1e-8, and one fixed synthetic O(22) action and one fixed well-conditioned GL(22) congruence applied to every original trial sequence must preserve G within relative 1e-8. MOV_GRAM is a low-dimensional common-O invariant probe, not the complete frozen quotient cost or a complete simultaneous-conjugacy invariant.

`MOV_SENSOR` concatenates `svec(Z1),...,svec(Z4)` in temporal order (1012 dimensions). It is montage-registered and gauge/sensor-coordinate sensitive, and is only a secondary localization feature.

Exactly seven conditions are used: STATIC 253, MOV_LEN 4, MOV_GRAM 10, MOV_SENSOR 1012, STATIC_PLUS_LEN 257, STATIC_PLUS_GRAM 263, and STATIC_PLUS_SENSOR 1265. Combined arrays are concatenated before scaling. PCA, spectra, commutators, extra/overlapping windows, learned fusion, and result-selected summaries are forbidden.

## Frozen decoder and metrics

Every target/condition fits one source-only `StandardScaler`, then `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=20000, tol=1e-6, random_state=20260817, class_weight=None)`. The same source-fitted scaler/model is reused for T1 and both T2 directions. Target calibration and evaluation rows never fit the scaler or classifier. There is no tuning or PCA.

Any convergence warning is saved with `n_iter_` and invalidates that target/condition. Any invalid STATIC or primary-comparison cell makes the primary terminal `UNASSESSED_CONVERGENCE_FAILURE`; parameters are not changed.

Primary metric is balanced accuracy. Accuracy, macro-F1, fixed-order per-class recall, 4x4 confusion matrix, predicted class, and fixed-order probabilities are also saved. All conditions use identical target evaluation UIDs. T2 A/B predictions are pooled only after the prescribed direction-specific target transforms; because halves and class counts are equal, pooled BA is required to equal the arithmetic mean of direction BAs within numerical tolerance.

## Frozen subject-level inference

For each subject, the primary paired deltas are STATIC_PLUS_LEN minus STATIC, STATIC_PLUS_GRAM minus STATIC, and STATIC_PLUS_SENSOR minus STATIC. Inference is only on the nine subject deltas. The exact one-sided sign-flip p-value enumerates all 512 sign patterns and counts statistics at least as large as the observed mean. Holm correction controls the three-primary-comparison family.

The descriptive 95% subject bootstrap interval is exact: all 9^9 ordered with-replacement resamples are represented by the 24,310 count compositions and their exact multinomial weights. Influence reports all nine leave-one-subject-out means. Top contributor share is `max(abs(delta))/sum(abs(delta))`, or zero if every delta is zero.

Support requires mean delta > 0, median delta > 0, and Holm p <= 0.05. Breadth additionally requires at least 6/9 positive subjects and every leave-one-subject-out mean > 0. Static mean BA must be at least 0.30; failure before movement interpretation is `UNASSESSED_STATIC_SANITY_FAILURE`.

Secondary paired comparisons are STATIC_PLUS_GRAM minus STATIC_PLUS_LEN and STATIC_PLUS_SENSOR minus STATIC_PLUS_GRAM with the same subject table and exact raw sign-flip p-value, no Holm vote, and no terminal vote. Movement-only conditions are descriptive relative to chance 0.25.

## Frozen terminal order

After all hard gates pass:

1. supported broad DELTA_GRAM: `GO_INVARIANT_MOVEMENT_INCREMENTAL_UTILITY`;
2. supported non-broad DELTA_GRAM: `GO_HETEROGENEOUS_INVARIANT_MOVEMENT_UTILITY`;
3. unsupported DELTA_GRAM but supported DELTA_SENSOR: `GO_SENSOR_ONLY_MOVEMENT_INCREMENTAL_UTILITY`;
4. unsupported DELTA_GRAM and DELTA_SENSOR but supported DELTA_LEN: `GO_SPEED_ONLY_INCREMENTAL_UTILITY`;
5. none supported: `STOP_NO_TRIAL_MOVEMENT_INCREMENTAL_UTILITY`.

Data, identity, geometry, leakage, convergence, output-contract, or static-sanity failure produces a specific `UNASSESSED_...` label and no positive claim.

## Reproduction and output contract

The prior V2 AIRM T2 aggregate is compared subject by subject. Exact equality is not expected because this audit prespecifies a source-only StandardScaler and tighter logistic tolerance/max iterations, while V2 had no scaler and used its own frozen decoder. The comparison is a split/centering-direction reproduction and exact numerical differences are saved; it must not be described as exact decoder reproduction.

All scientific artifacts live only in `outputs/bnci2014_001_trial_movement_utility_v0/` with the required `protocol`, `features`, `predictions`, `tables`, `figures`, `report`, and `decisions` subdirectories. Prior configs, protocols, outputs, and frozen inputs are hashed before and after execution. Required tables, predictions, feature archive, PNG/PDF/source-CSV figures, report, and terminal JSON are schema-checked before finalization.

## Claim limits

A positive result may say that trial-level ordered SPD movement contains class-discriminative information complementary to marginal-centered whole covariance under this frozen LOSO protocol. A Gram result concerns magnitudes and relative tangent inner products. A sensor-only result concerns montage-registered full movement coordinates. The audit cannot establish a neural mechanism, causality, continuous-time dynamics, identifiable subject action, a complete quotient geometry, a new adaptation method, or guaranteed future SPDNet benefit. A negative fixed-linear result does not prove all nonlinear networks fail.
