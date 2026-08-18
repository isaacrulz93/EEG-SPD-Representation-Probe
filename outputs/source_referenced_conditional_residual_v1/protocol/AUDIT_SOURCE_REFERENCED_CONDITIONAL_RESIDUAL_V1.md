# Audit: Source-Referenced Conditional Residual V1

Status: **PASS WITH HASH-LOCKED PARENT CACHE REUSE REQUIRED**

Audit date: 2026-08-18 (Asia/Seoul)

This Stage 0 audit is limited to lineage, hashes, schemas, ordering, algebraic availability, leakage boundaries, and resource estimates. It did not calculate `gamma`, a source correction, a source order, a new recovery metric, a confidence interval, or a real-data decision.

## 1. Exact parent and immutable history

- Repository: `isaacrulz93/EEG-SPD-Representation-Probe`.
- New isolated branch: `pilot/source-referenced-conditional-residual-v1`.
- Exact parent head: `8346a3e0f731c80668bd7147a2fe0fd12da6b914`.
- Parent branch/PR: `pilot/unlabeled-conditional-mode-identifiability-v0`, PR #17.
- Stacked base required for the new draft PR: `pilot/unlabeled-conditional-mode-identifiability-v0`.
- Parent PR #17 is open, draft, and its remote head is the exact parent head.
- PR #17 output manifest contains 68 files. All 68 file SHA-256 values were recomputed at audit time and matched.
- The inherited PR #16 objects retain their parent hashes, including OpenBMI V0 objects SHA-256 `f7e2fd7517fe1f55f84ef7729823b2d3f10452833ec2399a4b7014f769c98572` and V1.1 observed core SHA-256 `ecf0657d1ccf51e7aa3392a5e95608d849e9069dda2f27e2a76e321429fac168`.

Every freeze and real-data entry point will snapshot and revalidate all files under the PR #16 V1/V1.1 output directories and all 68 PR #17 manifest entries. No parent output is a write target.

## 2. Subject, session, class, and fold contract

- Subjects are integers 1--54 in ascending array order.
- Sessions are stored as `"0"`, `"1"`, corresponding to source sessions 1 and 2.
- Classes are ordered `[left_hand, right_hand]`, class indices `[0, 1]`.
- Every subject/session has 100 trials, 50 per class.
- Channels are the frozen 20-channel motor montage: `FC5, FC3, FC1, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6`.
- The six exact outer folds are inherited without modification from PR #16/#17; canonical fold hash: `bcbef19ed8d1b5bf385a600e967a929c185fe70582d8ee0557e30f7096afb500`.
- Rank-1 modes have shape `(6, 2, 210)` and SHA-256 `b0d71aeaddd73723d45cb0c009ea2bb036f72e1fe70ff25ac9c3c067eab179b7`.

## 3. Required parent-object schemas

| object | relevant keys and shapes | availability |
|---|---|---|
| `rank1_session_view_modes.npz` | `modes (6,2,210)`, `fold_of_subject (54,)`, `subjects (54,)` | committed, sufficient |
| `unlabeled_projected_trials.npz` | `projected_y (54,2,100)`, opaque `trial_ids (54,2,100)` | committed, sufficient for target full-trial contrasts |
| `evaluation_labels_separate.npz` | `class_index (54,2,100)`, matching opaque IDs | committed, evaluation only |
| `oracle_prototype_coordinates.npz` | `alpha (54,2)`, `beta (54,2)`, `raw_x (54,2,210)` | committed, held-out evaluation only |
| `unsigned_recovery_core.npz` | primary/mixture energy and `delta_trial`, each `(54,2)` | committed, sufficient |
| `minimal_anchor_subsamples.npz` | five positive budgets; `direct`, `orientation`, and historical estimates `(5,54,2,200)` | committed, sufficient |
| OpenBMI V0 interaction object | full `AIRM__session_specific__F__U (54,2,2,20,20)` and class proportions | committed, sufficient for total prototype contrasts and training templates |

All compact-object subject/session IDs must agree exactly before any numerical result is produced.

## 4. Source trial-contrast availability

The target subject's full-trial `delta_trial` is committed under the held-out subject's own outer-fold mode. The source correction, however, requires every outer-training subject to be projected under **each candidate outer fold's** frozen mode. The committed scalar target projections do not contain those counterfactual fold projections, so they are insufficient by themselves.

An existing PR #17 cache is available read-only:

- combined covariance cache: shape `(10800,20,20)`, SHA-256 `8fc43ff837c9ab1651f639c30a2aa4a09e760a6a2192b4c5407848fd9da04b28`, exactly equal to the committed PR #17 trial-covariance manifest;
- tangent feature cache: `features (54,2,100,210)`, `marginal_means (54,2,20,20)`, `trial_ids (54,2,100)`.

The tangent file is not itself a committed scientific artifact. It is therefore usable only after all of the following gates pass:

1. its trial IDs exactly equal the committed projected-trial and evaluation-label IDs;
2. its marginal means reproduce the committed marginal-mean object within the frozen tolerance;
3. projection by each subject's frozen held-out-fold mode reproduces every committed `projected_y` value within the frozen tolerance;
4. all entries are finite and shapes/dtypes/order are exact;
5. the combined covariance cache hash still matches the committed manifest.

Failure of any gate is `UNASSESSED_SOURCE_REFERENCE_OBJECT_INSUFFICIENT`; it is not permission to use `analysis_core.npz`, alter preprocessing, or rebuild with different settings. No raw EEG rebuild is currently required.

## 5. Reconstructible training-only quantities

For fold `f`, session `q`, mode `b[f,q]`, and the exact outer-training set `T`:

1. `d_proto[s,q]` is the projection of the total subject-marginal-recentered `U` class contrast.
2. `gamma[f,q]` is the projection of the mean training-subject `U` class contrast and therefore uses no held-out subject.
3. Each training subject's `delta_trial[r,q;f]` is reconstructed from its labeled trials only after the tangent coordinates have been frozen, projected using `b[f,q]`.
4. `correction[f,q]` is the arithmetic mean over `r in T` of `d_proto[r,q] - delta_trial[r,q;f]`.
5. Median correction and source-only affine calibration are frozen non-primary sensitivities and use only `T`.

No held-out target statistic, target label, target energy, or result can enter the mode, `gamma`, correction, affine calibration, threshold, or estimator selection.

## 6. Algebraic identities to be gated after freeze

For balanced binary classes, the fold-safe population template cancels identically from the class-independent residual, yielding

`0.5 * (Z_R - Z_L) = 0.5 * (U_R - U_L) - 0.5 * (mu_T,R - mu_T,L)`.

After Frobenius-isometric `svec` and projection on `b[f,q]`, this requires

`beta[s,q] = d_proto[s,q] - gamma[f,q]`.

The identity will be checked against the committed PR #17 `beta` with a strict absolute/relative floating-point tolerance. It is a data contract, not a statistical hypothesis. Failure is `UNASSESSED_BETA_REFERENCE_IDENTITY_FAILURE`.

## 7. Leakage boundaries

- PR #17 labels remain in the separate evaluation object.
- Target labels may form `delta_trial_full`, oracle `d_proto`/`beta`, and calibration-subset contrasts only after mode, source reference, correction, estimator declarations, nulls, and budgets are frozen.
- All source quantities are recomputed independently per outer fold using the exact outer-training subjects.
- A held-out subject is never included in `gamma`, correction, affine calibration, source order, or leave-one-training-subject source-order checks.
- The zero-label estimator is explicitly conditional on the frozen source-ordering assumption; this does not alter the class-permutation non-identifiability theorem.
- `sign(delta)` and `sign(beta)` are stored and scored separately.

## 8. Resource estimate

- Existing read-only PR #17 cache: approximately 1.2 GiB, dominated by retained canonical source downloads; no copy is required.
- Arrays actually read for this experiment: approximately 17 MiB tangent features plus compact parent objects.
- New compact outputs: expected below 25 MiB.
- Peak resident memory: expected below 2 GiB with one process and one BLAS thread.
- Source-reference reconstruction: below 5 minutes after cache validation.
- 1,999 paired sign-flip nulls, 10,000 subject bootstraps, 200 frozen subsamples across five positive budgets, plots, and report: anticipated 5--20 minutes.
- Free space at audit: approximately 56 GiB.

## 9. Stage 0 decision

The exact identity, training-only source reference, and source-only correction are leakage-safely reconstructible. New raw preprocessing is unnecessary. Real access remains prohibited until the protocol, configuration, formulas, decision gates, implementation, tests, synthetic gates, and complete PR #17 hash snapshot are committed in a clean protocol-freeze commit.
