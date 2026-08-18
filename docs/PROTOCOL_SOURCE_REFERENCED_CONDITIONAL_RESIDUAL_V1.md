# Protocol: Source-Referenced Conditional Residual V1

## Scope and frozen parent

This retrospective OpenBMI-only mechanistic follow-up is stacked on PR #17 at exact head `8346a3e0f731c80668bd7147a2fe0fd12da6b914`. It does not recompute or reinterpret PR #16's rank-1 population-structure terminal or PR #17's unsigned-energy and minimal-anchor terminals. It tests whether PR #17's apparent anchor mismatch arises because trial contrast is a total semantic coordinate while `beta` is residualized against the outer-training population class template.

No classifier, domain adaptation, TTA, pseudo-labeling, BNCI, Stieger2021, HGD, or ASD analysis is permitted. This is not prospective confirmation.

## Immutable coordinates

For each exact PR #16/#17 outer fold `f`, session `q`, and frozen unit mode `b[f,q]`:

- `d_proto = b^T svec(0.5 (U_R-U_L))`;
- `gamma = b^T svec(0.5 (mean_T U_R-mean_T U_L))`;
- `beta = d_proto-gamma`;
- `delta_trial = 0.5(mean(y|R)-mean(y|L))`.

The class order is literally `[left_hand, right_hand]`. `svec` is the frozen Frobenius-isometric upper-triangle vectorization. The mode, folds, trial coordinates, labels, energy estimates, and calibration subsets are inherited without refitting or resampling.

The identity `beta=d_proto-gamma` is checked at absolute and relative tolerance `3e-12` for every held-out subject/session. Failure is `UNASSESSED_BETA_REFERENCE_IDENTITY_FAILURE`.

## Parent cache gate

The PR #17 committed scalar target projections cannot reconstruct training-source contrasts under every outer-fold mode. The existing PR #17 tangent cache may be read only if: the combined covariance cache matches committed SHA-256 `8fc43f...04b28`; trial IDs match both committed projected trials and labels; marginal means match the committed object; and each subject's held-out-fold projection reproduces the committed `projected_y` within `2e-12`. Otherwise the terminal is `UNASSESSED_SOURCE_REFERENCE_OBJECT_INSUFFICIENT`. No raw rebuild or alternative preprocessing is authorized.

## Source reference and correction

The primary correction is `SOURCE_MEAN_ADDITIVE_CORRECTION`:

`correction[f,q] = mean_{r in T}(d_proto[r,q]-delta_trial[r,q;f])`.

The primary held-out prediction is `delta_trial_full + correction - gamma`. A source median correction is a secondary sensitivity. A least-squares slope/intercept mapping from source `delta_trial` to source `d_proto`, fit only on `T`, is a non-voting affine control. Nothing is selected using held-out targets.

The primary correction decision passes only if direct trial-delta MAE minus corrected MAE is positive, its 10,000 subject-bootstrap 95% CI excludes zero, a 1,999-replicate paired sign-flip p-value is at most .05, both session improvements are positive, the subject-bootstrap lower bound for beta-sign accuracy exceeds .5, and all leave-one-held-out-subject improvements remain positive.

## Explicit semantic-ordering assumption

The class-permutation non-identifiability theorem remains valid. An additional, falsifiable assumption is evaluated retrospectively:

`source_order[f,q] = sign(mean_{r in T} delta_trial[r,q;f])`.

Target ordering is correct when `sign(delta_trial_target)==source_order`. The convention is frozen without target labels. Ordering is supported only when accuracy exceeds .5 in both sessions, the pooled subject-bootstrap CI lower bound exceeds .5, the exact one-sided binomial p-value is at most .05, and leave-one-training-subject source-order refits never change the sign.

## Corrected zero/few-label estimators

The PR #17 primary unsigned energy `E_hat` remains fixed. Its primary magnitude is `sqrt(max(E_hat,0))`; the symmetric-mixture magnitude is a secondary sensitivity.

- Zero labels, under the explicit assumption only: `beta_hat_order = source_order*sqrt(E_hat)+correction-gamma`.
- For `m>0`: `beta_hat_UL_m = sign(delta_calibration_m)*sqrt(E_hat)+correction-gamma`.
- Fair direct baseline: `beta_hat_direct_m = delta_calibration_m+correction-gamma`.

Budgets are `[0,2,4,8,16,32]`, with exactly 200 inherited PR #17 deterministic subsamples per positive budget. The historical uncentered PR #17 estimator is descriptive only.

Zero-label recovery under the assumption is supported only if correction and ordering pass and, versus `beta=0`, MAE improvement has p<=.05, bootstrap CI above zero, positive improvement in both sessions, beta-sign-accuracy CI lower bound above .5, and positive leave-one-subject influence.

Minimal-anchor efficiency passes only if at some frozen budget `m<=8`, proposed MAE is lower than the fair corrected direct baseline, paired p<=.05, improvement CI excludes zero, both sessions improve, beta-sign-accuracy CI lower bound exceeds .5, and leave-one-subject improvement stays positive. The smallest passing budget is reported.

## Metrics, nulls, and oracle ceilings

Every method/budget reports beta MAE, normalized MAE, Pearson, Spearman, signed R2, beta-sign accuracy, semantic-sign accuracy relative to full target `delta`, projected prototype reconstruction error, session values, subject-bootstrap CI, paired sign-flip p-value, and leave-one-subject influence. `sign(delta)` and `sign(beta)` are never conflated.

All Monte Carlo nulls use 1,999 deterministic replicates; all subject bootstraps use 10,000 deterministic replicates. Source-reference comparisons average 200 subsample errors within subject/session before inference.

Non-voting factorial ceilings are: oracle sign plus estimated magnitude; estimated sign plus oracle absolute delta; oracle delta plus source reference; estimated delta plus oracle gamma; and full trial delta plus source reference. They separate magnitude, orientation, reference, calibration-sampling, and trial/prototype curvature errors.

## Decisions and interpretation

Four decisions are stored separately: source-reference correction, retrospective source ordering, zero-label recovery under the source-order assumption, and source-referenced minimal-anchor efficiency. A negative decision does not negate the PR #16 interaction or PR #17 unsigned-energy result.

The only allowed positive interpretation is: in retrospective OpenBMI analysis, the source population class template provides the missing reference needed to convert recoverable target projected class energy into a subject-specific residual coordinate under an explicit source-ordering assumption or minimal semantic calibration.

This does not establish a full conditional distribution, physiology, source anatomy, causality, universal coordinates, downstream classification benefit, pseudo-label validity, TTA recoverability, multiclass validity, or an ASD biomarker.

If the source-reference mechanism is supported, the exact next scientific question is: Can source-referenced conditional residualization and semantic ordering prospectively replicate across repeated sessions and multiple classes in Stieger2021 under a pre-frozen multiclass permutation protocol?

## Freeze and execution order

Before any new real statistic, commit this protocol, configuration, audit, identity note, exact formulas, implementations, tests, synthetic gates, exact folds/budgets/nulls, and a complete PR #16/#17 artifact-hash snapshot under commit subject `freeze source referenced conditional residual v1`.

After that clean commit: validate parent/cache contracts; run beta identity; run source correction; run source ordering; run corrected zero/few-label and oracle decomposition; generate report/figures; rerun focused and full tests; verify parent hashes and clean tree; then create a separate scientific-result commit and stacked draft PR.
