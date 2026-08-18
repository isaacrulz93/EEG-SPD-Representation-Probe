# Protocol: Unlabeled Conditional-Mode Identifiability and Minimal Semantic Anchoring V0

Status: **PRE-RESULT FROZEN PROTOCOL**

Parent: PR #16, `pilot/subject-class-population-structure-v1`, commit `9dee7642ac573f37756b8427a75864a50c32044e`.

## 1. Question and boundary

This experiment asks which aspects of an unseen OpenBMI subject's displacement along the frozen cross-session rank-1 subject×class interaction mode can be recovered from pooled unlabeled target trials, and how much semantic target supervision is needed to orient that displacement.

It does not train or evaluate a classifier, adaptation model, neural network, TTA procedure, loss, or pseudo-labeling method. It does not optimize downstream accuracy. It does not revisit the PR #16 terminal.

The scientific target is limited to montage-registered, mean-level conditional displacement. No full conditional distribution, dispersion identity beyond the declared projected second moment, physiology, causal mechanism, source anatomy, universal individuality axis, clinical inference, or unlabeled signed semantic coordinate is claimed.

OpenBMI / Lee2019-MI is the only executed dataset. BNCI, Stieger2021, HGD, ASD datasets, and all downstream method development are excluded.

## 2. Identification theorem

The pooled binary target distribution is

\[
P_s^U=\pi_LP_{s,L}+\pi_RP_{s,R}.
\]

Exchanging the semantic component names leaves this unlabeled mixture invariant but negates the signed class contrast and its signed mode coordinates. Thus a zero-label signed coordinate is mathematically non-identifiable without a semantic anchor or an independent symmetry-breaking assumption. The frozen zero-label signed decision is `NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION`; no signed zero-label predictor is trained.

Potentially identifiable objects are `|beta|`, `beta^2`, an unordered separation, and projected between-class energy. The detailed proof is in `docs/SIGNED_COORDINATE_NONIDENTIFIABILITY.md`.

## 3. Immutable inputs

The complete V1/V1.1 output directories are hash-snapshotted before new real access and revalidated by every runner. The exact parent mode archive, V0 OpenBMI interaction objects, source manifest, protocol manifest, V1 config, six outer folds, all inner folds, svec geometry, normalization, and rank-1 construction are immutable.

No PR #16 output is written, regenerated, or copied over. All new tracked artifacts live under `outputs/unlabeled_conditional_mode_identifiability_v0/`; the regenerable trial covariance cache is untracked under `cache/unlabeled_conditional_mode_identifiability_v0/`.

## 4. Trial object gate

No tracked frozen trial-covariance object is available. The exact 108-file parent source manifest and exact V0 preprocessing implementation are available, so the trial covariances will be rebuilt without changing any filtering, epoch, channel, covariance, regularization, reference, or trial-selection parameter.

Every raw SHA-256, covariance-array SHA-256, and metadata SHA-256 must equal its parent source-manifest record. The full AIRM marginal means, class means, and `U` matrices must reproduce the immutable parent objects within relative Frobenius tolerance `2e-12`. Failure gives `UNASSESSED_TRIAL_LEVEL_OBJECT_INSUFFICIENT` before recovery access.

The rebuild is the frozen V0 chain: offline run, 20 ordered channels, continuous 8--30 Hz fifth-order Butterworth zero-phase filtering, resample to 100 Hz, epoch `[1.0,3.5)` seconds, and float64 OAS covariance.

## 5. Frozen rank-1 modes

For outer fold `f`, reconstruct training-only normalized interaction features and the two-view cross-covariance SVD. The mode directions are

\[
b_{f,0}=L_f[:,1],\qquad b_{f,1}=R_f[:,1].
\]

They must agree sign-invariantly with the first columns saved by V1.1. All parent folds selected rank 1, and this branch does not reselect rank.

Mode stability is descriptive. It reports fold-pair and session-view absolute cosine, leave-one-training-subject refit absolute cosine/principal angle, a 1,999-replicate Haar baseline, and subject influence. Because sign is arbitrary, inferential summaries use absolute cosine only. For visualization alone, each fold's paired views are multiplied by one common sign chosen so session-0 has nonnegative dot product with fold-0 session-0; exact zero uses `+1`.

Inverse-svec matrices, diagonal/off-diagonal energy, channel node strength, and absolute edges are montage-coordinate descriptions, not source anatomy.

## 6. Operational target coordinates

From outer-fold-safe held-out `Z`, define

\[
X^{\mathrm{raw}}_{s,q}=\operatorname{svec}\left((Z_{s,q,R}-Z_{s,q,L})/2\right),
\quad x_{s,q}=X^{\mathrm{raw}}_{s,q}/\|X^{\mathrm{raw}}_{s,q}\|_2,
\]

\[
\alpha_{s,q}=b_{f,q}^{\top}x_{s,q},\qquad
\beta_{s,q}=b_{f,q}^{\top}X^{\mathrm{raw}}_{s,q}.
\]

Labels are used for these oracle coordinates only after every unlabeled projection and estimate is frozen.

## 7. Pooled unlabeled projection

For each subject/session, all 100 target covariances are pooled. Labels, events, run/block fields, acquisition order, and semantic order are excluded, then rows are deterministically shuffled. The AIRM marginal mean `M` is fit to these pooled covariances. Each trial becomes

\[
Y_i=\operatorname{svec}\{\log(M^{-1/2}C_iM^{-1/2})\},\qquad y_i=b_{f,q}^{\top}Y_i.
\]

No target label can enter `M`, `Y`, `y`, a zero-label estimator, a hyperparameter, or a stopping rule. Evaluation labels are a separate array joined only by opaque row ID after all unlabeled outputs are saved.

## 8. Trial/prototype bridge

After projection is frozen,

\[
\Delta^{\mathrm{trial}}=(E[y\mid R]-E[y\mid L])/2
\]

is compared with `beta`. Pearson, Spearman, sign agreement, OLS slope/intercept, normalized absolute error, both sessions, and leave-one-subject influence are reported.

The predeclared bridge statistic is the minimum of the two session-specific Spearman correlations. Subject permutation keeps the same subject permutation across sessions. The matched random control replaces each fold/session mode by a deterministic Haar unit direction and recomputes both projected trial contrast and raw prototype coordinate. Both nulls use 1,999 replicates and plus-one one-sided p-values.

The bridge is `TRIAL_TO_PROTOTYPE_MODE_COMPATIBLE` only if both session Spearman correlations are positive, subject-permutation and random-direction p-values are at most 0.05, and every leave-one-subject session correlation retains positive sign. Otherwise it is `UNASSESSED_TRIAL_TO_PROTOTYPE_MODE_INCOMPATIBILITY`, and no second-moment recovery claim is made.

## 9. Exact variance decomposition

With population variance (`ddof=0`) and evaluation labels joined only after projection,

\[
V_{total}=Var(y),\quad
V_{within}=\sum_c\pi_cVar(y\mid c),\quad
V_{between}=\sum_c\pi_c(E[y\mid c]-E[y])^2.
\]

Machine-precision gates require `V_total=V_within+V_between` and, for binary classes, `V_between=pi_L pi_R (E[y|R]-E[y|L])^2`.

`V_between` is primary. `|Delta_trial|`, `Delta_trial^2`, `|beta|`, and `beta^2` are secondary oracle targets.

## 10. Zero-label unsigned estimators

The predeclared primary estimator is `SOURCE_WITHIN_CORRECTED_PROJECTED_VARIANCE`. For each outer fold/session, training labels estimate the arithmetic mean training-subject within-class projected variance `W_hat`. For a held-out subject,

\[
\widehat V_{between}=\max(V_{total}-\widehat W,0).
\]

Secondary estimators are total projected variance, a deterministic equal-weight unordered two-component 1-D Gaussian mixture with shared variance, and deterministic unordered two-means separation. Components are never assigned semantic names.

For each estimator report pooled and session-specific Spearman/Pearson, raw and clipped-nonnegative R-squared, MAE, normalized MAE, calibration slope/intercept, subject bootstrap CI, and leave-one-subject influence.

The primary recovery controls are 1,999 subject-target permutations and 1,999 fold/session-matched Haar directions. Descriptive controls are the first training-session PCA direction, log Frobenius norm of the subject/session AIRM marginal, summed coordinate-wise population variance of all tangent trials, a source-within model shifted cyclically by one outer fold within session, the exact constant count of 100 trials, and reversal of projected trial order. None votes on the unsigned terminal.

`UNSIGNED_CONDITIONAL_ENERGY_RECOVERY_SUPPORTED` requires the compatible bridge, positive session associations, pooled 95% subject-bootstrap CI excluding zero, both primary null p-values at most 0.05, positive leave-one-subject associations, and the symmetry gate. Otherwise the decision is negative or unassessed according to the frozen gate order.

## 11. Exact label-swap symmetry

For every subject/session, the evaluation label names are completely swapped. Pooled covariances, `M`, every `Y` and `y`, all zero-label estimates, `V_between`, and `|beta|` must remain equal within `2e-12`; signed `alpha` and `beta` must negate within the same tolerance. Failure gives `UNASSESSED_LABEL_LEAKAGE_OR_SYMMETRY_FAILURE`, and signed identification becomes `UNASSESSED_SYMMETRY_CONTRACT_FAILURE`.

## 12. Minimal semantic anchor

Calibration budgets are frozen at `m=[0,2,4,8,16,32]`; positive budgets use equal labels per class and 200 deterministic subsamples per subject/session/budget. The zero-label magnitude is held fixed. Calibration labels supply only the sign of the projected class contrast; no target label refits a population mode or magnitude estimator.

The proposed signed estimate is `sign(calibration contrast) * sqrt(primary unsigned V_between estimate)`. The equal-label baseline is the direct `m`-label half class contrast. The source baseline uses the outer-training mean `|beta|` with the same calibration orientation.

Report signed-beta MAE, sign accuracy, projected prototype error, session direction, subject-paired improvement, bootstrap CI, 1,999 paired sign-flip permutation p-value, and leave-one-subject influence. Budget zero remains `NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION` and has no signed performance estimate.

`MINIMAL_SEMANTIC_ANCHOR_EFFICIENT` requires the smallest predeclared budget in `{2,4,8}` to show lower proposed MAE than the equal-label baseline, p at most 0.05, a positive 95% subject-bootstrap improvement CI, improvement in both sessions, sign-accuracy CI above 0.5, and positive leave-one-subject improvement. Otherwise it is `MINIMAL_SEMANTIC_ANCHOR_NOT_EFFICIENT`.

## 13. Synthetic gates

Before real recovery access the executable tests must pass: known balanced rank-1 mixture, exact class-swap invariance, recoverable unsigned separation, signed non-identifiability, varying within-class noise, no separation, random-direction equivalence, minimal-anchor sign resolution, and a target-label leakage sentinel.

## 14. Freeze and outputs

The protocol, config, exact parent-fold contract, estimator declarations, 1,999 null counts, calibration budgets, decision logic, implementation, scripts, tests, and synthetic reports are committed with subject `freeze unlabeled conditional mode identifiability v0` before a real mode or recovery statistic is accessed. Every runner verifies clean freeze ancestry and hashes of all scientific source files.

Required result tables and 12 predeclared figures are written only to `outputs/unlabeled_conditional_mode_identifiability_v0/`. Presentation fixes may not alter scientific CSV/NPZ hashes.

No scientific setting changes after freeze. Numerical difficulty does not authorize jitter, threshold changes, fold removal, estimator replacement, additional labels, or result-conditioned tuning.
