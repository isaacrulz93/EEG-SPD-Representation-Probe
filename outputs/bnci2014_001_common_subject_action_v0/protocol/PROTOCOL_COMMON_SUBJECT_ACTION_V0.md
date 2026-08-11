# Protocol: Common Subject Action Falsification V0

Status: corrected pre-data protocol, frozen before any BNCI common-action score
is loaded or computed. Base commit:
`272d775678644aad062df424a70586d4b42de652`. Synthetic implementation commit:
`f5ffc3bd554e2327040839aa4ebf55889092dbf5`.

## 1. Question and boundary

This is an anatomy/falsification experiment, not a new alignment method. It
asks whether one class-independent subject action can explain multiple motor-
imagery class effects and predict a class excluded from action fitting, first
within a session and then across sessions. If that common action is supported,
it asks whether a stable class-dependent residual remains.

The only primary dataset is BNCI2014_001: nine subjects, sessions `0train` and
`1test`, four classes in the fixed order `left_hand, right_hand, feet, tongue`,
22 fixed EEG channels, and all six runs. The preprocessing, OAS whole-trial
covariances, float64 arithmetic, AIRM means, and identity-tangent construction
are inherited unchanged from Subject-Class Interaction V0. OpenBMI and HGD are
not used.

PCA, low-rank models, neural networks, TTA, new losses, trajectory expansion,
full five-window bag anatomy, feature selection, class-subset selection, and
classifier studies are outside this protocol.

## 2. Frozen object and reproduction gate

For subject (s), session (q), and class (c), the inherited objects are

\[
\widetilde M_{s,q,c}=M_{s,q}^{-1/2}M_{s,q,c}M_{s,q}^{-1/2},\qquad
U_{s,q,c}=\log(\widetilde M_{s,q,c}).
\]

The primary object is (U), not the former population-template residual (Z).
The frozen source file is
`outputs/subject_class_interaction_v0/objects/core_interaction_objects.npz`
with SHA-256
`1bb487a8446308e1dd29965ac57901a810d6e92da25b020ae284569176f33f6d`.
The full, run-half A, and run-half B keys are respectively
`AIRM__session_specific__F__U`, `AIRM__session_specific__A__U`, and
`AIRM__session_specific__B__U`.

Before inference, require exact subjects, sessions, classes, channels, counts,
metadata ordering, and source-object file hash. Recompute the inherited
full-session construction from the frozen covariances and verify marginal
means, class means, and (U) against the PR #2 objects. The maximum absolute and
relative Frobenius tolerances are both (10^{-12}); symmetry relative error is at
most (10^{-12}); all values must be finite and all inherited geometry gates
must pass. The same order/count checks apply to the saved half objects. A gate
failure terminates as `UNASSESSED_NUMERICAL_OR_DATA_FAILURE`.

Each half independently used only its own runs: A is runs 0,1,2 and B is runs
3,4,5. Its marginal mean, class mean, whitening, and (U) were recomputed from
that half; no full-session marginal is reused.

## 3. Scientific action and objective

The subject action is exactly the sensor-coordinate orthogonal congruence

\[
U\mapsto Q U Q^T,\qquad Q\in O(d),\qquad d=22.
\]

It is never an arbitrary 253-by-253 rotation of `svec(U)`, an unconstrained
linear transform, or an SO-only model. For paired identity-tangent target
effects (A_c) and latent templates (B_c), fit

\[
L(Q)=\sum_{c\in C_{fit}}\lVert A_c-QB_cQ^T\rVert_F^2.
\]

This is named the **identity-tangent sensor-space orthogonal-conjugation
objective**. It is not called the AIRM RPA loss.

The squared-conjugation form is the same Euclidean pair-cost form used in the
[original RPA author implementation](https://github.com/plcrodrigues/RPA/blob/cfcddb3d31b482941a23353dfbe46dffb118d02d/rpa/helpers/transfer_learning/manopt.py).
The original RPA setting, objects, and classification task differ; the full
experiment is therefore described only as RPA-like.

Installed pyRiemann 0.12 is not a numerical oracle. Its Euclidean runtime loss
uses an unsquared distance while its analytic gradient and documentation
correspond to squared distance. The pinned source files, hashes, and synthetic
discrepancy are recorded in
`docs/ENGINEERING_NOTE_PYRIEMANN_012_ROTATION_DISCREPANCY.md`. Vendor code is
not modified and the discrepancy is not emulated.

## 4. Exact optimizer runtime contract

The scientific solver is Pymanopt 2.2.1. The manifold object is
`Stiefel(d, d, retraction="polar")`; square Stiefel matrices constitute full
(O(d)). The optimizer is `ConjugateGradient(beta_rule="HestenesStiefel")`
with an explicitly supplied `BackTrackingLineSearcher`.

The line-search constructor parameters are exactly:

- `contraction_factor=0.5`;
- `optimism=2.0`;
- `sufficient_decrease=1e-4`;
- `max_iterations=50`;
- `initial_step_size=1.0`.

The optimizer stopping parameters, which are not line-search parameters, are:

- `max_iterations=1000`;
- `min_gradient_norm=1e-5`;
- `min_step_size=1e-12`;
- `max_cost_evaluations=1001`;
- `max_time=3600` seconds.

The Euclidean gradient is analytic and is checked against finite differences
and Autograd. Pymanopt supplies tangent projection, polar retraction, vector
transport, and nonlinear conjugate-gradient updates. The independent
projected-Armijo solver is validation-only. Runtime objects and their stored
parameters must match the frozen YAML in a unit test.

Use eight deterministic starts: four in det(+1) and four in det(-1). Every
start records its initial/final determinant, final objective, projected
gradient norm, iterations, stopping reason, convergence, and induced held-out
prediction hash/error. At least one converged solution is required in each
determinant sector. Otherwise the required fit is
`UNASSESSED_TECHNICAL_FAILURE`. Converged local minima outside the near-optimal
criterion are recorded but excluded from the scientific equivalence set.

## 5. Structural and predictive identifiability

The fitted matrix (Q) is not assumed unique. For fit templates define

\[
\operatorname{Stab}(B_{fit})=
\{S\in O(d):SB_cS^T=B_c\ \forall c\in C_{fit}\}.
\]

For the continuous identity component, express a skew matrix (Omega) in an
orthonormal basis of (\mathfrak o(d)) and compute the singular spectrum of the
simultaneous linear map

\[
\Omega\mapsto(\Omega B_c-B_c\Omega)_{c\in C_{fit}}.
\]

Numerical rank uses the standard matrix-size-times-machine-epsilon SVD
tolerance. Approximate nullity additionally uses relative threshold (10^{-8})
times the largest singular value. This Lie-algebra diagnostic does not capture
all discrete symmetries, so deterministic multi-start predictive comparison is
mandatory.

### 5.1 Exact near-optimal set

For each required LOCO cell, let (L_{best}) be the best converged fit objective
and define exactly

\[
Q_{eq}=\{Q_j:L(Q_j)\le L_{best}+a+r|L_{best}|\},
\quad a=10^{-10},\quad r=10^{-8}.
\]

No unit floor is applied to the relative term. Exact continuous-stabilizer
actions generated from the numerical nullspace are included only if they also
satisfy this inequality. If near-optimal source generalized-Procrustes starts
produce different held-out templates, all induced target near-optimal
predictions across those source fits are included in the cell prediction bank;
global gauge differences alone are ignored.

For every included action define

\[
P_j=Q_jB_{held}Q_j^T.
\]

Let (P_{best}) be the prediction from the best full-session pipeline, and set

\[
\epsilon=\epsilon_{mach}\sqrt{d^2},\qquad
N_P=\max(\lVert P_{best}\rVert_F,\epsilon).
\]

Equivalent-solution dispersion is

\[
D_{eq}=\max_{j,k}\frac{\lVert P_j-P_k\rVert_F}{N_P}.
\]

Hashes are provenance only and are never used as a scientific magnitude. If
only one (Q) qualifies, (D_{eq}=0).

### 5.2 Split-half variability and exact decision

For the same cell, independently run the complete source/template/target
action construction from half A and half B objects, producing held-out
predictions (P_A) and (P_B). For Stage B, half A uses independently recomputed
half-A objects in both the action-estimation session and predicted session;
half B analogously uses half-B objects. Define with the same full-cell
normalizer

\[
D_{split}=\frac{\lVert P_A-P_B\rVert_F}{N_P}.
\]

There is no multiplicative factor. With the synthetic-calibrated numerical
floor (D_{floor}=10^{-5}), the cell is predictively identifiable iff

\[
D_{eq}\le\max(D_{floor},D_{split}).
\]

If (P_{best}), (P_A), or (P_B) has Frobenius norm at or below (\epsilon), or is
nonfinite, the chain is `UNASSESSED_NUMERICAL_OR_DATA_FAILURE`. A failed
split-half source/action fit is `UNASSESSED_TECHNICAL_FAILURE`. If a determinant
sector lacks a converged solution, the cell is technical failure. If both
sectors are near-optimal, both enter (Q_{eq}). Different (Q) values whose
predictions agree are harmless non-uniqueness, including continuous or
discrete stabilizers. Near-equivalent actions with dispersion above the frozen
threshold are `PREDICTIVE_NONIDENTIFIABILITY`, not a negative scientific
result.

Every required Stage-A or unlocked Stage-B cell is mandatory. One
predictively non-identifiable required cell makes the corresponding
inferential chain `UNASSESSED_ACTION_NOT_IDENTIFIABLE`. One technical cell
failure makes it `UNASSESSED_TECHNICAL_FAILURE`. No cell is dropped and there
is no available-case aggregation.

## 6. Source generalized Procrustes and gauge

For target subject (s), sources are every (r\ne s); the target never enters a
source fit. Every class is held out once, and the remaining three are the fit
classes.

Stage A fits each session separately:

\[
\min_{\{Q_{r,q}\},\{B_{q,c}\}}
\sum_{r\ne s}\sum_{c\ne c^*}
\lVert U_{r,q,c}-Q_{r,q}B_{q,c}Q_{r,q}^T\rVert_F^2.
\]

Stage B fits both sessions jointly with one action per source:

\[
\min_{\{Q_r\},\{B_{q,c}\}}
\sum_{r\ne s}\sum_q\sum_{c\ne c^*}
\lVert U_{r,q,c}-Q_rB_{q,c}Q_r^T\rVert_F^2.
\]

Fix the lowest-ID source subject action to identity. For fixed actions, the
exact update is

\[
B_{q,c}=\operatorname{mean}_{r\ne s}(Q_r^TU_{r,q,c}Q_r).
\]

For fixed templates, update every non-anchor action with the validated solver.
Use four deterministic outer starts, at most 120 alternations, and relative
objective tolerance (10^{-6}). Record every outer start. Gauge fixing does not
eliminate the common stabilizer.

Only after source actions are fitted from three classes is the held-out source
template formed:

\[
B_{q,c^*}=\operatorname{mean}_{r\ne s}(Q_r^TU_{r,q,c^*}Q_r).
\]

The held-out source label is used only here. The target held-out object never
enters target-action fitting.

## 7. Prediction, baseline, and group statistic

The raw baseline is the LOSO source mean in the observed sensor coordinates:

\[
\bar U^{raw}_{-s,q,c^*}=\operatorname{mean}_{r\ne s}U_{r,q,c^*}.
\]

For target (U_t) and prediction (\widehat U), define

\[
e=\frac{\lVert U_t-\widehat U\rVert_F^2}{\lVert U_t\rVert_F^2},
\qquad g=e_{raw}-e_Q.
\]

A numerical-zero target norm is a data/numerical failure. Positive gain means
the common-action prediction improves on the unaligned source population mean.

For each subject, aggregate the eight mandatory cell gains by their median;
the group statistic is the median of the nine subject medians. Subjects, not
cells or trials, are the statistical units.

## 8. Stage A: within-session held-out class

For every subject, session, and held-out class, estimate the target action from
the other three classes:

\[
\widehat Q_{s,q}^{(-c^*)}=\arg\min_Q
\sum_{c\ne c^*}\lVert U_{s,q,c}-QB_{q,c}Q^T\rVert_F^2,
\]

then predict the held-out class in the same session. This gives two sessions by
four held-out classes, eight cells per subject. Let the subject median be
(G_A(s)) and (T_A=\operatorname{median}_sG_A(s)).

Stage A passes iff (T_A>0), its unrelated-action effect is positive with
(p\le0.05), and its semantic-mismatch effect is positive with (p\le0.05),
after all hard, technical, and identifiability gates pass.

## 9. Stage B: cross-session stable action

Stage B has an inferential vote only if Stage A passes. For direction 0 to 1,
estimate the target action from session 0's three fit classes and use it with
the session-1 held-out source template. Reverse the sessions for direction 1
to 0. No other target-session class enters the action fit. The eight cells per
subject are two directions by four held-out classes. Define subject median
(G_B(s)) and group statistic (T_B).

Stage B passes iff Stage A passes, (T_B>0), and both unrelated-action and
semantic-mismatch effects are positive with their one-sided plus-one
(p\le0.05), after all required gates.

Failure means only that one session-independent orthogonal action in this
symmetric-whitened (U) representation did not predict the cross-session held-
out class. It does not prove that stable sensor mixing is absent.

## 10. Frozen nulls

All Monte Carlo streams use master seed 20260810, NumPy `PCG64DXSM`, fixed
stream tags, and 1999 replicates.

For the unrelated-action null, precompute every non-target source action in
the same fitted fold/gauge as the target. Each replicate deterministically
chooses one unrelated source action per cell, computes its held-out gain, then
uses the same subject-median/group-median aggregation. The observed action must
exceed this null with positive effect and one-sided greater-or-equal plus-one
(p\le0.05).

For semantic mismatch, enumerate exactly the five nonidentity permutations of
the three fit classes. Refit the target action under each mismatch, retain the
same held-out template, and precompute its gain. Each replicate selects one of
the five per cell and applies the identical aggregation. True semantic
correspondence must exceed this null with positive effect and (p\le0.05).

There is no best-class selection, retry, threshold tuning, or result-driven
change after Stage A is observed.

## 11. Stage C: post-action residual

Stage C runs only after Stage B passes. For each held-out class use the
opposite-session, leave-one-class-out action:

\[
E_{s,1,c}=U_{s,1,c}-\widehat Q_{s,0}^{(-c)}B_{1,c}
\widehat Q_{s,0}^{(-c)T},
\]

and symmetrically for session 0. Stack Frobenius-isometric `svec(E)` blocks in
the fixed four-class order and L2-normalize, failing on numerical-zero norm.

Residual test 1 compares the median same-subject cross-session cosine against
all exact nine-subject derangements. Residual test 2 independently applies a
nonidentity S4 class-block permutation to session 1 for every subject in each
of 1999 deterministic replicates. This is named the class-correspondence null,
not label destruction. Residual support requires both tests to have positive
effects and one-sided (p\le0.05).

## 12. Terminal logic

Apply in this order:

1. data/geometry/reproduction failure:
   `UNASSESSED_NUMERICAL_OR_DATA_FAILURE`;
2. incomplete required optimization/split-half grid:
   `UNASSESSED_TECHNICAL_FAILURE`;
3. any required predictively non-identifiable Stage-A or unlocked Stage-B
   cell: `UNASSESSED_ACTION_NOT_IDENTIFIABLE`;
4. Stage A fail: `COMMON_ACTION_NOT_SUPPORTED_WITHIN_SESSION`;
5. Stage A pass and Stage B fail: `SESSION_SPECIFIC_COMMON_ACTION_ONLY`;
6. Stage A/B pass and Stage C pass:
   `COMMON_ACTION_SUPPORTED_RESIDUAL_INDIVIDUALITY_REMAINS`;
7. Stage A/B pass and Stage C fail:
   `COMMON_ACTION_SUPPORTED_NO_STABLE_RESIDUAL_EVIDENCE`.

An unassessed outcome is not converted into a scientific fail. A Stage-A fail
stops B/C; a Stage-B fail stops C.

## 13. Output and provenance contract

All new artifacts live only below
`outputs/bnci2014_001_common_subject_action_v0/`. Save frozen protocol/config,
environment and provenance; full and split-half (U); source and target fits;
cross-fitted residuals; complete cell, subject, null, solver, split-half,
reproduction, and decision tables; deterministic seed manifests; all six
specified figures as CSV/PNG/PDF; terminal JSON; and the final Markdown report.

Raw EEG, frozen PR #2 outputs, and regenerable cache are not modified or
committed. Required failures are recorded without silent row deletion. The
report shows every one of the nine subjects and does not highlight only the
best subject.

## 14. Claim restrictions

At most, a positive result supports a predictively identifiable
class-independent sensor-space orthogonal action on marginally recentered
identity-tangent class effects, within session and, if Stage B passes, across
sessions. Stage C may support an additional stable class-correspondence
residual candidate.

It does not establish physiology, causal dynamics, source-space structure,
that all individuality is rotation, that every possible subject transform has
been rejected, unlabeled identifiability, a personalized-performance gain, or
that the former (Z) was caused by this action.
