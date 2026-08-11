# Protocol Amendment: Pairwise Common Action V1

Status: **pre-result amendment**. This document supersedes the primary
estimator in `PROTOCOL_COMMON_SUBJECT_ACTION_V0.md` but preserves its scientific
sensor-space action, frozen U objects, preprocessing, hard reproduction gate,
and claim restrictions.

Audit commit: `6ea711e3b4eca048fe5c092e0828d74d3c406367`.
Original corrected V0 freeze:
`9a4dd836fd24c47ffc2d0f80459c478827e93d3d`.

## 1. Why amendment is allowed

The V0 execution failed technically inside nested source generalized
Procrustes. No one of the 36 Stage-A tasks completed; consequently no
target-subject/session/held-out-class scientific cell, Stage-A statistic, or
p-value existed. The failure path could invoke 30,240 Pymanopt optimizer runs
inside one source fit, and it used nine actual starts where the configuration
declared eight. No result-derived scientific adaptation was possible because
there was no result.

The failure and formulation audit remain visible in
`COMMON_ACTION_FORMULATION_AUDIT_V1.md`. Nothing is deleted or rewritten.

Previous primary implementation:

> nested latent generalized Procrustes source model followed by target action

Amended primary implementation:

> pairwise held-out-class necessary-consequence gate, followed conditionally by
> equivalence-aware global consistency and cross-session questions

This changes formulation and computation, not the hypothesis
\(U\mapsto QUQ^{\mathsf T}\), \(Q\in O(22)\), or the held-out-class
falsification.

## 2. Frozen object and hard gate

Use BNCI2014_001 only: subjects 1–9, sessions `0train` and `1test`, classes in
the fixed order `left_hand, right_hand, feet, tongue`, and 22 frozen channels.
The inherited object is the marginally recentered identity-tangent effect

\[
U_{s,q,c}=\log(M_{s,q}^{-1/2}M_{s,q,c}M_{s,q}^{-1/2}).
\]

Load Full, half-A, and half-B U from the frozen Subject-Class Interaction V0
object whose SHA-256 is
`1bb487a8446308e1dd29965ac57901a810d6e92da25b020ae284569176f33f6d`.
Re-run the same source hash, shape, class-count, symmetry, finite-value, and
full-U exact-reproduction gates, with maximum absolute and relative Frobenius
tolerances \(10^{-12}\). Any failure is
`UNASSESSED_NUMERICAL_OR_DATA_FAILURE`.

Half A uses runs 0,1,2 and half B uses runs 3,4,5. Marginal mean, class mean,
whitening, and U have already been recomputed independently within each half;
the full marginal is never reused.

## 3. Primary pairwise LOCO action

For target subject \(s\), source subject \(r\ne s\), session \(q\), and held-out
class \(c^*\), let \(C_{fit}=C\setminus\{c^*\}\) in fixed semantic order and fit

\[
\widehat R_{s\leftarrow r,q}^{(-c^*)}
=\arg\min_{R\in O(22)}
\sum_{c\in C_{fit}}
\|U_{s,q,c}-RU_{r,q,c}R^{\mathsf T}\|_F^2.
\]

Predict the excluded class:

\[
P_{s\leftarrow r,q,c^*}
=\widehat R_{s\leftarrow r,q}^{(-c^*)}
U_{r,q,c^*}
\widehat R_{s\leftarrow r,q}^{(-c^*)\mathsf T}.
\]

The complete primary grid is
\(9\times8\times2\times4=576\) ordered pairwise cells. This is a necessary
consequence of a latent common-action model, not proof that global latent
\(\{Q_s,B_c\}\) objects exist.

## 4. Exact single-action optimizer and starts

The scientific loss is squared Frobenius sensor-space conjugation in identity
tangent coordinates. Use Pymanopt 2.2.1 on
`Stiefel(22, 22, retraction="polar")`, which is full \(O(22)\), with
`ConjugateGradient(beta_rule="HestenesStiefel")` and the explicit
`BackTrackingLineSearcher`:

- initial step 1.0;
- contraction 0.5;
- sufficient decrease \(10^{-4}\);
- optimism 2.0;
- maximum 50 line-search iterations.

Optimizer limits are maximum 1000 iterations, projected-gradient tolerance
\(10^{-5}\), minimum optimizer step \(10^{-12}\), maximum 1001 cost
evaluations, and maximum 3600 seconds per start.

Pymanopt is instantiated with `log_verbosity=1`. Pymanopt 2.2.1 can perform a
line-search update before applying its stopping check, so the runtime wrapper
returns the last logged/evaluated iterate whose projected gradient is at most
\(10^{-5}\), when one exists, instead of an unevaluated post-stop update.
Convergence is declared by either that primary projected-gradient condition or
the following pre-data numerical plateau condition, which is identical to the
already-audited independent solver's secondary stop: Pymanopt itself must
terminate at its frozen minimum step, the relative change between its final
two logged objective values must be at most \(10^{-13}\), and the final
projected gradient must be at most \(10\times10^{-5}=10^{-4}\). The plateau
rule cannot make a solution near-optimal; it only records that a determinant
sector was numerically explored. Near-optimal membership remains governed by
the much tighter objective rule in Section 5. Both termination paths and the
returned iterate are recorded per start.

Use exactly **four total starts**. Define two input-dependent spectral bases:

1. `spectral(sign_resolved=True)` from the weighted eigenspaces of the three
   target/source fit banks, with secondary-bank sign resolution;
2. `spectral(sign_resolved=False)` from the same primary weighted eigenspaces
   with all spectral signs initially positive.

Let \(F=\operatorname{diag}(-1,1,\ldots,1)\). The ordered starts are

\[
[Q_{spec,+},\;Q_{spec,+}F,\;Q_{spec,0},\;Q_{spec,0}F],
\]

each passed through the existing orthogonal retraction used by the audited
constructor. Every base/reflection pair lies in opposite determinant
components, so the runtime must contain exactly two initial det(+1) and two
initial det(-1) starts. A spectral or warm start counts toward the total; there
is no hidden fifth start. Runtime assertions require both
`len(initial_starts)==4` and `len(optimizer_results)==4` for every fit.

At least one converged result is required in both determinant sectors. No
post-optimization determinant correction is allowed. Failure is
`UNASSESSED_TECHNICAL_FAILURE`.

## 5. Predictive identifiability

For each full pairwise fit, let \(L_{best}\) be the best converged objective and
retain

\[
R_{eq}=\{R_j:L(R_j)\le L_{best}+10^{-10}+10^{-8}|L_{best}|\}.
\]

Analyze the continuous common stabilizer of the three **source** fit matrices
by the singular spectrum of

\[
\Omega\mapsto(\Omega U_{r,q,c}-U_{r,q,c}\Omega)_{c\in C_{fit}},
\quad\Omega=-\Omega^{\mathsf T}.
\]

Use machine-size SVD rank tolerance and the prespecified approximate tolerance
\(10^{-8}\) times the largest singular value. For each exact numerical
nullspace direction, the fixed angles \(+\pi/2,-\pi/2,\pi\) are evaluated and
retained only if their full fit objective satisfies the same near-optimal
criterion. This Lie diagnostic does not claim to enumerate every discrete
symmetry; all converged multi-start near-optima remain mandatory.

For every retained action,

\[
P_j=R_jU_{r,q,c^*}R_j^{\mathsf T}.
\]

Let \(P_{best}\) be the best full prediction,
\(\epsilon=\epsilon_{mach}\sqrt{22^2}\), and
\(N_P=\max(\|P_{best}\|_F,\epsilon)\). Define

\[
D_{eq}=\max_{j,k}\frac{\|P_j-P_k\|_F}{N_P}.
\]

Independently fit the same pairwise action from half A and half B and predict
their corresponding independently recomputed held-out source matrices,
producing \(P_A,P_B\). Define

\[
D_{split}=\frac{\|P_A-P_B\|_F}{N_P}.
\]

The pairwise held-out prediction is identifiable iff

\[
D_{eq}\le\max(10^{-5},D_{split}).
\]

Different R matrices with agreeing predictions are
`HARMLESS_Q_NONUNIQUENESS`. Dispersion above the threshold is
`PREDICTIVE_NONIDENTIFIABILITY`. Optimizer/sector/split failure is
`TECHNICAL_FAILURE`. Raw R disagreement is never the scientific magnitude.

Every one of the 576 primary cells is required. Any predictively
nonidentifiable required cell makes Stage A
`UNASSESSED_ACTION_NOT_IDENTIFIABLE`; any technical cell failure makes it
`UNASSESSED_TECHNICAL_FAILURE`. No available-case analysis is allowed.

Split-half raw error, action error, gain, prediction norms, and D statistics
are stored descriptively. Prediction reproducibility, not raw-R equality, is
the object.

## 6. Pairwise errors and gain

For target held-out matrix \(T=U_{s,q,c^*}\), source held-out matrix
\(S=U_{r,q,c^*}\), and action prediction \(P\), define exactly

\[
e_{raw}=\frac{\|T-S\|_F^2}{\|T\|_F^2},\qquad
e_R=\frac{\|T-P\|_F^2}{\|T\|_F^2},\qquad
g=e_{raw}-e_R.
\]

If \(\|T\|_F^2\le\epsilon_{mach}^2\,22^2\), terminate as
`UNASSESSED_NUMERICAL_OR_DATA_FAILURE`. Positive gain means that the action
estimated from three classes improves the fourth-class prediction over leaving
the same source held-out U unchanged.

## 7. Subject-level aggregation

Source–target pairs are not inferential samples. Aggregate in this exact order:

\[
G_{s,q,c^*}=\operatorname{median}_{r\ne s}g_{s,r,q,c^*},
\]

\[
G_A(s)=\operatorname{median}_{q,c^*}G_{s,q,c^*},
\qquad
T_A=\operatorname{median}_{s}G_A(s).
\]

Thus eight sources first produce one target/session/class cell; eight cells
produce one target-subject score; nine target subjects produce the group
statistic. The target subject is the inferential unit. Report all source pairs
and all nine subject scores, but never treat the 576 pairs as independent.

## 8. Stage-A primary unrelated-target-action null

The primary null asks whether the action fitted for target \(s\) is more useful
than an action fitted for an unrelated target while keeping the source,
session, and held-out fold fixed.

For each fixed \((r,q,c^*)\), the eight possible targets are all subjects other
than \(r\). At each of 1999 deterministic replicates, draw a derangement of
these eight target identities and give evaluation target \(s\) the action
\(R_{\pi(s)\leftarrow r,q}^{(-c^*)}\), where \(\pi(s)\ne s\). The same raw
source-to-target error remains fixed. Independently draw one derangement for
each source/session/held-out context, then repeat the exact source→target-cell→
target-subject→group aggregation. This permutation uses the already fitted
true-correspondence actions and needs no additional optimizer calls.

Master seed is 20260810; stream is `stage_A_unrelated_target`. The one-sided
greater-or-equal plus-one p-value is

\[
p_A=(1+\#\{T_A^{null}\ge T_A\})/(1+1999).
\]

The primary Stage-A gate passes iff

\[
T_A>0,\quad T_A-\operatorname{median}(T_A^{null})>0,\quad p_A\le0.05.
\]

## 9. Conditional semantic-mismatch gate

If and only if the primary gate in Section 8 passes, unlock the semantic
control. Otherwise record exactly
`NOT_UNLOCKED_BY_PRIMARY_GATE`; do not fit semantic comparators.

For the ordered three fit classes enumerate all five nonidentity S3
permutations \(\pi\). The mismatch fit is exactly

\[
\min_R\sum_{i=1}^3
\|U_{s,q,c_{\pi(i)}}-RU_{r,q,c_i}R^{\mathsf T}\|_F^2.
\]

It still predicts the original held-out semantic class. All five full-U fits
are required for every source pair. At each of 1999 replicates choose one of
the five permutations independently for every target/session/held-out cell,
but share that choice across its eight sources. Reapply the exact nested
aggregation. Master seed is 20260810; stream is `stage_A_semantic`.

These permuted fits are null comparators, not candidate scientific actions.
Every fit still executes exactly four starts, records all converged
near-optimal starts, and requires successful exploration of both determinant
sectors. Its reported prediction uses the minimum-objective converged action
with the frozen gradient/start-index tie break; the held-out class is never
used to choose among starts. The primary true-correspondence cells alone carry
the stabilizer, split-half predictive-identifiability gate, because that gate
asks whether an asserted action prediction is scientifically interpretable.
No semantic-null comparator is promoted to an asserted action or silently
substituted for a missing primary cell.

The final Stage-A semantic gate passes iff the observed true-correspondence
\(T_A\) is positive, its effect above the semantic-null median is positive,
and its one-sided plus-one p-value is at most 0.05. Pairwise Stage A is fully
supported only when both primary and semantic gates pass.

## 10. Conditional cross-session Stage B

Stage B is unlocked only if both Stage-A gates pass. For directions
0train→1test and 1test→0train, reuse the action estimated from the three fit
classes in the training session and predict the opposite-session held-out
source matrix:

\[
P_{s\leftarrow r,q\to q',c^*}
=R_{s\leftarrow r,q}^{(-c^*)}U_{r,q',c^*}
R_{s\leftarrow r,q}^{(-c^*)\mathsf T}.
\]

The target and raw baseline are respectively \(U_{s,q',c^*}\) and
\(U_{r,q',c^*}\), with the same target-norm errors. Equivalent full actions
and best independent half-A/half-B actions from Stage A are applied to the
opposite-session held-out objects; raw R matrices are not compared.

Aggregate eight sources, then two directions×four classes, then nine targets
by median exactly as Stage A. Use the analogous unrelated-target derangement
null with stream `stage_B_unrelated_target`. If its primary gate passes, reuse
the five Stage-A semantic mismatch actions on the opposite session and use the
conditional semantic stream `stage_B_semantic`. Stage B support requires both
its primary and semantic gates.

Stage B failure means no evidence for a session-stable pairwise common action
in this symmetric-whitened U representation. It does not negate subject
individuality.

## 11. Secondary global consistency

Pairwise Stage-A support unlocks an equivalence-aware cycle diagnostic. For
distinct \(s,r,t\), compare the induced predictions from
\(R_{s\leftarrow r}R_{r\leftarrow t}\) and \(R_{s\leftarrow t}\) over a fixed
four-class U probe bank, minimizing over the finite near-optimal action sets.
Store normalized Frobenius discrepancies. Raw action matrices are never
compared.

This cycle diagnostic is descriptive in this amendment and has no terminal
vote. The exact analytically profiled global objective validated in
`COMMON_ACTION_FORMULATION_AUDIT_V1.md` remains the only allowed latent global
model, but it is **not automatically fitted here**. Its product-manifold
determinant-component initialization requires a separate pre-result freeze.
The old nested alternating implementation is permanently forbidden.

## 12. Terminal logic

Apply in order:

1. data/reproduction failure → `UNASSESSED_NUMERICAL_OR_DATA_FAILURE`;
2. required optimizer/split/grid failure → `UNASSESSED_TECHNICAL_FAILURE`;
3. any required predictive nonidentifiability →
   `UNASSESSED_ACTION_NOT_IDENTIFIABLE`;
4. Stage-A primary FAIL →
   `PAIRWISE_COMMON_ACTION_NOT_SUPPORTED_WITHIN_SESSION`;
5. Stage-A primary PASS but semantic FAIL →
   `PAIRWISE_COMMON_ACTION_NOT_SEMANTICALLY_SUPPORTED`;
6. Stage A PASS but Stage-B primary FAIL →
   `PAIRWISE_COMMON_ACTION_WITHIN_SESSION_ONLY`;
7. Stage-B primary PASS but semantic FAIL →
   `PAIRWISE_COMMON_ACTION_NOT_CROSS_SESSION_SEMANTICALLY_SUPPORTED`;
8. Stage A/B primary and semantic gates PASS →
   `PAIRWISE_COMMON_ACTION_NECESSARY_CONSEQUENCE_SUPPORTED`.

Even the last outcome supports only a stable pairwise necessary consequence;
it does not establish a global latent model.

## 13. Computational contract

Primary Stage A performs 576 Full, 576 half-A, and 576 half-B action-fit
objects. At exactly four total starts this is **6,912 Pymanopt optimizer
runs**. The unrelated-target null adds zero optimizer runs.

If semantic control is unlocked, 576×5 full mismatch fits add 2,880 fit
objects or **11,520 Pymanopt runs**, for a maximum Stage-A total of **18,432**.
Cross-session Stage B reuses Stage-A actions and adds no action fits.

The synthetic d=22 benchmark measured 0.194 s and 0.237 s per four-start noisy
fit in the formulation audit. A final amendment-wrapper rerun measured 0.161 s
and 0.225 s for det(+1) and det(-1) truths, with held-out relative errors
\(1.21\times10^{-4}\) and \(1.44\times10^{-4}\) at injected symmetric noise
scale \(2\times10^{-4}\). The projected eight-worker Stage-A primary runtime is
2–6 minutes; semantic-unlocked maximum is 4–12 minutes. These ranges include
process, checkpoint, and stabilizer-diagnostic allowance and are engineering
projections, not outcome thresholds.

The final synthetic execution dry run also completed the exact BNCI-shaped
pairwise grid: all 576 primary cells, Full/A/B, and exactly 6,912 Pymanopt
runs, with maximum held-out action error \(1.33\times10^{-29}\). The
conditional five-permutation semantic grid completed all 2,880 fit objects
and 11,520 Pymanopt runs. No BNCI object was loaded for either validation.

## 14. Output and checkpoint contract

Write only below `outputs/bnci2014_001_pairwise_common_action_v1/`. Required:

- frozen protocol/config/environment and provenance;
- U reproduction/data-contract tables;
- resumable per-target primary and conditional-semantic checkpoints;
- all 576 pairwise scores and solver diagnostics;
- target-cell and all-nine-subject summaries;
- identifiability/stabilizer summaries and compact action objects;
- primary and conditional semantic null statistics plus seed manifests;
- conditional cross-session scores/nulls;
- descriptive cycle diagnostics if unlocked;
- decision chain, terminal JSON, and plain-language report.

No failed V0 output is overwritten. Ignored U/cache objects remain regenerable
and are not added to Git.

## 15. Claims and next branch

This is global common-action anatomy, not the final Paper-A claim. Never state
that individuality stops at Q, that Q is physiology, that all individuality is
rotation, or that a personalized model improves performance.

At most, a positive result says that, under the frozen identity-tangent
sensor-space representation, class-independent pairwise orthogonal actions
predict held-out class structure within session and, if Stage B passes, across
sessions.

Regardless of outcome, the next planned scientific branch remains
**SPLIT-EPOCH COVARIANCE-SET ANATOMY**, separating candidate location,
dispersion, pose/orientation, and intrinsic-configuration components. No such
analysis is added to this amendment.
