# Common Subject Action — Formulation Audit V1

Status: **synthetic/formulation audit only; no new BNCI scientific score was
read or computed.** This memo does not amend or freeze a new scientific
protocol.

## Preserved history and reason for this audit

- Branch: `pilot/common-subject-action-falsification-v0`
- Current pre-audit HEAD: `9a4dd836fd24c47ffc2d0f80459c478827e93d3d`
- Original implementation commit: `f5ffc3bd554e2327040839aa4ebf55889092dbf5`
- Corrected V0 protocol-freeze commit:
  `9a4dd836fd24c47ffc2d0f80459c478827e93d3d`
- The failed run left its immutable reproduction artifacts under
  `outputs/bnci2014_001_common_subject_action_v0/`. There is no separate local
  run log; the thrown `ActionSolverError` is preserved in the execution
  transcript. No Stage-A task, statistic, or p-value completed.
- The failure occurred inside source generalized Procrustes before a
  scientific cell returned. One source fit could make 3,360 calls to the
  multi-start single-action wrapper and 30,240 Pymanopt optimizer runs. The
  configured eight starts were actually a warm start plus eight deterministic
  starts. The present audit adds a fail-fast exact-total-start assertion and
  makes the old update explicitly warm-one-plus-deterministic-seven. It does
  not relaunch that model.

The hypothesis therefore remains **unassessed**, rather than scientifically
negative.

## 1. Exact scientific hypothesis

For subject (s), session (q), and class (c), the proposed explanation is

\[
U_{s,q,c} \approx Q_{s,q} B_{q,c} Q_{s,q}^{\mathsf T},
\qquad Q_{s,q}\in O(d).
\]

Here (U\in\operatorname{Sym}(d)) is the already defined marginally recentered
identity-tangent class effect. For BNCI, (d=22). The action is a
22-dimensional **sensor-space orthogonal conjugation**, not an arbitrary
rotation of a 253-dimensional `svec` representation.

Stage A permits a session-specific (Q_{s,q}). A later cross-session model may
ask whether one (Q_s) can be used across (q). In every case the falsifying
question is the same: can a class-independent action estimated without class
(c^*\) predict that held-out class?

This is a model-anatomy question. It is not a search for a new rotation
algorithm and does not imply physiology, source-space structure, or a useful
unlabeled personalization method.

## 2. Why three symmetric matrices can constrain an action strongly

Three matrices in \(\operatorname{Sym}(22)\) are not three ordinary points in
\(\mathbb R^{22}\). Each generic symmetric matrix carries 22 eigenvalues and
an orthogonal eigenspace frame. For one matrix with a simple spectrum, an
orthogonal conjugation must map its eigenspaces to the target eigenspaces, up
to eigenvector signs and any permitted eigenvalue ordering. Repeated
eigenvalues enlarge the freedom to rotate inside repeated-eigenvalue blocks.

For fit templates \(B_1,\ldots,B_k\), the remaining exact freedom is their
common stabilizer

\[
\operatorname{Stab}(B_{\rm fit})=
\{S\in O(d):SB_iS^{\mathsf T}=B_i\;\text{for all }i\}.
\]

Generic noncommuting matrices usually have eigenspaces in incompatible
orientations. Intersecting their individual stabilizers can therefore reduce
the common stabilizer to a small discrete set, sometimes only
\(\{+I,-I\}\). In contrast, commuting matrices, common repeated blocks, or
near-repeated spectra can leave substantial discrete or continuous freedom.
Consequently three class matrices *can* constrain the induced action far more
than three ordinary points, but identifiability must be measured rather than
assumed.

## 3. Stabilizer and predictive-identifiability condition

If (Q) fits the observed classes and (S\in\operatorname{Stab}(B_{\rm fit})),
then (QS) is observationally equivalent on those classes. The scientific
object is therefore not a unique matrix (Q). It is the induced held-out
prediction

\[
P(Q)=QB_{c^*}Q^{\mathsf T}.
\]

The required distinction is:

1. **Harmless nonuniqueness:** different admissible (Q) values induce the
   same held-out (P(Q)).
2. **Predictive nonidentifiability:** equal or near-equal fit objectives induce
   materially different held-out predictions.
3. **Technical failure:** the optimizer cannot reproducibly reach the relevant
   objective class.

The implemented structural diagnostic expands a skew-symmetric direction
\(\Omega\) in an orthonormal basis of \(\mathfrak{so}(d)\) and stacks the linear
maps

\[
\Omega\longmapsto \Omega B_i-B_i\Omega.
\]

Its singular spectrum, machine-rank nullity, relative-tolerance approximate
nullity, and null directions identify continuous or near-continuous
stabilizers. It cannot enumerate all discrete symmetries. Deterministic
multi-start solutions and their induced held-out predictions therefore remain
mandatory.

The previous cell-level prediction-dispersion idea remains mathematically
appropriate: collect all converged solutions within a prespecified objective
tolerance, compute normalized pairwise dispersion of their held-out
predictions, and compare it with a synthetic numerical floor and independently
recomputed split-half prediction variability. Its exact threshold and
fail-closed group policy must be re-frozen in the amended protocol; no BNCI
value may be used to select them.

## 4. Is pairwise LOCO a valid necessary-condition falsification?

Yes. Under the exact latent model for subjects (s) and (r), define

\[
R_{s\leftarrow r}=Q_sQ_r^{\mathsf T}.
\]

Then for every class

\[
R_{s\leftarrow r}U_{r,c}R_{s\leftarrow r}^{\mathsf T}
=Q_sB_cQ_s^{\mathsf T}=U_{s,c}.
\]

It follows that fitting one (R_{s\leftarrow r}\in O(d)) on three classes and
testing it on the fourth is a necessary consequence of the global
common-action model. If the fit-class action is predictively identifiable and
the held-out implication fails under a pre-frozen statistical rule, the exact
global model is falsified for that cell. A group-level rejection rule can then
reject this particular common orthogonal-action explanation without fitting a
latent population template.

For noisy least squares, “failure” needs a predeclared effect/null/aggregation
rule. A poor local optimizer or a predictively nonidentifiable fit must produce
`TECHNICAL_FAILURE` or `PREDICTIVE_NONIDENTIFIABILITY`, never a scientific
rejection.

## 5. What pairwise success does not prove

Ordinary empirical pairwise success does not by itself prove:

- that one set of latent \(\{Q_s,B_c\}\) explains all subjects jointly;
- that a subject action is consistent across different source partners;
- that actions estimated in different held-out-class folds are the same
  induced action;
- that the action is stable across sessions;
- that all residual subject×class individuality has been explained.

There is an important exact-case nuance. If, for every target, an exact map
from one common anchor maps the **entire four-class bank**, then a latent model
can be constructed directly: set (Q_a=I\), (B_c=U_{a,c}\), and
(Q_s=R_{s\leftarrow a}\). In that strong noiseless condition a separate cycle
test is redundant for existence. Real “PASS” decisions are approximate,
statistical, held-out-fold-specific, and equivalence-set-valued, so this exact
construction does not justify a global claim in the planned experiment.

Pairwise success alone supports only the narrower statement that a
source–target relative sensor action estimated from selected classes predicts
an unseen class for the tested pair under the frozen rule.

## 6. Global and cycle consistency after pairwise success

Exact representatives satisfy

\[
R_{s\leftarrow r}R_{r\leftarrow t}=R_{s\leftarrow t},
\qquad
R_{r\leftarrow s}=R_{s\leftarrow r}^{\mathsf T}.
\]

Raw-matrix differences are invalid in the presence of stabilizers. For finite
near-optimal equivalence sets \(\mathcal E_{sr}\), an induced-action diagnostic
on a prespecified probe bank \(\mathcal M_t\) is

\[
D_{srt}=\min_{A\in\mathcal E_{st},\,B\in\mathcal E_{sr},\,C\in\mathcal E_{rt}}
\frac{\|\operatorname{Ad}_{BC}(\mathcal M_t)-
\operatorname{Ad}_{A}(\mathcal M_t)\|_F}
{\max(\|\operatorname{Ad}_{A}(\mathcal M_t)\|_F,\epsilon)},
\]

where \(\operatorname{Ad}_R(M)=RMR^{\mathsf T}\). The synthetic implementation
searches representatives and compares induced predictions, never raw (R)
matrices. Cross-fold coherence should likewise ask whether the maps induced by
different held-out folds agree on a fixed matrix bank.

Under exact zero-error conditions on a connected graph, compatible
representatives, inverse consistency, and the cycle relation allow an anchor
construction of latent subject actions. With noise, finite samples, incomplete
equivalence sets, or tolerance-based success, pairwise prediction plus a small
cycle discrepancy is a necessary coherence diagnostic, not a proof. The
profiled global objective is the cleaner direct test of whether one joint
latent model fits after the cheap gate passes.

## 7. Is profiled generalized Procrustes exactly equivalent to Formulation A?

Yes, for the specified equal-weight squared-Frobenius problem.

Formulation A is

\[
F(Q,B)=\sum_{r,c}\|U_{r,c}-Q_rB_cQ_r^{\mathsf T}\|_F^2.
\]

Orthogonal invariance gives, with
\(V_{r,c}=Q_r^{\mathsf T}U_{r,c}Q_r\),

\[
F(Q,B)=\sum_{r,c}\|V_{r,c}-B_c\|_F^2.
\]

For fixed actions, differentiating in each (B_c\) gives the unique Euclidean
least-squares minimizer

\[
B_c^*(Q)=\bar V_c=\frac1R\sum_rV_{r,c}.
\]

Moreover,

\[
F(Q,B)=F_{\rm prof}(Q)+R\sum_c\|B_c-\bar V_c\|_F^2,
\]

so analytic substitution loses no solution and introduces no approximation:

\[
F_{\rm prof}(Q)=\sum_{r,c}\|V_{r,c}-\bar V_c\|_F^2.
\]

It also has the exact pairwise-dispersion form

\[
F_{\rm prof}(Q)=\frac1{2R}\sum_{c,r,t}\|V_{r,c}-V_{t,c}\|_F^2.
\]

After fixing one global gauge action to identity, the optimization space is
\(O(d)^{R-1}\). Formulations A and B therefore make the same least-squares
scientific claim. B merely removes templates analytically and replaces nested
alternation with one product-manifold problem.

Synthetic checks confirmed the Pythagorean identity and pairwise identity to
floating-point tolerance. A small direct joint product-manifold reference and
the profiled reference reached objectives (7.08812\times10^{-9}\) and
(7.08868\times10^{-9}\) on a noisy fixture, an absolute difference of
(5.55\times10^{-13}\) attributable to numerical stopping. On an exact fixture
both reached below (10^{-20}\).

## 8. Recommended primary formulation

**Recommendation: pairwise LOCO necessary-condition gate first, followed by
the profiled product-manifold global model only if the pairwise gate passes.**

The original alternating generalized Procrustes implementation should not be
the primary formulation. Its scientific objective is defensible, but its
nested numerical realization was both opaque and pathological.

The pairwise gate is preferable as the first scientific test because it:

- follows directly as a necessary implication of the latent hypothesis;
- retains the same held-out-class falsification;
- uses the already validated single-(O(22)) objective;
- makes every source–target fit independent and checkpointable;
- exposes predictive nonidentifiability at the exact pair where it occurs;
- can reject the common global explanation without estimating nuisance
  population templates.

If it passes, use Formulation B—not the old nested A implementation—to test the
stronger joint latent claim. Add equivalence-aware cross-fold/cycle diagnostics
as coherence checks. Only after global and cross-session support should a
post-action residual analysis be unlocked.

This recommendation changes the testing sequence and implementation, not the
scientific action (U\mapsto QUQ^{\mathsf T}\) or the held-out-class question.

## 9. Why this recommendation is logically conservative

A necessary-condition gate can terminate early only against the specified
global orthogonal-action hypothesis. It cannot reject arbitrary nonlinear,
nonorthogonal, class-dependent, or source-space transforms. Conversely, a
pairwise PASS is deliberately not promoted to a global-action conclusion.

This asymmetry is useful: negative evidence can be obtained cheaply when an
identifiable necessary implication fails, while a positive global claim still
requires the stronger profiled joint model. The sequence therefore reduces
computation without weakening the positive-evidence standard.

## 10. Expected runtime and solver-call counts

BNCI Stage A has (9\times2\times4=72\) target-session-held-out cells, eight
source subjects per cell, and Full/A/B objects for identifiability.

| formulation | optimization unit | full + split source/core work | extra target/semantic work | expected behavior |
|---|---:|---:|---:|---|
| Failed alternating A, actual | one (O(22)) solve, 231 DOF | 216 source fits × 30,240 Pymanopt runs = **6,531,840** runs | additional target fits | observed: no task completed after about 6 h 40 min; source fit failed |
| Alternating A with corrected total-8 count | same nested unit | 216 × 26,880 = **5,806,080** runs | additional target fits | still pathological; not recommended |
| Profiled B, four product starts as a count illustration | one (O(22)^7) solve, 1,617 DOF | 216 × 4 = **864** product optimizer runs | 576 single-action fit objects × 4 = **2,304** runs | synthetic noisy (d=22,R=8): 0.281 s per one product start; determinant design is not yet calibrated for production |
| Pairwise C core, four total starts | one (O(22)) solve, 231 DOF | 576 full pair fits = **2,304** Pymanopt runs; Full/A/B = **6,912** | five full semantic mismatches per pair would raise the complete total to **18,432** runs | synthetic noisy (d=22): 0.194–0.237 s per four-start fit |

The call counts are exact consequences of the candidate grids; wall-time
projections are descriptive synthetic engineering estimates, not protocol
promises. On the audited machine, the pairwise full core is about 2–2.5 minutes
serial from the measured fit time; Full/A/B is about 6–7 minutes serial; the
full core plus five semantic fits is about 15–18 minutes serial. Eight
independent workers ideally divide those times by eight, but a conservative
allowance for process, BLAS, checkpoint, and harder-objective overhead is:

- pairwise full core: roughly 1–3 minutes;
- pairwise Full/A/B diagnostic: roughly 2–6 minutes;
- pairwise Full/A/B plus all five semantic fits: roughly 4–12 minutes.

The profiled noisy product reference took 0.281 s for one 1,617-DOF start and
90 iterations; its exact fixture took one iteration. Four product starts plus
the target fits suggest a low-single-digit-minute Stage-A computation with
parallelism, but this is **not yet an executable estimate** because
\(O(22)^7\) has disconnected product components and its start design has not
been frozen. Another multi-hour run is not justified until a complete product
sector/start benchmark exists.

Pairwise fits are embarrassingly parallel over 576 source–target cells and can
checkpoint independently. A profiled global fit is parallel over 216
LOSO/LOCO/full-or-half fits but not inside its 1,617-DOF product solve. The old
alternating implementation nested optimization inside every outer iteration
and is the least auditable option.

## 11. Exact recommended use of Pymanopt

For every **single-action** pairwise fit, retain the validated scientific loss

\[
L(R)=\sum_{c\in C_{\rm fit}}\|U_{s,c}-RU_{r,c}R^{\mathsf T}\|_F^2
\]

on `Stiefel(d, d, retraction="polar")`, which is full (O(d)) for square
matrices. Use Pymanopt 2.2.1 `ConjugateGradient` with the analytic Euclidean
gradient, manifold projection/retraction/transport, and explicit
`BackTrackingLineSearcher`. The audited candidate numerical settings remain:

- `beta_rule="HestenesStiefel"`;
- maximum iterations 1000;
- projected-gradient tolerance (10^{-5});
- optimizer minimum step (10^{-12});
- maximum cost evaluations 1001;
- maximum time 3600 s per start;
- backtracking initial step 1.0, contraction 0.5, sufficient decrease
  (10^{-4}), optimism 2.0, maximum 50 line-search iterations.

The independent projected-Armijo solver remains a synthetic validation oracle,
not an additional production start. No optimizer call may be nested inside a
global alternating loop in the next primary gate.

For a later profiled global model, Pymanopt can optimize one `Product` of seven
square-Stiefel manifolds after fixing the anchor. The exact profiled gradient
was derived and finite-difference tested. That reference implementation is not
yet production because product determinant-component initialization and
global-fit predictive-identifiability rules require a separate freeze.

## 12. Exact determinant strategy

The scientific space remains (O(d)), not (SO(d)). For a single action,
deterministic initialization must explicitly cover both disconnected
components. Never apply a post-optimization last-axis sign correction.

The recommended synthetic-calibrated single-action design uses four **total**
starts:

1. a sign-resolved spectral start;
2. that start right-multiplied by a fixed reflection;
3. a sign-unresolved spectral start;
4. that start right-multiplied by the same fixed reflection.

Each base/reflection pair occupies opposite determinant components, giving
exactly two starts in each component. There is no hidden warm start. Runtime
assertions require `actual_start_count == configured_total_start_count`.

For the profiled (O(d)^{R-1}\) problem, the disconnected-component structure
has up to (2^{R-1}\) determinant patterns after gauge fixing. Four global
starts cannot be assumed to cover that space. Spectral anchor-relative starts
recovered the synthetic exact/noisy fixtures, but a production product-start
policy must be separately calibrated. This is another reason to make the
single-action pairwise gate primary.

## 13. Synthetic-only multi-start calibration

Four, six, and eight total starts were compared without loading BNCI. Fixtures
included det(+1), det(-1), (d=22), exact and noisy generic noncommuting
templates, a known local-minimum landscape, a continuous-stabilizer case, and
harmless/dangerous held-out stabilizer actions.

For the reproducible noisy-(d=22) benchmark:

| total starts | det truth | wall time per fit | held-out relative error | converged starts |
|---:|---:|---:|---:|---:|
| 4 | +1 | 0.194 s | (1.43\times10^{-4}) | 3/4 |
| 4 | -1 | 0.237 s | (1.56\times10^{-4}) | 3/4 |
| 6 | +1 | 0.308 s | (1.36\times10^{-4}) | 5/6 |
| 6 | -1 | 0.357 s | (1.29\times10^{-4}) | 5/6 |
| 8 | +1 | 0.403 s | (1.33\times10^{-4}) | 8/8 |
| 8 | -1 | 0.614 s | (1.20\times10^{-4}) | 7/8 |

The injected symmetric noise scale was (2\times10^{-4}). Four starts also
recovered exact det± held-out actions to machine tolerance, found the exact
solution in the known local-minimum fixture while retaining non-equivalent
local minima as diagnostics, and correctly treated a continuous stabilizer as
harmless when its held-out prediction was invariant. Six and eight starts did
not change the scientific classification in this calibration.

Therefore **four total starts are the minimum reasonable recommendation for
the next single-action pairwise protocol**, subject to review before freeze.
This finite fixture suite does not prove global optimization. Any later change
to the fixture family or start design must happen before real-data access and
be recorded as a protocol amendment.

Other synthetic validation results:

- generic noncommuting templates: commutator numerical nullity 0;
- shared repeated two-dimensional block: nullity 1;
- commuting distinct diagonal templates: continuous nullity 0 but discrete
  sign stabilizers remain, demonstrating the limit of the Lie test;
- equivalent actions with invariant held-out predictions: harmless;
- equivalent actions with different held-out predictions: predictive
  nonidentifiability;
- exact equivalence-aware cycle: relative discrepancy below (10^{-14});
- intentionally inconsistent direct cycle action: discrepancy above 0.1;
- held-out violation fixture: fit-three relative error below (10^{-8}),
  held-out relative error above 0.30.

## 14. Protocol amendment versus scientific-hypothesis change

The following are transparent **protocol/implementation amendments** while
preserving the hypothesis:

- making pairwise LOCO the necessary-condition first gate;
- using the exact profiled objective for a later global fit;
- removing nested alternating solver invocation;
- reducing and redefining starts as an exact total based only on synthetic
  calibration;
- adding pairwise/cross-fold/cycle induced-action diagnostics;
- adding per-fit checkpoints, timing, and exact call-count assertions;
- re-freezing aggregation, nulls, identifiability thresholds, and terminal
  logic before any new BNCI run.

The following would change the **scientific hypothesis** and require a new
experiment rather than an amendment:

- replacing (Q\in O(22)) by a general linear or nonlinear transform;
- allowing a different action for every class in the primary model;
- rotating `svec(U)` in 253 dimensions;
- changing the squared-Frobenius identity-tangent objective to AIRM merely to
  match another implementation;
- adding learned low-rank, PCA, neural-network, or target-adaptive structure;
- selecting classes, subjects, thresholds, or starts from BNCI outcomes.

## Audit stop

The formulation derivation, synthetic validation, call-count audit, and
runtime benchmark are complete. The recommended next step is human review,
then a separately committed protocol amendment if accepted. **No real BNCI
Stage A/B/C run is authorized or launched by this memo.**
