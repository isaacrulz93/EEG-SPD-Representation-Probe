# Protocol: Pairwise Common Subject Action V2

Status: **pre-result optimizer-only amendment**. This document must be
committed before any V2 BNCI scientific run. It supplements, and does not
rewrite, the frozen pairwise protocol in
`docs/PROTOCOL_AMENDMENT_PAIRWISE_COMMON_ACTION_V1.md`.

## 1. Amendment history and reason

The following history is immutable:

- original pairwise freeze:
  `27eaf074ac25acba1afc0f8f37671283267b16c9`;
- V1 technical-failure result:
  `83c3e1bddc40b433ca5494902da1ec6c64a06db5`;
- synthetic optimizer-audit precommit:
  `6e9f08a7879dae893c9d005f039e0ac20037f48a`;
- optimizer-audit result:
  `53aab81598c4d92e2f860385980cf9637a7fc380`.

V1 remains `UNASSESSED_TECHNICAL_FAILURE`. Its real run stopped before any
target-level Stage-A task completed because both det(+1) ConjugateGradient
starts in the first incomplete fit failed the frozen component-wise
convergence requirement. No Stage-A target statistic, null, p-value, semantic
control, or Stage-B result existed.

The preregistered d=22 synthetic-only forensic audit compared the unchanged
action objective using identical deterministic starts:

| optimizer | converged starts | both-sector certified fixtures |
|---|---:|---:|
| ConjugateGradient | 42/48 | 10/12 |
| TrustRegions | 48/48 | 12/12 |
| SteepestDescent | 37/48 | 10/12 |

No synthetic false-failure fixture had a stable induced action while merely
missing the gradient tolerance. This supports changing the numerical
optimizer to the already-tested TrustRegions implementation. It does not
support relaxing any tolerance.

**This is not a post-result scientific adjustment. No Stage-A target-level
scientific statistic, null, or p-value existed before this amendment.**

## 2. Exactly one changed implementation choice

Changed:

> Pymanopt `ConjugateGradient` is replaced by Pymanopt `TrustRegions` for the
> single-action fit.

Unchanged:

- hypothesis and sensor-space action;
- squared-Frobenius loss;
- (O(22)), including both determinant components;
- all four deterministic starts and their order;
- preprocessing, U construction, sessions, subjects, and classes;
- source/target pair grid and all four LOCO folds;
- Full/A/B split-half construction;
- baseline, normalization, error, and gain;
- source aggregation, target-subject inferential unit, and group statistic;
- null definitions, seeds, 1999 replicates, p-value, and thresholds;
- semantic-control and Stage-B unlock gates;
- stabilizer analysis and predictive-identifiability criterion;
- terminal logic and claim restrictions.

The failed nested generalized-Procrustes estimator remains forbidden.

## 3. Frozen scientific object

For target subject (s), source subject (r\ne s), session (q), and held-out
class (c^*), fit

\[
\widehat R_{s\leftarrow r,q}^{(-c^*)}
=\arg\min_{R\in O(22)}
\sum_{c\ne c^*}
\|U_{s,q,c}-R U_{r,q,c}R^{\mathsf T}\|_F^2.
\]

The held-out prediction is

\[
P_{s\leftarrow r,q,c^*}
=\widehat R U_{r,q,c^*}\widehat R^{\mathsf T}.
\]

This is a necessary-consequence falsification of a global latent common-action
model. Pairwise success alone does not prove that global latent
\(\{Q_s,B_c\}\) objects exist.

## 4. Exact audited TrustRegions runtime

Use Pymanopt 2.2.1 with
`Stiefel(22, 22, retraction="polar")`, which is full (O(22)). The problem
supplies the analytic Euclidean gradient and analytic Euclidean Hessian-vector
product of the declared squared loss. Pymanopt performs the Riemannian
gradient/Hessian conversion, tangent operations, polar retraction, and
trust-region updates.

Instantiate `TrustRegions` exactly as in the accepted synthetic audit:

- `miniter=3`;
- `kappa=0.1`;
- `theta=1.0`;
- `rho_prime=0.1`;
- `use_rand=False`;
- `rho_regularization=1000.0`;
- `max_iterations=1000`;
- `min_gradient_norm=1e-5`;
- `min_step_size=1e-12`;
- `max_cost_evaluations=1001`;
- `max_time=3600` seconds;
- `verbosity=0`;
- `log_verbosity=0`.

The audit called `TrustRegions.run()` without overriding its run parameters.
For `Stiefel(22,22)`, Pymanopt 2.2.1 therefore resolves the formerly implicit
defaults as:

- `mininner=1`;
- `maxinner=manifold.dim=231`;
- `Delta_bar=manifold.typical_dist=sqrt(22)=4.69041575982343`;
- `Delta0=Delta_bar/8=0.5863019699779287`.

The installed implementation shrinks the trust radius when the model ratio is
below 0.25, expands it when above 0.75 and the inner solver hits the boundary,
uses shrink factor 0.25 and expansion factor 2, and accepts a model-decreasing
proposal only when `rho > rho_prime`. These are library-internal behaviors,
not tuned V2 choices.

The analytic Euclidean Hessian-vector product for direction (H) is

\[
DG(Q)[H]=4\sum_i\left[
(HB_iQ^T+QB_iH^T)QB_i+(QB_iQ^T-A_i)HB_i
\right].
\]

Finite-difference agreement is a required unit test.

## 5. Convergence and determinant certification

The accepted-solution numerical requirement remains

\[
\|\operatorname{grad}L(R)\|\le10^{-5}.
\]

No objective-only, step-only, or post-hoc rescue is introduced. Pymanopt may
internally stop for its other standard limits, but a returned action is not a
certified candidate unless its independently recomputed Riemannian gradient
meets (10^{-5}).

Use exactly four total starts, unchanged from V1:

1. spectral, sign resolved;
2. start 1 right-multiplied by
   (F=\operatorname{diag}(-1,1,\ldots,1));
3. spectral, sign unresolved;
4. start 3 right-multiplied by (F).

Every deterministic or spectral initialization counts toward the four. There
is no hidden fifth or warm start. Runtime assertions require exactly two
initial det(+1), two initial det(-1), and exactly four optimizer results.
Determinant must remain in its initial disconnected component; post-hoc SVD
sign correction is forbidden.

At least one finite, orthogonal, gradient-certified candidate is required in
each determinant component. Otherwise V2 stops at
`UNASSESSED_TECHNICAL_FAILURE`; the available component cannot vote alone.

## 6. Predictive identifiability remains frozen

Among converged solutions retain

\[
R_{eq}=\{R_j:L_j\le L_{best}+10^{-10}+10^{-8}|L_{best}|\}.
\]

The source fit matrices are analyzed through the stacked skew-commutator
operator. Machine-rank nullity and the frozen relative (10^{-8}) approximate
nullity are recorded. Fixed stabilizer probe angles are
(+\pi/2,-\pi/2,\pi).

For held-out predictions (P_j=R_jU_{r,q,c^*}R_j^T), define

\[
D_{eq}=\max_{j,k}\frac{\|P_j-P_k\|_F}{N_P},
\qquad
D_{split}=\frac{\|P_A-P_B\|_F}{N_P},
\]

where (N_P=\max(\|P_{best}\|_F,\epsilon_{mach}\sqrt{22^2})). The cell is
predictively identifiable iff

\[
D_{eq}\le\max(10^{-5},D_{split}).
\]

Different raw (R) matrices are harmless when their held-out predictions
agree. Any required nonidentifiable pair fails closed as
`UNASSESSED_ACTION_NOT_IDENTIFIABLE`. No available-case dropping is allowed.

## 7. Prediction, aggregation, and frozen inference

For every pairwise cell,

\[
e_{raw}=\frac{\|U_{target,held}-U_{source,held}\|_F^2}
{\|U_{target,held}\|_F^2},
\quad
e_R=\frac{\|U_{target,held}-P_{held}\|_F^2}
{\|U_{target,held}\|_F^2},
\quad g=e_{raw}-e_R.
\]

The required Stage-A grid is (9\times8\times2\times4=576) ordered pairs.
First take the median over eight sources for each target/session/class, then
the median over the eight cells for each target, then the median over nine
target subjects. The target subject is the inferential unit.

The primary unrelated-target null, master seed 20260810, 1999 replicates,
one-sided greater-or-equal plus-one p-value, PASS rule, conditional five-way
semantic mismatch, Stage-B unlock, cross-session logic, and terminal mapping
are byte-for-byte scientific carryovers from V1.

## 8. Pre-run synthetic reproduction gate

Before V2 BNCI execution, the exact production TrustRegions path must pass:

- 48/48 d=22 stress starts gradient-certified;
- 12/12 fixtures certified in both determinant sectors;
- det(+1) and det(-1) exact recovery;
- noisy identifiable recovery;
- harmless nonuniqueness classification;
- predictive-nonidentifiability classification;
- stabilizer diagnostics;
- exact four-start assertions;
- Hessian finite differences and all U-independent unit tests;
- the full repository test suite.

For the preregistered 12-fixture stress grid, the exact-case held-out relative
error must be at most (10^{-8}) and every controlled-noise/ill-conditioned
case must be at most (5\times10^{-3}). These are synthetic recovery gates,
not BNCI scientific thresholds.

These checks must pass before the amendment commit. No BNCI scientific score
may be computed before that commit.

## 9. Fresh V2 execution and provenance

Write only under
`outputs/bnci2014_001_pairwise_common_action_v2/`. V1 checkpoints and
scientific caches are forbidden. V1 artifacts remain provenance only.

Every V2 checkpoint identity contains:

- frozen V2 config SHA-256;
- combined V2 source SHA-256;
- `pymanopt_2.2.1_stiefel_trust_regions` optimizer identity;
- clean V2 amendment commit SHA at run start.

A mismatch is a hard error, not a resumable cache hit. Output provenance also
records all four immutable history commits and the protocol/config hashes.

After the clean amendment commit, run from scratch:

1. exact U reproduction gate;
2. Stage-A Full/A/B pairwise LOCO;
3. predictive-identifiability gate;
4. target-subject aggregation and unrelated-target inference;
5. semantic comparator only if Stage-A primary unlocks it;
6. Stage B and cycle diagnostic only if Stage-A final passes.

If TrustRegions has any component-wise technical failure, stop and preserve
`UNASSESSED_TECHNICAL_FAILURE`. Do not create V3 automatically.

## 10. Claims and post-access lock

No scientific setting may change after V2 real-data access. Even success does
not establish physiology, a complete model of individuality, source-space
structure, causal dynamics, unlabeled identifiability, or a performance
improvement. Pairwise success is not proof of the global latent model. The
planned next scientific branch remains split-epoch covariance-set anatomy.
