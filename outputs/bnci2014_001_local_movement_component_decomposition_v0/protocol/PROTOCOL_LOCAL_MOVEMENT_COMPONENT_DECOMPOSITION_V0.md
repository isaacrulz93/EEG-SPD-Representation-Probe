# Protocol: Local Ordered Movement Component Decomposition V0

Frozen 2026-08-12 (Asia/Seoul) before any component statistic from the real
BNCI2014_001 movement objects is viewed. Result-contingent adaptation is
forbidden.

## Scope and question

This is a strict squared-cost decomposition of the finalized Local Mean
Covariance Movement V0 result. It does not define a new trajectory,
covariance feature, window, AIRM reference, classifier, network, adaptation,
GPA, subject-action model, or population/class/subject ANOVA. It never refits
AIRM means, raw covariance matrices, the anti-development, `M1`, or the
ordered reverse-prefix parallel transports.

The primary question is whether the common-O(22)-invariant,
directional/joint-matrix component of the frozen ordered AIRM displacement
sequence retains a reproducible subject×class relational interaction after
the ordered speed-profile contribution is removed exactly. The secondary
localization question concerns the additional common-frame,
simultaneous-conjugation-sensitive component.

## Immutable lineage and namespace

- Authoritative parent branch:
  `pilot/local-mean-movement-antidevelopment-v0`.
- Exact authoritative parent HEAD:
  `12c19f38266bc76875cffae056e7f9403df299c1`.
- Movement V0 protocol freeze:
  `e24312147ef3020854ef6f6cd174071d1c6ead02`.
- Movement V0 scientific result:
  `c3f1d5ff9cf23db2007bbf839cf4b266e2cb8960`.
- Movement V0 terminal:
  `GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT`.
- Component branch:
  `pilot/local-movement-component-decomposition-v0`.
- Output namespace:
  `outputs/bnci2014_001_local_movement_component_decomposition_v0/`.

The original repository worktree and cancelled
`pilot/local-ordered-airm-movement-v0` work are out of scope and must remain
untouched.

## Exact frozen inputs

The following files must be byte-identical and exactly array-equal to their
versions in the Movement V0 scientific-result commit. Their SHA-256 hashes are
frozen here:

| artifact | SHA-256 |
| --- | --- |
| `arrays/full_ordered_antidevelopment.npz` | `f3771773a194088e84d765b543ba50db95fc9571ab85838aa43a581eca32a2d3` |
| `arrays/split_half_ordered_antidevelopment.npz` | `ab64bfdb279805cf5d0e0b9bc12c65eccd1c602ee14e3084f7429e9588e98520` |
| `arrays/cross_session_movement_matrices.npz` | `f3470799a2a532d98f9406902cc7bb75f9385f899991ade350d63e3ca78ef5dc` |
| `nulls/movement_null_distributions.npz` | `954fbaaf761157332a67dfebcc8085ae1a4a8c38b2921340e1226a37e30b2fa1` |
| `tables/d_mov_matrix.csv` | `be1abec7e3081df6ffc8ac919ce12ceb6b94f527d2371984e23a8c92315ca64e` |
| `tables/d_len_matrix.csv` | `b57ad89bf6225df5728135424ab48848cb142bbae9aba0bb58cdab84a46f1e2f` |
| `tables/d_direct_matrix.csv` | `1d4ea0b4692565018f33f2889623e8bf372244788d9ff8c45987c442cbf4501c` |

All paths above are relative to
`outputs/bnci2014_001_local_mean_movement_v0/`. The full tuple bank has shape
`(2,9,4,4,22,22)` and the split bank has shape
`(2 halves,2 sessions,9,4,4,22,22)`. The exact canonical cell order is subject
1 through 9, with class order
`[left_hand,right_hand,feet,tongue]` within subject. Session 0 anchors rows and
session 1 supplies columns. The fixed discretization remains five windows of
0.8 seconds and four temporally ordered adjacent displacements.

If reproduction fails, terminate
`UNASSESSED_COMPONENT_DECOMPOSITION_TECHNICAL_FAILURE` without scientific
interpretation.

## Squared-cost mathematical object

For frozen tuples `A=(A1,...,A4)` and `B=(B1,...,B4)`, with every matrix in
`Sym(22)`, define only squared costs:

`c_sensor(A,B)=(1/4) sum_i ||Ai-Bi||_F^2=d_direct(A,B)^2`.

`c_full(A,B)=min_{Q in O(22)} (1/4) sum_i ||Ai-QBiQ^T||_F^2=d_mov(A,B)^2`.

One nuisance `Q` is shared by all four fixed transitions. There is no temporal
permutation, DTW, reassignment, or scientific interpretation of `Q`. The
primary full matrix is the square of the exact saved `d_mov` matrix; it is not
refit. Deterministic quotient refits on full data are permitted only for
internal reproduction of the four predeclared fixed diagnostic pairs, and
their objectives must match the saved `c_full` values within the frozen
tolerance.

Let `ai=||Ai||_F` and `bi=||Bi||_F`. Define
`c_len(A,B)=(1/4) sum_i (ai-bi)^2=d_len(A,B)^2`.

The exact components are

- `c_ang=c_full-c_len`;
- `c_ori=c_sensor-c_full`;
- hence `c_sensor=c_len+c_ang+c_ori` and `c_full=c_len+c_ang`.

`c_len` is ordered movement magnitude/speed-profile mismatch. `c_ang` is the
length-weighted, common-O(22)-invariant directional/joint-matrix mismatch. It
is not one geometric angle. `c_ori` is the mismatch removed by a common
sequence-level orthogonal conjugation and is described only as common-frame or
orientation-sensitive; it is not subject, neural, or anatomical pose.

## Numerical gates

The combined tolerance is fixed at `atol=1e-8`, `rtol=1e-8`. For every one of
the 1,296 full cross-session pairs require raw `c_len>=0`, `c_ang>=0`, and
`c_ori>=0` within the combined tolerance. The negative bound for a difference
component is `atol + rtol*max(abs(minuend),abs(subtrahend))` pair by pair.
Meaningful negatives fail. Tolerance-scale negative values remain raw in all
diagnostics and inferential arithmetic; no scientific array is clipped.

Require raw pairwise reconstruction, saved-root squaring, and independent
recalculation of `c_len` and `c_sensor` from the frozen Z tuples. The exact
saved root matrices and readable CSVs must agree. A failure terminates
`UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE` before interpretation.

Split-half `c_full` uses the exact frozen Movement V0 TrustRegions definition:
square Stiefel/O(22), polar retraction, six deterministic starts, three starts
in each determinant sector, analytic objective/gradient/Hessian, at most 250
iterations, gradient tolerance `1e-6`, minimum step `1e-12`, at most 5,000
cost evaluations, at most 120 seconds per start, and both determinant sectors
certified for each fit. Four workers are fixed. Split quotient failure is a
component numerical failure.

## Relation-cell statistics and exact reconstruction

For every squared-cost component `x` in
`{len,ang,ori,full,sensor}` and anchor `(s,c)`, define:

- `a_sc^x=cost_x[(s,c),(s,c)]`;
- `b_sc^x=mean_{k!=c} cost_x[(s,c),(s,k)]`;
- `c_sc^x=mean_{t!=s} cost_x[(s,c),(t,c)]`;
- `d_sc^x=mean_{t!=s,k!=c} cost_x[(s,c),(t,k)]`.

Smaller cost means greater similarity. Define
`S_sc^x=c_sc^x-a_sc^x`, `C_sc^x=b_sc^x-a_sc^x`, and the relational
difference-in-differences contrast
`J_sc^x=b_sc^x+c_sc^x-a_sc^x-d_sc^x`. `J` is not interaction variance,
random-effect variance, or residual norm.

Average classes within subject to obtain `S_s^x`, `C_s^x`, `J_s^x`, then
average subjects to obtain `T_subject^x`, `T_class^x`, and `T_J^x`.

Because these are linear means and contrasts, require `full=len+ang` and
`sensor=len+ang+ori` for every anchor-level S/C/J, every subject-level S/C/J,
and every group T. The same identities are additionally audited for every
indexed null draw. Failure is
`UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE`.

## Exact paired null mappings

Use exactly 1,999 draws and one-sided plus-one p-values
`(1+count(T_null>=T_observed))/2000`.

The subject-break stream is exactly
`default_rng(SeedSequence([20260810,1102]))`. Within each class it permutes
session-1 subject identities and moves complete frozen four-step tuples.

The class-break stream is exactly
`default_rng(SeedSequence([20260810,1101]))`. Within each session-1 subject it
permutes the four class identities and moves complete frozen tuples.

The explicit mappings saved by Movement V0 are preferred and must equal exact
regeneration. Draw `r` uses the same mapping for `len`, `ang`, `ori`, `full`,
and `sensor`. No geometry or optimizer runs inside null draws. Alpha is 0.05.

## Primary, secondary, and speed tests

The primary statistic is `T_J^ang`. Angular support requires it to be positive
and both its subject-break and class-break p-values to be below 0.05. If
supported, the allowed claim is: “After exactly removing the ordered
speed-profile contribution, the common-O(22)-invariant
directional/joint-matrix component of ordered mean-covariance movement retains
reproducible subject×class interaction.” It is never neural, physiological, or
source-space direction.

The secondary statistic is `T_J^ori`, with the same two-null positive support
rule. Support says only that a common-frame/simultaneous-conjugation-sensitive
component contributes additional interaction evidence. It does not establish
a reusable subject pose and cannot alone establish the primary directional
claim.

The speed statistic is the squared-cost-scale `T_J^len`, with the same two
nulls. No previous root-distance statistic is subtracted, and no inference of
unique direction is made from a comparison of significance states.

## Descriptive contributions and step localization

For every pair with safely nonzero `c_sensor>1e-12`, save
`c_len/c_sensor`, `c_ang/c_sensor`, and `c_ori/c_sensor`. Summarize them
descriptively by the four relation categories. They are not inferential
percentages.

The four fixed comparisons are S1 Left0 to S1 Left1, S2 Left1, S1 Feet1, and
S2 Feet1. Report their exact costs and fractions.

For transitions `1->2`, `2->3`, `3->4`, and `4->5`, report step-norm
distributions over all 72 full tuples and the distribution of per-step
`c_len` contributions over all 1,296 pairs. Reproduce the frozen common-O
objective only for the four fixed pairs and report each per-step
`(1/4)[||Ai-QBiQ^T||_F^2-(ai-bi)^2]` contribution. No best-transition test is
created and step diagnostics cannot change the terminal.

## Prespecified split-half stability

Use only `split_half_ordered_antidevelopment.npz`. Replicate A compares
session-0 Half A to session-1 Half A. Replicate B compares session-0 Half B to
session-1 Half B. Never mix halves or refit covariances.

For each replicate compute all 36×36 `c_sensor`, `c_len`, and newly optimized
`c_full` matrices, derive `c_ang` and `c_ori`, and report `T_J^ang` plus all
nine `J_s^ang`. Define
`split_half_ang_sign_stable=(T_J_ang_A>0) AND (T_J_ang_B>0)`. There is no
half-specific p-value threshold. If full angular support passes but this sign
rule fails, terminate `UNASSESSED_COMPONENT_DECOMPOSITION_UNRELIABLE`.

## Terminal hierarchy

First use exactly these failure terminals:

1. `UNASSESSED_COMPONENT_DECOMPOSITION_TECHNICAL_FAILURE` for frozen
   artifact or lineage failure.
2. `UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE` for nonnegativity,
   reconstruction, root reproduction, or quotient certification failure.
3. `UNASSESSED_COMPONENT_DECOMPOSITION_UNRELIABLE` when full angular support
   passes but split-half angular sign stability fails.

Otherwise define angular, orientation, and length support by their respective
positive `T_J` and two p-values below 0.05, and choose exactly one:

1. `BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS` if angular and
   orientation support pass.
2. `GO_DIRECTIONAL_JOINT_MOVEMENT_INTERACTION` if angular support alone among
   those two passes.
3. `SENSOR_FRAME_ORIENTATION_ONLY_SUPPORT` if orientation support alone
   passes.
4. `SPEED_PROFILE_SUFFICIENT_AT_CURRENT_RESOLUTION` if neither angular nor
   orientation support passes and speed support passes.
5. `NO_COMPONENT_INTERACTION_AT_SQUARED_COST_RESOLUTION` otherwise.

Only angular support votes positively for a future Paper-A directional claim.

## Claim restrictions

The object is always qualified as BNCI2014_001 window-wise mean covariance
movement, ordered AIRM displacement sequence, discrete anti-development, or
mean covariance movement pattern under the fixed 5 × 0.8-s discretization.

Never claim unique neural direction, physiological movement direction, motor
strategy direction, source-space direction, stable subject/neural/anatomical
pose, continuous-time dynamics, causal transitions, or biological privilege
of AIRM. If angular support fails, state only that current interaction evidence
does not require an additional directional/joint-matrix component beyond
ordered speed at the current resolution. If orientation support passes, state
only that a common-frame-sensitive component provides additional interaction
evidence.

In short, neither support pattern licenses a claim of neural, physiological, or source-space direction.

## Execution and post-result immutability

The prepare phase may verify only lineage, artifact bytes/hashes/arrays,
schemas, canonical ordering, and null mappings. It cannot square the real root
matrices, derive components, or view component statistics. Focused component
tests must pass, and the full repository suite must be executed and recorded,
before scientific execution.

The scientific run requires a clean worktree whose HEAD exactly equals the
supplied protocol-freeze SHA. After the full 1,296-pair component matrices are
first observed, cost squaring, components, tolerance, pair set, relation-cell
averaging, null mappings/seeds/count, alpha, split-half rule, optimizer,
terminal logic, and claims are immutable. No rescue normalization,
standardization, regression residualization, direction normalization,
matched-length subset, alternate windowing, or other post-result analysis is
allowed.

All focused component implementation and numerical tests must pass. The full
repository suite must also be executed before and after the scientific run.
Because this isolated worktree is forbidden from reusing the original
worktree's ignored caches, an unrelated legacy test that requires such a cache
may be reported as an external-artifact availability failure; it cannot excuse
any failure in the component-focused suite or any test whose inputs are
committed in this worktree.

Post-result operations may only record already-run tests, write the frozen
outputs/report/provenance, insert the scientific-result commit SHA, refresh
artifact hashes, push, and open a Draft PR. No automatic merge is authorized.
