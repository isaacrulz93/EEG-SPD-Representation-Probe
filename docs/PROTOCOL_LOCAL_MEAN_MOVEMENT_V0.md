# Protocol: Local Mean Covariance Movement V0

Status: pre-result frozen protocol, 2026-08-12 (Asia/Seoul).

## Scientific scope and lineage

This experiment asks whether ordered adjacent AIRM movement of the window-wise
mean covariance trajectory is cross-session reproducible and subject×class
specific after removing absolute starting location. It studies the four fixed
transitions `M1→M2`, `M2→M3`, `M3→M4`, and `M4→M5`. It is not GPA, unordered
shape, an S4/S5 quotient, point reassignment, DTW, a classifier, adaptation, or
continuous-time dynamics.

The branch is `pilot/local-mean-movement-antidevelopment-v0`. Its parent is the
finalized `pilot/local-temporal-sequence-correspondence-v0` tip
`6d5ad6a0bdd4f2d19bfee8ce6fcbb97a5c499a5d`. The frozen temporal protocol is
`70981aa89ddbadceca42f354c3c51d05bf6dbf0c`; its scientific artifact commit is
`43e926073fab0ba76fd5baa881804538f0d7beee`. All new artifacts are confined to
`outputs/bnci2014_001_local_mean_movement_v0/`; previous outputs are immutable.

## Exact frozen input and reproduction

Load, without refitting, the 72 full cell-level AIRM Fréchet mean sequences
from `ordered_mean_sequences.npz` and the 144 independently fitted run-blocked
half sequences from `split_half_mean_sequences.npz` in the temporal V0 output.
The full artifact SHA-256 is
`e03b94daef3eb37f9209ee7a7482ea575b1eb353804505a3e91013339da1913f`;
the split artifact SHA-256 is
`355f098de7ff3dcf274e5a62cf6d92022bc1b1f6ed4301a2dc53f9a19f3cd868`.
Compare the working files and every NPZ array against the blobs at the temporal
scientific-result commit. Require exact equality for all 360 full and 720 split
mean matrices and exact metadata order: subjects 1–9, sessions `0train` and
`1test`, classes `left_hand`, `right_hand`, `feet`, `tongue`, temporal positions
1–5, and halves `A`, `B`. A failure terminates as
`UNASSESSED_TECHNICAL_FAILURE`; no scientifically different mean is fitted.

The object is movement of window-wise cell-level mean covariances. It is not an
average of trial-level tangent velocities.

## Time interval, AIRM logarithm, and local cross-check

Freeze `Delta_t=0.8` seconds. For step `i=1,…,4`,

`V_i = Log_{M_i}(M_{i+1}) / Delta_t`,

where

`Log_P(Q)=P^(1/2) log(P^(-1/2) Q P^(-1/2)) P^(1/2)`.

Also compute only as a mathematical cross-check

`U_i=log(M_i^(-1/2) M_{i+1} M_i^(-1/2))/Delta_t`.

Require `||V_i||_{M_i}=||U_i||_F=d_AIRM(M_i,M_{i+1})/Delta_t`. The `U_i`
sequence is not compared scientifically because its symmetric-whitening gauge
changes at every step.

## Ordered reverse-prefix parallel transport and anti-development

Use standard AIRM Levi-Civita transport along each unique AIRM geodesic. For
`P→Q`, use `PT(S)=E S E^T`, where the principal
`E=(Q P^(-1))^(1/2)` is evaluated as

`E=P^(1/2)(P^(-1/2) Q P^(-1/2))^(1/2)P^(-1/2)`.

Transport along the actual ordered reverse prefix:

- `W_1=V_1`;
- `W_2=PT_{M2→M1}(V_2)`;
- `W_3=PT_{M2→M1}∘PT_{M3→M2}(V_3)`;
- `W_4=PT_{M2→M1}∘PT_{M3→M2}∘PT_{M4→M3}(V_4)`.

Independent radial transport to `M1` is forbidden. Map to identity coordinates
with `Z_i=M1^(-1/2) W_i M1^(-1/2)`. The primary cell representation is the
ordered tuple `(Z_1,Z_2,Z_3,Z_4)` in `Sym(22)^4`, called an *ordered discrete
AIRM anti-development* or *ordered parallel-transported AIRM displacement
sequence*.

## Mathematical gates and frozen tolerances

Before any BNCI movement discrepancy is computed, synthetic d=22 checks must
pass. For every real full and split sequence, every `Z_i` must be symmetric,
every edgewise parallel transport must preserve AIRM norm, and the `V/U/Z`
norm identity must hold. Frozen maxima are: relative symmetry `1e-10`, absolute
norm error `1e-8`, and edgewise transport relative error `1e-8`.

For a generic common congruence `M_i'=A M_i A^T`, compute

`O=(A M1 A^T)^(-1/2) A M1^(1/2)`.

Require orthogonality and `Z_i'=O Z_i O^T` for the same `O` at all four steps,
with Frobenius relative tolerance `1e-8`. Failure of log, transport, or
anti-development gates terminates as
`UNASSESSED_MOVEMENT_GEOMETRY_NUMERICAL_FAILURE`.

## Primary common-O(22) movement discrepancy

For ordered tuples `A` and `B`,

`d_mov(A,B)^2=min_{Q∈O(22)} (1/4) sum_i ||Z_i^A-Q Z_i^B Q^T||_F^2`.

One common `Q` is shared by all steps. Step identity is fixed. No S4
permutation, DTW, or point reassignment is allowed. `Q` is a nuisance quotient
variable and is never interpreted as a physical, subject, or intrinsic pose.

The exact new Frobenius objective, analytic ambient gradient, and analytic
ambient Hessian are implemented separately from the previous AIRM point-loss.
Optimization uses pymanopt 2.2.1 TrustRegions on square Stiefel/O(d), polar
retraction, the audited trust-region constants `miniter=3`, `kappa=0.1`,
`theta=1`, `rho_prime=0.1`, `use_rand=false`, `rho_regularization=1000`, at
most 250 iterations, gradient tolerance `1e-6`, minimum step `1e-12`, at most
5,000 cost evaluations, and at most 120 seconds per start.

Use exactly six deterministic starts: a joint-spectral start, a deterministic
nearby spectral perturbation, and a pair-hash-seeded Haar start, each mirrored
by a fixed reflection. This gives exactly three starts in each determinant
component. The best converged objective is used, but every required fit must
have at least one certified candidate in both determinant sectors. Canonically
order each tuple pair by a SHA-256 of its float64 contents before optimization,
which makes forward/reverse evaluation exactly identical.

Pre-data d=22 certification requires: known common-Q distance at most `1e-8`,
known-Q action recovery up to the unavoidable global sign at most `1e-8`, both
determinant sectors covered, objective spread between the spectral and nearby
spectral starts in the true sector at most `1e-8`, exact forward/reverse
distance equality, and tangent directional finite-difference agreement of the
analytic gradient with absolute tolerance `2e-6` and relative tolerance
`2e-5`. Failure terminates as
`UNASSESSED_MOVEMENT_QUOTIENT_NUMERICAL_FAILURE`.

## Cross-session matrices and frozen controls

Index session cells by subject then the frozen class order. Compute and save the
complete `36×36` matrix

`M_mov[(s,c),(t,k)]=d_mov(A(s,0,c),A(t,1,k))`.

The magnitude-only control uses the ordered speed profile
`ell_i=||Z_i||_F=d_AIRM(M_i,M_{i+1})/Delta_t` and

`d_len=sqrt((1/4) sum_i (ell_i^A-ell_i^B)^2)`.

The direct montage-registered control is

`d_direct=sqrt((1/4) sum_i ||Z_i^A-Z_i^B||_F^2)`.

Temporal indices remain fixed for both controls. Controls receive the same
statistics and relabeling maps and cannot change the primary terminal.

## Subject, class, and interaction statistics

For each session-0 anchor `(s,c)`, define `a_sc` as same subject/same class,
`b_sc` as the mean same-subject/different-class distance, `c_sc` as the mean
different-subject/same-class distance, and `d_sc` as the mean
different-subject/different-class distance.

- `S_sc=c_sc-a_sc`, `S_s=mean_c S_sc`, `T_subject=mean_s S_s`;
- `C_sc=b_sc-a_sc`, `C_s=mean_c C_sc`, `T_class=mean_s C_s`;
- `J_sc=b_sc+c_sc-a_sc-d_sc`, `J_s=mean_c J_sc`, `T_J=mean_s J_s`.

Use exactly 1,999 inherited subject-break mappings from
`default_rng(SeedSequence([20260810,1102]))`: permute session-1 subjects within
class and move complete four-step tuples. Use exactly 1,999 inherited
class-break mappings from `default_rng(SeedSequence([20260810,1101]))`: permute
four complete class tuples within each session-1 subject. Do not recompute
anti-developments or distances inside null draws. All p-values are one-sided
plus-one values with null values greater than or equal to observed counted.
Alpha is `0.05`.

Subject support requires `T_subject>0` and `p_subject<0.05`. Class support
requires `T_class>0` and `p_class<0.05`. Interaction support requires `T_J>0`,
`p_J_subjectbreak<0.05`, and `p_J_classbreak<0.05`; subject and class support
alone never establish interaction.

## Split-half movement reliability

Use the already saved independent temporal V0 halves: Half A runs `{0,2,4}`
and Half B runs `{1,3,5}`. Anti-develop each half sequence independently using
its own `M1`, then compute the common-O quotient distance between halves for
every subject×session×class cell. Save and report all 72 distances. This is
non-gating and receives no post-hoc threshold.

## Descriptive visualization and fixed comparisons

Apply Frobenius-isometric `svec` to all `72×4` full `Z_i` matrices and fit one
global two-component PCA with full deterministic SVD. Plot ordered `Z1→Z2→Z3→Z4`
paths for Subject 1 Left, Subject 1 Feet, Subject 2 Left, and Subject 2 Feet in
both sessions. Also save the d_mov heatmap, subject contrast bars, split-half
reliability, and all speed profiles. Inference remains in full `Sym(22)^4`.

Predeclare fixed table comparisons anchored at session-0 Subject 1 Left:
session-1 S1 Left, S2 Left, S1 Feet, and S2 Feet, for all three discrepancies.

## Terminal hierarchy

Use exactly one primary terminal:

1. `GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT` when subject support, class
   support, and interaction support all pass.
2. `GO_REPRODUCIBLE_ORDERED_MOVEMENT_WITHOUT_INTERACTION` when both subject and
   class support pass but the interaction criterion fails.
3. `STOP_NO_REPRODUCIBLE_ORDERED_MOVEMENT_V0` otherwise, including partial
   specificity or interaction without both prespecified specificity supports.
4. `UNASSESSED_MOVEMENT_GEOMETRY_NUMERICAL_FAILURE` for log/PT/anti-development
   failure.
5. `UNASSESSED_MOVEMENT_QUOTIENT_NUMERICAL_FAILURE` for common-O optimization
   failure.
6. `UNASSESSED_TECHNICAL_FAILURE` for another blocking failure.

## Claim boundaries and prior results

If the full terminal passes, the allowed claim is: “After removing the initial
SPD location and one common sequence-level orthogonal gauge, the ordered
displacement pattern of the window-wise mean covariance trajectory is
cross-session reproducible and subject×class-specific.” If only the direct
interaction passes, say the information is concentrated in the
montage-registered sequence orientation removed by the quotient. If the length
control passes, say magnitude contributes information; do not infer a unique
directional contribution without a separate paired contribution test.

Do not claim individual-trial neural velocity, continuous-time dynamics,
causal or stable physiological transitions, source-space dynamics, physical
sensor orientation, absolute subject pose, biological privilege of AIRM, or
completeness of five windows. Preferred nouns are *window-wise mean covariance
trajectory*, *ordered AIRM displacement sequence*, *discrete anti-development*,
and *mean covariance movement pattern*.

The prior unordered local metric analysis supported subject and class
specificity but not explicit interaction. The ordered raw mean-sequence
analysis supported temporal correspondence, subject specificity, class
specificity, and interaction. This experiment asks whether that ordered result
survives after absolute placement is replaced by adjacent relative movement;
it is not a rescue of the unordered result.

## Execution freeze and post-result immutability

The `prepare` phase may only verify parent artifacts and run synthetic gates.
Focused and full repository tests must pass and their exact results are saved.
Commit the protocol, config, implementation, tests, and pre-data artifacts
before execution. The scientific `run` requires a clean worktree whose HEAD is
exactly the supplied protocol-freeze SHA.

After the first complete cross-session movement matrix is observed, reference
`M1`, reverse-prefix path convention, `Delta_t`, quotient, optimizer, controls,
statistics, null mappings, alpha, split halves, visualization basis, terminals,
and claim restrictions are immutable. No rescue analysis is allowed. Result
finalization may only insert the scientific-result commit SHA into the report,
write post-result provenance, and refresh artifact hashes.
