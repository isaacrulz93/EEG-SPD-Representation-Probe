# Protocol: Local GPA Consensus V1 Optimizer-Only Amendment

## Status and immutable history

This document incorporates the complete scientific protocol in `PROTOCOL_LOCAL_GPA_CONSENSUS_V0.md` by reference and freezes one numerical amendment for V1. It is a **post-technical-failure optimizer-only numerical amendment**. It is not a scientific redesign.

The immutable V0 lineage is:

- scientific base: `796f04e7970972175a660a521caff47c83e0295f`;
- V0 protocol freeze: `ad3dd429a69e7cea77e3d4421987989ab45171b9`;
- V0 numerical-failure result: `9aba8a8cee1c84d9b69b7fdf3d53d570927a7cc6`;
- V0 final: `abc4b8127da174709060c82507329bc8c5137069`.

V0 stopped at `UNASSESSED_GPA_NUMERICAL_FAILURE` before a complete cell consensus, between-cell matrix, `T_J`, subject statistic, correspondence null, or p-value existed. The exact first replayed registration certified det(+1), but both det(−1) ConjugateGradient starts missed the unchanged `1e-6` gradient gate. The failure is numerical and is not a negative scientific result.

The optimizer audit was defined at `a87563bd3788754172c0b73699a81a4bb10fb232` and finalized at `d846650ea9d4ead205b580d004205327ac1fd3fd`. It used known-Q synthetic registrations, the exact failed registration, and an outcome-blind real registration bank. No Stage-2A scientific result informed this amendment.

## The only amendment

The fixed-action optimizer changes from Pymanopt 2.2.1 `ConjugateGradient(beta_rule="HestenesStiefel")` to Pymanopt 2.2.1 `TrustRegions`.

The action remains on `Stiefel(d,d,retraction="polar")`, which represents both disconnected determinant components of `O(d)`. For BNCI, `d=22`. Exactly four total deterministic action starts remain required: the two frozen S5/spectral basins, each paired with its fixed first-axis reflection, yielding exactly two initial actions in det(+1) and two in det(−1). There is no hidden fifth start, determinant correction, tolerance relaxation, or component deletion.

For a fixed S5 permutation `pi`, the scientific cost remains exactly

\[
L(Q)=\frac{1}{5}\sum_{i=1}^{5}d_{\mathrm{AIRM}}^2
\left(X_i,QY_{\pi(i)}Q^T\right),\qquad Q\in O(22).
\]

The fixed 120-permutation S5 assignment, action/assignment alternation, and both-component certification are unchanged.

## Audited TrustRegions implementation

TrustRegions minimizes the exact cost above with its exact analytic Euclidean gradient. Its Hessian-vector input is the audited central finite difference of the exact Riemannian gradient, not a surrogate objective:

1. normalize the requested tangent direction;
2. retract from the current action by `+1e-5` and `-1e-5` with the same polar retraction;
3. evaluate the exact analytic Riemannian gradient at both points;
4. transport both gradients to the original tangent space;
5. take the centered difference, rescale by the original tangent norm, and project to the base tangent space.

The frozen TrustRegions settings are:

- `miniter=3`;
- `kappa=0.1`;
- `theta=1.0`;
- `rho_prime=0.1`;
- `use_rand=False`;
- `rho_regularization=1000.0`;
- `mininner=1`, `maxinner=30`;
- `max_iterations=250`;
- `max_time=120 seconds`;
- `min_gradient_norm=1e-6`;
- `min_step_size=1e-12`;
- `max_cost_evaluations=5000`.

The protocol-level acceptance condition remains a recomputed Riemannian gradient norm no greater than `1e-6`. No library stopping message alone certifies a start.

The audit found Hessian self-adjoint relative error `1.02e-9` and directional second-difference relative error `3.14e-7`. TrustRegions passed 4/4 starts on the failed V0 registration, 12/12 known-Q synthetic starts, and 16/16 outcome-blind real-bank starts: 32/32 total. The corresponding frozen CG run certified det(+1) but failed both det(−1) starts on the failed registration. These engineering results justify only the optimizer substitution.

## Unchanged scientific contract

The following remain byte/field-equivalent to V0 except for versioned provenance and output namespace:

- BNCI2014_001, all 9 subjects, sessions `0train` and `1test`, four MI classes, and 5,184 trials;
- the frozen 22-channel, 250 Hz, 8–32 Hz, 0–3.996 s, five non-overlapping 200-sample, float64 pyRiemann-OAS covariance representation;
- exact frozen input hashes and the machine-precision reproduction gate;
- trial-only AIRM Fréchet centering and no subject/session centering or temporal identity;
- the scientific quotient object `SPD(22)^5/(O(22)×S5)`;
- the constrained prototype parameterization `P_i=exp(Z_i)`, `sum_i Z_i=0`;
- two GPA starts, 24 outer iterations, the frozen prototype update, convergence limits, first/final four-start registration, and intermediate one-basin continuation;
- all 72 full cells and 144 run-blocked split-half cells, with A=`{0,2,4}` and B=`{1,3,5}`;
- the 36×36 cross-session quotient-consensus matrix and session-role symmetrization;
- `J_sc=b_sc+c_sc-a_sc-d_sc`, four-class `J_s`, nine-subject `T_J`, and supporting `S`/`C` contrasts;
- 1,999 class-break and 1,999 subject-break correspondence nulls, seed `20260810`, plus-one one-sided p-values, and strict alpha `0.05`;
- the GO/STOP scientific decisions and all claim restrictions.

Registration actions and permutations remain nuisance/debug variables. They are never averaged, compared across cells, or interpreted as subject pose, neural orientation, or anatomy.

## Machine-checked amendment scope

Before the real V1 run, the executable must compare the V0 and V1 configs and write a machine-readable scope report. Dataset, input reproduction, local centering, quotient object, all `gpa.settings`, split halves, interaction, nulls, scientific GO/STOP terminals, and runtime grid must match exactly. Allowed differences are limited to protocol/version provenance, output/cache namespace, implementation file list, optimizer identity, the audited HVP/TrustRegions fields, and the versioned numerical-failure label. Any other difference terminates as `STOP_AMENDMENT_SCOPE_VIOLATION`.

## Pre-run gates

Before the V1 freeze and before any Stage-2A scientific statistic, the following must pass:

1. immutable WINDOW5 input/hash reproduction;
2. independent local centering of all 5,184 trials with no gate failure;
3. known-Q O(d) registration in both determinant sectors;
4. known S5 relabeling;
5. known-answer constrained GPA;
6. frozen quotient symmetry tolerance;
7. exact V0 failed registration replay with TrustRegions certification in both determinant sectors;
8. exact four-start assertions and full repository tests.

The V1 cache key includes the V1 config hash, V1 implementation hash, optimizer identity, and V1 protocol-freeze SHA. V0 cell checkpoints or scientific caches cannot be used.

## Scientific execution and terminal logic

Only after the clean V1 freeze commit may the frozen 72 full-cell and 144 split-half consensuses be estimated. If any required TrustRegions registration fails the unchanged certification, execution stops at `UNASSESSED_GPA_NUMERICAL_FAILURE_V1`; no tolerance, budget, starts, determinant requirement, assignment, or optimizer may be changed or automatically rerun.

If numerically assessed:

- `GO_STABLE_SUBJECT_CLASS_QUOTIENT_MEAN_CONFIGURATION` requires `T_J>0`, `p_classbreak<0.05`, and `p_subjectbreak<0.05`;
- otherwise the terminal is `STOP_NO_STABLE_QUOTIENT_MEAN_CONFIGURATION_INTERACTION`.

Split-half reliability is descriptive and non-gating. A positive result concerns a pose-quotiented cell mean configuration orbit, not a mean pose. A negative result does not reinterpret the V0 numerical failure or alter the immutable Stage-1 result.

After the first V1 scientific output, every scientific definition above is immutable. Any blocking problem is preserved and reported without patch-and-rerun.
