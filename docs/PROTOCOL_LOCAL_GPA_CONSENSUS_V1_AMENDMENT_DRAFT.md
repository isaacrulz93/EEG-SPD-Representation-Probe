# Draft: Local GPA Consensus V1 Optimizer-Only Amendment

## Status

This document is a **prepared amendment specification only**. It is not a protocol freeze and does not authorize a Stage-2A scientific rerun.

This is a post-technical-failure numerical amendment after real BNCI data entered registration, not a pre-data amendment. The immutable V0 protocol freeze is `ad3dd429a69e7cea77e3d4421987989ab45171b9`; V0 remains `UNASSESSED_GPA_NUMERICAL_FAILURE` and must be cited in every V1 report.

## Proposed single amendment

Replace only the single-action numerical optimizer:

- V0: Pymanopt 2.2.1 ConjugateGradient with Hestenes–Stiefel update;
- V1 candidate: Pymanopt 2.2.1 TrustRegions on `Stiefel(d,d,retraction="polar")`.

The TrustRegions candidate uses:

- exact unchanged squared-AIRM registration objective;
- exact unchanged analytic Euclidean/Riemannian gradient;
- central finite-difference Riemannian Hessian-vector operator using polar retraction, tangent transport, and radius `1e-5`;
- `miniter=3`, `kappa=0.1`, `theta=1.0`, `rho_prime=0.1`, `use_rand=False`, `rho_regularization=1000`;
- `max_iterations=250`, `max_time=120 s`, gradient certification `1e-6`;
- `mininner=1`, `maxinner=30`;
- exactly the same four total deterministic starts and both determinant sectors;
- no hidden warm start and no post-hoc determinant correction.

The existing 250-iteration and 120-second budgets are retained. The audit maximum was 37 TrustRegions outer iterations and 0.792 seconds per start, so no budget expansion is proposed.

## Explicitly unchanged

V1 may not change:

- EEG representation, covariance estimation, local centering, or frozen input hashes;
- `SPD(22)^5/(O(22)×S5)` scientific object;
- constrained quotient-GPA definition or prototype constraint;
- AIRM registration objective;
- exact 120-permutation S5 update or action/assignment alternation;
- four-start construction or both-component certification;
- `1e-6` gradient threshold;
- GPA outer/prototype settings;
- full/split cell grid and run-blocked halves;
- `M01`, symmetrization, `J`, supporting contrasts, subject aggregation;
- 1,999-draw correspondence nulls, p-values, alpha, terminal rules, or claim restrictions.

## Required gates before any V1 run

1. Freeze a versioned V1 protocol/config with the exact settings above.
2. Re-run the existing known-Q, determinant-component, S5, quotient-symmetry, constrained-GPA, and full repository tests.
3. Assert the V0 output tree remains byte-immutable.
4. Use a new V1 output and cache namespace; do not reuse V0 cell-fit checkpoints.
5. Record the V0 failure and this post-failure optimizer audit in V1 provenance.
6. Only after a separate reviewed freeze commit may V1 real science begin.

If TrustRegions fails a required registration in V1, stop and preserve that failure. Do not relax tolerance, remove a determinant component, or create another amendment automatically.
