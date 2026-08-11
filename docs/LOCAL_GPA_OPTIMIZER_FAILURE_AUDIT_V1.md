# Local GPA Optimizer Failure Audit V1

## 1. Scope and immutable scientific contract

The frozen Stage-2A V0 ended correctly at `UNASSESSED_GPA_NUMERICAL_FAILURE`. This audit does not reinterpret that outcome as a scientific negative and does not rerun Stage-2A science.

The audit branch is `audit/local-gpa-registration-failure-v1`, based on V0 final commit `abc4b8127da174709060c82507329bc8c5137069`. The scientific base (`796f04e7970972175a660a521caff47c83e0295f`), V0 freeze (`ad3dd429a69e7cea77e3d4421987989ab45171b9`), failure result (`9aba8a8cee1c84d9b69b7fdf3d53d570927a7cc6`), and all V0 artifacts remain byte-immutable.

The following were held exactly fixed:

- frozen EEG WINDOW5 covariances and trial-local AIRM centering;
- the constrained quotient-GPA object in `SPD(22)^5/(O(22)×S5)`;
- the exact mean-five squared-AIRM registration loss;
- all 120 `S5` assignment evaluation and the frozen assignment/action alternation;
- the four deterministic initial actions/permutations, with two starts in each determinant component;
- the requirement for at least one converged candidate in each determinant component;
- the Riemannian gradient certification threshold `1e-6`;
- all split-half, J, null, and terminal scientific definitions.

No `T_J`, correspondence null, `J_s`, `J_sc`, or other Stage-2A scientific statistic was computed.

## 2. What the original V0 artifact did and did not record

The original failure artifact records the exception `required registration lacks converged candidates in both determinant sectors`, elapsed time, and frozen provenance. It does **not** contain the cell identity, start identity, action trajectory, or per-start result. Those unavailable historical details were not reconstructed and presented as if they had been recorded.

V0 used eight parallel workers. Consequently, the artifact cannot identify which worker raised first. Deterministic replay in frozen input order showed that the first `Full` task and the immediately following half-A task begin with the **same target/source registration** and hashes. Thus the exact registration object is identified even though the original artifact cannot distinguish which of those two parallel task wrappers surfaced the exception first.

All detailed traces below are explicitly from the deterministic forensic reproduction, not from the original artifact.

## 3. Exact failed registration identity

The first frozen task and first sequential registration reproduce the V0 exception:

| field | value |
|---|---|
| subject | 1 |
| session | `0train` |
| class | `left_hand` |
| split used for deterministic replay | `Full` |
| equivalent parallel task sharing the same first registration | half A |
| GPA start | 0 |
| GPA outer iteration | 1 |
| trial position | 0 |
| global sample index | 3 |
| trial UID | `S01_0train_T004` |
| registration phase | initial full four-start certification |
| target/prototype SHA-256 | `42fbff415a91b8fec4941c827338c4e5057dbd8df0e783c617fbd72662eed8f7` |
| source configuration SHA-256 | `324e9aad5de12568fc062f30b528805d24587a1a67884a541cae390024c86a61` |

The target is the exact constrained-prototype parameterization of this same locally centered source trial. Their maximum entrywise difference is `1.0119327598090422e-10`; the small difference comes from enforcing the frozen zero-sum-log prototype constraint.

## 4. Frozen ConjugateGradient four-start forensic

The frozen Pymanopt 2.2.1 `ConjugateGradient(beta_rule="HestenesStiefel")`, polar square-Stiefel manifold, explicit backtracking line search, 250 iterations, and `1e-6` certification were replayed without changing any numerical update. Observation-only logging captured accepted iteration objectives; it does not change the optimizer path.

| start | det | initial π | initial L | final π | final L | final grad | grad / 1e-6 | iterations | counted cost calls | result |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| 0 | +1 | 0,1,2,3,4 | 1.487806e-16 | 0,1,2,3,4 | 1.487806e-16 | 2.311387e-08 | 0.023 | 1 | 1 | converged |
| 1 | −1 | 0,1,2,3,4 | 1.783441 | 0,1,2,3,4 | 0.097178557 | 1.888339e-05 | 18.883 | 250 | 655 | max iterations |
| 2 | −1 | 2,1,0,3,4 | 7.488083 | 2,1,0,3,4 | 2.891878794 | 1.260047e-01 | 126,004.7 | 250 | 624 | max iterations |
| 3 | +1 | 2,1,0,3,4 | 7.818089 | 2,1,0,3,4 | 2.846108723 | 9.190093e-04 | 919.0 | 250 | 638 | max iterations |

All four registrations remained in their initial determinant component and retained their initial point assignment. There was one action solve/assignment alternation per start.

The failed certification component was **det(−1)**. Both det(−1) starts failed:

- Start 1 reached essentially the same objective later reached by TrustRegions, but its final gradient remained 18.883 times the frozen tolerance. Its accepted-iterate objective decreased by only `1.89e-9` over the final 10 iterations and `7.57e-8` over the final 50. Under the audit's declared descriptive rule this trace was plateaued with a nonzero gradient.
- Start 2 remained far from stationarity: its final gradient was approximately 126,005 times tolerance and its objective decreased by `0.00434` over the final 10 and `0.01593` over the final 50 iterations. It was still descending when the 250-iteration budget ended.

Start 3 also failed, but the det(+1) component was nevertheless certified by start 0. Start 3 was still descending descriptively at termination. No line-search accepted/rejected-step history existed in the V0 artifact; the machine-readable forensic trace contains accepted CG iteration objectives and gradients, counted cost calls, and the solve-level stopping result, but does not invent historical line-search decisions.

Therefore the V0 failure mechanism was not absence of a component-wise minimizer. It was failure of frozen Hestenes–Stiefel CG to meet the unchanged gradient certification in the det(−1) component within 250 iterations.

## 5. TrustRegions implementation and exact objective

TrustRegions used the same `Stiefel(22,22,retraction="polar")`, target/source matrices, four initial Q matrices, initial permutations, exact AIRM loss, exact analytic gradient, 120-permutation update, and alternation logic.

The exact second derivative of the AIRM matrix-log composition was not already available in the V0 implementation. The audit therefore supplied Pymanopt with a central finite-difference **Riemannian Hessian-vector operator**, not a surrogate loss:

1. normalize the requested tangent direction;
2. move by `±1e-5` using the same polar retraction;
3. evaluate the exact analytic Riemannian gradients of the exact AIRM loss;
4. transport both gradients back to the original tangent space;
5. take the central difference and rescale by the tangent norm.

The Hessian-vector audit gave:

- self-adjointness relative error: `1.0228e-9`;
- directional second-difference relative error: `3.1364e-7`.

The TrustRegions audit settings were the installed Pymanopt defaults `miniter=3`, `kappa=0.1`, `theta=1`, `rho_prime=0.1`, `use_rand=False`, and `rho_regularization=1000`, with outer limit 250, time limit 120 seconds, gradient threshold `1e-6`, and at most 30 inner truncated-CG iterations. The scientific loss was unchanged.

## 6. TrustRegions four-start result on the failed registration

| start | det | initial π | initial L | final π | final L | final grad | grad / 1e-6 | outer iterations | counted cost calls | Hessian-vector calls | result |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | +1 | 0,1,2,3,4 | 1.487806e-16 | 0,1,2,3,4 | 7.696520e-20 | 7.215221e-12 | 0.000007 | 1 | 2 | 30 | converged |
| 1 | −1 | 0,1,2,3,4 | 1.783441 | 0,1,2,3,4 | 0.097178555 | 1.044350e-07 | 0.104 | 14 | 15 | 219 | converged |
| 2 | −1 | 2,1,0,3,4 | 7.488083 | 2,1,0,3,4 | 2.820092633 | 1.533517e-07 | 0.153 | 18 | 19 | 215 | converged |
| 3 | +1 | 2,1,0,3,4 | 7.818089 | 2,1,0,3,4 | 2.846107973 | 7.971123e-07 | 0.797 | 29 | 30 | 398 | converged |

TrustRegions converged 4/4 starts and certified both determinant components. Pymanopt TrustRegions does not report a `cost_evaluations` field in its result object, so the table uses independently counted exact-cost calls and labels them accordingly.

## 7. Small preregistered audit bank

The additional real registrations were selected before either solver was run on them. A NumPy `SeedSequence([20260811, 0x47504131])` chose four distinct frozen `Full` cell tasks and one nonzero within-cell source-trial position. The target was always the feasible constrained prototype from trial position 0, at GPA start 0, outer iteration 1. Selection did not use objective, convergence, or any scientific outcome.

The bank contained:

- the exact failed real registration;
- three existing frozen known-Q fixtures: d=4 det(−1), d=4 det(+1), and d=22 det(−1);
- four outcome-blind deterministic real registrations.

| fixture group | solver | certified registrations | converged starts | total starts | total runtime (s) | median registration runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| failed real | CG | 0/1 | 1 | 4 | 1.654 | 1.654 |
| failed real | TrustRegions | 1/1 | 4 | 4 | 1.180 | 1.180 |
| known-Q synthetic | CG | 3/3 | 12 | 12 | 1.081 | 0.140 |
| known-Q synthetic | TrustRegions | 3/3 | 12 | 12 | 0.829 | 0.078 |
| deterministic real bank | CG | 0/4 | 0 | 16 | 8.755 | 2.190 |
| deterministic real bank | TrustRegions | 4/4 | 16 | 16 | 8.450 | 2.103 |

Across all 32 audited starts, CG converged 13/32 and TrustRegions converged 32/32. All 16 CG starts in the additional real bank stopped at the 250-iteration limit, whereas TrustRegions certified both determinant components for every real registration.

TrustRegions used a median 18.5 outer iterations, maximum 37, and maximum per-start runtime 0.792 seconds. Its aggregate audited start runtime (`10.350 s`) was slightly lower than CG (`11.379 s`) despite the finite-difference Hessian-vector calculations.

## 8. Numerical budget recommendation

No tolerance relaxation is supported. The gradient threshold remains exactly `1e-6`; all determinant components and all four starts remain mandatory.

The audited TrustRegions maximum was 37 outer iterations, well below the existing 250-iteration numerical budget. Therefore the proposed V1 amendment does **not** need a larger outer-iteration or time budget:

- `max_iterations = 250`;
- `max_time = 120 s` per action solve;
- `maxinner = 30` truncated-CG iterations;
- Hessian-vector finite-difference radius `1e-5`;
- all other TrustRegions parameters exactly as audited above.

The `maxinner` and finite-difference radius are TrustRegions implementation parameters and must be frozen in a V1 amendment before any scientific rerun.

## 9. Audit decision

**`RECOMMEND_GPA_OPTIMIZER_ONLY_V1_AMENDMENT`**

TrustRegions passed all existing known-Q gates, certified both determinant components on the exact failed registration, certified 4/4 outcome-blind additional real registrations, preserved the exact AIRM scientific objective, and satisfied the unchanged `1e-6` gradient contract.

This supports preparing an optimizer-only amendment. It does not complete or preview Stage-2A science. The V0 outcome remains immutable and unassessed.

## 10. Machine-readable artifacts

- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/protocol/failure_identity.json`
- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/protocol/audit_bank_selection.json`
- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/protocol/trust_regions_hessian_validation.json`
- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/tables/registration_summary.csv`
- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/tables/four_start_forensic.csv`
- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/tables/alternation_solve_trace.csv`
- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/tables/cg_iteration_trace.csv`
- `outputs/bnci2014_001_local_gpa_optimizer_audit_v1/decisions/audit_decision.json`
