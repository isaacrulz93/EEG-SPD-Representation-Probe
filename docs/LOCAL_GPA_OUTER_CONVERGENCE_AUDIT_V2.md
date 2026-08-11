# Local GPA Outer-Loop Convergence Audit V2

## 1. Scope, provenance, and frozen contract

The frozen Local GPA Consensus V1 ended at `UNASSESSED_GPA_NUMERICAL_FAILURE_V1` before any Stage-2A scientific statistic was computed. This audit asks only whether its higher-level constrained-GPA failure was a 24-outer-iteration budget problem or a failure to reach the unchanged alternating-loop convergence gates within a bounded continuation to 96 iterations.

- Audit branch: `audit/local-gpa-outer-convergence-v2`
- Immutable V1 final/base: `122eacff868aa8f656ad6360716c1816f453979f`
- V1 protocol freeze: `e6e8887036d1e04a2da7db36d84db58d8c80d9ef`
- V1 numerical-failure result: `acfe94d805293119092413401e9261853273a1e8`
- Audit-definition commit: `69c7b0e34721bba4d1093638d0fff9c1e6882f0e`

The input representation, trial-local centering, quotient space, GPA objective and prototype constraint, TrustRegions registration solver/HVP, fixed-action certification, exact `S5` handling, both deterministic GPA starts, prototype inner limit 16, line search, and both convergence tolerances remained unchanged. The diagnostic continuation resumed the same in-memory trajectory after outer 24 and stopped no later than outer 96. It did not create a consensus checkpoint or compute a between-cell distance, `J`, null, or p-value.

Before reading any comparative trajectory, the audit fixed:

- prototype line-search initial-step candidates `{1.0, 2.0, 4.0}`;
- default convergence gates: projected prototype gradient `<= 2e-5` and relative objective change `<= 1e-7`;
- common-budget milestones 48, 64, and 96;
- trajectory-classification rules;
- the outcome-blind bank selection and the rule that the bank would unlock only if both default-step exact-cell starts converged stably by 96.

The line-search quantity above is an **initial Armijo proposal step**, not a fixed learning rate. The contraction factor remained 0.5 and all other prototype settings were unchanged.

## 2. Exact failed task and deterministic reproduction

The first task in the exact frozen V1 execution order was reproduced:

| field | value |
|---|---|
| subject | 1 |
| session | `0train` |
| class | `left_hand` |
| task/split | `Full` |
| frozen task index | 0 |
| trials | 72 |
| cell-configuration SHA-256 | `1f775f813c0e634327519322a5e1e8db7f34f91a034b737a6ef327283ea4914b` |
| start-0 initial trial position | 0 |
| start-0 prototype SHA-256 | `42fbff415a91b8fec4941c827338c4e5057dbd8df0e783c617fbd72662eed8f7` |
| start-1 initial trial position | 36 |
| start-1 prototype SHA-256 | `89750d4e8ad81acad0f8dcbe32d36fdb996b8d243637607c4208864584e8f8c9` |

With the frozen initial step 1.0, start 0 again failed to satisfy both gates at outer 24. This reproduces the V1 failure. Start 1 was then run independently for the audit; production V1 had failed closed before reaching it.

## 3. Frozen 24-outer trace and bounded continuation

### Start 0

| outer | joint objective | projected gradient | relative change | consecutive prototype quotient distance | changed `S5` registrations |
|---:|---:|---:|---:|---:|---:|
| 24 | 3.199558554 | 1.23394e-5 | 3.21618e-4 | 2.23360e-2 | 0 |
| 48 | 3.177787553 | 1.44749e-5 | 1.55602e-4 | 1.54135e-2 | 0 |
| 64 | 3.171386296 | 1.75624e-5 | 8.53465e-5 | 1.09844e-2 | 0 |
| 96 | 3.167805412 | 1.64774e-5 | 2.95105e-6 | 2.10110e-3 | 0 |

The gradient gate passed at outer 96, but the relative-change gate remained about 29.5 times its `1e-7` threshold. The normalized objective decrease over the final five blocks was `1.48252e-5`, above the preregistered `1e-5` strong-descent boundary. The trace classification is therefore `STILL_DESCENDING_STRONGLY`, not a 24-to-96 converged trajectory.

### Start 1

| outer | joint objective | projected gradient | relative change | consecutive prototype quotient distance | changed `S5` registrations |
|---:|---:|---:|---:|---:|---:|
| 24 | 3.159841800 | 1.43735e-5 | 4.44398e-4 | 2.59954e-2 | 0 |
| 48 | 3.144050477 | 1.52844e-5 | 6.29437e-5 | 9.58513e-3 | 0 |
| 64 | 3.142148320 | 1.23566e-5 | 1.36770e-5 | 4.51929e-3 | 0 |
| 96 | 3.139726160 | 1.59324e-5 | 7.73100e-6 | 3.42734e-3 | 0 |

The gradient gate also passed here, but relative change remained about 77.3 times threshold. The normalized objective decrease over the final five blocks was `3.39524e-5`. This trace is also `STILL_DESCENDING_STRONGLY`.

Neither start first satisfied both frozen gates at any outer iteration through 96. The objectives decreased monotonically at every recorded outer transition. Starting with the first comparable block, no trial registration changed its `S5` permutation in either default trace; all final eight blocks likewise had zero changes. Constraint residuals remained at approximately `2e-14` to `3e-14`. Thus the observed behavior was not assignment/prototype oscillation, line-search failure, objective increase, or numerical blow-up. It was slow continued descent of the stable-assignment alternating loop without attaining the frozen relative-change gate.

## 4. Bounded initial-step audit

All six exact-cell trajectories used the same initialization and settings except for the preregistered Armijo proposal step.

| initial proposal | GPA start | objective at 24 | objective at 96 | gradient at 96 | relative change at 96 | total backtracking reductions | first convergence | runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1.0 | 0 | 3.199558554 | 3.167805412 | 1.64774e-5 | 2.95105e-6 | 0 | none | 1112.72 |
| 1.0 | 1 | 3.159841800 | 3.139726160 | 1.59324e-5 | 7.73100e-6 | 0 | none | 1123.30 |
| 2.0 | 0 | 3.199554623 | 3.167804319 | 1.09413e-5 | 2.91052e-6 | 0 | none | 1050.73 |
| 2.0 | 1 | 3.158996721 | 3.140295173 | 6.00275e-6 | 2.87297e-5 | 0 | none | 1058.49 |
| 4.0 | 0 | 3.199144284 | 3.171255185 | 4.98365e-6 | 1.39617e-5 | 81 | none | 1116.90 |
| 4.0 | 1 | 3.163794930 | 3.138932160 | 1.01382e-4 | 8.74108e-5 | 7 | none | 1128.46 |

Initial 1.0 accepted 1.0 without a backtracking reduction; initial 2.0 likewise accepted 2.0 without reduction. Initial 4.0 predominantly accepted 4.0, with 81 and 7 reductions to 2.0 for starts 0 and 1 respectively. Zero-valued final inner steps represent the already-satisfied inner gradient condition and were not counted as backtracking.

The larger proposals therefore did **not** merely reproduce the 1.0 accepted-step sequence, so the special `INITIAL_STEP_NOT_THE_BOTTLENECK` branch of the preregistered rule is not invoked. More importantly, none reached the unchanged outer gates by 96; 4.0/start 1 also remained above the gradient gate. The outcome is:

`NO_INITIAL_STEP_AMENDMENT_SUPPORTED`

There is no `INITIAL_STEP_AMENDMENT_CANDIDATE`. No candidate was selected using its outer-24 objective.

## 5. Outcome-blind bank and common outer budget

The four-cell bank was deterministically fixed before trajectory inspection (task indices 21, 29, 157, and 182), but its execution prerequisite failed: both default-step exact-cell starts had to converge stably by 96. Running the bank after that failure could not rescue the required exact-cell condition and was prohibited by the preregistered audit flow.

Accordingly:

- bank status: `NOT_UNLOCKED_BY_EXACT_DEFAULT_TRAJECTORY`;
- converged by 24/48/64/96 for both exact starts: no/no/no/no;
- supported common outer budget: none;
- recommended V2 budget amendment: none.

## 6. Audit decision

**`GPA_ALTERNATING_FORMULATION_NUMERICALLY_UNSTABLE`**

This label follows the frozen decision rule because both required exact-cell starts failed to converge by the audit cap of 96. The descriptive traces are stable-assignment, monotone, slowly descending trajectories rather than oscillatory ones; nevertheless, no smallest defensible common finite budget up to 96 can be frozen. Therefore `RECOMMEND_GPA_OUTER_BUDGET_ONLY_V2_AMENDMENT` is not justified.

The current constrained quotient-GPA alternating implementation is not numerically qualified for a Stage-2A BNCI assessment under the unchanged contract. This audit does not propose tolerance relaxation, fewer starts, a different prototype optimizer or parameterization, or an automatic further amendment.

The scientific question remains unassessed. No Stage-2A consensus output, cross-session matrix, `T_J`, `J_s`, `J_sc`, correspondence null, or p-value was computed, viewed, or inferred.

## 7. Runtime and machine-readable evidence

The complete six-trajectory audit took `6593.88 s` (`1 h 49 min 53.88 s`). Individual trajectories took `1050.73–1128.46 s`.

Machine-readable evidence is stored under `outputs/bnci2014_001_local_gpa_outer_convergence_audit_v2/`:

- `protocol/failed_task_identity.json`;
- `protocol/audit_bank_selection.json`;
- `protocol/prototype_initial_step_candidates.json`;
- `protocol/trajectory_classification_contract.json`;
- `protocol/audit_provenance.json`;
- `tables/outer_convergence_summary.csv`;
- six per-candidate/start outer-trace CSV and summary JSON files;
- `traces/outer_loop_traces.csv` and `traces/outer_loop_traces.npz`;
- `decisions/audit_decision.json`.

The CSV traces contain every outer block's objective, aligned-registration summary, projected gradient, relative change, prototype inner iterations, accepted proposal steps, backtracking counts, final line-search step, changed permutations, consecutive-prototype quotient distance, constraint residual, and convergence flag.

## 8. Verification and repository state

- Focused pre-execution audit tests: `20 passed`.
- Full post-audit repository suite: `230 passed in 33.19 s`.
- Machine consistency check: six summaries, 576 outer-trace rows, and 24 NPZ arrays; all six trajectories were unconverged by outer 96 and all recorded zero changed permutations.
- `git diff --check`: PASS.
- Final handoff working tree: clean after the audit-result commit.
- Stage-2A scientific rerun: not performed.
