# Local GPA Consensus V1

## Outcome first

**Terminal: `UNASSESSED_GPA_NUMERICAL_FAILURE_V1`.**

V1 did not produce a scientific Stage-2A answer. The audited TrustRegions substitution passed every pre-run gate, including both determinant sectors of the exact V0 failed fixed-action registration. During the frozen real run, however, a required **constrained quotient-GPA outer loop** stopped because GPA start 0 did not converge within the unchanged 24 outer iterations. The process failed closed before one full cell consensus was checkpointed. No `T_J`, `J_s`, `J_sc`, correspondence null, or p-value exists.

This is not `STOP_NO_STABLE_QUOTIENT_MEAN_CONFIGURATION_INTERACTION`; the scientific hypothesis remains unassessed.

## 1–5. Branch and provenance

- Branch: `pilot/local-gpa-consensus-v1`
- Scientific base: `796f04e7970972175a660a521caff47c83e0295f`
- Immutable V0 protocol freeze: `ad3dd429a69e7cea77e3d4421987989ab45171b9`
- Immutable V0 failure: `9aba8a8cee1c84d9b69b7fdf3d53d570927a7cc6` (`UNASSESSED_GPA_NUMERICAL_FAILURE`)
- Immutable V0 final: `abc4b8127da174709060c82507329bc8c5137069`
- Optimizer-audit definition: `a87563bd3788754172c0b73699a81a4bb10fb232`
- Optimizer-audit final: `d846650ea9d4ead205b580d004205327ac1fd3fd`
- V1 amendment freeze: `e6e8887036d1e04a2da7db36d84db58d8c80d9ef`
- V1 numerical-failure result: `acfe94d805293119092413401e9261853273a1e8`
- Final provenance/report SHA: pending until this report commit is created.

The machine-readable V0→V1 comparison is `protocol/amendment_scope_diff.json`. Dataset, input hashes, local centering, quotient space, all GPA scientific settings, exact four starts, determinant certification, S5 assignment, action/permutation alternation, split halves, `J`, supporting contrasts, null mappings/count/seed, alpha, and GO/STOP rules were equal. The permitted differences were limited to:

1. fixed-action optimizer: CG/Hestenes–Stiefel → audited TrustRegions/HVP;
2. optimizer/HVP provenance fields;
3. V1 protocol/output/cache namespace;
4. the versioned numerical-failure label.

## 6. Frozen input reproduction

The immutable BNCI WINDOW5 objects reproduced before the run:

- 5,184 trials;
- session-0 state SHA-256 `c75044f48552f12ad088306b505b074e930f396fdcb544307fff394717e2ca86`;
- session-1 state SHA-256 `1afc8cd52d82310a05857d1ffa67859427c4c9aa1302897a140ebda64d0442f8`;
- frozen AIRM-distance content SHA-256 `681d8a075eff1218e5e2b2d0e292631ead67badaccc00ec075ba428c9d5aed64`;
- maximum distance reproduction difference: `0`, tolerance `1e-12`;
- all metadata/run-block contracts: PASS.

All 5,184 trial-local AIRM centerings passed. Maximum normalized Karcher residual was `8.539076980603567e-11`, below the frozen `1e-7` gate.

## 7. TrustRegions pre-run gates

- Exact V0 failed registration hashes matched the optimizer audit.
- TrustRegions starts converged 4/4.
- Converged determinant components: `{-1,+1}`.
- Per-start final gradients: `7.22e-12`, `1.04e-7`, `1.53e-7`, `7.97e-7`, all below `1e-6`.
- Known-Q O(d) registrations in det(−1) and det(+1): PASS.
- Exact S5 relabeling: PASS.
- Known-answer constrained GPA: PASS.
- Quotient symmetry: PASS at the frozen tolerance.
- Exact four-total-start/two-per-sector contract: PASS.
- Pre-run full repository suite: `225 passed`.

The amendment justification remained the frozen audit result: Hessian self-adjoint relative error `1.02e-9`, directional second-difference relative error `3.14e-7`, exact failed registration 4/4 PASS, known-Q 12/12 PASS, outcome-blind real bank 16/16 PASS, total 32/32 PASS.

## 8–10. Cell completion and numerical diagnostics

- Full-cell consensuses completed/checkpointed: **0/72**.
- Split-half consensuses completed/checkpointed: **0/144**.
- Combined full/split cell tasks completed/checkpointed: **0/216**.
- Scientific run elapsed time before fail-closed termination: `321.10 s`.
- Failure: `GPA start 0 did not converge in frozen outer iterations`.
- Failure level: constrained GPA outer loop, not the audited fixed-action TrustRegions solver.
- Fixed GPA outer budget: 24, unchanged from V0.
- Automatic rerun: none.

The parallel worker exception did not serialize the cell identity or outer-loop objective history before propagating. Those unavailable details are not reconstructed or presented as recorded diagnostics. The exact worker traceback is preserved in `decisions/technical_failure.json`.

## 11. Split-half reliability

Not computed because the required consensus bank was incomplete. Reliability remains diagnostic/non-gating, but there are no V1 reliability distances to report.

## 12–16. Primary and supporting scientific statistics

- `T_J`: not computed.
- `p_classbreak`: not computed.
- `p_subjectbreak`: not computed.
- Conservative p-value: not computed.
- All 9 `J_s`: unavailable; none were computed.
- All 36 `J_sc`: unavailable; none were computed.
- Supporting subject specificity `T_S`: not computed.
- Supporting class specificity `T_C`: not computed.

No zero, NaN, partial-cell, or available-case value substitutes for these missing scientific statistics.

## 17–19. Scientific interpretation and terminal

The optimizer-only amendment resolved the specific V0 failure at the fixed-action registration layer, but the unchanged higher-level GPA convergence contract failed before cell consensus completion. Therefore V1 supplies no evidence for or against a stable cross-session subject×class quotient mean-configuration interaction.

The only valid terminal is:

`UNASSESSED_GPA_NUMERICAL_FAILURE_V1`

It is not valid to report either scientific GO or scientific STOP. Registration actions remain nuisance variables and were not averaged, compared across cells, or interpreted as mean pose, subject pose, neural orientation, or anatomy.

## 20–23. Runtime, tests, repository state, and immutability

- Scientific runtime to fail-closed termination: `321.10 s`.
- Pre-run full repository tests: `225 passed in 46.71 s`.
- Post-run full repository tests: `225 passed in 32.97 s`.
- `git diff --check`: required to pass at final handoff.
- Git status: required to be clean after the final report commit.
- No scientific definition changed after V1 real-data access.
- No tolerance, iteration budget, determinant requirement, start count, optimizer, assignment rule, GPA rule, statistic, null, or terminal mapping was altered.
- No V1 automatic rerun was performed.

The immutable V0 outputs and history were not overwritten. V1 used a distinct output and cache namespace.
