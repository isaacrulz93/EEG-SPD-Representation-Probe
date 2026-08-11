# Local GPA Consensus V0

Stage 1 used only the ten internal AIRM edge lengths and ended at `STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0`. Stage 2A was preregistered to test a richer object: the full, trial-locally-centered five-point SPD configuration averaged within each subject×session×class cell after quotienting by `O(22)×S5`. Registration matrices were nuisance variables only; no mean pose or pose comparison was performed.

## Provenance

- Branch: `pilot/local-gpa-consensus-v0`
- Scientific base SHA: `796f04e7970972175a660a521caff47c83e0295f`
- Protocol-freeze SHA: `ad3dd429a69e7cea77e3d4421987989ab45171b9`
- Scientific-result SHA: `9aba8a8cee1c84d9b69b7fdf3d53d570927a7cc6`
- Final SHA: this report-finalization commit; recorded in the Git handoff and Draft PR

## Frozen input reproduction

The two frozen WINDOW5 covariance arrays reproduced the existing 5,184 AIRM distance matrices exactly:

- session-0 covariance content SHA-256: `c75044f48552f12ad088306b505b074e930f396fdcb544307fff394717e2ca86`
- session-1 covariance content SHA-256: `1afc8cd52d82310a05857d1ffa67859427c4c9aa1302897a140ebda64d0442f8`
- frozen/recomputed AIRM-distance content SHA-256: `681d8a075eff1218e5e2b2d0e292631ead67badaccc00ec075ba428c9d5aed64`
- maximum absolute reproduction difference: `0`

Trial-local centering completed for all 5,184 trials. The maximum normalized centered Karcher residual was `8.539076980603567e-11`, below the frozen `1e-7` gate; the median was `5.832384513594664e-11`.

## Pre-data numerical sanity

Before protocol freeze, known `O(d)` actions in both determinant components and known `S5` relabelings produced quotient distances of approximately `8.0e-15` and `9.6e-15`. A known-answer GPA cell recovered the correct orbit with objective `5.99e-29`; the two deterministic GPA starts differed by quotient distance `1.14e-14`. The d=22 exact four-start registration objective was `5.12e-26`. The complete pre-data repository test suite passed (`219 passed`).

## Frozen real-run numerical failure

The real execution stopped during the first parallel wave, before any full-cell consensus checkpoint completed. A required trial-to-prototype registration did not yield a converged candidate in both disconnected determinant components of `O(22)` under the frozen 250-iteration, gradient-`1e-6` certification contract.

This is a numerical/optimizer failure, not evidence that a quotient mean configuration does not exist. The pipeline stopped fail-closed as preregistered. It did not increase iterations, relax tolerances, remove a determinant component, alter starts, patch the optimizer, or rerun.

## Required scientific results

- Full-cell consensus convergence: `0 / 72` completed before the blocking failure.
- Split-half consensus convergence: `0 / 144`; not reached.
- Split-half reliability: not assessed.
- Cross-session 36×36 consensus-distance matrix: not assessed.
- `T_J`: not assessed.
- `p_classbreak`: not assessed.
- `p_subjectbreak`: not assessed.
- All 9 `J_s`: not available because the required consensus grid was incomplete.
- All 36 `J_sc`: not available because the required consensus grid was incomplete.
- Subject specificity and class specificity: not assessed.
- Within-cell dispersion: no complete cell result was available for scientific reporting.

## Terminal decision

**`UNASSESSED_GPA_NUMERICAL_FAILURE`**

The correct interpretation is that Stage 2A remains scientifically unassessed under the frozen numerical implementation. It is neither a positive result nor `STOP_NO_STABLE_QUOTIENT_MEAN_CONFIGURATION_INTERACTION`.

## Runtime and immutability

- Real-run elapsed time to fail-closed termination: `17.9153 s`.
- Local centering portion: `12.1064 s` with 8 workers.
- No Stage-2A interaction statistic, correspondence null, or subject result was computed.
- No scientific definition changed after real-data access.
- No automatic rerun was performed.
- The immutable Stage-1 result was not modified, rerun, rescued, or reinterpreted.

## Claim boundaries

No stable subject pose, class pose, `Q_s`, physical orientation, neural orientation, or anatomical orientation was estimated or claimed. The failure also says nothing about whether a future, separately reviewed numerical formulation could assess the same scientific question.
