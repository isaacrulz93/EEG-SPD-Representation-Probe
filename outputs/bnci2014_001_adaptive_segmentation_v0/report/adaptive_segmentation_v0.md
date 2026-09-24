# Adaptive Covariance Segmentation falsification v0 — execution record

## Terminal

**`UNASSESSED_ADAPTIVE_BOUNDARY_V0`**. No classifier fit or target-subject score was run.

During deterministic ADAPT5 construction, at least one trial produced five equal segment lengths. The only distinct ordering of this multiset is the identity, which cannot produce the required non-identity RANDOM_MATCHED boundary set. Construction stopped after 1,100 trials; no workaround, identity substitute, or protocol/configuration change was made. Since the key matched-length comparator could not be defined for the observed input, no paired classification tests or GO/STOP terminal evidence exist. This is a structural degeneracy in the preregistered control, not evidence for or against adaptive segmentation.

## Data and leakage gate

The prepared epoch cache was absent at start. The builder used the frozen 5window YAML's MotorImagery preprocessing values and MOABB's converters/pipelines, resolving only the nine training files `A01T.mat` through `A09T.mat`. It did not call MOABB's general BNCI loader, which opens both T and E files. No E file URL was requested, no `1test` data was loaded, and no within-subject evaluation was added.

Regenerated data passed the frozen content contract: shape `(2592,22,1000)`, `float32`, 250 Hz; only session `0train`; configured 22-channel order; expected X-content SHA-256 `0ea4250c…92b3972`; expected trial-metadata SHA-256 `4ce1ce3f…05d39ae`. The source training MAT files occupy 369 MiB in the ignored local MOABB cache. Prepared epochs and metadata are retained under this experiment output for audit.

Probe OAS covariance, trial-local symmetric EVD tangent features, svec, and exact DP ran through 1,100 trials without a numerical exception; the run then stopped at the matched-length control edge case. No classifier/scaler was fit, and target labels did not enter fitting or tuning.

## Tests

- New focused tests: **4 passed** (piecewise DP/brute-force optimum, SPD and svec, coverage/minimum lengths, random controls, matched length multiset, exact 512 sign configurations, and scaler source-only invariance).
- Python compilation and `git diff --check`: passed.
- Full repository test collection: blocked by missing `pymanopt` in seven pre-existing test modules.
- Remaining existing suite with those seven modules excluded: **221 passed, 2 skipped, 1 failed**. The failure requires the absent ignored cache `cache/bnci2014_001_trajectory_within_subject_v1/combined_trajectory_features.npz`; unrelated to this experiment. One other frozen-cache test and one cache-dependent test skipped.

## Frozen implementation and outputs

Pre-result implementation commit: `290626ddb15f902ac0de716cc0ab227797aab576`.
Branch: `pilot/adaptive-segmentation-v0`, based on `origin/pilot/local-mean-movement-antidevelopment-v0`.
No post-result changes were made to protocol, configuration, thresholds, or implementation. The output directory contains data provenance, prepared 0train epochs, and terminal decision. Because boundary construction did not complete, full per-trial boundary arrays, LOSO tables, plots, and a metric report were not generated.
