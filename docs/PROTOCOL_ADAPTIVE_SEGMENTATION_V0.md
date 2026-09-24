# Adaptive Covariance Segmentation Falsification v0

## Aim and frozen scope

This experiment tests whether covariance-change boundaries preserve cross-subject motor-imagery class information better than equal fixed intervals and random boundary controls. It is a segmentation falsification study, not a model proposal. No neural network, attention, Transformer, or SPDNet is used.

Only BNCI2014_001 session `0train` is eligible. The existing frozen preparation settings are reused: 1,000 samples/trial, 22 EEG channels in the configured order, 250 Hz, 8–32 Hz bandpass, OAS covariance, float64 geometry. Session `1test` is forbidden. The prepared array contract is `(2592,22,1000)`.

## Boundary signal

For each trial, OAS covariances are computed from 125-sample windows stepped by 25 samples (36 windows). The whole-trial OAS covariance is the trial-local reference. For each probe covariance, `Z=log(G^(-1/2) S G^(-1/2))`; symmetric eigendecompositions are used throughout, without eigenvalue clipping. Frobenius-isometric svec has 253 coordinates.

## Five-segment conditions

- `GLOBAL`: one OAS covariance over the full trial.
- `FIXED5`: five contiguous 200-sample intervals.
- `ADAPT5`: exact dynamic programming over 25-sample grid boundaries, exactly five segments, each at least 125 samples. Cost is within-segment sum of squared Euclidean deviations of probe svec vectors. Ties resolve by first encountered start index.
- `RANDOM5`: 20 deterministic, label-independent valid boundary sets per trial, with the same grid, segment count and minimum length.
- `RANDOM_MATCHED`: for each trial, up to 20 deterministic unique non-identity permutations of the ADAPT5 segment-length multiset. Segment count and length multiset are preserved exactly.

The final classifier covariances are recomputed from raw EEG within each resulting segment; probe covariances are never substituted for these.

## Classifier and evaluation

Each final covariance becomes `svec(log(C))`. Each condition uses a source-only StandardScaler and multinomial logistic regression (`C=1`, `lbfgs`, `max_iter=5000`). The outer split is nine-subject LOSO. Multi-segment training duplicates the trial class label to each segment; each test trial probability is the duration-weighted average of segment probabilities.

Primary metric is target-subject balanced accuracy. Exact one-sided paired sign-flip tests enumerate all 512 signs for the nine target-subject BA differences. RANDOM5 and RANDOM_MATCHED are averaged over available replicate scores within each target before pairing.

Boundary validity is tested independently from classification: each adaptive boundary compares OAS covariances from up to 125 samples immediately before/after, against same-width valid non-boundary positions. AIRM distances are averaged trial-wise then subject-wise and tested with the same exact paired sign-flip test. Across-segment versus within-segment AIRM distances are also saved descriptively.

## Terminal rule

`GO_ADAPTIVE_BOUNDARY_V0` requires all three: ADAPT5 mean BA exceeds FIXED5 with one-sided paired `p<.05`; ADAPT5 exceeds RANDOM_MATCHED with `p<.05`; and boundary AIRM change exceeds matched random positions with `p<.05`. Otherwise record `STOP_ADAPTIVE_BOUNDARY_V0`. Any data or numerical gate failure records `UNASSESSED_ADAPTIVE_BOUNDARY_V0`.

All configuration and code are committed before scientific evaluation. Results do not authorize rescue experiments or post-result changes.
