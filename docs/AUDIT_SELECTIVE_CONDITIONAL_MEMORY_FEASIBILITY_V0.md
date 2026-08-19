# Audit — Selective Conditional Memory Feasibility V0

## Lineage and immutability

The isolated worktree starts exactly at PR #20 head
`9c95e5b19eb4c44acc411c1e0d72a5cdd4d9ef63` on branch
`pilot/selective-conditional-memory-feasibility-v0`. The parent terminal remains
`STOP_NO_CROSS_SESSION_DOWNSTREAM_UTILITY`. Every tracked file below PR #16--#20
output namespaces is snapshotted by SHA-256 into the new output namespace and
reverified after execution. No parent path is a write target.

The five parent manifest file SHA-256 values are locked in the config. PR #20's
95-file canonical result-manifest hash is
`cc8d6484dfc1cf355455a9e38a00b09319583299dd4b616717ee51b311d7d68f`.
Its Stieger and OpenBMI frozen prediction objects are also hash-locked and are
used only for the oracle ceiling and historical baseline consistency audit.

## Data and cache contract

Stieger uses the PR #19 manifest of 124 subject-session tangent files, subjects
1--62, sessions 2 and 3, task 3, literal class order right hand, left hand, both
hands, rest, and 210 finite float64 coordinates. Each file retains
`acquisition_index`; therefore A/B enrollment splits require no raw access.

OpenBMI uses the PR #17 cache with finite shape `(54,2,100,210)`, exact 50/50
class counts, source chronology session 1 to 2, and the PR #20 folds. The cache's
opaque shuffled trial IDs are deterministically mapped to acquisition order by
hashing `subject|array_session|original_index`, exactly as the frozen PR #17
loader did, and joining the committed metadata. The mapping must be one-to-one.

The cache links reference the existing external-drive objects; they do not copy
or regenerate data. Raw download, filtering, covariance computation, tangent
reconstruction, channel changes, and coordinate scaling are forbidden.

## Leakage boundary

Enrollment labels and acquisition order may construct reliability features.
The outer-source templates, feature mean/scale, L2/global-kappa selection, and
gate fit exclude outer-test subjects. Deployment features do not enter the gate
input. Deployment labels remain separately typed and join predictions only for
evaluation. The only target-label-dependent objects are explicitly non-voting
oracle ceilings.

The executable audit reads hashes, schemas, shapes, trial counts, finite flags,
class balance, acquisition-order mappings, and folds only. It does not access
oracle balanced accuracy, gate performance, null statistics, or scientific
decisions before protocol freeze.

## Resource estimate and fail-closed gate

No additional raw/cache storage is required. Compact results are expected below
100 MiB. Peak RAM is expected below 4 GiB. Observed fits should complete within
minutes; 1,999 replicate controls may require several hours with one worker.
Missing or hash-invalid caches terminate as
`UNASSESSED_REQUIRED_TRIAL_CACHE_MISSING_OR_INVALID`. Any label leakage or fold
mismatch terminates as `UNASSESSED_TARGET_LABEL_LEAKAGE_OR_SPLIT_FAILURE`; other
numerical/data-contract failures fail closed.
