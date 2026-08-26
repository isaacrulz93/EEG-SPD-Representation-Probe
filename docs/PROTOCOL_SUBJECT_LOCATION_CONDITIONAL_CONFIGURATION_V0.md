# Protocol: Cross-Session Subject Location → Conditional Configuration V0

## Frozen question

Can a subject's label-free location in a source-population SPD coordinate
system predict that same subject's class-centered conditional configuration
in the other session, when the subject is held out from fitting?

Primary directions are session 2 → session 3 and session 3 → session 2. The
dataset, cohort, task, class order, folds, and geometry are the exact locked
PR #19 Stieger2021 objects: 62 subjects, sessions 2 and 3, task 3, four classes,
six outer folds, and the `primary/F` SPD(20) geometry.

## Fold construction

For each outer fold, all held-out subjects are excluded before any reference,
population configuration, output basis, or regression parameter is computed.
For source subject s, first form the equal-session subject reference

`R_s = FM_AIRM(M_s^(2), M_s^(3))`.

The fold reference is `M_0 = FM_AIRM({R_s : s in S_train})`, with exactly one
vote per source subject. Inner validation recomputes `M_0`, the population
class configuration, and every coordinate using only the inner-training
subjects.

The label-free input is

`q_s^(u) = svec(log(M_0^(-1/2) M_s^(u) M_0^(-1/2)))`.

For output session v, class coordinates are equal-class centered within
subject, then source-centered:

`d_s,c^(v) = z_s,c^(v) - (1/4) sum_j z_s,j^(v)`,

`Delta_s,c^(v) = d_s,c^(v) - mean_{r in S_train} d_r,c^(v)`.

Both `d` and `Delta` must sum to zero over the four classes. The flattened
prediction target has 840 coordinates.

## Model selection and prediction

The only primary model is source-only reduced-rank dual ridge regression.
Candidate output ranks are 0, 1, 2, 3, 5, 8, and 13. Rank is clipped by the
inner training size and numerical output rank. Rank zero predicts the exact
zero residual.

For centered input Q, `K = Q Q^T`, `kappa = trace(K)/n`, and
`lambda = multiplier × max(kappa, machine_tiny)`. Ridge multipliers are
`1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000`. A stable dual linear solve is
used; no explicit inverse or coordinate standardization is permitted.

Selection minimizes mean inherited-inner-fold normalized conditional SSE.
The one-standard-error set is formed from the minimum-loss candidate's
standard error; the smallest effective rank is selected, then the largest
ridge multiplier. No held-out outcome participates.

## Outcome barrier and evaluation

Each fold/direction has a label-free source/input packet and a separate sealed
held-out conditional-outcome vault. Predictions, model provenance, and hashes
must be complete for all 12 fold-direction cells before the evaluator may open
the vault. Prediction files contain only subject IDs and predicted Delta.

The baseline is `Delta_hat = 0`. Primary statistics are forward, reverse, and
pooled conditional R², subject error gains, 10,000 paired-subject bootstrap
replicates, paired sign flips, and leave-one-subject influence. The primary
source-pairing null and secondary held-out-location derangement null each use
1,999 deterministic replicates. Location norm and same-session models are
non-voting controls.

## Frozen terminals

`GO_LOCATION_PREDICTS_CROSS_SESSION_CONDITIONAL_CONFIGURATION` requires every
registered materiality, direction, bootstrap, null, sign-flip, positive-unit,
influence, and engineering gate. A positive result that fails the full gate is
`WEAK_LOCATION_CONDITIONAL_ASSOCIATION_NOT_METHOD_READY`. Nonpositive or
unsupported cross-session prediction is
`STOP_LOCATION_DOES_NOT_PREDICT_CONDITIONAL_CONFIGURATION`. Contract failures
remain `UNASSESSED_*`.

The state order is exactly: PARENT_VALIDATED, PROTOCOL_FROZEN,
OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED, PRIMARY_PREDICTIONS_FROZEN,
TARGET_OUTCOMES_RELEASED_FOR_EVALUATION, NULLS_COMPLETE, TERMINAL_WRITTEN,
STOPPED. Valid earlier artifacts are never deleted or silently regenerated.
