# Protocol — Selective Conditional Memory Feasibility Audit V0

## Status and question

This protocol is stacked on PR #20 head
`9c95e5b19eb4c44acc411c1e0d72a5cdd4d9ef63`. Its frozen terminal,
`STOP_NO_CROSS_SESSION_DOWNSTREAM_UTILITY`, is unchanged. The present question
is narrower: can enrollment-only reliability determine how much of a returning
user's previous-session class-conditional residual should be reused at a later,
zero-label deployment session?

This is a feasibility gate. It contains no neural network, residual decoder,
low-rank transfer map, raw-data access, preprocessing, or downstream rescue of
PR #20.

## Datasets and immutable coordinates

The voting analysis is Stieger2021 task 3, session 2 labeled enrollment to
session 3 zero-label deployment, using the frozen 62-subject cohort and exact
PR #19/#20 outer and inner folds. OpenBMI source session 1 to source session 2
is the external binary replication with the frozen 54 subjects and exact PR #20
folds. Reverse directions are descriptive only.

All methods use the hash-validated 210-dimensional session-marginal tangent
features, labels, class order, NCM metric implementation, and subject inference
unit inherited from PR #20. Deployment labels remain a sealed evaluation
object. Parent results and output namespaces are read-only.

## Frozen baselines and oracle gate

The PR #20 predictions for `POPULATION_ONLY`, `IDENTITY_RESIDUAL_CARRY`,
`PAST_PROTOTYPE_DIRECT`, and `LRCM` are immutable historical inputs. For each
subject, `ORACLE_IDENTITY_OR_POPULATION` selects the higher subject-level
balanced accuracy of population-only and identity carry after both predictions
are frozen. It is non-deployable and non-voting.

The oracle analysis reports mean balanced accuracy, gain over identity carry,
10,000-subject-bootstrap confidence interval, selection proportions, class
recall patterns, both datasets, and reverse-direction descriptions. If the
Stieger chronological oracle gain is nonpositive or its confidence-interval
lower bound is nonpositive, the experiment stops as
`STOP_NO_SELECTIVE_MEMORY_ORACLE_HEADROOM`; no gate is fitted.

## Enrollment-only reliability features

For every subject and enrollment class, trials are sorted by frozen acquisition
order, then assigned alternately within class: even within-class rank to A and
odd within-class rank to B. OpenBMI opaque IDs are mapped back to the committed
metadata acquisition order using the frozen SHA-256 ID rule. Neither deployment
features nor labels enter this operation.

Let `gamma_E[c]` be the outer-source enrollment population offset. With the
frozen full-session unlabeled enrollment mean `zbar_E`,

```text
r_A[c] = prototype_A[c] - zbar_E - gamma_E[c]
r_B[c] = prototype_B[c] - zbar_E - gamma_E[c].
```

The eight class features are:

1. cosine between `r_A` and `r_B` (zero if either norm is numerically zero);
2. squared norm `||r_A-r_B||^2`;
3. squared norm `||(r_A+r_B)/2||^2`;
4. `max(||mean||^2-||(r_A-r_B)/2||^2,0)/(||mean||^2+1e-12)`;
5. within-class tangent variance trace (coordinate variances, `ddof=1`);
6. trial count;
7. distance from the enrollment class offset to `gamma_E[c]`;
8. minimum distance to another target enrollment class prototype.

Each feature is aggregated across classes by mean, minimum, and maximum, giving
a literal 24-dimensional subject vector. Class-wise features are retained for
the secondary gate. Scaling uses outer-source means and population standard
deviations only; exact zero scales are replaced by one as in a conventional
constant-column standardizer.

For an outer target, `gamma_E` uses every outer-source subject. For a source
training example, templates exclude that source subject. In inner validation,
all templates, scaling, and parameters use inner-training subjects only.

## Selective gate

The primary subject-global gate is

```text
kappa_s = sigmoid(w^T h_s + b),  0 <= kappa_s <= 1.
```

It predicts deployment prototypes as

```text
mu_hat_D[c] = zbar_D^U + gamma_D[c] + kappa_s r_E[c],
r_E[c]      = prototype_E[c] - zbar_E^U - gamma_E[c].
```

Only the current deployment unlabeled mean is legal. Gate input contains no
deployment quantity. `w,b` minimize the mean of per-source-subject deployment
cross-entropies plus `L2*||w||^2/2`; the intercept is unpenalized. Optimization
uses deterministic zero initialization and SciPy L-BFGS-B with maximum 500
iterations, `ftol=1e-12`, and `gtol=1e-8`. The sigmoid creates only a convex
interpolation of the two frozen endpoints; this phrase does not assert that the
parameter objective is globally convex.

L2 is chosen in deterministic paired-subject five-fold inner CV from
`[1e-4,1e-3,1e-2,1e-1,1,10]` by mean subject balanced accuracy, breaking exact
ties toward the largest L2. All selection uses source subjects only.

## Controls

`GLOBAL_KAPPA` selects one scalar from `0,0.1,...,1` in the same source-only
inner CV. `FIXED_RELIABILITY_KAPPA` uses the subject mean class reliability
ratio clipped to `[0,1]`. `CLASSWISE_SELECTIVE_GATE` uses one class coefficient
from the class reliability ratio and is secondary; it cannot rescue the global
gate. `ORACLE_CONTINUOUS_KAPPA` searches 101 equally spaced values on `[0,1]`
with target deployment labels and is a non-voting ceiling.

## Metrics and inference

Primary comparisons are `SELECTIVE_GATE` minus, respectively,
`IDENTITY_RESIDUAL_CARRY`, `POPULATION_ONLY`, and `GLOBAL_KAPPA`, at the
subject-level balanced-accuracy unit. Secondary outcomes are macro-F1, NLL,
Brier score, ECE with 10 fixed bins, subject win rate, and worst-quartile gain.
Each paired result uses 10,000 subject bootstraps and 1,999 deterministic
one-sided sign-flip permutations. Holm correction covers the three Stieger
primary comparisons.

## Required nulls

Each Stieger null has 1,999 deterministic replicates and reruns target
prediction. `ENROLLMENT_SUBJECT_MEMORY_PERMUTATION` assigns another outer-test
subject's residual but retains the original gate. `RELIABILITY_FEATURE_PERMUTATION`
permutes gate features but retains the correct residual.
`ENROLLMENT_CLASS_SEMANTICS_PERMUTATION` applies independent nonidentity class
permutations to enrollment residuals. `UNPAIRED_SOURCE_SESSION_GATE_TRAINING`
breaks the outer-source enrollment/deployment pairing and refits the gate with
the already source-only-selected L2. This is a conditional randomization test;
no target outcome selects nuisance settings.

Null p-values are `(1 + #null >= observed)/(1 + 1999)` for the selective-gate
gain over identity carry.

## Frozen decisions

`GO_SELECTIVE_MEMORY_FOR_SPDNET` requires on Stieger: positive oracle CI;
selective-gate gains over identity, population, and global kappa with positive
CIs and Holm p-values at most .05; positive macro-F1 gain over identity; NLL no
worse than population by more than .01; all four null p-values at most .05; and
positive leave-one-subject identity gain. It additionally requires OpenBMI
selective-gate gain over identity with positive CI and p at most .05.

If all Stieger conditions pass but OpenBMI fails, the decision is
`GO_STIEGER_SELECTIVE_MEMORY_NEEDS_REPLICATION`. If oracle headroom exists but
the deployable gate fails, it is `STOP_RELIABILITY_GATE_CANNOT_SELECT_MEMORY`.
Absent oracle headroom, it is `STOP_NO_SELECTIVE_MEMORY_ORACLE_HEADROOM`.
Cache, leakage/split, and other numerical/data-contract failures use the frozen
`UNASSESSED_*` labels from the config.

A GO permits only development of `SelectiveMemSPDNet` with shared SPD backbone,
current-session marginal normalization, persistent enrollment residual memory,
and the validated reliability gate. A STOP terminates this persistent
conditional-memory line and does not authorize another low-rank rescue.

## Freeze and execution order

Before real oracle or gate statistics: audit parent hashes/cache schemas; freeze
this protocol, config, exact feature definitions, optimizer, inference, source;
run synthetic tests; copy folds; and commit with subject
`freeze selective conditional memory feasibility v0`. Then push and open a
stacked draft PR. After freeze: run the frozen oracle gate; only if it passes,
fit Stieger observed/controls/nulls; run OpenBMI replication; generate report;
verify all parent hashes, result hashes, tests, diff, and a clean worktree; and
commit results separately. No automatic merge is allowed.
