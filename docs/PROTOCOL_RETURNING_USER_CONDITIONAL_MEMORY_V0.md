# Returning-User Low-Rank Conditional Memory Downstream Pilot V0

## Status and scope

This prospectively frozen downstream pilot is stacked on PR #19 at
`6abb73d82a0f616e0ca9d3eaa44e23d911a2123f`. It tests an offline returning-user
scenario: one labeled enrollment session followed by a deployment session with
zero new task labels. The deployment batch contributes only its already-frozen
session-wise unlabeled marginal. This is **initial calibration followed by
zero-recalibration for a returning user**, not calibration-free use for a new
person, online causal adaptation, pseudo-labeling, TTA, or a neural method.

All PR #16--#19 artifacts are immutable. New artifacts are confined to
`outputs/returning_user_conditional_memory_v0/`. A missing or hash-invalid parent
cache yields `UNASSESSED_REQUIRED_TRIAL_CACHE_MISSING_OR_INVALID`; no raw data are
downloaded or rebuilt.

## Datasets and frozen directions

The voting primary is Stieger2021 task 3, literal classes
`[right_hand,left_hand,both_hand,rest]`, frozen subjects 1--62 and PR #19 folds.
Session 2 is labeled enrollment and session 3 is zero-label deployment. The
independently fitted reverse direction is descriptive.

OpenBMI is the external binary replication with literal classes
`[left_hand,right_hand]`. Committed metadata identify source session 1 as the
earlier session and source session 2 as the later session; their frozen array
indices are 0 and 1. The chronological 1-to-2 direction votes only on the stated
replication gate. The reverse direction is descriptive. No other dataset is run.

## Frozen trial representation and leakage barrier

Every method receives exactly the parent marginal-centered tangent trial vector

\[
z_{sqi}=\operatorname{svec}\log(M_{sq}^{-1/2}C_{sqi}M_{sq}^{-1/2})\in\mathbb R^{210}.
\]

There is no new filtering, epoching, covariance estimation, regularization,
reference, channel selection, scaling, or feature normalization. Coordinate-wise
standardization is forbidden. Enrollment labels are available to construct the
held-out user's memory. Deployment labels are loaded only by a sealed evaluator
after every prediction, distance, probability, hyperparameter, and stopping
decision has been frozen. Source subjects' enrollment and deployment labels may
be used within source-only nested validation.

## Conditional signature

For each labeled session, class means form `P` and a deterministic orthonormal
Helmert matrix forms

\[
X=\operatorname{vec}(H_CP)\in\mathbb R^{(C-1)210}.
\]

`X` is not unit normalized. The inverse centered prototypes are
`H_C^T reshape(X,C-1,210)`. The Helmert matrix, literal class order, and exact
zero-class-mean round trip are contractual.

## Low-Rank Conditional Memory (LRCM)

For outer-source enrollment and deployment signatures, source means alone give
`A=X_E-mean_E` and `B=X_D-mean_D`. For each frozen ridge value,

\[
W_\lambda=A^T(AA^T+\lambda I)^{-1}B,
\]

implemented by a symmetric solve without an explicit matrix inverse. From
`AW=U Sigma V^T`, rank `R` gives `W_R=W V_R V_R^T`. The target prediction is

\[
\hat x_D=\bar x_D+(x_{t,E}-\bar x_E)W_R.
\]

The retained user memory is
`a_t=(x_{t,E}-mean_E) W V_R`, exactly `R` float64 values (`8R` bytes). The
predicted signature is Helmert-inverted and translated by the unlabeled
deployment tangent mean to produce class prototypes. Raw enrollment trials are
not part of the compact deployed memory.

## Nested selection and classifier

The immutable grids are ranks `[1,2,3,5,8,13]`, ridge values
`[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]`, and temperatures
`[0.05,0.1,0.2,0.5,1,2,5]`, subject to sample-identifiable rank limits. Each outer
fold uses the inherited deterministic five source-subject inner folds. The exact
enrollment-to-deployment scenario is simulated inside each split. Selection is
by mean subject balanced accuracy. Ties use smallest rank, largest ridge, then
temperature closest to one. Since temperature does not alter nearest-centroid
labels, all tied temperatures deterministically select one.

The classifier is equal-prior predicted-prototype nearest centroid with squared
Euclidean distance. Probabilities are the softmax of negative distances divided
by source-selected temperature.

## Baselines

The voting baselines are `POPULATION_ONLY`, `PAST_PROTOTYPE_DIRECT`, and
`IDENTITY_RESIDUAL_CARRY`; the last is the primary personalized comparator.
Low-rank audits are `FULL_RIDGE_TRANSFER`, `ENROLLMENT_PCA_TRANSFER`, and a Haar
`RANDOM_RANK_MATCHED` control. `CURRENT_SESSION_KSHOT` uses K per class in
`[1,2,4,8,16]`, 200 deterministic class-balanced support draws and disjoint query
trials. `CURRENT_SESSION_FULL_ORACLE` is an upper bound. A well-defined
class-independent offset is reported non-voting and is expected to collapse
toward population-only under marginal centering.

## Inference and nulls

The unit of inference is the subject. Each paired comparison reports mean and
median difference, 10,000-subject-bootstrap 95% CI, 1,999 one-sided paired
sign-flip permutations, win rate, and leave-one-subject mean-difference range.
Holm correction is applied across the three Stieger primary comparisons. NLL
non-inferiority is 0.01 nats/trial and full-ridge balanced-accuracy
non-inferiority is 0.005.

Each null has 1,999 deterministic replicates: target enrollment-memory
derangement within each outer fold; independent nonidentity enrollment-class
permutations; Haar rank-matched output subspaces; and unpaired source-session
maps. The last is a conditional randomization test: the observed source-only
hyperparameter selection is treated as a fixed nuisance statistic, while the
longitudinal source pairing alone is randomized. This exactly isolates pairing
without consulting an outer target and is the predeclared computationally
equivalent alternative to millions of redundant inner searches.

## Metrics

Primary performance is subject-level balanced accuracy. Required secondary
metrics are macro-F1, accuracy, per-class recall, NLL, Brier score, fixed-bin ECE,
subject win rate, median and worst-quartile gain, memory dimension/bytes, runtime,
and the interpolation-free calibration-equivalent K-shot bracket.

## Frozen decisions

`GO_LOW_RANK_CONDITIONAL_MEMORY_REPLICATED` requires all Stieger utility gates,
three memory nulls, every low-rank gate, and positive significant OpenBMI
replication over identity residual carry. `GO_STIEGER_ONLY_NEEDS_EXTERNAL_REPLICATION`
requires all Stieger gates but failed OpenBMI replication.
`GO_CROSS_SESSION_MAP_BUT_LOW_RANK_NOT_SUPPORTED` requires downstream map utility
but a failed low-rank gate. If LRCM does not beat identity residual carry under
the frozen primary gate, the terminal is
`STOP_NO_CROSS_SESSION_DOWNSTREAM_UTILITY`. Cache, leakage/split, and other
numerical/data-contract failures remain separate `UNASSESSED_*` outcomes. Parent
structural terminals cannot rescue this downstream test.

## Interpretation boundary

Even a GO establishes only an offline batch returning-user result after one
labeled enrollment session. It does not establish new-user zero calibration,
full conditional recovery, causal online deployment, semantic recovery from the
current unlabeled session alone, physiology, anatomy, universal coordinates,
pseudo-label validity, TTA, ASD generalization, or clinical efficacy.

