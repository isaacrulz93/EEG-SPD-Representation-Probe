# Pairwise Common Action V2

## Plain-language question

This audit asks whether one class-independent orthogonal sensor action,
estimated from three motor-imagery classes for a source–target subject pair,
improves prediction of the fourth unseen class. Pairwise results are collapsed
within each target before group inference; the target subject, not the pair,
is the inferential unit.

V1 did not answer this question: ConjugateGradient failed technically before
one Stage-A target task completed. V2 changes only that numerical optimizer to
the synthetically audited Pymanopt TrustRegions implementation. No scientific
definition or threshold changed after V2 data access.

## Reproduction and numerical gates

The frozen U object reproduced exactly: maximum absolute and relative
differences were both 0. All Full/A/B symmetry gates passed.

TrustRegions completed all 18,432 required starts. All 9,216 det(−1) and all
9,216 det(+1) starts met the frozen Riemannian-gradient threshold
\(10^{-5}\). The median was 21 iterations, the maximum was 60, and no start
reached the 1,000-iteration limit.

All 576 Stage-A and 576 Stage-B prediction cells were classified
`PREDICTIVELY_IDENTIFIABLE`. No numerical or approximate continuous
stabilizer direction was detected in any of the 576 Stage-A fit banks.

## Stage A: within-session unseen-class prediction

The nested target-subject median raw error was 1.20958203 and the action error
was 0.87740532. The primary subject-level gain statistic was 0.22759557.

- unrelated-target null median: −0.13915509;
- effect above null median: 0.36675067;
- one-sided plus-one Monte Carlo p: 0.0005;
- decision: **PASS**.

The semantic mismatch control also passed:

- true-correspondence statistic: 0.22759557;
- semantic-null median: 0.09780589;
- effect: 0.12978968;
- p: 0.0005.

Thus an action fitted using the correct three class correspondences predicted
the fourth class better than both an unrelated target action and arbitrary
fit-class correspondence under the frozen aggregation.

## Stage B: opposite-session unseen-class prediction

The nested target-subject median raw error was 1.20958203 and the action error
was 1.02410049. The cross-session gain statistic was positive but small:
0.00061939.

- unrelated-target null median: −0.17223676;
- effect above null median: 0.17285615;
- p: 0.0005;
- decision: **PASS**.

The cross-session semantic control also passed:

- true-correspondence statistic: 0.00061939;
- semantic-null median: −0.07071865;
- effect: 0.07133803;
- p: 0.0005.

Five of nine target subjects had a positive Stage-B descriptive median. The
small observed group median and subject heterogeneity must be reported rather
than replaced by the null-relative effect.

## Target-subject results

| target subject | Stage-A median gain | Stage-B median gain |
|---:|---:|---:|
| 1 | 0.106297 | 0.048058 |
| 2 | 0.689942 | 0.335360 |
| 3 | 0.108490 | 0.000619 |
| 4 | 0.315268 | 0.025191 |
| 5 | 0.896487 | −0.089599 |
| 6 | 0.454159 | −0.011112 |
| 7 | 0.227596 | 0.070421 |
| 8 | −0.016090 | −0.142948 |
| 9 | 0.073492 | −0.014591 |

Stage A was positive in eight of nine subjects; Stage B was positive in five
of nine. These counts are descriptive and do not replace the frozen group
tests.

## Global-consistency boundary

The equivalence-aware pairwise cycle discrepancy had descriptive median
0.71613120. The profiled global latent-template model was not run because it
requires a separate product-manifold start freeze. Therefore this experiment
supports only the pairwise necessary consequence; it does not establish a
globally cycle-consistent latent \(\{Q_s,B_c\}\) model.

## Terminal decision

**PAIRWISE_COMMON_ACTION_NECESSARY_CONSEQUENCE_SUPPORTED**

Under the frozen identity-tangent, sensor-space \(O(22)\) model, a
class-independent pairwise subject action inferred from three classes
predicts an unseen fourth class within session and, more weakly and
heterogeneously, across sessions. Correct semantic correspondence matters.

This does not show that individuality stops at a common action, that every
subject has a positive cross-session effect, or that a global latent action
model exists. It does not establish physiology, source-space structure,
causality, unlabeled identifiability, or performance improvement. The planned
next scientific branch remains split-epoch covariance-set anatomy.

Total V2 wall time was 582.94 seconds (9.72 minutes) with eight workers.
