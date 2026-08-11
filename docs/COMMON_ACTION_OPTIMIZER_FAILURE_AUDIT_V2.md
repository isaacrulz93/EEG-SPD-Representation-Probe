# Common Action Optimizer Failure Audit V2

Status: **engineering/convergence audit only**. This memo does not amend the
frozen pairwise protocol and does not contain a BNCI held-out gain, aggregate
Stage-A statistic, null, p-value, semantic-control result, or Stage-B result.
No BNCI optimizer or scientific pipeline was rerun for this audit.

Preserved failure commit:
`83c3e1bddc40b433ca5494902da1ec6c64a06db5`.
The same commit remains on the remote branch and in draft PR #4. The
preregistered synthetic stress implementation was committed separately as
`6e9f08a` before the stress results were generated.

## 1. Exact scientific hypothesis and why the real run failed

The unchanged scientific action and loss are

\[
U\mapsto Q U Q^{\mathsf T},\qquad Q\in O(22),
\]

\[
L(Q)=\sum_{i=1}^{3}\|A_i-QB_iQ^{\mathsf T}\|_F^2.
\]

The real pairwise run stopped in the Full-split fit for target subject 7,
source subject 1, session `0train`, held-out class `feet`. The frozen contract
required at least one converged start in each determinant component. One
det(-1) start converged, but both det(+1) starts reached 1,000 iterations with
projected Riemannian gradient norm above \(10^{-5}\). The correct terminal
state was therefore `UNASSESSED_TECHNICAL_FAILURE`.

This is not evidence that a det(+1) solution does not exist. Each determinant
component of \(O(22)\) is compact and the objective is continuous, so a
component-wise minimum exists. The observed failure concerns the current
optimizer/stopping implementation, not existence.

No target-subject task completed. Consequently there is no complete Stage-A
scientific grid and no scientific result to interpret.

## 2. Forensic limits of the preserved real-data trace

The worker used Pymanopt logging in memory, but the pipeline checkpointed only
completed target tasks. The process raised before any target task completed,
and its iteration log was not serialized. The preserved exception and
`technical_failure.csv` contain only the final per-start diagnostics shown
below.

Reconstructing the missing iteration history would require rerunning the real
BNCI fit. That is explicitly outside this audit, so unavailable quantities are
reported as `NOT_RECORDED`; they are not inferred from the final point.

| start | initial/final det | status | iterations | reported final objective | reported final gradient | stopping criterion |
|---:|---:|---|---:|---:|---:|---|
| 0 | -1 / -1 | failed | 1000 | 1.0113924173547508 | 1.106774462156307e-3 | max iterations |
| 1 | +1 / +1 | failed | 1000 | 1.0122124822161918 | 1.1862512748811898e-3 | max iterations |
| 2 | -1 / -1 | converged | 604 | 1.0076277912040363 | 9.233935028016979e-6 | min gradient norm |
| 3 | +1 / +1 | failed | 1000 | 1.0062940720007654 | 1.4809489254446199e-4 | max iterations |

For each failed det(+1) start, the following requested fields were not
persisted:

| field | det(+1) start 1 | det(+1) start 3 |
|---|---|---|
| initial objective | NOT_RECORDED | NOT_RECORDED |
| best objective reached | NOT_RECORDED | NOT_RECORDED |
| objective change over last 10/50/100/250 iterations | NOT_RECORDED | NOT_RECORDED |
| initial gradient norm | NOT_RECORDED | NOT_RECORDED |
| minimum gradient norm reached | NOT_RECORDED | NOT_RECORDED |
| accepted/rejected line-search trials | NOT_RECORDED | NOT_RECORDED |
| step-size distribution | NOT_RECORDED | NOT_RECORDED |
| orthogonality error throughout | NOT_RECORDED | NOT_RECORDED |
| determinant throughout | only initial/final +1 recorded | only initial/final +1 recorded |

The recorded stopping string for both det(+1) runs was only `max iterations`.
Their final gradients also did not meet either the primary \(10^{-5}\)
criterion or the existing bounded-gradient \(10^{-4}\) part of the
minimum-step plateau clause. However, without the actual iteration sequence it
is impossible to classify either real trajectory as still descending,
plateaued, oscillating, line-search stalled, or numerically unstable. The
forensic answer to that requested classification is **UNDETERMINED FROM
PRESERVED ARTIFACTS**.

The same limitation applies to the det(-1) traces. The available evidence says
only that start 2 reached the strict gradient criterion at iteration 604 while
start 0 reached the iteration limit. It does not provide their earlier
objective, gradient, step, or orthogonality trajectories.

## 3. Preregistered synthetic-only stress suite

Before generating results, commit `6e9f08a` fixed a 12-fixture grid:

- dimension 22 and four symmetric matrices per fixture;
- generic noncommuting, exact: det(+1), det(-1);
- generic noncommuting, symmetric noise scale \(2\times10^{-4}\): det(+1),
  det(-1);
- nearly commuting, noise scale \(10^{-5}\): det(+1), det(-1);
- clustered/near-repeated eigenvalues, noise scale \(10^{-5}\): det(+1),
  det(-1);
- approximate common stabilizer, noise scale \(10^{-6}\): det(+1), det(-1);
- deliberately ill-conditioned spectra spanning eight orders of magnitude,
  noise scale \(10^{-7}\): det(+1), det(-1).

Every optimizer received byte-identical deterministic start matrices: two
starts in det(+1) and two in det(-1), with no hidden warm start and no
post-optimization sign correction. The exact loss above was used throughout.
Synthetic held-out recovery compared the induced conjugation action with the
known generating action; raw \(Q\) equality was not required.

The compared Pymanopt 2.2.1 optimizers were:

1. `ConjugateGradient(beta_rule="HestenesStiefel")` with the frozen explicit
   backtracking line search;
2. `TrustRegions` with its documented defaults (`miniter=3`, `kappa=0.1`,
   `theta=1`, `rho_prime=0.1`, deterministic inner solve) and an analytic
   Hessian-vector product for the unchanged loss;
3. `SteepestDescent` with the same explicit backtracking line search.

All used maximum 1,000 iterations and strict projected-gradient tolerance
\(10^{-5}\). The analytic Euclidean Hessian-vector product supplied to the
standard TrustRegions implementation was

\[
D G(Q)[H]=4\sum_i\left[
(H B_iQ^{\mathsf T}+Q B_iH^{\mathsf T})QB_i
+(QB_iQ^{\mathsf T}-A_i)HB_i
\right],
\]

and was independently finite-difference tested. This adds the derivative of
the declared loss; it is not a new optimizer or geometry.

Generated artifacts:

- `outputs/common_action_optimizer_failure_audit_v2/synthetic_optimizer_runs.csv`
  (SHA-256 `99fc9b4416aabff4bf9746a83c5d49d9520f538f74f78539d6ad1115d9410bbe`);
- `synthetic_fixture_summary.csv`
  (SHA-256 `bcc32c036d79d9ea2e0f899ff001e430047b3f36cfb46425b9d6448b9458542d`);
- `synthetic_optimizer_summary.json`
  (SHA-256 `9949803a55932f13c304c03a6a80599f7aa59a621aeb29e2fb9e21f5e5abb8eb`).

The audit module has no data loader, and a test explicitly rejects imports or
calls to the BNCI U reproduction pipeline.

## 4. Synthetic robustness results

| optimizer | strict-converged starts | fixtures with both det sectors certified | median iterations | max iterations | median seconds/start | total seconds | median best held-out relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConjugateGradient | 42/48 | 10/12 | 66.0 | 1000 | 0.0212 | 2.783 | 1.086e-4 |
| TrustRegions | 48/48 | 12/12 | 9.0 | 33 | 0.0224 | 5.284 | 7.600e-5 |
| SteepestDescent | 37/48 | 10/12 | 212.5 | 1000 | 0.0530 | 4.858 | 1.116e-4 |

Current CG certified both sectors for generic exact/noisy, nearly commuting,
clustered-spectrum, and approximate-stabilizer fixtures. It failed sector
certification for both ill-conditioned fixtures: six starts reached 1,000
iterations. All six retained trajectories were still decreasing over their
last 100 iterations and had non-negligible last-50 induced-prediction spread;
none was a numerically stable action marginally missing only the gradient
threshold.

SteepestDescent had 11 iteration-limit starts. Other starts still certified
both sectors in all but the two ill-conditioned fixtures, so its fixture-level
certification was also 10/12. It did not improve on CG robustness.

TrustRegions strictly converged every start in every determinant sector. Its
largest iteration count was 33. It was slower in total because the
ill-conditioned second-order solves were more expensive, but its median
per-start time was essentially the same as CG and it materially improved the
predeclared robustness endpoint from 10/12 to 12/12 certified fixtures.

Family-level median four-start fit times for TrustRegions were 0.037 s
(nearly commuting), 0.093 s (generic exact), 0.104 s (clustered), 0.118 s
(generic noisy), 0.145 s (approximate stabilizer), and 2.144 s
(ill-conditioned). Thus the robustness improvement is not obtained by
silently skipping the difficult family.

## 5. Is the frozen stopping contract too strong?

The frozen primary condition is projected gradient norm at most \(10^{-5}\).
CG/Steepest also have an existing secondary condition limited to a
minimum-step termination with objective change at most \(10^{-13}\) and
gradient at most \(10^{-4}\). No stress run used that clause.

This audit prespecified a diagnostic false-failure pattern:

- strict gradient criterion not met;
- relative objective change over the last 50 accepted iterates at most
  \(10^{-10}\); and
- induced held-out-action spread over the same window at most \(10^{-8}\).

Zero CG, TrustRegions, or SteepestDescent runs met that stable-action/
above-gradient pattern. All nonconverged logged runs were classified as still
descending rather than plateaued. Therefore the synthetic evidence does **not**
support loosening the gradient tolerance or adopting an objective/step-only
success rule to rescue the real cell.

A generally defensible future diagnostic contract could record all three of
gradient norm, windowed relative objective change, and induced-action change.
But on the present stress suite there is no calibrated reason to let the latter
two override a failed gradient condition. The V2 recommendation is therefore
to retain \(10^{-5}\) as the actual convergence requirement and store the
windowed quantities for diagnosis. If a future, independently preregistered
synthetic suite produces stable-action/above-gradient cases, a combined rule
could be reviewed then; it is not adopted here.

## 6. Determinant-sector certification

For a possible V2 amendment, certify each disconnected component separately:

1. initialize exactly two deterministic starts in det(+1) and two in det(-1);
2. prohibit determinant correction or cross-component projection;
3. require at least one finite, orthogonal, strict-gradient-converged candidate
   in each initial sector;
4. require determinant sign preservation throughout the recorded iterates;
5. retain all near-optimal converged candidates under the already frozen
   objective-equivalence tolerance;
6. if either sector lacks a certified candidate, return
   `UNASSESSED_TECHNICAL_FAILURE` with no available-sector scientific vote.

The stress audit observed determinant preservation and orthogonality error
below \(10^{-10}\) for every optimizer run. TrustRegions supplied both-sector
certification in all 12 fixtures. No SVD last-axis correction was used.

## 7. Recommendation and expected runtime

For review in a separate V2 protocol amendment, the recommended primary
single-action optimizer is **Pymanopt TrustRegions with the analytic
Hessian-vector product of the unchanged squared-Frobenius conjugation loss**.
Use the same four deterministic starts, both determinant sectors, maximum
1,000 iterations, and strict \(10^{-5}\) gradient requirement. Keep current CG
as an engineering cross-check, not as a fallback chosen after seeing a real
cell. SteepestDescent is not recommended.

The recommendation is based only on the preregistered synthetic robustness
endpoint: TrustRegions certified 12/12 fixtures and 48/48 starts, compared with
10/12 and 42/48 for CG. It is not based on BNCI gain or prediction performance;
none was computed.

Synthetic compute-only extrapolation for the 1,728 four-start primary fit
objects is approximately 95 seconds at the mean stress-suite cost on eight
workers, or about 8.9 minutes if every fit cost matched the slowest synthetic
fixture. For the maximum 4,608 primary-plus-semantic fit objects, the analogous
figures are about 4.2 and 23.6 minutes. Allowing process startup, checkpointing,
commutant SVDs, and real-matrix variability, a cautious planning range is
**3–12 minutes for primary Stage A** and **8–30 minutes if the semantic stage
unlocks**. These are engineering projections, not scientific thresholds or a
promise about a future run.

Before any V2 real-data run, a separate protocol amendment would also need to
freeze and test persistence of per-iteration objective, gradient, accepted/
rejected trust-region decision, trust radius/step norm, orthogonality, and
determinant diagnostics. Logging must observe the standard optimizer without
changing its update.

## 8. What remains unchanged

This audit recommends a possible numerical implementation amendment only.
The following remain exactly unchanged:

- the sensor-space action \(U\mapsto Q U Q^{\mathsf T}\), \(Q\in O(22)\);
- the squared-Frobenius identity-tangent scientific objective;
- the pairwise three-fit-class/one-held-out-class LOCO design;
- source/target/session exclusions and all four held-out folds;
- split-half predictive-identifiability testing;
- target-subject aggregation and inferential unit;
- unrelated-action and semantic controls, seeds, null counts, and PASS logic;
- claim restrictions and the interpretation that pairwise success is only a
  necessary consequence, not proof of a global latent model.

No frozen protocol is amended by this memo. No BNCI rerun is authorized or
performed. The next action, if any, is human review of whether to author a
separate pre-result V2 amendment.
