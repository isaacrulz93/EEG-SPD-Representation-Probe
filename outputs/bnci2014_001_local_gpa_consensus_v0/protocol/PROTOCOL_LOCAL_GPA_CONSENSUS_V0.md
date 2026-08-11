# Protocol: Local GPA Consensus V0

## 1. Frozen question and lineage

This is Stage 2A of the local SPD-configuration anatomy. The immutable Stage-1 reference is commit `796f04e7970972175a660a521caff47c83e0295f`, whose terminal decision was `STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0`. Stage 1 is neither rerun nor reinterpreted here.

Stage 2A asks whether the **cell-level mean orbit** of a full, locally centered, five-point SPD configuration is reproducible for the same BNCI2014_001 subject and MI class across sessions after removing one common orthogonal registration and the five point labels. It does not estimate or compare mean poses.

The branch is `pilot/local-gpa-consensus-v0`. All new scientific artifacts are confined to `outputs/bnci2014_001_local_gpa_consensus_v0/`.

## 2. Frozen EEG object and reproduction gate

The input is exactly the frozen WINDOW5 object used by the trajectory and local-metric audits:

- BNCI2014_001; all 9 subjects; sessions `0train` and `1test`;
- four classes in the order `left_hand`, `right_hand`, `feet`, `tongue`;
- six runs per session and 72 trials per subject × session × class;
- 22 ordered EEG channels, 250 Hz, 8–32 Hz, cue-relative 0–3.996 s;
- 1,000 samples per trial, five nonoverlapping 200-sample windows;
- float64 pyRiemann OAS covariance, no added regularization;
- AIRM geometry.

The session-0 and session-1 covariance-file SHA-256 values, covariance-array content hashes, metadata balance, and trial order are checked. From the covariance arrays, all 5,184 frozen 5×5 AIRM distance matrices must reproduce with maximum absolute difference no greater than `1e-12`. Failure yields `UNASSESSED_TECHNICAL_FAILURE_REPRODUCTION` and no Stage-2A statistic.

## 3. Trial-local centering

For trial (t), let (B_t=(C_{t1},\ldots,C_{t5})\). Its AIRM Fréchet mean is

\[
G_t=\operatorname{FM}_{\rm AIRM}(C_{t1},\ldots,C_{t5}),
\qquad
\bar C_{ti}=G_t^{-1/2}C_{ti}G_t^{-1/2}.
\]

The public pyRiemann AIRM mean is called with tolerance `1e-9`, maximum 100 iterations, `init=None`, and float64 input. A warning is a hard numerical failure. The mean is recomputed on the centered states and its normalized Karcher residual must be at most `1e-7`; its identity error must be at most `1e-6`.

This is trial-local centering only. No subject/session marginal center is applied. Window order is not used.

## 4. Quotient configuration distance

For centered configurations (X,Y\in\mathrm{SPD}(22)^5\),

\[
\rho^2(X,Y)=\frac15\sum_{i=1}^5d_{\rm AIRM}^2(X_i,Y_i),
\]

and

\[
d_Q([X],[Y])=
\min_{Q\in O(22),\,\pi\in S_5}
\rho\!\left(X,(Q,\pi)\cdot Y\right).
\]

The point assignment is exactly one of the 120 vertex permutations in (S_5\), never an arbitrary permutation of ten edges. The orthogonal action is (Y_i\mapsto QY_iQ^\top\). Both determinant components of (O(22)\) are searched.

## 5. Fixed-assignment orthogonal registration

For a fixed point permutation, the numerical subproblem is the exact scientific AIRM objective

\[
L(Q)=\frac15\sum_i d_{\rm AIRM}^2(X_i,QY_{\pi(i)}Q^\top).
\]

It is optimized on Pymanopt 2.2.1 `Stiefel(d,d,retraction="polar")`, which represents both connected components of (O(d)\). The optimizer is `ConjugateGradient(beta_rule="HestenesStiefel")` with an explicitly constructed `BackTrackingLineSearcher`:

- initial step 1.0;
- contraction factor 0.5;
- sufficient decrease `1e-4`;
- optimism 2.0;
- line-search iterations 50;
- action iterations 250;
- minimum Riemannian gradient norm `1e-6`;
- minimum step size `1e-12`;
- maximum cost evaluations 251;
- maximum time 120 seconds.

The Euclidean gradient is analytic and was checked by tangent directional finite differences. Pymanopt provides tangent projection, the polar retraction, line search, and CG update.

At fixed (Q), all 120 assignments are evaluated exactly. Assignment and action updates alternate for at most 8 iterations and stop only when the assignment is unchanged and the action solve is certified.

### Deterministic four-start contract

There are exactly four starts in each full registration certification, with no hidden fifth start:

1. rank all 120 point assignments by mismatch of the two internal 5×5 AIRM distance matrices;
2. take the two lowest-cost assignments, using stable lexicographic tie order;
3. for each assignment, construct a deterministic spectral action from weighted sums of identity-tangent logs and deterministic eigenvector signs;
4. pair each spectral action with its product by the fixed reflection `diag(-1,1,...,1)`.

Thus there are two starts in each determinant component. Internal distance geometry is used only to select numerical basins; the fitted objective and returned distance remain the full SPD AIRM configuration objective. A full certification requires at least one converged candidate in each determinant component. Every start's objective, gradient norm, determinant, iterations, and convergence status is retained; the best converged objective is used.

For exactly symmetric computation, a pair is placed in deterministic SHA-256 array order before fitting. Therefore calls with ((X,Y)) and ((Y,X)) execute the same symmetric scientific minimization and return exactly the same numerical value.

## 6. Exact centered-prototype parameterization

Every accepted prototype is parameterized as

\[
P_i=\exp(Z_i),\qquad Z_i=Z_i^\top,\qquad \sum_iZ_i=0.
\]

At base point (I), the AIRM Karcher equation is (sum_i\log(P_i)=0\). Since the SPD/AIRM manifold has a unique Fréchet mean, this parameterization enforces `FM_AIRM(P)=I`. The implementation never fits an unconstrained prototype and then silently recenters it.

With registrations fixed, the exact AIRM objective is differentiated through the symmetric matrix exponential with SciPy's Fréchet derivative. The five log-matrix gradients are projected by subtracting their pointwise mean, and an Armijo step remains in the zero-sum subspace. Settings are: initial step 1, contraction 0.5, sufficient decrease `1e-4`, 30 line-search iterations, minimum step `1e-12`, and at most 16 prototype steps per GPA outer block.

## 7. Deterministic cell GPA

For a cell containing (n\) centered trials, the two total GPA starts use trial indices 0 and `n//2` as feasible prototype representatives. Each start performs at most 24 block-coordinate outer iterations.

- First outer block: every trial registration receives the full four-start/two-component certification.
- Intermediate blocks: the currently best nuisance basin is continued with exactly one warm start. This is explicitly a continuation, not a hidden multistart.
- Prototype block: up to 16 exact constrained projected-gradient steps.
- Accepted final prototype: every trial is registered again with the full four-start/two-component certification.

The GPA start converges only when its fixed-registration projected prototype gradient is at most `2e-5` and relative objective change is at most `1e-7`. Both GPA starts must converge. The lower objective is the frozen consensus representative; no global-optimum claim is made.

The two GPA objectives are numerically near-equivalent when both are at most

\[
L_{\rm best}+10^{-10}+10^{-8}|L_{\rm best}|.
\]

When this occurs, the two returned prototypes must have quotient distance at most `1e-4`. Otherwise the cell yields `UNASSESSED_GPA_NUMERICAL_FAILURE`. If the two objectives are not near-equivalent, their spread is reported and the lower deterministic multistart objective is retained. This rule was frozen from exact/noisy synthetic checks, not BNCI cell outcomes.

For every full or split cell, the prototype constraint residual, objective, RMS within-cell Procrustes dispersion, best/second objectives, objective spread, outer iterations, gradients, and registration diagnostics are saved. Registration (Q,\pi\) values are numerical/debug artifacts only. They are never averaged, compared across cells, or interpreted.

## 8. Synthetic numerical gate

Before the protocol commit and before any Stage-2A BNCI consensus is computed, the implementation must pass:

- identical configurations give quotient distance approximately zero;
- known (Q\) and known (S_5\) relabeling give approximately zero for det +1 and det −1 truth;
- quotient distance symmetry;
- action and constrained-prototype gradient finite differences;
- two GPA initializations recover the same orbit on a known-answer cell;
- d=22 exact registration with four total starts and both determinant components.

The d=22 action benchmark, all tested settings, and exact known-answer errors are written to `synthetic_numerical_validation.json`.

## 9. Full and split-half cell consensuses

Exactly 72 full-cell consensus orbits are fitted: 9 subjects × 2 sessions × 4 classes, each using all 72 trials.

The non-gating split-half diagnostic independently fits 144 more consensuses:

- half A: runs `{0,2,4}` (36 trials/cell);
- half B: runs `{1,3,5}` (36 trials/cell).

The halves are disjoint and exhaustive. For every cell,

\[
R_a=d_Q([P_a^A],[P_a^B]).
\]

All 72 (R_a\) values are reported without a post-hoc reliability threshold. Numerical GPA failures remain gating; scientifically large split-half variation is descriptive and non-gating.

## 10. Cross-session cell distance and structural contrasts

For the 36 subject×class labels (a,b\),

\[
M_{01}[a,b]=d_Q([P_{a,0}], [P_{b,1}]),
\qquad
M=\tfrac12(M_{01}+M_{01}^\top).
\]

For anchor ((s,c)\), define (a_{sc}\) as the same-subject/same-class diagonal; (b_{sc}\) as the mean over the other three classes of the same subject; (c_{sc}\) as the mean over the other eight subjects of the same class; and (d_{sc}\) as the mean over the other eight subjects and other three classes. Then

\[
J_{sc}=b_{sc}+c_{sc}-a_{sc}-d_{sc},\quad
J_s=\frac14\sum_cJ_{sc},\quad
T_J=\frac19\sum_sJ_s.
\]

The target subject is the group unit. Supporting contrasts are (S_{sc}=c_{sc}-a_{sc}\) and (C_{sc}=b_{sc}-a_{sc}\), aggregated first over class and then subject. Supporting specificity alone is not an interaction claim.

## 11. Frozen correspondence nulls

The already estimated consensus cells are relabeled; GPA is never rerun within a null. Both families use master seed `20260810`, 1,999 draws, and one-sided greater-or-equal plus-one p-values.

- Class-break: independently permute the four session-1 class cell identities within each subject, preserving subject.
- Subject-break: independently permute the nine session-1 subject cell identities within each class, preserving class.

The primary interaction passes only when (T_J>0\), `p_classbreak < 0.05`, and `p_subjectbreak < 0.05`. The conservative reported p-value is their maximum. The subject-break null supports (T_S\), and the class-break null supports (T_C\), without overriding the primary decision.

## 12. Terminal logic

- Passing all three primary requirements: `GO_STABLE_SUBJECT_CLASS_QUOTIENT_MEAN_CONFIGURATION`.
- Otherwise: `STOP_NO_STABLE_QUOTIENT_MEAN_CONFIGURATION_INTERACTION`.
- Untrustworthy GPA/quotient optimization: `UNASSESSED_GPA_NUMERICAL_FAILURE`.
- Other blocking failure: `UNASSESSED_TECHNICAL_FAILURE` (or the explicit reproduction failure above).

After the first real Stage-2A consensus/statistic is observed, local centering, the constrained GPA objective and parameterization, optimizers, starts, (S_5\) handling, split halves, cell aggregation, (J\), nulls, and alpha are immutable.

## 13. Claims and exclusions

Allowed positive wording is limited to a cross-session-reproducible subject-specific class interaction in the pose-quotiented cell-level mean local SPD configuration. Stage 2A does not identify or compare stable subject poses, class poses, (Q_s\), neural orientation, physical head orientation, or anatomical orientation. It does not perform mean-Q analysis, synchronization, cycle consistency, hierarchical GPA, adaptation, or classifier development.

A positive result is not a rescue of Stage 1; it would locate an interaction in a richer object. A negative result means neither the distance-only Stage-1 object nor the frozen Stage-2A quotient mean-configuration object supports the interaction.
