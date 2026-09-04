# Frozen Protocol: Baseline-Referenced SPD Trajectory Identifiability V0

Version 0.0. Freeze date 2026-09-04, Asia/Seoul. This protocol becomes
immutable before any real classification, clustering, semantic-permutation,
or action-bridge result is inspected.

## Question and stopping rule

The experiment asks whether a cue-pre baseline-relative local SPD object adds
transferable class/correspondence information beyond full-trial covariance.
It does not implement a canonicalizer, conditional alignment, SPDHSW/SPDSW
loss, pseudo-labeling, entropy minimization, target optimization, Transformer,
Mamba, or a hyperparameter/window/filter/covariance sweep.

Phase 1 is forbidden unless the predeclared Phase 0 decision unlocks it.

## Data contract

BNCI2014_001: subjects 1-9, sessions `0train` and `1test`, six runs/session,
four classes, 22 ordered EEG channels, 250 Hz. Raw stim events mark trial
start; raw annotations mark cue onset exactly 2 s later. Continuous runs are
filtered once at 8-32 Hz with the lineage MNE IIR implementation before
epoching. The order-4 Butterworth SOS filter is zero-phase forward-backward;
therefore it can mix information across cue time. Run-edge padding is
`reflect_limited`; no trial/window is independently filtered.

C0 is `[cue-1.0 s, cue)` (250 samples). Post-cue is `[cue, cue+4.0 s)`
(1,000 samples). C1..C5 are five ordered, nonoverlapping 200-sample windows.
OAS covariance is float64, numerically symmetrized, without trace
normalization, added jitter, or outcome-tuned floor. Any nonfinite/non-SPD
matrix fails closed. The real data audit must reproduce frozen post-cue
WINDOW5 at `atol=rtol=1e-12` or stop with a separately named version, filter,
raw-event, or numerical mismatch.

## Geometry and features

All spectral operations use float64 symmetric eigendecomposition without
clipping.

- F0: `svec(log(Rtrain^-1/2 Cfull Rtrain^-1/2))`.
- F1: concatenate the five `svec(log(Rtrain^-1/2 Ct Rtrain^-1/2))` states.
- F2-S: let `Bt=C0^-1/2 Ct C0^-1/2`, `Lt=log(Bt)`; concatenate
  `svec(L1)..svec(L5)`.
- F2-V: `D1=L1`, `Dt=Lt-L(t-1)` for t=2..5; concatenate `svec(Dt)`.
- F3-G: upper triangle of `Kst=<Ls,Lt>F`, 15 dimensions.
- F3-D: five AIRM distances `d(C0,Ct)` plus four adjacent speeds.
- F2-S-SHUFFLE: each trial receives one deterministic, independent,
  label-free, nonidentity permutation of its five states. The mapping is
  persisted.
- F2-S-REVERSE: exact reversal of the five states.

`Rtrain` is the AIRM mean of training trials only. F2-S is not called exactly
GL-invariant: a common congruence leaves an orthogonal-conjugation ambiguity
after baseline whitening. It is a partial canonicalization/equivariant
coordinate. F3-G is invariant to that common congruence, but is an extreme
invariant that may discard sensor orientation/laterality. F3-D is only a
low-dimensional interpretive control. No post-result concatenated rescue
feature is allowed.

## Fixed classifier and metrics

Every vector feature uses exactly:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])
```

Scaler/LDA fit training rows only. Primary metric is balanced accuracy;
accuracy, macro-F1, and fixed-class-order confusion matrices are secondary.
There is no tuning and no target-label model/feature selection.

## Evaluation protocols

- P0: within each subject/session, stratified trial-grouped 5-fold sanity.
- P1: each subject in both directions, `0train -> 1test` and reverse; average
  directions within subject.
- P2 primary: LOSO, target subject wholly absent from all transforms and fits;
  equal source subject-by-class weighting; fully inductive.
- P3 secondary: LOSO subject RA. Each subject's unlabeled full-trial AIRM mean
  defines one congruence applied to C0, C1..C5, and Cfull. The target mean uses
  every unlabeled target trial and is marked `TRANSDUCTIVE CALIBRATION`.
  Target labels are unused. P3 never replaces P2. F3 invariance before/after
  RA is a numerical test.

## Phase 0-A task gates

Co-primary candidates are F2-S and F3-G. For each versus F0, report mean and
median subject delta, wins/losses/ties, the exact 512-pattern paired sign-flip
test, Holm correction over the two candidates, and descriptive 95% subject
bootstrap CI.

T-noRA requires one candidate on P2 no-RA to have mean delta at least +0.050,
at least 6/9 positive subjects, and Holm-adjusted p at most 0.05.

T-RA requires one candidate on P3 to have mean delta at least +0.020, at least
6/9 positive subjects, and Holm-adjusted p at most 0.05.

The same passing candidate must be cross-session consistent on P1: mean delta
at least -0.010 and at least 5/9 positive subjects.

ORDER is evaluated on primary P2 no-RA, F2-S minus F2-S-SHUFFLE. It requires
mean delta at least +0.010, 6/9 positive subjects, and exact sign-flip p at
most 0.05. A task result without ORDER is called
`BASELINE_RELATIVE_LOCAL_CONFIGURATION`, never a trajectory-dynamics result.

## Phase 0-B unlabeled class identifiability

For every LOSO target and F0/F1/F2-S/F2-V/F3-G, fit StandardScaler and PCA on
source rows only. PCA uses `min(20,n_features,n_source-1)`, no whitening, and
is diagnostic only. Target KMeans uses K=4, `n_init=50`, seeds 0..19.

After clustering only, target labels score NMI, ARI, Hungarian-matched
accuracy, purity, entropy, and size imbalance. Target-only leave-one-out
neighbour metrics are 1-NN same-class fraction, 5-NN class purity, and
different-class-neighbour fraction. Source-prototype semantic assignment uses
equal-weight source-subject class centroids and a Hungarian centroid-cost
assignment; target labels do not enter assignment and score only its final
mapping accuracy/BA.

IDENT-CLUSTER requires F2-S or F3-G versus F0 to have median-subject NMI
improvement at least +0.050, median-subject Hungarian BA improvement at least
+0.050, and at least 6/9 positive Hungarian-BA subjects. Subject 2 and the
worst F0 subject are mandatory descriptive rows, not individual gates.

## Phase 0-C common-action semantic bridge

The known-correspondence V2 result is not rerun. Its TrustRegions scientific
contract is reused: O(22), four deterministic starts, two in each determinant
sector, analytic gradient/Hessian, maximum 1,000 iterations, gradient gate
`1e-5`, no optimizer change after results.

A0 is static WHOLE zero-label KMeans/permutation recovery. A1 forms true-label
target class components and then hides their names. The 36 oracle cells are
predeclared as 9 target subjects x 2 source-template sessions x 2 target
sessions. For every one of 24 permutations, each of four classes is held out
in turn; one common action is fit on the remaining three class correspondences
and the four held-out errors are summed. True names only score the selected
permutation. A2 replaces true components with K=4 target clusters in the
source-fitted F2-S coordinate, records all seeds 0..19, and applies the same
24-permutation leave-one-component-out score. The permutation selector has no
target-label argument.

Static WHOLE, absolute F1, baseline-relative F2-S, semantic-permutation null,
and unrelated-target action null use identical optimizer/start/determinant
contracts.

ACTION-ORACLE requires F2-S versus A0 semantic-permutation accuracy improvement
at least +0.10 and true permutation ranked first in at least 27/36 cells.
ACTION-ZERO requires median target-subject mapping-BA improvement at least
+0.050 and at least 6/9 positive subjects.

Oracle pass with zero-label failure is
`ORACLE_STRUCTURE_PRESENT_BUT_COMPONENT_RECOVERY_BOTTLENECK`; both fail is
`NO_TRAJECTORY_SEMANTIC_IDENTIFIABILITY`; zero-label pass is
`TRAJECTORY_BREAKS_SEMANTIC_PERMUTATION`.

## Optional Phase-P bridge

T3DA D-star predictability is run only after exact equality of dataset,
subject/session/run, trial and label order, epoch, channel order, and class
encoding. Without the exact fold cache/trial identity it is recorded as
`NOT_RUN_EXACT_TRIAL_IDENTITY_UNAVAILABLE`; approximate matching is forbidden.
The continuation signal is mean kNN R2 improvement at least +0.10 and subject
2 improvement at least +0.10. It cannot rescue a Phase 0 gate.

## Frozen decision tree

- CASE 0: T-noRA, T-RA, IDENT-CLUSTER, and ACTION-ZERO all fail ->
  `STOP_BASELINE_RELATIVE_TRAJECTORY_LINE`.
- CASE 1: a task or identifiability gate passes but ORDER fails ->
  `GO_BASELINE_RELATIVE_LOCAL_CONFIGURATION`.
- CASE 2: a task gate and ORDER pass ->
  `GO_ORDERED_BASELINE_RELATIVE_TRAJECTORY`.
- CASE 3: IDENT-CLUSTER or ACTION-ZERO passes while task gates fail ->
  `GO_IDENTIFIABILITY_CANONICALIZATION_ONLY`; no weekend GRU.
- CASE 4: ACTION-ORACLE alone passes ->
  `STOP_ZERO_LABEL; ORACLE_TRAJECTORY_STRUCTURE_ONLY`.

Phase 1 is allowed only for CASE 2; CASE 1 additionally needs LOSO task delta
at least +0.030 and 6/9 wins. The predeclared near miss is no-RA delta at least
+0.020, ORDER pass, and P1 degradation at least -0.010. Otherwise model status
is `NOT_RUN`.

If unlocked, M0 is a dimension-8 one-layer static SPDNet/LogEig linear head;
M1 is parameter-matched DeepSets over five baseline-relative latent logs; M2
is a one-layer unidirectional GRU (input 36, hidden 64); M3 receives movement;
M4 receives projected state+movement; M5 is a paired deterministic nonidentity
shuffle; M6 is reverse-evaluation only. BiMap is 22->8, ReEig threshold `1e-4`.
SPD spectral operations are float64 and GRU/head may be float32 with an audited
boundary. LOSO outer validation is source-subject grouped only. Seeds 0/1/2,
100 epochs, Adam 1e-3, weight decay 1e-4, batch 64, gradient clip 5, patience
15, source validation BA. Target labels are final evaluation only. No sweep.

The model gate requires M2 or M4 versus M0 mean LOSO delta +0.020, 6/9 wins,
exact p<=0.05, and mean delta versus M1 +0.010. An ordered claim additionally
requires +0.010 versus M5 and 6/9 wins.

## Interpretation boundary

No result identifies physiological sources or a unique subject mixing matrix,
establishes causal chronological order, proves F3 is a complete quotient,
recovers a full conditional distribution without labels, or by itself
justifies a canonicalizer or SPDHSW. A failure is limited to this frozen
BNCI2014_001 band, epoch, covariance, descriptor, and zero-label bridge.
