# BNCI2014_001 Trajectory Anatomy v0 — Preregistered Protocol

Protocol name: **Trajectory Anatomy v0**  
Protocol version: **0.0**  
Freeze date: **2026-08-09** (Asia/Seoul)  
Branch: **pilot/trajectory-anatomy-v0**  
Base commit: **f76785d4c06ddc3cf2f1f7a6310d4fb153a19ba9**  
Master seed: **20260809**  
Primary dataset/session: **BNCI2014_001 / 0train only**  
Primary geometry: affine-invariant Riemannian metric (**AIRM**)  
Secondary robustness geometry: **Log-Euclidean (LE)**  
Primary evaluation metric: balanced accuracy (**BA**)

이 문서는 결과 계산 전에 데이터 범위, five-state covariance construction, ordered
trajectory와 unordered controls, intrinsic quantities, null distributions, numerical gates,
output schemas와 판정 규칙을 고정한다. 이 pilot은 새 모델이나 alignment method를
제안하지 않는다. 결과를 본 뒤 representation, seed, replicate 수, threshold,
classifier, split 또는 verdict rule을 바꾸지 않는다. 변경이 필요하면 기존 산출물을
덮어쓰지 않고 별도 protocol/version/output namespace를 만든다.

## 1. Scientific scope and frozen questions

한 motor-imagery trial의 다섯 local covariance states를 세 수준으로 분리한다.

1. **PATH_D10**: 관측된 window order를 보존한 pairwise-distance representation.
2. **BAG_CANON_D10**: 동일한 five-state set의 order를 완전히 제거한
   permutation-invariant distance representation.
3. **SCALARS_11**: path length, endpoint displacement, turning, fixed-time endpoint
   geodesic deviation와 Fréchet spread를 해석 가능한 scalars로 분해한 representation.

Frozen research questions are:

- **RQ1**: Session 0train의 unordered five-state covariance set 자체에 class signal이
  있는가?
- **RQ2**: 같은 state set에서 관측된 temporal order가 class signal에 추가로
  기여하는가?
- **RQ3**: intrinsic scalars의 variation은 subject, class, subject×class interaction,
  residual 중 어디에 놓이는가?
- **RQ4**: AIRM 방향이 LE robustness repetition에서도 같은가?

높은 accuracy나 SOTA는 성공 조건이 아니다. Negative, mixed, failed result를 정해진
상태 그대로 보고한다.

## 2. Session-0-only data barrier

### 2.1 Included data

MOABB **BNCI2014_001**, session key **0train**만 사용한다.

- subjects: integer 1 through 9
- fixed class order: left_hand, right_hand, feet, tongue
- total trials: 2592
- trials per subject: 288
- trials per subject×class: 72
- exact run metadata strings: "0" through "5", parsed to integers 0 through 5
- trials per subject×run: 48
- trials per subject×run×class: 12

Run은 exact string을 integer로 변환한 뒤 항상 0,1,2,3,4,5 순으로 정렬한다.
Subject마다 정확히 여섯 runs와 동일한 3/3 run-half split을 만들 수 있어야 한다.
Count, class vocabulary, run set 또는 balance가 다르면 hard data failure다.

### 2.2 Forbidden session access

Session **1test**는 discovery protocol에서 열지 않는다. 다음을 모두 금지한다.

- 1test EEG, covariance, label loading
- 1test metadata를 이용한 threshold, feature 또는 plotting decision
- 1test replication, tuning, validation 또는 qualitative inspection
- 0train/1test pooling

Session filter 뒤 0train 외 row가 feature, classifier, null, factor summary 또는 figure
input에 하나라도 들어가면 hard data failure다. 1test는 이 protocol 완료 뒤 별도
preregistered replication에서만 접근할 수 있다.

## 3. Frozen preprocessing and five-state construction

V1과 동일한 0train preprocessing을 그대로 재사용한다.

- EEG only: 22 channels; EOG excluded
- band-pass: 8–32 Hz
- epoch relative to MI cue: 0.000–3.996 s
- sampling frequency: 250 Hz; no resampling
- samples per trial: 1000
- covariance estimator: OAS
- covariance dtype: float64
- covariance estimator regime: frozen V1 implementation
- no extra diagonal loading
- no eigenvalue clipping
- deterministic symmetrization only as frozen in V1

각 preprocessed trial \(X_i\in\mathbb{R}^{22\times1000}\)을 sample order 그대로 다섯
non-overlapping windows로 자른다.

\[
X_i=(X_{i,1},X_{i,2},X_{i,3},X_{i,4},X_{i,5}),\qquad
X_{i,j}\in\mathbb{R}^{22\times200}.
\]

- window count: exactly 5
- samples per window: exactly 200
- boundaries: [0,200), [200,400), [400,600), [600,800), [800,1000)
- overlap: none
- remainder: none
- observed order: 1→2→3→4→5

각 window에 동일 OAS estimator를 독립 적용한다.

\[
C_{i,j}=\operatorname{OAS}(X_{i,j})\in\operatorname{SPD}(22),
\qquad j=1,\ldots,5.
\]

Trial identity는 sample_index, subject, session, run, trial_id, trial_uid,
class_label을 보존한다. **sample_index**는 모든 null과 fold에서 변하지 않는 고정 row
identity다.

## 4. Frozen geometries

### 4.1 Primary AIRM

\[
d_{\mathrm{AI}}(A,B)
=\left\|\log\!\left(A^{-1/2}BA^{-1/2}\right)\right\|_F.
\]

\[
\operatorname{Log}_{C}(B)
=C^{1/2}\log(C^{-1/2}BC^{-1/2})C^{1/2},
\]

\[
\langle U,V\rangle_C
=\operatorname{Tr}(C^{-1}UC^{-1}V).
\]

Symmetric EVD로 matrix power/log를 계산하고 output을 symmetrize한다. 유효 positive
eigenvalue를 조용히 clip하지 않는다.

Five-state AIRM Fréchet mean:

\[
M_i^{\mathrm{AI}}
=\arg\min_{M\in\operatorname{SPD}(22)}
\sum_{j=1}^{5}d_{\mathrm{AI}}(M,C_{i,j})^2.
\]

Solver is frozen to tolerance \(10^{-9}\), maximum 100 iterations. Independently computed
normalized Karcher residual must be at most \(10^{-7}\).

### 4.2 Secondary Log-Euclidean

\[
d_{\mathrm{LE}}(A,B)=\|\log A-\log B\|_F.
\]

Let \(S_j=\log C_j\). LE scalar robustness uses metric-consistent log-space geometry:

- tangent displacement at \(S_j\): ordinary symmetric-matrix difference
- angle inner product: Frobenius inner product
- endpoint geodesic:
  \[
  G_{\mathrm{LE}}(t)=
  \exp\!\left((1-t)\log C_1+t\log C_5\right)
  \]
- barycenter:
  \[
  M_{\mathrm{LE}}=\exp\!\left(\frac15\sum_{j=1}^{5}\log C_j\right).
  \]

LE is a fixed secondary robustness repetition. It is not an extra vote that can rescue or replace
the primary AIRM verdict.

## 5. Distance matrix and frozen representations

For geometry \(g\in\{\mathrm{AIRM},\mathrm{LE}\}\):

\[
D^{(g)}_{i,jk}=d_g(C_{i,j},C_{i,k}),
\qquad j,k\in\{1,\ldots,5\}.
\]

### 5.1 PATH_D10

\[
\operatorname{PATH\_D10}_i=
[d_{12},d_{13},d_{14},d_{15},d_{23},d_{24},d_{25},d_{34},d_{35},d_{45}].
\]

Stored column order is exactly
**d12,d13,d14,d15,d23,d24,d25,d34,d35,d45**.

### 5.2 BAG_CANON_D10

Sort all 120 \(S_5\) permutations lexicographically. For permutation \(\pi\):

\[
v_i(\pi)=
[D_{\pi_1\pi_2},D_{\pi_1\pi_3},D_{\pi_1\pi_4},D_{\pi_1\pi_5},
D_{\pi_2\pi_3},D_{\pi_2\pi_4},D_{\pi_2\pi_5},D_{\pi_3\pi_4},
D_{\pi_3\pi_5},D_{\pi_4\pi_5}].
\]

\[
\operatorname{BAG\_CANON\_D10}_i
=\min_{\mathrm{lex}}\{v_i(\pi):\pi\in S_5\}.
\]

If several permutations attain the same vector, store the lexicographically smallest permutation
as audit identity. Columns are **bag01** through **bag10**, followed by
**canonical_permutation**, serialized in 1-based form such as "1-2-3-4-5".

### 5.3 BAG_SORTED_D10 secondary control

\[
\operatorname{BAG\_SORTED\_D10}_i
=\operatorname{sort}_{\uparrow}
\{D_{i,jk}:1\le j<k\le5\}.
\]

Columns are **sorted01** through **sorted10**. This is a coarser secondary unordered control.
The frozen BAG hypothesis always refers to BAG_CANON_D10.

### 5.4 LOCAL_BARYCENTER and WHOLE-1000

**LOCAL_BARYCENTER** is the AIRM mean \(M_i^{\mathrm{AI}}\) of the same five local states.
It is evaluated with secondary **MDM(metric="riemann")**.

**WHOLE-1000** is the V1 OAS covariance estimated directly from all 1000 trial samples and is
evaluated with the same Riemannian MDM as contextual control. WHOLE-1000 uses one 1000-sample OAS,
whereas LOCAL_BARYCENTER is derived from five 200-sample OAS estimates. Their difference is
therefore confounded by covariance-estimator sample-size regime and cannot establish method
superiority.

## 6. Intrinsic trajectory quantities

### 6.1 Steps, length, endpoint and efficiency

\[
s_{i,j}=d(C_{i,j},C_{i,j+1}),\qquad j=1,\ldots,4,
\]

\[
L_i=\sum_{j=1}^{4}s_{i,j},\qquad
E_i=d(C_{i,1},C_{i,5}),
\]

\[
\operatorname{efficiency}_i=E_i/L_i,\qquad
\operatorname{excess}_i=L_i-E_i.
\]

If \(L_i\le10^{-12}\), flag the trial degenerate and fail the gate. Do not define, delete or impute
its efficiency.

### 6.2 Intrinsic turning angles

For AIRM and interior state \(j=2,3,4\):

\[
u_{i,j}=-\operatorname{Log}_{C_{i,j}}(C_{i,j-1}),\qquad
v_{i,j}= \operatorname{Log}_{C_{i,j}}(C_{i,j+1}),
\]

\[
\theta_{i,j}=
\arccos\!\left(
\operatorname{clip}_{[-1,1]}
\frac{\langle u_{i,j},v_{i,j}\rangle_{C_{i,j}}}
{\|u_{i,j}\|_{C_{i,j}}\|v_{i,j}\|_{C_{i,j}}}
\right).
\]

Zero tangent norm is a numerical failure. For LE, use log-space vectors
\(-(S_{j-1}-S_j)\) and \(S_{j+1}-S_j\) with Frobenius inner product and the same clipped-cosine
formula.

### 6.3 Fixed-time endpoint-geodesic deviation

AIRM endpoint geodesic:

\[
G_i(t)=A_iH_i^tA_i,\qquad
A_i=C_{i,1}^{1/2},\quad
H_i=C_{i,1}^{-1/2}C_{i,5}C_{i,1}^{-1/2}.
\]

\[
\operatorname{dev}_{i,2}=d(C_{i,2},G_i(1/4)),\quad
\operatorname{dev}_{i,3}=d(C_{i,3},G_i(2/4)),\quad
\operatorname{dev}_{i,4}=d(C_{i,4},G_i(3/4)).
\]

These are distances to the **fixed time point \(G_i(j/4)\)**. They are not minimum distances to
the endpoint geodesic set, and no optimization over \(t\) is allowed. LE uses its Section 4
fixed-time geodesic at the same \(1/4,2/4,3/4\).

Synthetic checks require relative endpoint errors for \(G(0)=C_1\) and \(G(1)=C_5\) at most
\(10^{-10}\).

### 6.4 Fréchet spread and diameter

For geometry-native mean \(M_i\):

\[
\operatorname{frechet\_variance}_i
=\frac15\sum_{j=1}^{5}d(C_{i,j},M_i)^2,
\]

\[
\operatorname{frechet\_radius\_mean}_i
=\frac15\sum_{j=1}^{5}d(C_{i,j},M_i),
\]

\[
\operatorname{diameter}_i=\max_{1\le j<k\le5}D_{i,jk}.
\]

### 6.5 Exact 11-scalar probe

The classifier/factor vector order is frozen:

1. total_path_length
2. endpoint_distance
3. efficiency
4. excess
5. mean_turn
6. max_turn
7. mean_geodesic_deviation
8. max_geodesic_deviation
9. frechet_variance
10. frechet_radius_mean
11. diameter

Here mean_turn is the mean of theta2 through theta4, and mean_geodesic_deviation is the mean of
dev2 through dev4. **s1–s4, theta2–theta4, dev2–dev4 are descriptive-only**: they are stored but
not added to SCALARS_11 or the 11-scalar factor grid.

## 7. AIRM subject-centering isometry cross-check

Fit the V2 WHOLE AIRM subject mean \(M_s^{\mathrm{WHOLE,AI}}\) from that subject's 288 unlabeled
0train WHOLE covariances. Apply it only to every local state of the same subject:

\[
\widetilde C_{i,j}=
(M_s^{\mathrm{WHOLE,AI}})^{-1/2}\,
C_{i,j}\,
(M_s^{\mathrm{WHOLE,AI}})^{-1/2}.
\]

For all 2592 trials, original versus centered-local \(D\), PATH_D10, BAG_CANON_D10 and \(L\)
must be invariant. This is an implementation cross-check, not a classifier condition. Do not fit a
duplicate centered-local classifier or give it a verdict vote.

## 8. Numerical and data hard gates

All scientific probes and nulls wait for every required gate.

### 8.1 Counts, identities and SPD usability

- exact counts and 0train barrier from Section 2
- globally unique trial_uid and sample_index
- five local covariances with window indices exactly 1 through 5
- all covariances/logs/distances/features finite
- relative symmetry Frobenius error at most \(10^{-12}\)
- minimum eigenvalue strictly greater than zero
- condition number at most \(10^{12}\)
- no loading, clipping, silent replacement, deletion or imputation

### 8.2 Distance metrics

For AIRM and LE, every trial must satisfy:

- maximum absolute symmetry error at most \(10^{-10}\)
- maximum absolute diagonal error at most \(10^{-12}\)
- every distance at least \(-10^{-12}\)
- all 10 unordered triples from five states satisfy all three cyclic triangle inequalities
  (30 inequalities per trial); a material violation exceeds \(10^{-10}\) absolute plus
  \(10^{-10}\) relative to the compared two-edge path

### 8.3 Means, barycenters and invariance

- every LOCAL_BARYCENTER finite, symmetric, SPD and condition number at most \(10^{12}\)
- AIRM mean tolerance \(10^{-9}\), maximum 100 iterations
- normalized Karcher residual at most \(10^{-7}\)
- any required mean warning closes the gate
- across all 2592 trials, AIRM-centered versus original \(D\), PATH, BAG and \(L\) maximum
  absolute and relative errors are each at most \(10^{-10}\)

### 8.4 All-permutation BAG gate

Use exactly 36 trials: within each numeric subject×fixed-order class, choose the smallest
chronological trial_id, breaking ties by sample_index and then trial_uid. For AIRM and LE, rebuild
BAG_CANON_D10 after each of all 120 lexicographic permutations and require maximum absolute
difference from the original canonical vector at most \(10^{-12}\).

### 8.5 Intrinsic constraints

- \(L\ge E\) within \(10^{-10}\) absolute plus \(10^{-10}\) relative tolerance
- no \(L\le10^{-12}\) trial
- efficiency in [0,1] with \(10^{-12}\) bound tolerance
- angles finite and in [0,\(\pi\)] with \(10^{-12}\) bound tolerance
- deviations finite and at least \(-10^{-12}\)
- synthetic endpoint-geodesic relative errors at most \(10^{-10}\)

Any failure above stops scientific evaluation. Set terminal verdict to **UNASSESSED** and
failure status exactly to **UNASSESSED — NUMERICAL/DATA FAILURE**. Persist failed identities and
diagnostics; never compute hypothesis decisions from partial data.

## 9. Frozen class LOSO information probes

### 9.1 Required observed grid

Primary AIRM conditions:

- AIRM / PATH_D10
- AIRM / BAG_CANON_D10

Secondary AIRM conditions:

- AIRM / BAG_SORTED_D10
- AIRM / SCALARS_11

LE robustness conditions:

- LE / PATH_D10
- LE / BAG_CANON_D10
- LE / SCALARS_11

LOCAL_BARYCENTER and WHOLE-1000 MDM are separately labeled contextual controls.

### 9.2 Split and fixed classifier

Nine LOSO folds are fixed. For target subject \(s\), train on the other eight subjects and evaluate
all 288 target-subject trials.

- target: fixed four class labels
- source-only preprocessing: StandardScaler fit only on source feature rows
- classifier: multinomial logistic regression
- C = 1
- solver = lbfgs
- max_iter = 5000
- tolerance = \(10^{-4}\)
- random_state = 20260809
- no class weighting, PCA, extra normalization or tuning

The evaluation boundary is the only function that receives target class labels. A convergence
warning makes the condition/fold **FAILED**; no alternative solver, scaling, iteration setting or
retry is allowed.

Primary metric is BA. Secondary metrics are accuracy, macro-F1 and per-class recall in fixed class
order; store the full confusion matrix. Group summaries contain mean, SD with ddof 1, median,
minimum and maximum across nine target subjects. Null statistics use **median subject BA**, not the
mean.

Any missing or FAILED row in a required scientific probe/null grid prohibits available-case
inference. Set terminal verdict to **UNASSESSED** and failure status exactly to
**UNASSESSED—PROTOCOL/TECHNICAL FAILURE**. PASS-only values may be listed descriptively with their
denominator, but an 8/9 substitute never receives a frozen verdict.

## 10. Order-shuffle falsification null

### 10.1 Replicates and exact seed derivation

- master seed: 20260809
- replicates: \(B=199\)
- stream tag: hexadecimal 0x4F52444552
- child construction:
  **SeedSequence([20260809, 0x4F52444552]).spawn(199)**
- stored child seed:
  **int(child.generate_state(1, dtype=np.uint64)[0])**
- replicate RNG: **default_rng(stored_seed)**

Replicate indices are 1 through 199 in child-list order. The ordered seed list and exact derivation
are stored in **nulls/order_shuffle_seeds.json**.

### 10.2 Per-trial nonidentity permutation

Construct all \(S_5\) permutations as lexicographically sorted tuples. Identity
\((1,2,3,4,5)\) is index zero. For every replicate, traverse trials in ascending sample_index and
draw one uniform integer from 1 through 119 inclusive for every trial. Apply that selected
nonidentity permutation independently to the trial's five states.

The complete mapping (replicate, sample_index) to permutation is built once before LOSO. Therefore
a trial has the same shuffled order regardless of its train/test role in a fold, while different
trials are independently permuted. Recompute PATH_D10 from the permuted distance matrix.
Recompute BAG_CANON_D10 only as the invariant negative control.

The required order-null grid is AIRM PATH_D10; LE repeats PATH_D10 as secondary robustness.
BAG_CANON_D10 is not refit as an order-null classifier because its feature vector must be
identical. SCALARS_11 and BAG_SORTED_D10 are not additional order-null votes.

Each replicate uses the exact observed LOSO/scaler/logistic procedure. For representation \(r\):

\[
T^{\mathrm{order}}_{r,\mathrm{obs}}
=\operatorname{median}_{s=1}^{9}BA_{r,s,\mathrm{obs}},
\]

\[
T^{\mathrm{order}}_{r,b}
=\operatorname{median}_{s=1}^{9}BA_{r,s,b},
\]

\[
p^{\mathrm{order}}_r
=\frac{1+\#\{b:T^{\mathrm{order}}_{r,b}\ge
T^{\mathrm{order}}_{r,\mathrm{obs}}\}}{200},
\]

\[
\operatorname{effect}^{\mathrm{order}}_r
=T^{\mathrm{order}}_{r,\mathrm{obs}}
-\operatorname{median}_{b=1}^{199}T^{\mathrm{order}}_{r,b}.
\]

Subject-level order effect:

\[
\Delta^{\mathrm{order}}_{r,s}
=BA_{r,s,\mathrm{obs}}-\operatorname{median}_{b}BA_{r,s,b}.
\]

No alternative sidedness, plus-one rule, statistic, seed count or replicate selection is allowed.

## 11. Label-destruction null

### 11.1 Replicates and exact seed derivation

- master seed: 20260809
- replicates: \(B=199\)
- stream tag: hexadecimal 0x4C4142454C
- child construction:
  **SeedSequence([20260809, 0x4C4142454C]).spawn(199)**
- child uint64 extraction and default_rng rule: identical to Section 10

The ordered list is stored in **nulls/label_permutation_seeds.json**.

### 11.2 Within-subject×run label destruction

For every replicate, sort groups by numeric (subject, run). Within each group, sort rows by
ascending sample_index and apply **rng.permutation(n_group)** to labels only. Features, identities,
subjects and runs remain fixed. This preserves class counts within every subject×run while
destroying trial-label association.

PATH_D10 and BAG_CANON_D10 use the exact same permuted label vector and identical LOSO folds in
each replicate. Required label-null conditions are AIRM/PATH_D10 and AIRM/BAG_CANON_D10. They are
the only null tests voting in H_PATH_CLASS and H_BAG_CLASS. LE observed class LOSO remains a fixed
robustness comparison, not a second label-null hypothesis family.

For \(r\in\{\mathrm{AIRM/PATH},\mathrm{AIRM/BAG\_CANON}\}\):

\[
T^{\mathrm{label}}_{r,\mathrm{obs}}
=\operatorname{median}_{s}BA_{r,s,\mathrm{obs}},\qquad
T^{\mathrm{label}}_{r,b}=\operatorname{median}_{s}BA_{r,s,b},
\]

\[
p^{\mathrm{label}}_r
=\frac{1+\#\{b:T^{\mathrm{label}}_{r,b}\ge
T^{\mathrm{label}}_{r,\mathrm{obs}}\}}{200},
\]

\[
\operatorname{effect}^{\mathrm{label}}_r
=T^{\mathrm{label}}_{r,\mathrm{obs}}
-\operatorname{median}_{b}T^{\mathrm{label}}_{r,b}.
\]

Store subject-level observed minus null-median BA. No post-result p-value family, correction or
alternative label grouping is added.

## 12. Run-half subject-information probe

This probe predicts subject ID rather than class within session 0train. Two deterministic
directions are required:

- **A_TO_B**: train runs 0,1,2; evaluate runs 3,4,5
- **B_TO_A**: train runs 3,4,5; evaluate runs 0,1,2

Each half has 144 trials per subject. Train/evaluation trial UIDs must be disjoint. Required
representations are primary-AIRM PATH_D10, BAG_CANON_D10 and SCALARS_11.

- target: subject ID 1 through 9
- StandardScaler: train-half only
- fixed logistic settings from Section 9
- metrics: BA primary, accuracy secondary
- chance reference: exactly \(1/9\)
- final representation score: arithmetic mean of the two direction scores

No random split, subject-specific tuning, 1test probe or required LE subject probe is added.

## 13. Balanced scalar variance anatomy

For each exact scalar and AIRM/LE separately, use the balanced subject×class design:
\(S=9\), \(C=4\), \(n=72\) trials per cell.

\[
SS_{\mathrm{subject}}
=Cn\sum_s(\bar y_{s..}-\bar y_{...})^2,
\]

\[
SS_{\mathrm{class}}
=Sn\sum_c(\bar y_{.c.}-\bar y_{...})^2,
\]

\[
SS_{\mathrm{interaction}}
=n\sum_{s,c}
(\bar y_{sc.}-\bar y_{s..}-\bar y_{.c.}+\bar y_{...})^2,
\]

\[
SS_{\mathrm{residual}}
=\sum_{s,c,k}(y_{sck}-\bar y_{sc.})^2,
\]

\[
SS_{\mathrm{total}}
=\sum_{s,c,k}(y_{sck}-\bar y_{...})^2.
\]

\[
\eta_q^2=SS_q/SS_{\mathrm{total}}.
\]

Required scalars, in exact order:

1. total_path_length
2. endpoint_distance
3. efficiency
4. excess
5. mean_turn
6. max_turn
7. mean_geodesic_deviation
8. max_geodesic_deviation
9. frechet_variance
10. frechet_radius_mean
11. diameter

Store all four SS and eta-squared components. SS closure relative error

\[
\frac{|SS_{\mathrm{total}}-(SS_s+SS_c+SS_{s\times c}+SS_e)|}
{SS_{\mathrm{total}}}
\]

is evaluated only when \(SS_{\mathrm{total}}>0\) and must be at most \(10^{-10}\). If
\(SS_{\mathrm{total}}=0\), mark that scalar degenerate and do not divide. This is a hard
interpretation gate: terminal verdict becomes UNASSESSED with numerical/data failure status.
No p-values, F-tests, post-hoc contrasts or result-driven scalar selection are allowed.

## 14. AIRM-primary and LE-secondary discipline

AIRM alone votes in H_PATH_CLASS, H_BAG_CLASS, H_ORDER and the terminal verdict. LE repeats:

- PATH_D10, BAG_CANON_D10 and SCALARS_11 construction
- observed class LOSO
- order-shuffle null for PATH_D10
- 11-scalar balanced factor decomposition

The robustness table stores paired subject BA, group statistics, order effects, scalar component
directions, agreement/disagreement and failures. Delta category tolerance is \(10^{-12}\):
improved if delta \(>10^{-12}\), worsened if \(<-10^{-12}\), otherwise tied. This numerical
category tolerance does not replace the hypothesis operand **effect > 0**.

LE agreement cannot rescue a failed AIRM hypothesis. LE disagreement must be reported without
selecting the better-looking geometry. LOCAL_BARYCENTER remains the AIRM mean control; it is not
redefined as LE.

## 15. Frozen hypotheses and terminal verdict

All operands use full precision, never display-rounded values.
Allowed terminal verdict tokens are exactly **GO_TRAJECTORY_ORDER**,
**GO_UNORDERED_DISTRIBUTION**, **MIXED_TRAJECTORY_SIGNAL**,
**STOP_LOCAL_TRAJECTORY_V0**, and **UNASSESSED**. Failure-status text is stored separately.

### 15.1 Hypotheses

**H_PATH_CLASS** is true iff both are true for AIRM/PATH_D10 under label destruction:

1. effect_label_PATH \(>0\)
2. p_label_PATH \(\le0.05\)

**H_BAG_CLASS** is true iff both are true for AIRM/BAG_CANON_D10 under label destruction:

1. effect_label_BAG \(>0\)
2. p_label_BAG \(\le0.05\)

Observed paired path-minus-bag operand:

\[
\Delta^{\mathrm{PATH-BAG}}_{\mathrm{med}}
=\operatorname{median}_{s=1}^{9}
(BA_{\mathrm{PATH},s}-BA_{\mathrm{BAG},s}).
\]

**H_ORDER** is true iff all three are true:

1. effect_order_PATH \(>0\)
2. p_order_PATH \(\le0.05\)
3. median_subject_PATH_minus_BAG \(>0\)

### 15.2 Terminal decision, evaluated in exact order

1. Any numerical/data hard-gate failure: verdict **UNASSESSED**; failure status
   **UNASSESSED — NUMERICAL/DATA FAILURE**.
2. Any required probe/null grid missing or FAILED: verdict **UNASSESSED**; failure status
   **UNASSESSED—PROTOCOL/TECHNICAL FAILURE**.
3. H_PATH_CLASS and H_ORDER true:
   **GO_TRAJECTORY_ORDER**.
4. Otherwise, H_BAG_CLASS true and H_ORDER false:
   **GO_UNORDERED_DISTRIBUTION**.
5. Otherwise, H_PATH_CLASS and H_BAG_CLASS both false:
   **STOP_LOCAL_TRAJECTORY_V0**.
6. Every other combination:
   **MIXED_TRAJECTORY_SIGNAL**.

No available-case substitute, alternative alpha, best representation, seed selection or visual
override can change the decision.

## 16. Required output namespace and artifact policy

All v0 outputs live only under **outputs/bnci2014_001_trajectory_v0/** with exact
subdirectories:

    outputs/bnci2014_001_trajectory_v0/
    ├── protocol/
    ├── tables/
    ├── figures/
    ├── nulls/
    ├── objects/
    └── report/

V1 and Geometry V2 outputs are read-only context and are never overwritten. Raw EEG, large
matrices, fitted objects, general NPZ files and cache remain ignored.

Exactly two narrow required-NPZ exceptions must later be added to .gitignore and committed:

- outputs/bnci2014_001_trajectory_v0/nulls/order_shuffle_group_stats.npz
- outputs/bnci2014_001_trajectory_v0/nulls/label_null_group_stats.npz

No other NPZ/cache/raw path is unignored. The two seed JSON files below are also tracked.

### 16.1 Common provenance and identities

Every CSV begins with:

**protocol_version,protocol_sha256,config_sha256,seed,session,generated_at_utc,status**

Scientific tables then carry every applicable identity selected from:

**sample_index,subject,run,trial_id,trial_uid,class_label,window_index,geometry,
representation,replicate,replicate_seed,target_subject,split**

generated_at_utc is provenance only; it never enters hashes, RNG, splits or numerical results.
Fitted-condition tables include **convergence_warning,warning_messages**. Failed required rows
retain identity and warnings and have NA metrics; they are not removed or imputed.

### 16.2 Exact 21 required tables

The following CSV files are required under tables/.

1. **dataset_contract.csv**  
   check, observed, expected, comparator, required, passed, failure_message.

2. **covariance_sanity.csv**  
   sample_index, subject, run, trial_id, trial_uid, class_label, window_index,
   symmetry_relative_error, min_eigenvalue, max_eigenvalue, condition_number, has_nan,
   has_inf, is_spd, required, passed.

3. **trajectory_geometry_correctness.csv**  
   geometry, subject, sample_index, trial_uid, check, statistic, value, threshold,
   comparator, absolute_error, relative_error, required, passed, failure_message.

4. **trial_airm_path_features.csv**  
   Trial identity, geometry, s1,s2,s3,s4,total_path_length,endpoint_distance,efficiency,
   excess,theta2,theta3,theta4,mean_turn,max_turn,dev2,dev3,dev4,
   mean_geodesic_deviation,max_geodesic_deviation,frechet_variance,
   frechet_radius_mean,diameter,degenerate.

5. **trial_le_path_features.csv**  
   Same feature schema as item 4 with geometry LE.

6. **airm_path_d10.csv**  
   Trial identity, geometry, representation,
   d12,d13,d14,d15,d23,d24,d25,d34,d35,d45.

7. **airm_bag_canon_d10.csv**  
   Trial identity, geometry, representation,
   bag01,bag02,bag03,bag04,bag05,bag06,bag07,bag08,bag09,bag10,
   canonical_permutation.

8. **airm_bag_sorted_d10.csv**  
   Trial identity, geometry, representation,
   sorted01,sorted02,sorted03,sorted04,sorted05,sorted06,sorted07,sorted08,
   sorted09,sorted10.

9. **le_path_d10.csv**  
   Same D10 schema as item 6 with geometry LE.

10. **le_bag_canon_d10.csv**  
    Same canonical BAG schema as item 7 with geometry LE.

11. **class_loso_metrics.csv**  
    geometry, representation, target_subject, source_subjects, train_n, test_n,
    train_uid_sha256, test_uid_sha256, scaler_fit_uid_sha256, balanced_accuracy,
    accuracy, macro_f1, recall_left_hand, recall_right_hand, recall_feet,
    recall_tongue, confusion_matrix_json, prediction_sha256, classifier_config_sha256,
    convergence_warning, warning_messages.

12. **subject_runhalf_probe.csv**  
    geometry, representation, split, train_runs, evaluation_runs, train_n, test_n,
    train_uid_sha256, test_uid_sha256, chance_level, balanced_accuracy, accuracy,
    direction_average_ba, direction_average_accuracy, prediction_sha256,
    classifier_config_sha256, convergence_warning, warning_messages.

13. **scalar_factor_decomposition.csv**  
    geometry, scalar, n_subjects, n_classes, n_per_cell, grand_mean, ss_subject,
    ss_class, ss_interaction, ss_residual, ss_total, eta2_subject, eta2_class,
    eta2_interaction, eta2_residual, ss_reconstruction_relative_error, degenerate,
    uses_p_value.

14. **order_shuffle_subject_metrics.csv**  
    geometry, representation, replicate, replicate_seed, target_subject,
    balanced_accuracy, accuracy, macro_f1, observed_ba, subject_null_median_ba,
    subject_effect, train_uid_sha256, test_uid_sha256, classifier_status,
    convergence_warning, warning_messages.

15. **order_shuffle_group_metrics.csv**  
    geometry, representation, observed_median_subject_ba, null_replicates,
    null_median, null_mean, null_sd_ddof1, null_min, null_max, effect, p_value,
    exceedance_count, median_subject_path_minus_bag, hypothesis_operand_pass.

16. **label_null_subject_metrics.csv**  
    geometry, representation, replicate, replicate_seed, target_subject,
    balanced_accuracy, accuracy, macro_f1, observed_ba, subject_null_median_ba,
    subject_effect, train_uid_sha256, test_uid_sha256, classifier_status,
    convergence_warning, warning_messages.

17. **label_null_group_metrics.csv**  
    geometry, representation, observed_median_subject_ba, null_replicates,
    null_median, null_mean, null_sd_ddof1, null_min, null_max, effect, p_value,
    exceedance_count, hypothesis_operand_pass.

18. **local_barycenter_mdm.csv**  
    representation, target_subject, train_n, test_n, train_uid_sha256, test_uid_sha256,
    metric, balanced_accuracy, accuracy, macro_f1, four fixed-order recalls,
    confusion_matrix_json, prediction_sha256, convergence_warning, warning_messages.

19. **whole_context_mdm.csv**  
    Same metric schema as item 18 plus covariance_samples_per_estimate,
    estimator_regime_confounded, interpretation_limit.

20. **airm_le_robustness.csv**  
    analysis, representation, subject, scalar, airm_value, le_value, paired_delta,
    delta_category, airm_status, le_status, agreement_category, interpretation.

21. **trajectory_v0_summary.csv**  
    row_type, hypothesis, operand, formula, value, threshold, comparator, pass_flag,
    verdict, source_table, failure_type, failure_detail, interpretation.

All confusion matrices use the fixed class order. Every table also includes the common provenance
and applicable identity columns from Section 16.1.

### 16.3 Exact null artifacts

Tracked JSON:

- nulls/order_shuffle_seeds.json
- nulls/label_permutation_seeds.json

Each JSON has:

**protocol_version,master_seed,stream_tag_hex,seedsequence_entropy,child_count,
seed_dtype,seed_extraction,replicates**

replicates is the ordered list of objects **{replicate, seed}** for indices 1 through 199.

Tracked NPZ:

- nulls/order_shuffle_group_stats.npz
- nulls/label_null_group_stats.npz

Each NPZ contains only small group-null identity/statistic arrays:

- replicate: int64, shape (199,)
- replicate_seed: uint64, shape (199,)
- one float64 shape-(199,) statistic array per required geometry×representation condition

Statistic key format is lowercase
**{geometry}__{representation}__median_subject_ba**. JSON and NPZ replicate seeds must match
exactly.

objects/ may contain locally ignored scaler/model/prediction audit objects. No object substitutes
for a required CSV, JSON or tracked NPZ.

## 17. Required figures

Exactly eight stems are required under figures/. Every stem has PNG, PDF and same-stem source CSV.
The source CSV contains all displayed points and intervals; figures do not read a selected subset.

1. **figure_1_class_loso_ba**  
   Subject 1 through 9 observed LOSO BA for AIRM PATH_D10, BAG_CANON_D10,
   BAG_SORTED_D10 and LOCAL_BARYCENTER MDM. WHOLE-1000 is dashed and explicitly labeled
   estimator-regime-confounded context. Failed rows are explicit.

2. **figure_2_order_shuffle_null**  
   All 199 group order-null statistics, observed statistic, null median, effect and p-value for
   AIRM PATH and secondary LE PATH. AIRM PATH is identified as the primary operand.

3. **figure_3_label_destruction_null**  
   All 199 group label-null statistics and observed lines for AIRM PATH/BAG using identical label
   permutations.

4. **figure_4_scalar_eta2**  
   All 11 scalars by subject, class, interaction and residual components for AIRM and LE. No
   best-scalar selection.

5. **figure_5_scalars_by_class**  
   Class distributions for total_path_length, endpoint_distance and efficiency, fixed class
   order, with no selected-subject panels.

6. **figure_6_scalars_by_subject**  
   Subject distributions for total_path_length, endpoint_distance and efficiency, subjects
   ordered 1 through 9.

7. **figure_7_subject_probe**  
   Both run-half directions and their average for AIRM PATH/BAG/SCALARS with chance \(1/9\).

8. **figure_8_airm_le_robustness**  
   Paired AIRM/LE observed and order-null operands without selecting the better geometry.

Rendering parameters are frozen in implementation tests, not selected after results. Optional
Figure 9 is explicitly omitted.

## 18. Required report contract

Report path:

**outputs/bnci2014_001_trajectory_v0/report/trajectory_anatomy_v0.md**

Exact title:

**# BNCI2014_001 Trajectory Anatomy v0**

It has exactly these 18 level-two sections in order:

1. Scientific question
2. Why V1 did not test this question
3. Frozen protocol
4. Geometry correctness
5. Five-state AIRM geometry
6. BAG vs PATH definition
7. Intrinsic path quantities
8. Class LOSO results
9. Order-shuffle falsification
10. Label-destruction null
11. Subject-information results
12. Class vs subject vs interaction effects
13. LOCAL_BARYCENTER / WHOLE contextual controls
14. AIRM vs LE robustness
15. Frozen verdict
16. What is actually justified
17. What is NOT justified
18. Single recommended next step

The report includes actual counts, all gates and failures, observed/null operands, exact Monte
Carlo exceedance counts, classifier warnings, denominators, AIRM/LE disagreements and the
estimator-regime confound. It does not convert MIXED_TRAJECTORY_SIGNAL,
STOP_LOCAL_TRAJECTORY_V0 or UNASSESSED into positive language.
Section 18 names exactly one next step. A failure verdict names exactly one repair-and-rerun of the
unchanged protocol rather than a scientific follow-up.

## 19. Reproducibility, ordering and hashes

- master/classifier seed: 20260809
- null streams and child extraction: exactly Sections 10 and 11
- subjects: numeric 1 through 9
- runs: exact strings parsed then numeric 0 through 5
- classes: Section 2 fixed order
- trials: ascending sample_index; identity ties are errors
- windows: integer 1 through 5
- permutations: lexicographic order
- no seed repetition, sweep or selection
- source/evaluation sets store sorted-trial-UID SHA-256
- frozen protocol/config/environment copies live under protocol/
- every figure/report value traces to a required CSV or tracked null artifact

Before the official run, synthetic tests must cover SPD/distance gates, matrix formulas, fixed-time
geodesic endpoints, all-permutation BAG invariance, RNG serialization/replay, LOSO leakage
boundaries, identical label permutations, SS closure, verdict boundaries, schemas and exact
report/figure contracts.

## 20. Failure and negative-result policy

Numerical/data gates run before scientific probes. Nulls never continue from a failed observed
grid. Required conditions never silently drop subjects, trials, folds or replicates. A convergence
warning is FAILED. Grid failure produces a report with identities/warnings and no available-case
verdict.

Negative results are terminally valid:

- no PATH/BAG signal can produce STOP_LOCAL_TRAJECTORY_V0
- BAG signal without order evidence can produce GO_UNORDERED_DISTRIBUTION
- conflicting operands can produce MIXED_TRAJECTORY_SIGNAL
- failures produce verdict UNASSESSED plus one of the two exact failure-status codes

No architecture, conditional geometry, domain adaptation, sequence learner or session replication
is proposed before this anatomy is resolved.

## 21. Prohibited analyses and interpretation

The following are outside v0:

- session 1test access or pooling
- conditional geometry or conditional alignment
- domain adaptation or target adaptation
- pseudo-labels
- neural networks, RNNs, Transformers, attention or trajectory classifiers
- new distribution classifiers
- hyperparameter, band, epoch, window-count, overlap, seed or metric sweep
- t-SNE or result-selected embedding
- best subject, class, scalar or geometry selection
- alternative window counts or temporal grids
- scalar-decomposition p-values or post-hoc tests
- optional Figure 9
- classifier/SOTA claims
- interpreting LOCAL_BARYCENTER versus WHOLE as an unconfounded method comparison

These analyses are not run, cached as exploratory results or cited as evidence.

## 22. Frozen commit strategy

Commit only after relevant tests pass and raw/cache/large files are absent. Exact milestone
messages:

1. **freeze trajectory anatomy v0 protocol**
2. **implement intrinsic five-state trajectory geometry**
3. **add bag/path controls and null tests**
4. **run trajectory anatomy discovery experiment**
5. **add final report and figures**

The first milestone freezes this document and protocol-only scaffolding needed for auditability.
Later implementation does not rewrite this protocol after scientific results are observed.
Tracked seed JSON, required narrow NPZ exceptions and small tables/figures/report enter their
applicable later milestone; raw EEG, cache, general NPZ and fitted objects do not.
