# BNCI2014_001 WHOLE-SPD Geometry Audit V2 — Preregistered Protocol

Protocol version: `2.0`  
Freeze date: `2026-08-09` (Asia/Seoul)  
Random seed: `20260809`  
Primary dataset/session: `BNCI2014_001 / 0train`  
Primary evaluation unit: one motor-imagery trial  
Primary metric: balanced accuracy (`BA`)

Git freeze provenance:

- Base branch: `main`.
- Base commit: `1af54c7a69161853654dc2bb1f959d1bf3d3593e`.
- Pre-branch status: clean, tracking `origin/main`.
- V2 branch: `pilot/geometry-audit-v2`.
- Pre-branch `git log --oneline -10` (repository에는 7 commits가 존재):
  `1af54c7`, `5aae27f`, `a5b5072`, `9abfba4`, `005bc9a`, `96b23ed`,
  `5749d4f`.

이 문서는 결과를 계산하기 전에 V2의 데이터 범위, 수학적 변환, 수치 허용오차,
누수 경계, 평가 절차와 판정 규칙을 고정하는 preregistration이다. 결과를 확인한 뒤
이 문서의 threshold, preprocessing, geometry, split, classifier 또는 판정식을 바꾸지
않는다. 변경이 필요하면 기존 산출물을 덮어쓰지 않고 별도의 V3 protocol과 output
namespace를 만든다.

## 1. 목적과 연구 질문

V1에서 subject-wise Log-Euclidean coordinate subtraction 후 class linear-information
probe가 개선되고 subject probe가 거의 제거되었다. V2의 목적은 새 모델을 제안하는
것이 아니라, 그 관찰이 다음 중 무엇인지 분리하는 것이다.

- subject marginal location을 제거한 일반적인 효과인지,
- Log-Euclidean geometry에 특수한 효과인지,
- target subject의 평가 covariate까지 mean 추정에 사용한 transductive overlap에
  민감한 효과인지.

두 연구 질문만 둔다. 아래 RQ와 Section 13의 operational verdict Q1–Q3는 서로 다른
층위이다. RQ는 연구 범위를 정의하고, Q1–Q3는 관찰을 판정하기 위한 고정 규칙이다.

**RQ1.** BNCI2014_001의 subject-specific covariance shift는 Log-Euclidean 또는
AIRM Fréchet mean을 identity로 옮기는 label-free marginal centering만으로 크게
제거되는가?

**RQ2.** centering 뒤 남는 representation은 source 8 subjects에서 학습한 동일한
decoder가 unseen target subject의 motor-imagery class로 transfer할 때 RAW보다 높은
balanced accuracy를 보이는가?

Section 13의 **Q1**은 RQ2의 geometry-robust transfer 판정, **Q2**는 그 효과가
Log-Euclidean에만 의존하는 artifact인지의 판정, **Q3**는 T1의 evaluation-covariate
포함이 결과에 필수적인지의 protocol-sensitivity 판정이다. Q3 audit은 제3의 독립 RQ로
확장하지 않는다.

G3 arithmetic-mean congruence는 control이고, MDM은 metric-native sanity check이다.
둘 다 Q1–Q3의 primary verdict를 결정하지 않는다.

## 2. 고정 데이터와 V1 동일 preprocessing

V2는 V1과 같은 WHOLE covariance만 사용한다. V1 결과를 본 뒤 preprocessing을
바꾸지 않는다.

- Dataset: MOABB `BNCI2014_001` 공식 데이터.
- Subjects: 정수 ID `1, …, 9` 전부.
- Session: `0train`만. `1test`는 로드, 탐색, pooling 또는 replication하지 않는다.
- Classes와 고정 출력 순서:
  `left_hand`, `right_hand`, `feet`, `tongue`.
- Channels: 아래 EEG 22개를 이 순서로 사용하고 EOG는 0개여야 한다.
  `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6,
  CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`.
- Epoch: motor-imagery cue 기준 `0.000–3.996 s`; source time `2.000–5.996 s`.
- Sampling: `250 Hz`, trial당 정확히 `1000` samples; resampling 없음.
- Filter: `8–32 Hz` band-pass.
- Baseline correction: 없음.
- Covariance: trial의 1000 samples 전체에 동일한 OAS estimator 적용.
  pyRiemann `Covariances(estimator="oas")`가 사용하는 scikit-learn OAS backend,
  `assume_centered=False`를 고정한다.
- Covariance 후 deterministic symmetrization은 허용하지만 추가 diagonal loading,
  eigenvalue clipping 또는 다른 regularization은 금지한다.
- Expected shape: EEG `(2592, 22, 1000)`, WHOLE SPD `(2592, 22, 22)`.
- Expected counts: subject당 288 trials, run당 48 trials, subject/class당 72 trials,
  subject/run/class당 12 trials.

V1의 frozen config와 covariance/metadata를 재사용할 수 있지만, config hash, covariance
shape, metadata identity와 preprocessing metadata가 위 값과 일치해야 한다. 하나라도
다르면 **V1 artifact reuse가 hard-fail**하며, 아래 규칙대로 새 V2 cache에 재계산해야
한다. V2는 WINDOW5 arrays나 V1 t-SNE를 입력으로 사용하지 않는다.

V1 input provenance는 다음 exact 값으로 고정한다.

- V1 config SHA-256:
  `41f1ad7b2d80a07ab19436de5ec547dfe63d86a320f06dfd05953620944cb232`.
- V1 `cache/bnci2014_001/covariances.npz` file SHA-256:
  `c6bd774ac5b3b53c497381433bcc7974af3b54b850a0ac0539aa2f797f6fa997`.
- NPZ의 WHOLE array-content SHA-256:
  `3be57ea13767a407030a7b56541d07a7f4ce4ccbc1f0579720245413067f36a4`.
- V1 `whole_metadata.csv` SHA-256:
  `285eaf926fbfcec8f1228c3fd660b0b0e9ce209a7c5a13d0acd6b4eabdba0a6e`.
- WHOLE array는 `(2592,22,22)`, `float64`; metadata는 2592 rows, run 문자열
  `"0".."5"`, global-unique `trial_uid`여야 한다.

V2 loader는 이 V1 cache를 read-only로 연다. exact hash/shape/session/channel
order/OAS contract가 하나라도 다르면 V1 cache를 수정하거나 덮어쓰지 않는다. 재계산이
필요한 경우 동일 frozen preprocessing으로 **새 V2 cache namespace에만** 쓰고,
재계산된 provenance와 mismatch 원인을 기록한다. mismatch 상태의 V1 artifact를 그대로
사용하는 것은 금지한다.

## 3. 표기와 공통 수치 연산

subject `s`의 trial `i` covariance를

\[
C_{s i}\in\mathcal S_{++}^{22}
\]

로 쓴다. `N_s=288`은 T1/all-source fit에서의 subject별 표본 수이다. T2 target
calibration에서는 `N_s=144`이다. 모든 geometry 연산은 `float64`와 symmetric EVD로
수행한다. EVD 기반 matrix function 결과는 `(A+A^T)/2`로 수치 대칭화할 수 있으나,
고유값을 조용히 clip하지 않는다. 입력 또는 중간 행렬에 non-positive eigenvalue가
있으면 실패로 기록하고 중단한다.

사용할 거리와 vectorization은 다음과 같다.

\[
d_{LE}(A,B)=\|\log A-\log B\|_F,
\]

\[
d_{AI}(A,B)=\left\|\log(A^{-1/2}BA^{-1/2})\right\|_F,
\]

\[
\operatorname{svec}(S) =
\big(S_{11},\ldots,S_{22,22},\sqrt2 S_{12},\ldots\big)\in\mathbb R^{253}.
\]

`svec`의 diagonal은 그대로, off-diagonal은 `sqrt(2)` 배이며
`dot(svec(A),svec(B)) = Tr(AB)`가 수치 허용오차 안에서 성립해야 한다.

## 4. 고정 geometry G0–G3

모든 subject mean은 class label을 전혀 사용하지 않고 지정된 fit covariate만으로
추정한다.

### G0 — RAW

\[
\widetilde C^{G0}_{s i}=C_{s i}.
\]

어떤 subject center도 fit하지 않는 baseline이다. primary logistic feature는
`svec(log(C_si))`이다.

### G1 — Log-Euclidean translation (LE)

fit set `F_s`에 대해

\[
L_s=\frac1{|F_s|}\sum_{i\in F_s}\log C_{s i},\qquad
M_s^{LE}=\exp(L_s),
\]

\[
\widetilde C^{LE}_{s i}
=\exp\!\left(\log C_{s i}-\log M_s^{LE}\right)
=\exp(\log C_{s i}-L_s).
\]

이는 Log-Euclidean group translation이다. feature는

\[
x^{LE}_{s i}=\operatorname{svec}(\log\widetilde C^{LE}_{s i}).
\]

V1의 `svec(log C_si) - mean_s(svec(log C))`와 수치적으로 동치여야 하며, 이
동치성은 classification 전 hard-gate test이다.

### G2 — AIRM Fréchet mean congruence (AIRM)

fit set `F_s`에 대해

\[
M_s^{AI}=\arg\min_{M\in\mathcal S_{++}^{22}}
\sum_{i\in F_s}d_{AI}(M,C_{s i})^2,
\]

\[
\widetilde C^{AI}_{s i}
=(M_s^{AI})^{-1/2}C_{s i}(M_s^{AI})^{-1/2}.
\]

`M_s^AI`는 pyRiemann `mean_riemann`으로 계산한다. 고정 solver 설정은
`tol=1e-9`, `maxiter=100`, `init=None`이다. inverse square root는 stable symmetric
EVD로 계산한다. feature는 identity tangent coordinate
`svec(log(tilde C_si^AI))`이다.

pyRiemann 0.12의 public API는 실제 iteration count나 termination reason을 반환하지
않는다. 따라서 warning은 포착하고, iteration/termination columns에는
`NA_API_UNAVAILABLE`를 기록하며, 반환 mean에서 Karcher residual을 독립적으로 다시
계산한다.

가능하면 pyRiemann `TLCenter(target_domain=s, metric="riemann")`와 별도
cross-check를 수행한다. public API/metadata contract 때문에 실행할 수 없으면
`SKIPPED_API_UNAVAILABLE`로 명시하며, 자체 구현과의 cross-check가 실행된 경우에는
아래 tolerance를 통과해야 한다. 이 cross-check를 위해 label을 사용할 수 없다.

### G3 — Arithmetic-mean congruence control (EA)

fit set `F_s`에 대해

\[
M_s^{E}=\frac1{|F_s|}\sum_{i\in F_s} C_{s i},
\]

\[
\widetilde C^{EA}_{s i}
=(M_s^{E})^{-1/2}C_{s i}(M_s^{E})^{-1/2}.
\]

feature는 `svec(log(tilde C_si^EA))`이다. G3의 고정 명칭은 반드시
**Arithmetic-mean congruence control**이다. 이는 AIRM Fréchet mean이 아니며,
"Riemannian centering method" 또는 geometry-correct AIRM alignment라고 부르지
않는다.

## 5. Geometry correctness hard gate

어떤 classification도 아래 required check를 전부 계산해
`geometry_correctness.csv`에 기록하고 모두 PASS하기 전에는 실행할 수 없다.
Tolerance는 결과 확인 전에 다음과 같이 고정한다.

Isometry는 class label을 보지 않고 고른 subject별 64개 unique unordered trial pair에서
검사한다. 후보 pair는 canonical `trial_uid` 순서로 만든 뒤
`SHA256(seed, subject, uid_i, uid_j)` digest의 오름차순 상위 64개로 고정한다. 같은 pair를
LE와 AIRM 전후 비교에 재사용한다.

| Check | 고정 statistic | PASS 조건 |
|---|---|---|
| Finite | 입력, mean, transformed SPD의 NaN/Inf 수 | 모두 0 |
| Symmetry | `||A-A^T||F / max(||A||F,tiny)` | 각 행렬 `<= 1e-12` |
| Positive definiteness | EVD의 minimum eigenvalue | 각 행렬 `> 0` |
| Conditioning | `lambda_max/lambda_min` | finite이며 각 행렬 `<= 1e12` |
| EVD reconstruction | `||A-Q diag(lambda)Q^T||F/max(||A||F,tiny)` | `<= 5e-12` |
| Log/exp round trip | `||A-exp(log A)||F/max(||A||F,tiny)` | `<= 1e-10` |
| Inverse-root whitening | `||M^-1/2 M M^-1/2-I||F/||I||F` | `<= 1e-10` |
| G1 custom/official mean | `||M_custom^LE-M_pyriemann^LE||F/||M_pyriemann^LE||F` | `<= 1e-10` |
| G1/V1 equivalence | centered log-coordinate max-absolute, relative L2 및 reconstructed-matrix relative Frobenius error | 모두 `<= 1e-10` |
| G1 fitted mean | `||mean_i log(tilde C_i)||F/sqrt(22)` | `<= 1e-10` |
| G1 isometry | before/after `d_LE` pair error | max absolute와 max relative 모두 `<= 1e-10` |
| G2 Karcher residual | `||mean_i log(M_AI^-1/2 C_i M_AI^-1/2)||F/sqrt(22)` | `<= 1e-7` |
| G2 fitted mean | `d_AI(mean_riemann(tilde C),I)/sqrt(22)` | `<= 1e-7` |
| G2 isometry | before/after `d_AI` pair error | max absolute와 max relative 모두 `<= 1e-10` |
| G3 fitted mean | `||mean_i tilde C_i-I||F/||I||F` | `<= 1e-10` |
| TLCenter cross-check, if run | sample별 normalized AIRM distance의 maximum | `<= 1e-7` |

각 check는 geometry, subject, protocol (`T1`/`T2`), split과 fit scope별로 기록한다.
Aggregate만 기록해서 subject-level failure를 숨기지 않는다. `required=true` row가 하나라도
FAIL하거나 missing이면 `classification_gate_pass=false`이며 logistic과 MDM을 모두
실행하지 않는다. TLCenter가 API 사유로 명시적으로 skip된 것만 required gate에서
제외한다. 실패 행렬을 삭제하거나 tolerance를 넓혀 재실행하지 않는다.

## 6. LOSO 공통 설계와 label barrier

LOSO target `t`마다 source는 정확히 나머지 8 subjects이다.

1. 각 source subject의 center는 그 subject의 모든 `0train` WHOLE covariates로
   독립적으로 unlabeled fit한다.
2. source subjects를 하나의 mean으로 먼저 pooling하지 않는다.
3. 각 source covariance는 자기 subject의 fitted transform으로 변환한 뒤 8 subjects를
   pool한다.
4. classifier fit에 들어가는 label은 source class labels뿐이다.
5. target center에는 어떤 target class label도 들어가지 않는다.
6. target labels는 오직 evaluation function만 받을 수 있다.

구현은 mean-fitting/transform, classifier-fitting, evaluation을 분리해야 한다.
mean-fitting 함수 signature에는 label argument가 존재하면 안 된다. classifier fit
함수에는 source arrays/labels만 전달한다. evaluation 함수 호출 전에는 target label을
담은 object를 target centering 또는 classifier namespace에서 접근할 수 없어야 한다.

각 결과 row에 `source_subjects`, `target_subject`, `fit_trial_uid_hash`,
`calibration_trial_uid_hash`, `evaluation_trial_uid_hash`, `transductive_overlap`을 저장한다.
hash는 정렬된 full trial UID 목록을 SHA-256으로 직렬화해 계산한다.

## 7. T1 — Transductive label-free target centering

T1에서 target center는 target subject의 `0train` WHOLE covariance 288개 전부를
label 없이 사용하여 fit한다. 그 288개와 동일한 trials를 evaluation한다.

- `target_mean_fit_n=288`, `target_eval_n=288`.
- `fit_trial_uids == evaluation_trial_uids`를 assert한다.
- `transductive_overlap=true`를 모든 T1 centered row에 기록한다.
- 이 설정의 고정 명칭은 **transductive label-free target centering**이다.
- 이를 inductive, held-out 또는 deployment-safe evaluation이라고 부르지 않는다.
- target labels는 center fit에 들어가지 않고 evaluation 단계에서만 사용한다.

G0 RAW는 center를 fit하지 않지만 동일 target trials에서 평가하고
`transductive_overlap=not_applicable`로 기록한다.

## 8. T2 — Calibration-to-held-out-run target centering

각 target subject에서 실제 `run` ID를 문자열이 아닌 정수 의미로 stable sort하고,
정확히 `[0,1,2,3,4,5]`인지 먼저 assert한다. run별 48 trials와 class별 12 trials가
아니면 중단한다.

- Split A: calibration runs `[0,1,2]`, evaluation runs `[3,4,5]`.
- Split B: calibration runs `[3,4,5]`, evaluation runs `[0,1,2]`.
- 각 split에서 target mean fit은 calibration covariances 144개만 사용한다.
- evaluation은 반대쪽 144개만 사용한다.
- 각 calibration/evaluation half는 class별 36 trials여야 한다.
- calibration/evaluation trial UID 교집합은 empty여야 하고 hash도 달라야 한다.
- `transductive_overlap=false`를 기록한다.
- target calibration labels는 mean fit 또는 classifier fit에 사용하지 않는다.
- evaluation target labels는 evaluation function만 받는다.

Source 8 subjects, source data와 source subject centers는 Split A/B에서 완전히 같다.
source centers는 각 source subject의 288 trials 전부를 사용할 수 있다. 따라서 같은
target/geometry/classifier에서 Split A와 B의 source feature hash 및 fitted classifier
configuration hash가 동일해야 한다. 이 assertion 실패는 leakage/identity failure이다.

T2의 subject-level BA는 먼저 subject 내 Split A/B BA를 산술평균하고, 그 후 9 subjects
평균을 낸다. Q3의 T2 mean은 이 9개 subject-level 평균의 산술평균이다. split row 18개를
한꺼번에 평균해도 balanced design상 같아야 하며 두 계산의 차이는 `<=1e-12`여야 한다.

## 9. Primary common logistic information probe

G0–G3 모두 253-D `svec(log(transformed SPD))`를 동일한 classifier에 입력한다.
geometry마다 classifier 설정을 바꾸지 않는다.

- Model: multinomial L2 logistic regression.
- `C=1.0`.
- Solver: `lbfgs`.
- `max_iter=5000`.
- `tol=1e-4`.
- `class_weight=None`.
- `random_state=20260809`.
- StandardScaler 없음, normalization 없음, PCA 없음.
- Hyperparameter tuning, inner CV, early model selection 없음.

Primary metric은 target-subject BA이다. Secondary metrics는 accuracy, macro-F1,
고정 class 순서의 per-class recall과 `4 x 4` confusion matrix이다. 모든 metric은 trial
단위 prediction으로 계산한다. BA, accuracy, macro-F1과 recall은 `[0,1]` 범위여야 한다.
confusion matrix row/column class 순서는
`left_hand, right_hand, feet, tongue`으로 고정한다.

9 LOSO subject 결과의 mean은 단순 macro mean이며 SD는 subject 간 sample SD (`ddof=1`)로
기록한다. Q1–Q3에는 BA만 사용한다. secondary metric이 primary verdict를 뒤집지 않는다.
convergence warning이 발생하면 alternate solver/C/scaling을 시도하지 않고 해당 condition을
FAILED로 기록한다.

## 10. Secondary metric-native MDM sanity check

pyRiemann MDM은 secondary sanity check로만 실행한다. tuning은 없다.

- G1 LE: `MDM(metric="logeuclid")`; 비교 RAW baseline도
  `MDM(metric="logeuclid")`로 별도 실행한다.
- G2 AIRM: `MDM(metric="riemann")`; 비교 RAW baseline도
  `MDM(metric="riemann")`로 별도 실행한다.
- G3 EA control: transformed covariance에 `MDM(metric="riemann")`을 적용하되,
  결과를 AIRM centering evidence로 부르지 않는다.

MDM에도 T1/T2와 동일한 source/target fit scope, LOSO, label barrier, metrics와 class
order를 적용한다. MDM 결과는 Q1–Q3 verdict에 투표하지 않으며 logistic conclusion과
다를 경우 `metric-native sanity check disagrees`라고 그대로 보고한다.

## 11. V1 leakage audit

V1 WHOLE class probe에서 subject LE mean을 전체 288 trials로 먼저 구한 뒤 fold CV를
수행한 것은 label leakage는 아니지만 held-out fold covariates가 centering에 들어가는
transductive covariate overlap이다. 다음 두 조건만 같은 고정 5 folds에서 비교한다.

1. `v1_all_sample`: 각 subject의 288 trials 전부로 LE center를 fit한 V1 방식.
2. `fold_safe`: 각 fold마다 subject별 training rows만으로 LE center를 fit하고, 그
   center를 해당 training/test rows에 적용한다. held-out fold covariance는 center fit에
   들어가지 않는다.

Fold는 seed `20260809`의 기존 deterministic subject × class stratified trial-group
assignment를 그대로 사용하고, 각 trial은 한 fold에만 속해야 한다. classifier는 Section
9의 동일 logistic 설정이며 target은 class이다. BA가 audit primary metric이고 accuracy,
macro-F1, per-class recall/confusion을 함께 저장한다.

V1 보고서의 centered WHOLE class-probe accuracy `0.6119`는 audit benchmark일 뿐 강제
재현값이나 pass/fail threshold가 아니다. software version, fold serialization 또는
표시 반올림으로 차이가 나도 값을 덮어쓰지 않는다. 실제 재실행 accuracy와 `0.6119`의
차이를 기록하고 원인을 감사하되, 그 benchmark에 맞추려고 설정을 바꾸지 않는다.
이 audit 자체는 Q1–Q3 verdict를 결정하지 않는다.

## 12. Mean 및 marginal-domain diagnostics

class label을 사용하지 않고 다음을 계산한다.

### LE 대 AIRM subject means

각 subject 및 각 fit scope에서 `M_s^LE`와 `M_s^AI`에 대해 다음을 저장한다.

- `d_LE(M_LE,M_AI)`와 `d_AI(M_LE,M_AI)`.
- `d_LE(M_LE,M_E)`와 `d_AI(M_AI,M_E)`.
- subject covariance의 LE dispersion
  `sqrt(mean_i d_LE(C_i,M_LE)^2)`와 AIRM dispersion
  `sqrt(mean_i d_AI(C_i,M_AI)^2)`.
- 위 네 center distance를 대응하는 subject dispersion으로 나눈 normalized distance.
- `||M_LE-M_AI||F / max(||M_AI||F,tiny)`.
- 각 mean의 min/max eigenvalue와 condition number.
- G1/G2 transformed coordinate 차이의 subject별 mean, median, maximum L2 norm.

이 값은 geometry 차이를 설명하는 descriptive statistic이며 class performance의
대체 evidence가 아니다.

### Marginal domain shift

각 LOSO target, G0–G3 및 T1/T2 split별로 source-centered pooled covariates와
target evaluation covariates에서 다음 class-free diagnostics를 기록한다.

- pooled source domain mean과 identity의 거리.
- target evaluation-domain mean과 identity의 거리.
- pooled source mean과 target evaluation mean 사이의 직접 거리.
- source와 target 각각의 identity 주위 RMS dispersion.
- source와 target 각각의 자기 domain mean 주위 RMS dispersion.
- source-target dispersion의 signed 및 absolute difference.

LE에는 LE mean/distance, AIRM에는 AIRM mean/distance를 사용한다. EA는 arithmetic
mean과 Frobenius distance를 사용하는 arithmetic-control로 명시한다. RAW는 단위를
숨기지 않기 위해 LE와 AIRM reference metric을 각각 별도 row로 계산한다. T2 target
diagnostics는 calibration mean으로 변환된 **held-out evaluation runs**에서 계산하며,
evaluation set 자체로 mean을 다시 fit하지 않는다.

RQ1의 보조 subject-identifiability audit로 common unscaled log-svec space의 subject
silhouette와 subject between/within RMS distance ratio도 WHOLE 2592 points 전부에서
계산한다. sampling, PCA, t-SNE 또는 class-label selection은 하지 않는다. 이 값들은
DA performance나 class-conditional alignment를 의미하지 않는다.

## 13. Frozen Q1–Q3 판정 규칙

모든 verdict는 geometry hard gate를 통과한 **primary logistic BA**만으로 계산한다.
`delta_{g,s}=BA_{T1}(g,s)-BA_{T1}(RAW,s)`이고 `g`는 LE 또는 AIRM이다.
수치 sign 비교 tolerance는 `1e-12`로 고정한다. `delta>1e-12`이면 improved,
`delta<-1e-12`이면 worsened, 그 사이는 tie이다.

### Q1 — geometry-robust improvement

geometry `g`가 자체 criterion을 충족하려면 둘 다 참이어야 한다.

1. 9-subject mean `delta_g >= 0.01`.
2. improved subjects가 `>= 6/9`.

판정은 정확히 다음과 같다.

- LE와 AIRM 둘 다 criterion 충족: `ROBUSTLY SUPPORTED`.
- 둘 중 정확히 하나만 충족: `GEOMETRY-SENSITIVE-MIXED`.
- 둘 다 실패: `NOT SUPPORTED`.

EA, MDM, secondary metrics 또는 V1 audit은 이 판정을 변경하지 않는다.

### Q2 — LE/AIRM geometry dependence

다음 세 조건 중 하나라도 참이면 `SUPPORTED`, 모두 거짓이면 `NOT SUPPORTED`이다.

1. `abs(mean(delta_LE)-mean(delta_AIRM)) >= 0.02`.
2. 두 mean delta의 sign이 반대이다. 한쪽이 tie이면 opposite sign으로 세지 않는다.
3. subject별 sign pattern disagreement가 `>=4/9`이다. 각 subject에서 LE와 AIRM의
   improved/tie/worsened category가 다르면 disagreement 1로 센다.

Q2 `SUPPORTED`는 어떤 geometry가 더 옳다는 뜻이 아니라 conclusion이 geometry
choice에 민감하다는 뜻이다.

### Q3 — T1/T2 protocol sensitivity

LE 또는 AIRM 중 하나라도

\[
|mean\ BA_{T1,g}-mean\ BA_{T2,g}|\ge 0.02
\]

이면 `POTENTIALLY IMPORTANT`, 둘 다 `0.02` 미만이면 `SMALL IN THIS PILOT`이다.
T2 mean은 Section 8의 subject-then-split averaging rule을 따른다. 방향과 geometry별
실제 차이를 모두 보고한다. 이 판정은 T1이 leakage-free라는 뜻도, T2가 실제 deployment
성능을 완전히 대표한다는 뜻도 아니다.

hard gate 또는 필수 primary row가 실패하면 해당 질문은 `UNASSESSED — TECHNICAL
FAILURE`이며 이를 `NOT SUPPORTED`로 바꾸지 않는다.

## 14. Required outputs

V1 outputs를 덮어쓰지 않는다. V2 root는 고정적으로
`outputs/bnci2014_001_geometry_v2/`이다. 큰 matrices, fitted objects와 raw predictions
cache는 ignored `cache/bnci2014_001_geometry_v2/`에 둔다. CSV에는 최소한 protocol
version, config hash, seed, software versions와 identity hashes를 연결할 수 있어야 한다.

### Required tables — 정확히 다음 9개 logical result tables

`outputs/bnci2014_001_geometry_v2/tables/` 아래에 다음을 저장한다.

1. `geometry_correctness.csv`
2. `geometry_mean_comparison.csv`
3. `loso_logistic_transductive.csv`
4. `loso_logistic_calibration.csv`
5. `loso_mdm_transductive.csv`
6. `loso_mdm_calibration.csv`
7. `v1_leakage_audit.csv`
8. `domain_shift_diagnostics.csv`
9. `geometry_v2_summary.csv`

LOSO result tables에는 최소한 classifier, geometry, native metric, target subject,
split, source/calibration/evaluation counts와 hashes, overlap flag, BA, accuracy,
macro-F1, 네 per-class recalls, 16 confusion cells, convergence/failure status가 있어야
한다. `geometry_v2_summary.csv`에는 Q1–Q3별 verdict, 모든 threshold operand, pass flags와
산출식 문자열을 저장한다.

### Required figures와 underlying CSV

모든 figure는 결과 선택 없이 전체 subject를 고정 순서 `1..9`로 표시한다. PNG와 같은
stem의 source CSV를 둘 다 저장한다.

1. `figure_1_loso_ba_by_subject.png/.csv`: x=`1..9`, primary logistic T1의
   `RAW, LE, AIRM, EA` BA series. y-axis `[0,1]`.
2. `figure_2_paired_delta_vs_raw.png/.csv`: subject별 T1 BA delta,
   `LE-RAW, AIRM-RAW, EA-RAW`, zero reference line.
3. `figure_3_t1_vs_t2_ba.png/.csv`: geometry별 T1과 T2 subject-level BA 비교.
   T2는 subject 내 A/B mean이며 subject points와 9-subject mean을 함께 표시한다.
4. `figure_4_le_vs_airm_centers.png/.csv`: subject별 normalized AIRM center distance와
   normalized center difference. 서로 다른 scale은 명시된 panels로 분리한다.
5. `figure_5_v1_leakage_audit.png/.csv`: 같은 folds에서 `v1_all_sample` 대
   `fold_safe` class-probe BA/accuracy; fold points와 aggregate mean을 모두 표시한다.

Figure 1–5 외 exploratory figure, t-SNE, best-subject subset 또는 결과에 따른 panel
선택은 만들지 않는다.

### Required report — 정확한 14개 section

최종 report는
`outputs/bnci2014_001_geometry_v2/report/geometry_audit_v2.md`이고, 제목 아래
section heading을 정확히 다음 순서로 둔다.

1. `Motivation`
2. `Frozen protocol`
3. `Geometry definitions`
4. `Geometry correctness`
5. `LE vs AIRM Fréchet means`
6. `V1 leakage audit`
7. `LOSO transductive results`
8. `Calibration-to-held-out-run results`
9. `Metric-native MDM sanity check`
10. `Marginal domain diagnostics`
11. `Frozen decision-rule verdicts`
12. `What is actually justified`
13. `What is NOT justified`
14. `Single recommended next experiment`

Report에는 actual data counts, 모든 hard-gate 결과, T1/T2의 overlap 의미, Q1–Q3
operands/verdict, negative/mixed result, convergence/failure와 limitation을 생략 없이
기록한다. recommended experiment는 정확히 하나만 쓴다.

## 15. Reproducibility, seed 및 ID discipline

- 전역 seed는 `20260809` 하나만 사용한다.
- 별도 seed 반복, seed selection 또는 stochastic robustness sweep은 하지 않는다.
- subject 정렬은 integer `1..9`, run 정렬은 integer `0..5`, class order는 Section 2를
  따른다.
- canonical trial key는 `(subject, session, run, trial_id, trial_uid)`이다.
- metadata row 순서에 의존하지 않고 canonical key로 stable sort한 뒤 hash/split한다.
- 중복 trial UID, covariance/metadata row mismatch, trial label 불일치가 하나라도 있으면
  중단한다.
- source, calibration, evaluation UID set과 교집합 assertion을 machine-readable하게
  저장한다.
- 환경에는 OS, Python, NumPy, SciPy, scikit-learn, pandas, matplotlib, MNE, MOABB,
  pyRiemann 버전을 저장한다.
- protocol file과 frozen config의 SHA-256을 모든 summary metadata에 기록한다.
- 동일 inputs로 재실행한 CSV numeric values와 predictions hash가 같아야 한다.
- figure/report는 result CSV만 읽으며 feature/classifier를 다시 계산하지 않는다.

## 16. 금지 분석

이번 V2에서는 다음을 하지 않는다.

- WINDOW5, local covariance, trajectory 또는 transition 분석.
- session `1test` 사용, session pooling 또는 session 2 replication.
- preprocessing, channels, epoch, band, sampling, covariance estimator 또는
  regularization 변경.
- 새로운 representation/geometry, Procrustes/rotation, parallel transport,
  supervised alignment 또는 class-conditional mean 추가.
- neural network, attention, Transformer, RNN, sequence/distribution classifier.
- domain-adaptation objective, OT/SPDSW/SPDHSW, pseudo-label.
- target label을 center fit, model fit, model selection 또는 hyperparameter choice에 사용.
- StandardScaler, PCA, feature selection, class weighting, alternate C/solver 또는 tuning.
- hyperparameter/tolerance/window/seed sweep.
- 결과를 본 뒤 subject/run 제외, outlier removal 또는 failure covariance 삭제.
- t-SNE/UMAP 또는 visualization을 quantitative evidence로 사용.
- p-value fishing, unregistered significance test 또는 post-hoc threshold.
- MDM/EA/secondary metric으로 primary Q1–Q3 verdict 뒤집기.
- 높은 accuracy를 새로운 method performance 또는 SOTA로 주장.

## 17. 실패 및 중단 정책

1. **Data contract failure:** V1 provenance hash mismatch는 read-only reuse를 중단하고
   새 V2 cache로의 frozen 재계산을 요구한다. 재계산 후에도 expected
   subjects/session/runs/classes/counts/channels/shape/config contract가 다르거나 재계산을
   완료할 수 없으면 즉시 중단한다. 임의 수정이나 row 삭제를 하지 않는다.
2. **Geometry hard-gate failure:** required correctness row 하나라도 실패/missing이면 모든
   classifier를 실행하지 않는다. 실패 statistic, subject, geometry, protocol/split을
   그대로 저장한다. tolerance 확대, eigen clipping 또는 regularization 재시도는 금지한다.
3. **Leakage assertion failure:** source/target, calibration/evaluation 또는 label barrier
   assertion 실패 시 해당 run뿐 아니라 V2 primary evaluation 전체를 invalid로 처리하고
   중단한다.
4. **Classifier failure:** convergence warning/exception은 해당 condition을 FAILED로 남긴다.
   solver, C, scaling 또는 iteration limit을 결과에 맞춰 바꾸지 않는다. required primary
   rows가 불완전하면 Q1–Q3는 `UNASSESSED — TECHNICAL FAILURE`이다.
5. **Secondary failure:** MDM 또는 optional TLCenter API cross-check만 실패하고 primary
   hard gate/logistic이 유효한 경우 primary 결과는 유지하되 secondary failure를 report에
   명시한다. TLCenter의 수치 불일치는 skip이 아니라 hard-gate failure이다.
6. **Negative result:** threshold를 못 넘은 결과는 정해진 `NOT SUPPORTED`, `MIXED` 또는
   `SMALL IN THIS PILOT`로 그대로 보고한다. 설정 변경이나 positive framing을 하지 않는다.
7. **Output completeness:** required table, figure-source CSV 또는 14-section report가
   missing이면 프로젝트 완료로 표시하지 않는다.

이 protocol이 허용하는 결론은 WHOLE covariance의 marginal geometry와 evaluation-scope
audit에 한정된다. classifier architecture, temporal representation, DA 또는
class-conditional alignment에 관한 방법론적 결론은 후속 preregistration 전까지 유보한다.
