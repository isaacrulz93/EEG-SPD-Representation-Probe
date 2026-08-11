# Common Subject Action — 목요일 설명 요약

## 문제

EEG에서 보이는 subject individuality는 무엇으로 설명되는가?

사람마다 전체적인 covariance 기준점이 다르다는 것만으로는 충분하지
않다. 더 중요한 질문은 같은 운동심상 class에 대한 반응 구조도 사람마다
다른지, 그리고 그 차이가 얼마나 단순한 형태로 설명될 수 있는지다.

## 이미 아는 것

현재까지 두 가지가 확인됐다.

1. subject마다 marginal/global sensor-space 차이가 강하다.
2. marginal 차이를 제거한 뒤에도 subject×class signature가 session을 넘어
   재현되고, 실제 class correspondence에 의존한다.

따라서 안정적인 sensor-space subject×class interaction evidence는 있다.
하지만 이것이 곧바로 class마다 완전히 별개의 individuality가 존재한다는
뜻은 아니다.

## 가장 단순한 alternative

한 사람에게 공통으로 작용하는 하나의 orthogonal sensor action이 모든
class 차이를 설명할 수도 있다.

\[
U_{s,c}\approx Q_sB_cQ_s^{\mathsf T}
\]

여기서 (B_c\)는 population/shared class structure이고, (Q_s\)는 class와
무관한 subject action이다. 이 설명이 맞다면 individuality의 중요한 부분은
“class마다 전혀 다른 효과”가 아니라 “여러 class에 공통으로 적용되는 한
사람의 coordinate action”일 수 있다.

## 핵심 falsification

subject action을 일부 class만으로 추정한 뒤, 추정에 사용하지 않은 새로운
class를 예측할 수 있는지 묻는다.

예를 들어 Left/Right/Feet로 action을 정하고 Tongue를 예측한다. 네 class가
각각 한 번씩 held out된다. 이 질문은 within-session에서 먼저 하고, 통과한
경우에만 cross-session 안정성을 묻는다.

단, optimizer가 (Q\) 하나를 반환했다는 이유만으로 예측 가능하다고 보면
안 된다. fit한 세 matrix가 허용하는 여러 동등한 action들이 held-out class에
대해 같은 예측을 만들 때만 그 예측을 identifiable하다고 본다. 동등하게
잘 맞는 action들이 서로 다른 held-out 예측을 만들면 결과는 FAIL이 아니라
`PREDICTIVE_NONIDENTIFIABILITY`, 즉 판단 보류다.

## 왜 matrix 3개가 Q를 강하게 묶을 수 있는가

각 (22\times22) symmetric matrix는 단순한 점 하나가 아니다. eigenvalue와
eigenspace orientation을 함께 담고 있다.

한 matrix만 보면 repeated eigenvalue block 안에서 회전하거나 eigenvector
sign을 바꾸는 자유가 남을 수 있다. 그러나 서로 commute하지 않는 여러
matrix를 동시에 맞추면 각 matrix가 허용하는 symmetry의 교집합이 빠르게
작아질 수 있다.

즉 Left, Right, Feet matrix가 서로 다른 eigenspace 제약을 주면 하나의
action이 상당히 강하게 제한될 수 있다. 반대로 세 matrix가 commute하거나
공통 repeated block을 가지면 큰 ambiguity가 남을 수 있다. 그래서
stabilizer와 실제 held-out prediction dispersion을 매 fold 확인해야 한다.

## 계산 formulation을 다시 정리한 이유

이전 실행은 scientific result가 나오기 전에 source generalized-Procrustes
계산이 폭발했다. 한 source fit 안에서 Pymanopt가 최대 30,240번 실행됐고,
Stage-A cell 하나도 완료되지 않았다. 따라서 hypothesis가 실패한 것이
아니라 계산 formulation이 기술적으로 실패한 것이다.

수학적으로는 latent template (B_c\)를 action들이 주어졌을 때 정확히
평균으로 제거할 수 있다. 이것은 approximation이 아니라 원래 least-squares
문제의 exact profiling이다. 그러나 global product model도 여전히 크고
nonconvex하다.

더 싼 첫 질문은 두 subject 사이의 relative action을 세 class로 fit하고
네 번째 class를 예측하는 것이다. global common-action model이 참이면 이런
pairwise 예측은 반드시 가능해야 한다. 따라서 identifiable한 pairwise
held-out prediction이 실패하면 global explanation을 먼저 기각할 수 있다.

## 권장 순서

1. **Pairwise necessary-condition gate**
   - 세 class로 한 source→target action을 fit한다.
   - unseen class를 예측한다.
   - stabilizer와 near-optimal prediction dispersion을 확인한다.

2. **Pairwise PASS 시 global consistency 확인**
   - source partner나 held-out fold가 달라도 induced action이 일관적인지 본다.
   - raw (Q\) 자체가 아니라 matrix에 작용한 결과로 cycle consistency를 본다.

3. **Exact profiled global model**
   - 하나의 joint \(\{Q_s,B_c\}\) 설명이 실제로 성립하는지 더 강하게 검사한다.

4. **Cross-session stability**
   - session이 바뀌어도 같은 subject action 설명이 유지되는지 묻는다.

5. **Post-action residual**
   - common action을 설명하고도 안정적인 class-specific residual individuality가
     남는지 검사한다.

Pairwise 성공만으로 global latent model을 확립하지 않는다. 반면
predictively identifiable한 pairwise necessary condition이 실패하면 이
minimal global orthogonal-action 설명은 값비싼 global fit 전에 기각할 수
있다.

## 다음 분기

### Common action이 충분한 경우

여러 class에 걸친 individuality의 상당 부분이 하나의 global/common subject
action으로 설명된다. 그 뒤 stable class-specific residual evidence가 없다면
현재 데이터에서는 추가적인 individuality 구조를 지지하지 못한다.

이것을 “모든 individuality가 rotation이다”라고 과장할 수는 없다.

### Common action 뒤에도 residual이 남는 경우

subject individuality는 다음처럼 두 층으로 해석할 후보가 된다.

\[
\text{individuality}
=\text{global/common action}
+\text{class-specific residual}
\]

이 경우 다음 main object는 원래 (Z\) 자체가 아니라 common action을
cross-fit으로 제거한 residual이다.

### Common action necessary condition이 실패하는 경우

현재 minimal (O(22)) class-independent conjugation은 여러 class 효과를
공통으로 설명하지 못한다. 이것은 모든 possible subject transform을
기각한다는 뜻은 아니며, 더 복잡한 모델을 자동으로 시작하라는 뜻도 아니다.

## 현재 결론

이번 단계에서는 실데이터 scientific score를 새로 계산하지 않았다.
수학·synthetic audit 결과, 다음 실험의 가장 보수적이고 효율적인 primary
sequence는 다음과 같다.

> identifiable pairwise held-out-class gate → equivalence-aware global
> consistency → exact profiled global model → cross-session → residual

이 순서를 새 protocol amendment로 먼저 고정한 뒤에만 BNCI를 다시 실행해야
한다.
