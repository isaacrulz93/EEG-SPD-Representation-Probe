# Protocol: OpenBMI External Replication of Ordered Movement Anatomy V0

Status: **FROZEN BEFORE ANY OPENBMI MOVEMENT STATISTIC**. This protocol is prospective, has no temporal-grid search, and cannot modify the finalized BNCI result.

## 1. Immutable lineage and donor

The implementation branches from exact BNCI component HEAD `edc1d344cb0657f2f2d87b2992049bceec4705d2` (`pilot/local-movement-component-decomposition-v0`). Its protocol freeze is `95c330de9596fa4c4eb4ee377d5af8d99896f4c3`, scientific result is `0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`, and terminal is `BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS`.

OpenBMI conventions are inherited read-only from `pilot/subject-class-interaction-v0` at `272d775678644aad062df424a70586d4b42de652`. Its protocol manifest, source manifest, and unlock bytes must have SHA256 `4b956d7e3b2b1a271ec07bddecc1ce0a93460ab1515ab8902b3d1ca35ebdb0ea`, `5133a0d8521f4cdd121362663c4980fed500e72875f26bb3dc4ce830d8c5e409`, and `aaf2bf598802e0e5bf84924b26d0c9ed7f64640605a594273894b888b6c043f0`. Every one of its 108 source-file SHA256 values must reproduce. Shared raw caches are read-only; task-private source staging is allowed only for exact donor-hash reproduction.

The pre-result preparation gate reproduced all 108 source hashes (106 read-only shared sources and two exact private Subject-5 copies). Its frozen five-bin covariance archive SHA256 is `abdf9682ae7d593544aca68e196711300e25c01104165f1415f07629dfdd21cf`, canonical covariance-array SHA256 is `af111eae37076d2f7ed80f9d3ab5c5033c5b8843c3b7b138f8b8ab8fa0f768af`, and metadata SHA256 is `6977a31be900f7b626744192113d2d9d71ea107d56a3b1538b43948152f72dd6`. Scientific execution requires those exact prepared inputs.

## 2. Frozen OpenBMI contract

Lee2019-MI uses subjects 1–54, sessions 0 and 1 (source sessions 1 and 2), only offline `1train`, left-hand event 2 and right-hand event 1, and 50 acquisition-ordered trials per subject/session/class. The exact 20-channel order is `FC5, FC3, FC1, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6`.

Continuous offline signals are filtered 8–30 Hz with the donor fifth-order Butterworth IIR, SOS output and zero phase, resampled from 1000 to exactly 100 Hz by MNE 1.12.1's frozen default polyphase handling, and then epoched on the half-open interval [1.0,3.5) s. There is no baseline or rejection. Covariances use pyRiemann OAS, float64, explicit symmetrization, no extra regularizer, and no eigenvalue clipping.

The frozen epoch is exactly 250 samples. It is divided once, before results, into five equal consecutive non-overlapping bins `[0:50), [50:100), [100:150), [150:200), [200:250)`. Thus every bin is exactly 0.5 seconds. No overlap, alternate bin count, grid search, or post-result 0.8-second sensitivity is permitted. Failure of this exact integer contract returns `STOP_AND_REPORT_TEMPORAL_CONTRACT_AMBIGUITY`.

## 3. Mean sequences and independent halves

For each of 54 × 2 × 2 cells and each of five bins, fit the AIRM Fréchet mean of the 50 OAS covariances using the finalized dimension-independent BNCI convention: public pyRiemann `mean_riemann`, `init=None`, tolerance 1e-9, at most 100 iterations, no warning, SPD output, and normalized Karcher residual at most 1e-7.

The frozen split is within-cell acquisition-order interleaving: Half A takes zero-based even positions and Half B zero-based odd positions, exactly 25 trials each. Half means are fitted independently. Replicate A compares session-0 Half A to session-1 Half A; replicate B compares the corresponding Half B objects.

## 4. Ordered anti-development and quotient

For `M1,...,M5`, define `V_i=Log_{M_i}(M_{i+1})/0.5` for four fixed adjacent steps. Parallel transport each vector backward to `T_{M1}` along the actual reverse prefix of the ordered piecewise-AIRM-geodesic path, then whiten symmetrically at `M1` to obtain `Z1,...,Z4` in `Sym(20)`. Norm, symmetry, transport, and AIRM-speed identities are hard numerical gates.

No transition permutation or DTW is allowed. For cross-session cells A and B:

`c_full = min_{Q in O(20)} (1/4) sum_i ||A_i-Q B_i Q^T||_F^2`.

One common Q serves all four steps and is a nuisance only. The exact finalized BNCI TrustRegions optimizer is inherited: six deterministic starts (three in each determinant sector), 250 iterations, gradient tolerance 1e-6, minimum step 1e-12, maximum time 120 s and 5,000 cost evaluations. Both determinant sectors require certified candidates. Full and both half 108 × 108 fits use atomic resumable checkpoints carrying completed bitmap, objective, certificates, determinant coverage, protocol/config hash, and input hashes.

## 5. Exact squared component decomposition

At squared-cost scale only:

- `c_sensor=(1/4) sum_i ||A_i-B_i||_F^2`;
- `c_len=(1/4) sum_i (||A_i||_F-||B_i||_F)^2`;
- `c_ang=c_full-c_len`;
- `c_ori=c_sensor-c_full`.

Raw components must be nonnegative within combined absolute/relative tolerance 1e-8, and `c_sensor=c_len+c_ang+c_ori`. Meaningful negative components are never clipped and return `UNASSESSED_OPENBMI_MOVEMENT_NUMERICAL_FAILURE`.

## 6. Frozen relation statistics and nulls

For every squared-cost matrix and anchor `(s,c)`, `a` is same-subject/same-class, `b` is same-subject/different-class, `c` is the mean over 53 other subjects for the same class, and `d` is the mean over 53 other subjects for the other class. Define `S=c-a`, `C=b-a`, and `J=b+c-a-d`; average classes within subject and then subjects.

The primary endpoint is `T_J_ang`. Subject-break uses exactly 1,999 draws from `SeedSequence([20260810,1102])`, independently permuting all 54 complete session-1 subject tuples within each class. Class-break uses exactly 1,999 draws from `SeedSequence([20260810,1101])`, independently applying a two-element class permutation within each session-1 subject. The indexed mappings are shared across components. One-sided plus-one p-values count null values greater than or equal to observed.

Primary support requires positive `T_J_ang` and both p-values below 0.05. Full quotient S/C/J, speed J, orientation J, sensor-frame summaries, and both split-half angular J results are secondary and cannot rescue the endpoint.

## 7. Ordered raw temporal control

The secondary descriptive temporal control compares the cross-session identity correspondence `1->1,...,5->5` with all 44 complete derangements. Each permutation cost is root-mean-square AIRM distance across the five mean states. Per cell, report median derangement cost minus identity cost and the identity rank among all 120 permutations. No PCA-space inference, p-value, or terminal change is allowed.

## 8. Terminal and claims

Use `REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION` exactly when the primary criterion passes, otherwise `NOT_REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION` when numerically valid. Geometry, optimizer, or decomposition failure returns `UNASSESSED_OPENBMI_MOVEMENT_NUMERICAL_FAILURE`; data/count/provenance failure returns `UNASSESSED_OPENBMI_DATA_CONTRACT_FAILURE`.

If supported, the allowed claim is that a speed-removed common-O(20)-invariant directional/joint component of ordered window-wise mean covariance movement shows cross-session subject×class interaction in independent OpenBMI. This is explicitly a two-class structural replication, not BNCI's four-class combinatorial structure and not the same physical bin duration. It makes no claim about equal effect size, physiology, neural trajectory, continuous dynamics, motor strategy, source-space direction, or stable subject pose.

After the first OpenBMI movement statistic is accessed, no representation, grid, optimizer, cost, averaging, null, seed, alpha, secondary endpoint, or terminal may change. No rescue analysis is allowed.
