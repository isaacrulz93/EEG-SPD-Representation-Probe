# Protocol: Subject Class Interaction V0

Status: **FROZEN BEFORE SCIENTIFIC COMPUTATION**

Freeze date: 2026-08-10 (Asia/Seoul)

Base commit: `6124adb907369bc8f76733e880ebb8c43db38e94`

Branch: `pilot/subject-class-interaction-v0`

Master seed: `20260810`

## 1. Purpose and scope

This is a minimal anatomy/premise-falsification experiment. It does not develop a domain-adaptation method, network, loss, classifier, low-rank model, or mixed-effects model. It asks only whether a **cross-session-reproducible subject×class interaction in marginally recentered covariance representation** is empirically present.

The operational question is:

> After removing marginal subject/session location, the session-specific population class effect, and the class-independent subject residual, is a class-dependent deviation specific to each subject reproducible across recording sessions?

The result must not be described as physiology, personality, a neural trait, a brain fingerprint, a biomarker, source anatomy, or an intrinsically biological class effect. TTA and unlabeled recoverability are long-term motivations only and are not tested here.

The following are forbidden in v0: TTA; neural networks; new losses; low-rank factor models; mixed-effects fitting; classification-SOTA experiments; WINDOW5; adding features, thresholds, datasets, channels, epochs, or analyses after viewing results.

## 2. Frozen dataset roles

### 2.1 BNCI2014_001: retrospective development only

BNCI2014_001 has already been inspected repeatedly in prior work and is never called strict confirmation. The frozen contract is 9 subjects (`1..9`), sessions `0train` and `1test`, class order `left_hand`, `right_hand`, `feet`, `tongue`, 22 fixed EEG channels, 8–32 Hz, cue-relative 0–3.996 s, 250 Hz, 1000 samples, no baseline, no resampling, OAS covariance, float64 covariance matrices, and WHOLE only.

The exact expected counts are 288 trials per subject/session, 72 per class, 6 runs per subject/session, 48 per run, and 12 per run/class. Any discrepancy is a data-contract failure; no trial is dropped or synthesized to force the count.

The frozen covariance and metadata artifacts are reused only if both file and content hashes match the config. The channel order is:

`Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`.

### 2.2 OpenBMI / Lee2019-MI: prospective external replication

OpenBMI remains scientifically locked during Phase A. It is opened only if all three BNCI primary effect directions defined below are strictly positive. Metadata-only protocol resolution is allowed before unlock; scientific scores, similarity matrices, and figures are not.

The intended primary design is all 54 subjects with two valid sessions, two left/right motor-imagery classes, offline training phase only, the balanced trials defined by the dataset, OAS float64 covariance, canonical 20 motor-area channels, 8–30 Hz, and cue-relative 1.0–3.5 s. Exact MOABB class/API name, subject/session identifiers, offline phase identifier, canonical channel order, sample-rate handling, expected counts, source/version, and raw-input hashes must be resolved from primary sources, installed MOABB metadata/code, or official dataset metadata and frozen in a separate manifest commit.

If the exact canonical 20-channel subset cannot be authoritatively resolved, analysis stops as `OPENBMI_PROTOCOL_RESOLUTION_FAILURE`. All 62 channels or a guessed subset is forbidden. After the manifest freeze, preprocessing, channels, epoch, null counts, alpha, and outcome logic cannot change.

## 3. Frozen geometry and objects

AIRM is primary. Log-Euclidean (LE) geometry is robustness-only and cannot rescue or change the AIRM terminal decision.

For subject (s), session (q), class (c), let (C) denote trial OAS covariances. Compute the AIRM Fréchet marginal mean (M_{s,q}) from all trials and class mean (M_{s,q,c}) from class trials. Then define

\[
\widetilde M_{s,q,c}=M_{s,q}^{-1/2}M_{s,q,c}M_{s,q}^{-1/2},\qquad
U_{s,q,c}=\log(\widetilde M_{s,q,c}).
\]

(U) is called the **marginally recentered class effect**, never an intrinsic biological effect. All (U) matrices share the identity tangent coordinate system.

For every session and class, define the leave-one-subject-out population template by entrywise/Frobenius Euclidean averaging in the identity tangent:

\[
\mu_{-s,q,c}=\frac{1}{N-1}\sum_{r\ne s}U_{r,q,c}.
\]

The target subject must never enter its own template. Define

\[
R_{s,q,c}=U_{s,q,c}-\mu_{-s,q,c},\qquad
\bar R_{s,q}=\sum_c\pi_{s,q,c}R_{s,q,c},
\]

where (pi_{s,q,c}) is the observed class count divided by the observed total for that subject/session. The primary object is

\[
Z_{s,q,c}=R_{s,q,c}-\bar R_{s,q}.
\]

Thus (Z) removes marginal SPD location, the session-specific population class template, and the class-independent subject residual. (R) is saved as a mandatory descriptive control. Stable (R) without stable (Z) is interpreted as a global/class-independent subject residual, not as the target interaction.

The primary population template is session-specific. A pooled-session (mu_{-s,c}) sensitivity is secondary only and cannot rescue a failed session-specific chain.

## 4. Frozen signatures

### 4.1 Primary montage-registered sensor signature

Use upper-triangle row-major Frobenius-isometric `svec`: diagonal entries unchanged and off-diagonal entries multiplied by (sqrt2). Concatenate in frozen class order:

\[
b^{raw}_{s,q}=[\operatorname{svec}(Z_{s,q,1});\ldots;\operatorname{svec}(Z_{s,q,K})],
\qquad b_{s,q}=b^{raw}_{s,q}/\lVert b^{raw}_{s,q}\rVert_2.
\]

The fixed degeneracy threshold is machine epsilon times the square root of vector dimension. A norm at or below it yields `DEGENERATE_INTERACTION_SIGNATURE` and makes that primary chain unassessed; silent epsilon normalization is forbidden. Save every (U,R,Z), per-class Frobenius norm, raw vector norm, and normalized vector.

This is a coordinate-dependent sensor-space signature registered to a common montage; it is not intrinsic or source-space.

### 4.2 Orthogonal-conjugation-invariant spectral control

For every symmetric (Z_{s,q,c}), use `eigvalsh` in ascending order, concatenate classes in frozen order, and L2-normalize under the same degeneracy rule. This spectrum signature is a secondary control. Sensor and spectrum passing is stronger orthogonal-gauge-robust evidence; sensor passing and spectrum failing supports only the montage-registered sensor representation. Spectrum results never rescue the primary sensor result.

The same signatures are computed descriptively from (R), with no primary vote.

## 5. Numerical and data hard gates

Before inference, require all of the following:

1. Exact data counts and metadata keys; unique, traceable trial IDs; no overlap or leakage between splits.
2. All inputs and derived arrays finite and symmetric; covariance minimum eigenvalue strictly positive; covariance condition number at most (10^{12}); covariance relative symmetry error at most (10^{-12}).
3. AIRM mean tolerance (10^{-9}), maximum 100 iterations, convergence without prohibited repair, and independently evaluated Karcher residual at most (10^{-7}).
4. Each recentered marginal mean maps to identity with relative error at most (10^{-7}).
5. (U,R,Z) symmetry relative error at most (10^{-12}).
6. Class proportions sum to one with absolute error at most (10^{-15}), and (lVert\sum_c\pi_cZ_c\rVert) relative error at most (10^{-12}).
7. Deterministic class order, `svec`, ascending eigenspectrum order, derangements, label permutations, and RNG mapping.
8. Reconstruction/provenance hashes for config, code, inputs, and saved core objects.

There is no eigenvalue clipping, epsilon repair, available-case analysis, silent row deletion, or result-driven retry. Failure of any required gate produces `UNASSESSED_NUMERICAL_OR_DATA_FAILURE` (or the explicitly named protocol-resolution/degeneracy failure) rather than a scientific negative.

## 6. Stage R: within-session measurement reliability

This stage asks whether the (Z) object can be estimated reliably from finite trials. Failure does not refute individuality; it rejects the current class-mean covariance object as sufficiently reliable for this premise test.

For BNCI, half A is runs 0,1,2 and half B is runs 3,4,5. For OpenBMI, within each subject/session/class, sort by acquisition order and put zero-based even positions in A and odd positions in B (the human-readable odd/even names refer to one-based order). This deterministically interleaves the time span; expected 50 trials/class gives 25/25. Halves must be disjoint and exhaust the eligible trials.

Each half independently recomputes (M,M_c,U,mu_{LOSO},R,\bar R,Z,b) from scratch. For each subject/session compute (cos(b_A,b_B)). The frozen group statistic is

\[
T_{half}=\operatorname{median}_{s}\left[\operatorname{mean}_{q}
\cos(b_{s,q,A},b_{s,q,B})\right].
\]

The label-destruction null has 1999 replicates. Within every subject/session, permute labels while exactly preserving the label multiset. Each half uses the same split membership but its permuted labels. Marginal (M) may be cached because it is label-independent; every (M_c,U,mu,R,\bar R,Z,b) must be recomputed. The null statistic uses the identical aggregation.

Use a one-sided greater-or-equal plus-one Monte Carlo p-value and

\[
E_{half}=T^{obs}_{half}-\operatorname{median}(T^{null}_{half}).
\]

OpenBMI Gate R passes iff hard gates pass, (E_{half}>0), and (p_{half}\le0.05). BNCI uses the same computation but only descriptively and as a direction gate.

## 7. Stage I: cross-session same-subject reproducibility

Use full-session signatures and define

\[
S_{s,t}=\cos(b_{s,0},b_{t,1}),\qquad
T_{same}=\operatorname{median}_s S_{s,s}.
\]

The unrelated-subject null uses all 133,496 derangements for BNCI and 100,000 deterministic random derangements for OpenBMI. A derangement has no fixed point. For each (pi), compute (T_{perm}=\operatorname{median}_s S_{s,\pi(s)}). Save the exact greater-or-equal tail probability for BNCI and a plus-one value for comparability; use the plus-one one-sided value for OpenBMI. Define

\[
E_{id}=T_{same}-\operatorname{median}(T_{perm}).
\]

OpenBMI Gate I passes iff Gate R passes, (E_{id}>0), and (p_{id}\le0.05). If Gate R fails, Stage I is reported as `DESCRIPTIVE_ONLY`.

## 8. Stage C: true-class dependence

This stage tests whether same-subject reproducibility depends on the true class-conditional structure rather than a generic residual fingerprint. It uses 1999 label-destruction replicates. Within each subject/session, independently permute labels while preserving counts, and recompute the full label-dependent full-session pipeline from (M_c) through (b). Compute the same-subject statistic for every replicate. Define

\[
E_{class}=T^{true}_{same}-\operatorname{median}(T^{null}_{class}).
\]

Use the one-sided greater-or-equal plus-one p-value. OpenBMI Gate C passes iff Gate I passes, (E_{class}>0), and (p_{class}\le0.05).

## 9. Controls and sensitivities

Repeat Stages R/I/C descriptively for the concatenated (R) sensor and spectral signatures. (R) never casts a primary vote. If (R) is stable and the (Z) primary chain fails, the designated interpretation is `STOP_SUBJECT_MAIN_EFFECT_ONLY`.

Repeat the core stages for the pooled-session population template as secondary sensitivity. Pooled-only success is not stable individual interaction evidence. Repeat the core stages with LE as robustness only. LE cannot rescue AIRM or alter the terminal decision.

No absolute cosine cutoff exists. All pass/fail inference uses frozen effect directions, p-values, prerequisite gates, and hard gates only.

## 10. Descriptive energy anatomy

Without fitting a mixed-effects model, save for each dataset/geometry/session the squared population class-effect norm, squared class-independent subject-residual norm, squared (Z) interaction norm, split-half estimation-discrepancy energy, and cross-session discrepancy energy. Normalize to fractions only for sums supported by the explicit algebra. These are labeled **DESCRIPTIVE ENERGY FRACTIONS**, not identified variance components.

## 11. Two-phase replication boundary

Phase A consists of this protocol/config freeze, generic implementation and tests, and BNCI development analysis. If any of (E_{half},E_{id},E_{class}) for the primary AIRM/session-specific/sensor/(Z) chain is nonpositive, terminate as `STOP_BNCI_DIRECTION_FAILURE`; do not run OpenBMI scientific analysis. Numerical bug fixes are permitted, but scientific definitions, thresholds, features, and outcome logic cannot be changed and rerun after results.

Only if all three directions are positive may Phase B begin. Resolve and freeze the exact OpenBMI manifest, commit it with message `freeze OpenBMI external replication manifest`, and write that exact commit SHA plus manifest/config/input hashes to `openbmi_unlock.json`. The lock validator must reject any scientific evaluation before that commit and reject dirty or mismatched manifests.

## 12. Frozen terminal logic

The primary chain is **AIRM + session-specific LOSO template + sensor-space (Z) signature**. Apply the following logic in order, while retaining the specific prerequisite status in the decision table:

1. Any hard-gate failure: `UNASSESSED_NUMERICAL_OR_DATA_FAILURE`.
2. OpenBMI Gate R failure: `UNASSESSED_CURRENT_OBJECT`.
3. Z Gate R pass but Z Gate I failure: `STOP_NO_STABLE_INDIVIDUAL_COMPONENT`.
4. Z Gate I pass but Z Gate C failure: `STOP_GENERIC_SUBJECT_FINGERPRINT`.
5. If the primary Z chain fails and the corresponding R control is stable: `STOP_SUBJECT_MAIN_EFFECT_ONLY` as the mechanistic refinement of the applicable stop.
6. OpenBMI Z R/I/C pass but any corresponding BNCI effect direction is nonpositive: `STOP_NOT_CROSS_DATASET_ROBUST` (normally unreachable because the Phase B lock prevents it).
7. AIRM sensor Z passes all R/I/C but spectrum lacks corresponding support: `GO_SENSOR_SPACE_ONLY`.
8. AIRM sensor Z and the spectrum control support all R/I/C: `GO_STABLE_SUBJECT_CLASS_INTERACTION`.

BNCI-only early termination is `STOP_BNCI_DIRECTION_FAILURE`, not an OpenBMI outcome. A GO does not establish physiology, source anatomy, TTA identifiability, unlabeled recoverability, low dimensionality, intrinsic Riemannian random effects, the full conditional distribution, or a causal mechanism.

## 13. Deterministic nulls and resumability

Every stochastic replicate is a pure function of `(master_seed, dataset, geometry, stage, signature, template, replicate_index)`. Encode the key as canonical sorted compact JSON, hash it with SHA-256, and use the first 128 bits as seed entropy for NumPy `PCG64DXSM`. Replicate indices start at zero.

Checkpoints atomically save completed indices/bitmap, key-to-seed mapping, partial statistics, config SHA-256, code SHA, and input hashes. A resume validates all identity fields before accepting partial results. Canonically sorted final rows and fixed float serialization must make the final summary byte-identical to an uninterrupted run.

## 14. Required outputs and reporting

Write all artifacts below `outputs/subject_class_interaction_v0/` with provenance, data/geometry tables, reliability, cross-session, class-dependence, gauge, decisions, report, and figures 1–9. Every figure must have PNG, PDF, and source CSV. No t-SNE or UMAP is used.

The report explains every statistic before its numeric value and distinguishes (U,R,Z). It states the dataset roles, all numerical/data gates, BNCI development, OpenBMI only if unlocked, R-versus-Z, session-specific versus pooled templates, sensor versus spectrum, LE robustness, descriptive energy anatomy, frozen decision, justified and unjustified claims, and the killed direction after STOP.

If and only if the final result is GO, the report contains exactly one next structural question and does not implement it:

> Is the stable subject×class interaction low-dimensional and structured across the population?
