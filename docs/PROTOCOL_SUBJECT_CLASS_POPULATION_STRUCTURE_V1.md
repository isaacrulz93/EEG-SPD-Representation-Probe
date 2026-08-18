# Protocol: Cross-Session Population Structure of Subject-Class Interaction V1

Status: **FROZEN BEFORE REAL-DATA SVD OR STATISTIC ACCESS**

Freeze date: 2026-08-18 (Asia/Seoul)

Base commit: `d9a67a130aeeea7eb8a93d76878e43f636802e93`

Branch: `pilot/subject-class-population-structure-v1`

Master seed: `20260818`

## 1. Question and scope

This experiment falsifies whether the already established montage-registered, mean-level subject-by-class interaction can be described by a small set of population-shared linear modes that predicts paired independent-session structure for held-out subjects. It does not assume such a factorization is true and does not use ordinary PCA variance explained as evidence.

The operational model is

\[
Z_{s,q,c}\approx\sum_{r=1}^{R}\alpha_{s,r}B_{q,r,c}+\varepsilon_{s,q,c},
\]

where `Z` has exactly the V0 operational meaning. The target is class-conditional location/prototype interaction only. Dispersion, multimodality, higher-order distributional shape, classifiers, networks, losses, domain adaptation, TTA, WINDOW5/local trajectory, GPA, anti-development, and temporal ordering are outside scope.

## 2. Parent facts and immutable base

PR #2 result `272d775678644aad062df424a70586d4b42de652` established the V0 sensor-space interaction and OpenBMI replication, with terminal `GO_SENSOR_SPACE_ONLY`; its invariant spectrum control failed OpenBMI Stage C. PR #4 result `d9a67a130aeeea7eb8a93d76878e43f636802e93` supports only the pairwise leave-one-class-out necessary consequence of a common orthogonal action. It did not establish a globally identifiable or cycle-consistent `Q_s` model.

The exact base is the PR #4 result. Later PR #5–#15 local/trajectory branches are excluded. Parent artifacts and hashes are frozen in the config and Phase-A audit.

## 3. Dataset roles

- OpenBMI / Lee2019-MI: **PRIMARY_RETROSPECTIVE_STRUCTURAL_ANALYSIS**. It has 54 subjects, two independent sessions, two balanced classes, and frozen Full/A/B `U` objects. This is seen data and is never called prospective confirmation.
- BNCI2014_001: **SECONDARY_MULTI_CLASS_MECHANISM_DIAGNOSTIC**. It has nine subjects, two sessions, and four classes. It cannot rescue or overturn OpenBMI and does not establish a population rank.
- Stieger2021: **FUTURE_PROSPECTIVE_CONFIRMATION_CANDIDATE**. Metadata/loader feasibility only; no download, preprocessing, covariance, or statistic.
- HGD and all ASD datasets: excluded from execution.

## 4. Outer-fold-safe reconstruction

Stored parent `Z` is never used as a V1 feature. For every training set `T`, training subject `s` uses the class template averaged over `T` excluding `s`; a held-out subject uses the class template averaged over all of `T`. Each `Rbar` uses that subject's frozen class proportions, and `Z=R-Rbar` is rebuilt separately for Full, A, and B. The same rule is nested inside rank selection and every null.

No held-out subject may contribute to a population template, feature mean, feature scale, latent basis, rank selection, random calibration, or null calibration. Pairing-null derangements are restricted separately to each current training and held-out partition so a raw held-out view cannot enter training. Unit tests inject sentinel held-out values and require training templates, centers, bases, scales, and selected ranks to be invariant.

## 5. Feature geometry

Use upper-triangle row-major Frobenius-isometric `svec`, multiplying off-diagonal coordinates by `sqrt(2)`. No coordinate-wise standardization is allowed.

OpenBMI class order is literal `[left_hand,right_hand]`, and

\[
X_{s,q}=\operatorname{svec}((Z_{s,q,right}-Z_{s,q,left})/2).
\]

The primary `x` is `X/||X||_2`; unnormalized `X` is a magnitude-preserving secondary sensitivity.

BNCI class order is literal `[left_hand,right_hand,feet,tongue]`. The literal orthonormal Helmert matrix and its canonical JSON hash are stored in the config. With class-by-svec matrix `V`, use `vec(H_4 V)` in C row-major order and normalize its global Frobenius norm.

Allowed preprocessing is training-only column mean removal, whole-vector norm normalization, and training-only projected-score SD scaling. Zero/near-zero feature norms or score SDs fail closed.

## 6. Primary two-view model

For outer-training feature matrices `X0,X1`, remove their separate training column means and compute `C01=X0.T@X1/(n_train-1)`. Its SVD is `L diag(sigma) R.T`. The implementation uses an algebraically exact compact subject-space decomposition but tests equality to direct SVD on synthetic matrices.

At rank `r`, project centered view 0 onto `L[:,:r]` and view 1 onto `R[:,:r]`. Divide each coordinate only by its outer-training projected-score sample SD (`ddof=1`). The SVD's paired nonnegative-singular-value convention is accepted as returned; no result-inspected sign flip or reordering is allowed.

Similarity and directional separations are exactly the formulas in the task: average paired coordinate products, subtract the median other-subject similarity in each direction, average directions per subject, and take the pooled subject median. OpenBMI impostors are only other subjects in the same outer test fold.

## 7. OpenBMI folds and rank selection

Use the six literal outer folds and the five literal inner validation folds in the config. Every subject appears in exactly one outer test fold, and both sessions remain paired. The fold-contract SHA-256 is `bcbef19ed8d1b5bf385a600e967a929c185fe70582d8ee0557e30f7096afb500`.

The data-independent rank grid is `[1,2,3,5,8,13,21,34]`, restricted at every fit to `r <= min(p,n_train-2)`. Within each outer-training set, each rank's score is computed in all five paired-subject inner folds. Its inner mean and standard error are calculated from the five fold medians. Let `r_best` maximize the mean (ties choose the smaller rank). The one-standard-error threshold is `mean_best-SE_best`; choose the smallest eligible rank. Test scores never enter this choice.

The low-dimensional cap in every OpenBMI outer fold is

\[
r_{low}=\min(8,\lfloor(n_{train}-1)/4\rfloor)=8.
\]

Reports include `r/p`, `r/(n_train-1)`, all-rank curves, fold rank stability, selected-mode A/B reproducibility, session-direction consistency, and descriptive singular values only.

## 8. Reliability prerequisite

Use immutable independently refitted A/B `U` banks. Reconstruct outer-fold-safe A/B sensor signatures separately. In each session, compute direct A-to-B and B-to-A same-subject-versus-other held-out separation. The group statistic is the median subject-average directional separation.

Use 1,999 deterministic paired-subject-label nulls. For each replicate, half-B identities are fixed-point-free deranged separately within every outer training/test partition and session; all B templates and signatures are rebuilt. A session passes iff its observed separation is positive and one-sided plus-one `p<=0.05`. Both sessions must pass and all leave-one-subject medians must retain positive sign. Failure terminates as `UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE`; no real population rank is interpreted.

## 9. Primary nulls and controls

All stochastic mappings are deterministic functions of the master seed, analysis name, replicate, fold, inner fold, and partition, using canonical JSON, SHA-256, and NumPy `PCG64DXSM`. Exact seeds/mappings are saved.

1. **Subject-pairing destruction:** 1,999 replicates. Session-1 identities are fixed-point-free deranged separately within every current train/test partition. `Z`, features, basis, score scales, inner rank selection, and test scoring are rerun. The null may not reuse observed ranks.
2. **Class-semantics destruction:** 1,999 replicates. Each subject/session receives an independent binary swap. An all-common mapping is rejected. Permuted `U` and proportions rebuild `Z`, features, nested rank selection, and scores.
3. **Equal-rank random subspace:** 1,999 replicates. In each outer fold, independently draw Haar-consistent QR bases for the two sessions at that fold's observed selected rank, with the same outer features, centering, scaling, and scoring. Compare by one-sided plus-one tail probability.
4. **Full sensor-space:** direct outer-fold-safe unit-signature dot products and directional separation. It is stable iff its statistic is positive and its pairing-null `p<=0.05`.
5. **Same-session PCA:** fit PCA in session 0 and apply it to both sessions, then reverse sessions, at the observed selected rank. This never votes on the primary.
6. **Invariant controls:** ordered eigenvalues of each outer-fold `Z`, and log generalized eigenvalues of `(M_class,M_marginal)`, passed through the same class contrast and nested outer-CV pipeline. They are secondary only and cannot rescue sensor-space failure.
7. **Common-action overlap:** only BNCI PR #4 subject-level gains are available. Report Spearman associations with held-out latent-coordinate norm and cross-session coordinate discrepancy plus action-positive/action-nonpositive group summaries. Do not fit a new `Q_s`. Global mode/action overlap is labeled `UNASSESSED_ACTION_OVERLAP_NO_GLOBAL_Q`.

The one-sided Monte Carlo p-value is `(1+#null>=observed)/(1+1999)`. The observed rank is not fixed in pairing/class nulls. Bootstrap uncertainty uses 10,000 deterministic subject resamples and the percentile 95% interval.

## 10. Influence and full-space requirements

Primary leave-one-subject influence recomputes the pooled median from the already held-out subject-level separations with each subject omitted; every value must remain positive. It does not refit the model and is explicitly an influence summary, not independent validation. Full-space stability must reproduce before either GO terminal is allowed.

## 11. Frozen terminal mapping

Apply in order:

1. Object/hash/order/finite/fold/reproducibility failure: `UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE`.
2. Reliability prerequisite failure: `UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE`.
3. Reduced-rank statistic nonpositive, either direction nonpositive, pairing `p>0.05`, class `p>0.05`, influence sign failure, or full-space instability: `STOP_NO_HELDOUT_POPULATION_STRUCTURE`.
4. Equal-rank random-subspace `p>0.05`: `STOP_RANDOM_SUBSPACE_EQUIVALENT`.
5. All structure gates pass but median rank exceeds the low cap or fewer than four of six folds select at/below it: `GO_STRUCTURED_BUT_NOT_LOW_DIMENSIONAL`.
6. All ten declared GO gates pass: `GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION`.

The parent-object audit failure terminal is separately `UNASSESSED_PARENT_OBJECT_INSUFFICIENT` and prevents this protocol from reaching real-data execution.

## 12. BNCI diagnostic

After the unchanged OpenBMI primary, use nine outer LOSO folds, sessions `[0train,1test]`, fixed Helmert contrasts, rank grid `[1,2,3]`, and inner LOSO over the eight outer-training subjects. A held-out subject is compared to outer-training impostors in the same fitted fold because each LOSO test set has one subject. Pairing and independent per-subject/session four-class permutation nulls use 1,999 replicates and rerun selection. This difference is explicit and diagnostic only.

Report class loading energy, per-class/per-pair contribution, directions, full-space, and leave-one-subject influence. No pair is selected after results, and no Left/Right-only result represents the four-class coupled system. The replication target is the phenomenon, never equality of bases across datasets.

## 13. Synthetic gates

Before any real-data execution, the implementation must pass and serialize:

- known paired low rank: positive held-out pairing, learned above pairing null, rank recovery tendency, principal-angle accuracy, and paired session convention;
- stable full-rank: `GO_STRUCTURED_BUT_NOT_LOW_DIMENSIONAL`;
- independent sessions: `STOP_NO_HELDOUT_POPULATION_STRUCTURE`;
- random-subspace-equivalent setting: learned subspace does not exceed the random control;
- unreliable A/B measurement: `UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE`.

The focused tests enforce svec, Helmert, binary redundancy removal, ordering, outer/inner leakage, derangements, rank-selection reruns, random orthogonality, deterministic seeds, fold coverage, parent immutability, terminals, report consistency, finite outputs, and reproducible rerun.

## 14. Execution lock and output contract

Phase A audit precedes implementation. Phase B writes config, this protocol, implementation, tests, validators, and synthetic outputs. Those artifacts are committed with message `freeze subject class population structure v1`; the worktree must be clean. Real-data access is rejected unless HEAD is that clean freeze commit (or its exact result-descendant with no scientific source/config change) and all frozen hashes match.

Real execution order is reliability, OpenBMI observed, full primary nulls/controls, terminal decision, BNCI diagnostic, metadata-only Stieger feasibility, ASD boundary, final report, and validation. Scientific settings cannot change after freeze. Technical failure does not authorize tuning or automatic V2.

Write only under `outputs/subject_class_population_structure_v1/`. Commit compact scientific arrays, CSV/JSON, fixed figures, reports, and provenance; never raw EEG or large regenerable caches.

## 15. Interpretation boundaries

The strongest allowed positive claim is: OpenBMI's montage-registered mean-level subject-by-class interaction contains low-dimensional population-shared linear structure that generalizes to unseen subjects and independent sessions.

This work cannot establish a full conditional law, dispersion structure, physiology, source anatomy, causality, diagnosis, ASD biomarker, target-unlabeled inference, TTA recoverability, a nonlinear/universal individuality manifold, globally identifiable `Q_s`, or equality of modes across datasets.

If positive, the only next question is: **Can an unseen subject's coordinates in the stable interaction subspace be identified from unlabeled marginal EEG without reliable pseudo-labels?**

If negative, the only next question is: **What additional supervision or physiological anchor is required when stable individual interaction does not admit a transferable population-shared low-rank representation?**
