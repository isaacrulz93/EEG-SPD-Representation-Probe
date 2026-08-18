# Audit: Subject-Class Population Structure V1

Status: **PASS — leakage-safe reconstruction is available from immutable compact parent objects**

Audit date: 2026-08-18 (Asia/Seoul)

This is a Phase-A schema, lineage, hash, finite/symmetry, and SPD-metadata audit only. No PCA, cross-covariance SVD, singular-value spectrum, effective-rank selection, subject-mode plot, real pairing statistic, or mode visualization was accessed or computed during this audit.

## 1. Clean parent and isolated branch

- Repository: `isaacrulz93/EEG-SPD-Representation-Probe`
- Isolated worktree: `EEG-SPD-Representation-Probe-population-structure-v1`
- Branch: `pilot/subject-class-population-structure-v1`
- Exact base: `d9a67a130aeeea7eb8a93d76878e43f636802e93`
- Base subject-action result: `PAIRWISE_COMMON_ACTION_NECESSARY_CONSEQUENCE_SUPPORTED`
- Subject-Class Interaction V0 result commit: `272d775678644aad062df424a70586d4b42de652`
- `git merge-base(272d775…, d9a67a…) = 272d775…`; the PR #4 head is nine commits after the PR #2 result and contains it as an ancestor.
- The base is the minimal requested clean parent containing the V0 objects and final PR #4 result. It does not contain the later WINDOW5/local-trajectory branches.
- The pre-existing main worktree is dirty only through its unrelated untracked `reference paper/` directory; it is not modified or staged by this branch.

## 2. Required parent lineage read

### PR #2 — Complete OpenBMI external replication

The frozen V0 protocol, implementation, compact objects, split-half construction, report, and terminal decision were read directly. The operational chain is:

1. AIRM marginal and class Frechet means `M[s,q]` and `M[s,q,c]`;
2. identity-tangent marginal recentering `U[s,q,c]`;
3. leave-one-subject-out session-specific population class template;
4. `R = U - population_template`;
5. class-weighted `Rbar`;
6. primary `Z = R - Rbar`.

The full and independently refitted acquisition-order split halves `A/B` are present. BNCI2014_001 and OpenBMI sensor-space chains passed the parent reliability, identity, and class-dependence gates. The OpenBMI ordered-eigenvalue control failed Stage C (`p=0.108`), so the immutable terminal is `GO_SENSOR_SPACE_ONLY`, not an invariant-spectrum GO. Result commit: `272d775678644aad062df424a70586d4b42de652`.

### PR #4 — Falsify a common subject action for class effects

The final V2 protocol, implementation lineage, subject summaries, and cycle diagnostic were read directly. It fits pairwise O(22) actions with leave-one-class-out prediction. Stage A and the weaker/heterogeneous cross-session Stage B both passed their pairing and semantic nulls (`p=0.0005` for each reported gate). The necessary-consequence terminal is `PAIRWISE_COMMON_ACTION_NECESSARY_CONSEQUENCE_SUPPORTED`. This does not establish a globally identifiable `Q_s` or a cycle-consistent latent action model; the descriptive equivalence-aware cycle median is 0.71613120 and the profiled global product-manifold model was not run. Result commit: `d9a67a130aeeea7eb8a93d76878e43f636802e93`.

### PR #5–#15 boundary audit

| PR | title | base lineage | terminal/status relevant here |
|---:|---|---|---|
| 5 | Audit local AIRM metric subject-class interaction | trajectory audit | `STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0` |
| 6 | Assess local quotient GPA consensus | PR #5 | `UNASSESSED_GPA_NUMERICAL_FAILURE` |
| 7 | Assess local GPA consensus with audited TrustRegions | PR #6 | `UNASSESSED_GPA_NUMERICAL_FAILURE_V1` |
| 8 | Local temporal sequence correspondence V0 | GPA convergence audit | `GO_REPRODUCIBLE_SUBJECT_CLASS_TEMPORAL_SEQUENCE` |
| 9 | Add local mean covariance movement V0 | PR #8 | `GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT` |
| 10 | Local ordered movement component decomposition V0 | PR #9 | `BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS` |
| 11 | OpenBMI ordered movement replication v0 | PR #10 | `UNASSESSED_OPENBMI_DATA_CONTRACT_FAILURE` |
| 12 | BNCI Left/Right-only angular factorial diagnostic v0 | PR #10 | `BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED` |
| 13 | BNCI angular relation anatomy v0 | PR #10 | descriptive anatomy only; parent inference unchanged |
| 14 | BNCI angular six-pair and dual relation anatomy v0 | PR #10 | descriptive anatomy only; parent inference unchanged |
| 15 | Trial-level ordered SPD movement incremental utility audit V0 | PR #13 | `STOP_NO_TRIAL_MOVEMENT_INCREMENTAL_UTILITY` |

These branches answer WINDOW5/local pose, GPA, anti-development, temporal ordering, angular anatomy, or trial-level incremental-utility questions. None is merged into this experiment. This work analyzes only global WHOLE-covariance `Z` population structure.

## 3. Immutable artifact contract and hashes

| role | path | SHA-256 |
|---|---|---|
| BNCI Full/A/B compact objects | `outputs/subject_class_interaction_v0/objects/core_interaction_objects.npz` | `1bb487a8446308e1dd29965ac57901a810d6e92da25b020ae284569176f33f6d` |
| OpenBMI Full/A/B compact objects | `outputs/subject_class_interaction_v0/objects/openbmi_core_interaction_objects.npz` | `f7e2fd7517fe1f55f84ef7729823b2d3f10452833ec2399a4b7014f769c98572` |
| V0 protocol | `outputs/subject_class_interaction_v0/provenance/PROTOCOL_SUBJECT_CLASS_INTERACTION_V0.md` | `88134949377265c1280e4efa292e7265fade4f4bd688cf98ce8b228af48d4657` |
| V0 frozen config | `outputs/subject_class_interaction_v0/provenance/frozen_config.yaml` | `06ea1c28399b081d14cb14c6185d39b3784f59c70b613da34976b820b7b7224d` |
| OpenBMI order/preprocessing manifest | `outputs/subject_class_interaction_v0/provenance/openbmi_protocol_manifest.json` | `4b956d7e3b2b1a271ec07bddecc1ce0a93460ab1515ab8902b3d1ca35ebdb0ea` |
| OpenBMI observed manifest | `outputs/subject_class_interaction_v0/provenance/openbmi_observed_manifest.json` | `8e40746ef847afb38f633da6e948b00ee9b3691cca8d8cf290a6a0816a65fb04` |
| BNCI observed manifest | `outputs/subject_class_interaction_v0/provenance/bnci_observed_manifest.json` | `994e358e9ce6cdecb997e59fa59e144b97897ed2e113a62a484a88499caddae8` |
| PR #4 V2 protocol | `outputs/bnci2014_001_pairwise_common_action_v2/protocol/PROTOCOL_COMMON_SUBJECT_ACTION_PAIRWISE_V2.md` | `c6856ed0dd16c3b76ef5232735e0d894f561ac9cd84a758c4d3e8d62c365677d` |
| PR #4 Stage-A subject gains | `outputs/bnci2014_001_pairwise_common_action_v2/tables/stage_A_subject_summary.csv` | `b1bdcb1061d72fb685ee76462ece593c56a362bbbbc81702791b474fdd9d8cd3` |
| PR #4 Stage-B subject gains | `outputs/bnci2014_001_pairwise_common_action_v2/tables/stage_B_subject_summary.csv` | `5e76e8cfe7c01ed69ca8bede7aad1acf3f237880560ed35f07e6a44985bd37f8` |
| PR #4 cycle diagnostic | `outputs/bnci2014_001_pairwise_common_action_v2/tables/global_cycle_diagnostics.csv` | `e4a50b9b1e34695b2c40ab4825d25260e94224c1ffeb93a406ced212abdc6cf7` |

The V1 code will read these paths but will never overwrite them. It records pre-run and post-run hashes, fails on mismatch, and writes only below `outputs/subject_class_population_structure_v1/`.

## 4. NPZ schema, shapes, dtypes, and numerical metadata

Both archives contain 24 fields for every `(geometry, template, split)` group: `marginal_means`, `class_means`, `class_counts`, `class_proportions`, `U`, `population_templates`, `R`, `Rbar`, `Z`, raw/normalized sensor and spectrum signatures and norms, and per-class norms. The groups cover AIRM/LE, session-specific/pooled-session, and `A/B/F`. V1 uses only the frozen AIRM/session-specific `U`, counts/proportions, and means; stored final `R/Z/signatures` are audit comparators, not outer-CV features.

| dataset | archive keys | uncompressed arrays | primary shapes | dtypes | finite/symmetry/SPD metadata |
|---|---:|---:|---|---|---|
| OpenBMI | 288 | 68,988,672 bytes | `U/Z/class_means=(54,2,2,20,20)`, `marginal_means=(54,2,20,20)`, counts/proportions `(54,2,2)` | matrices/proportions `float64`; counts `int64` | all inspected required arrays finite; exact stored symmetry (`max_abs_asym=0`); smallest inspected Full/A/B AIRM marginal/class-mean eigenvalue > `1.46e-13` |
| BNCI2014_001 | 289, including `metadata_json` | 26,597,160 bytes | `U/Z/class_means=(9,2,4,22,22)`, `marginal_means=(9,2,22,22)`, counts/proportions `(9,2,4)` | matrices/proportions `float64`; counts `int64` | all inspected required arrays finite; exact stored symmetry (`max_abs_asym=0`); smallest inspected Full/A/B AIRM marginal/class-mean eigenvalue > `5.86e-14` |

The eigenvalue checks above are SPD metadata checks on parent covariance means only. No eigenspectrum of `Z`, SVD, or rank-related quantity was inspected.

## 5. Ordering contract

### OpenBMI / Lee2019-MI

- Subjects: literal `1..54` in ascending order.
- Sessions: literal `['0', '1']`, corresponding to source session numbers `[1,2]`.
- Classes: literal `['left_hand', 'right_hand']`.
- Event IDs: left hand `2`, right hand `1`.
- Channels (fixed order): `FC5, FC3, FC1, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6`.
- Each subject/session/class has 50 Full trials and 25 acquisition-order-interleaved trials in each independently refitted half.

The OpenBMI NPZ lacks a separate `metadata_json` scalar. Its order is nevertheless fixed by the immutable protocol manifest and the parent constructor: subjects are sorted, sessions retain manifest/metadata first occurrence, and the literal `CLASSES` order is used. V1 will assert all shapes, counts, proportions, manifest literals, and constructor-order hashes before use.

### BNCI2014_001

- Subjects: literal `[1,2,3,4,5,6,7,8,9]`.
- Sessions: literal `['0train','1test']`.
- Classes: literal `['left_hand','right_hand','feet','tongue']`.
- Channels (fixed order): `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`.
- Each subject/session/class has 72 Full trials and 36 trials in each run-blocked half.

The BNCI archive embeds and agrees with this ordering in all 12 group records in `metadata_json`.

## 6. Leakage-safe reconstruction decision

**PASS.** For each Full/A/B bank, the immutable archive provides subject-local `U[s,q,c]` and class proportions. Therefore every outer or inner split can reconstruct:

- training `mu[T\\{s},q,c]`, `R`, `Rbar`, and `Z` using only the relevant training subset;
- held-out `mu[T,q,c]`, `R`, `Rbar`, and `Z` using no held-out contribution;
- OpenBMI binary sensor signature and BNCI fixed-Helmert signature after this reconstruction;
- split-half A/B signatures from independently refitted parent A/B `U` banks;
- generalized-eigen sensitivity from immutable class and marginal means.

Raw EEG and trial covariances are not required for the declared class-level semantic-destruction rule, which independently permutes the already frozen class-mean labels within each subject/session and then reconstructs `Z` and the complete nested-CV pipeline. No raw download or covariance recomputation is planned.

## 7. Why stored final Z is forbidden for outer evaluation

The V0 `Z[s,q,c]` uses a leave-one-subject-out template over the full dataset. For a subject that is held out in V1, the templates used for V1 training subjects would therefore include that held-out subject's `U`. The stored `Z` also freezes feature preprocessing before the outer split. Reusing it would leak held-out information into population templates and could favor the learned basis, centering, scale, and rank selection. V1 must start from `U` and proportions and refit `mu/R/Rbar/Z`, feature centering, latent basis, score scaling, and inner rank selection inside every outer/inner/null split.

## 8. Leakage risks and enforced mitigations

| risk | mitigation |
|---|---|
| full-dataset V0 template in stored `Z` | never use parent `Z` as a V1 feature; reconstruct from `U` |
| held-out subject in a training subject's leave-one-out template | explicit source-index audit and unit tests at every nesting level |
| held-out subject in feature mean/score scale | fit and record view means and score SDs from training rows only |
| test-guided rank | deterministic paired-subject inner CV only; test labels/scores are inaccessible to selector |
| null reuses observed rank | every pairing/class null reruns nested rank selection |
| global permutation moves test view into training | pairing derangements are generated separately within each current train/test partition |
| class null is a harmless global binary swap | independent subject/session swaps with literal stored mapping; global-common swap is rejected |
| coordinate reweighting | no coordinate-wise StandardScaler; Frobenius-isometric `svec` only |
| parent mutation | pre/post SHA-256 validation and read-only paths |
| result-driven protocol change | implementation/config/tests/synthetic outputs committed cleanly before any real SVD/statistic access |

## 9. Anticipated resources

- Immutable compact input: 56 MiB compressed; 95,585,832 uncompressed array bytes.
- OpenBMI sensor feature dimension: `20*21/2 = 210`.
- BNCI Helmert sensor feature dimension: `3*(22*23/2) = 759`.
- Maximum outer training subjects: 45; maximum OpenBMI identifiable rank under the frozen rule: 43.
- Null workload: 1,999 pairing + 1,999 class-semantic + 1,999 random-subspace replicates, each with six outer folds; pairing and class nulls include five inner folds and rank selection.
- Planned execution: four worker processes; BLAS threads pinned to one; compact subject-space SVD rather than a dense full-rank `p x p` decomposition.
- Exact resource budget: peak RSS must remain at or below 2.0 GiB; total wall-clock budget is 8 hours on the current host. Anticipated wall time is 2–6 hours, with checkpoint/resume at deterministic replicate boundaries. Exceeding the budget is a technical failure and does not authorize changing ranks, folds, null counts, thresholds, or preprocessing.

## 10. Missing-object blockers

None for the frozen primary, reliability prerequisite, declared controls, or BNCI diagnostic. The compact parent objects are sufficient without raw EEG.

PR #4 action overlap is available only for BNCI because PR #4 analyzed BNCI. It is a secondary association/control and cannot vote on the OpenBMI terminal. If its exact table schema cannot support the frozen association after implementation validation, it will be labeled `UNASSESSED_ACTION_OVERLAP` without changing the primary.

## 11. Phase-A decision

`PARENT_OBJECT_CONTRACT_PASS`

The fail-closed terminal `UNASSESSED_PARENT_OBJECT_INSUFFICIENT` is not triggered. Real-data rank/SVD/statistic access remains locked until the protocol/config/implementation/tests/synthetic outputs are committed in the protocol-freeze commit and the worktree is clean.
