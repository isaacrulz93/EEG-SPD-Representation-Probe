# Audit: Unlabeled Conditional-Mode Identifiability V0

Status: **PASS WITH EXACT REBUILD REQUIRED**

Audit date: 2026-08-18 (Asia/Seoul)

This audit is limited to lineage, file/schema/hash availability, ordering, storage, and preprocessing contracts. No real mode refit, recovery statistic, trial projection, mixture fit, null statistic, or calibration result was accessed.

## 1. Exact stacked parent

- Repository: `isaacrulz93/EEG-SPD-Representation-Probe`
- New isolated worktree branch: `pilot/unlabeled-conditional-mode-identifiability-v0`
- Exact parent head: `9dee7642ac573f37756b8427a75864a50c32044e`
- Parent branch: `pilot/subject-class-population-structure-v1`
- Parent terminal: `GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION`
- Parent V1.1 rank-1 object: `outputs/subject_class_population_structure_v1_1/objects/openbmi_observed_core.npz`, SHA-256 `ecf0657d1ccf51e7aa3392a5e95608d849e9069dda2f27e2a76e321429fac168`.
- Parent terminal artifact: `outputs/subject_class_population_structure_v1_1/decisions/terminal_decision.json`, SHA-256 `8d4eaa8e4effb1efc124965e5400d247dd472f1b3c06283f20dc4a583f41b631`.
- Parent OpenBMI interaction object: `outputs/subject_class_interaction_v0/objects/openbmi_core_interaction_objects.npz`, SHA-256 `f7e2fd7517fe1f55f84ef7729823b2d3f10452833ec2399a4b7014f769c98572`.
- Parent V1 scientific config: `configs/subject_class_population_structure_v1.yaml`, SHA-256 `72d6ab325ebe199be3320fc561ae7e161a26008d1b9d009b1eb3e4b07673f2d9`.

The new branch is stacked directly on the parent result commit. It does not modify or regenerate either `outputs/subject_class_population_structure_v1/` or `outputs/subject_class_population_structure_v1_1/`. The freeze step records a complete per-file SHA-256 snapshot of both directories and every later runner revalidates it.

## 2. Frozen folds and mode-object schema

The exact six outer folds and five inner folds per outer fold remain those in the immutable V1 config. The canonical fold hash is `bcbef19ed8d1b5bf385a600e967a929c185fe70582d8ee0557e30f7096afb500`. Every subject 1--54 occurs in exactly one outer test fold and both sessions remain paired.

The frozen V1.1 mode archive contains:

| key | shape | dtype | role |
|---|---:|---|---|
| `subjects` | `(54,)` | `int64` | subject ordering |
| `selected_ranks` | `(6,)` | `int64` | all six frozen selections are rank 1 |
| `mean0`, `mean1` | `(6, 210)` | `float64` | training-only feature means |
| `left`, `right` | `(6, 210, 34)` | `float64` | paired session-view bases |
| `singular_values` | `(6, 34)` | `float64` | descriptive cross-covariance spectrum |
| `scale0`, `scale1` | `(6, 34)` | `float64` | training-only projected-score scales |

This experiment uses only the first column of `left` and `right` after independently reconstructing each fold fit from the frozen V0 `U` objects and confirming sign-invariant identity with the stored direction.

## 3. Trial-level source availability

No immutable trial-covariance NPZ is tracked in the repository, and none is present in the three existing repository worktree caches. Therefore the trial object cannot be silently substituted from another pipeline.

An exact-rebuild source is available:

- Parent source manifest: `outputs/subject_class_interaction_v0/provenance/openbmi_source_manifest.json`, SHA-256 `5133a0d8521f4cdd121362663c4980fed500e72875f26bb3dc4ce830d8c5e409`.
- Parent protocol manifest: `outputs/subject_class_interaction_v0/provenance/openbmi_protocol_manifest.json`, SHA-256 `4b956d7e3b2b1a271ec07bddecc1ce0a93460ab1515ab8902b3d1ca35ebdb0ea`.
- Manifest records: 108 subject/session files.
- Local MNE tree records found: 108.
- 106 local files match the parent byte-size contract.
- Subject 5 sessions 1 and 2 do not match parent byte sizes. The parent manifest records both as `source_reused_from_mne_cache=false`; the V0 run downloaded the canonical URL to a temporary file. V0 behavior will be reproduced in the new cache without changing the MNE cache.
- The rebuild validates every source SHA-256, every derived covariance-array SHA-256, and every metadata SHA-256 against the parent manifest.

If any source or derived array fails these hashes, the experiment terminates `UNASSESSED_TRIAL_LEVEL_OBJECT_INSUFFICIENT` before a recovery statistic is computed.

## 4. Frozen preprocessing lineage

The only permitted rebuild calls the parent `_prepare_one` implementation without parameter overrides:

1. offline `EEG_MI_train` run only;
2. ordered 20-channel motor montage `FC5, FC3, FC1, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6`;
3. continuous 8--30 Hz fifth-order Butterworth IIR, zero phase;
4. resample 1000 Hz to 100 Hz;
5. epoch half-open `[1.0, 3.5)` seconds, 250 samples;
6. OAS covariance using `pyriemann.estimation.Covariances(estimator="oas")`;
7. `float64`, no extra regularization, no eigenvalue clipping;
8. 100 trials per subject/session, 50 per semantic class.

The combined expected trial covariance shape is `(10800, 20, 20)`. Subject ordering is 1--54; within subject it is source session 1 then 2; within session it is acquisition order 0--99. Evaluation metadata contains `subject`, session mapped to `"0"/"1"`, `trial_uid`, `class_label`, and acquisition order.

## 5. Label stripping and leakage boundary

The covariance matrices contain no label or event-code field. The runner creates two separate objects:

- an unlabeled object containing only shuffled covariance rows and opaque row IDs;
- an evaluation-only label vector stored separately and inaccessible to marginal fitting, tangent projection, mixture estimation, estimator selection, or stopping conditions.

The deterministic shuffle occurs within each subject/session after labels, event codes, block/run identifiers, acquisition order, and semantic trial order have been excluded. An acquisition-order sentinel recomputes every unlabeled estimate after an independent order permutation and requires machine-precision identity.

The parent metadata is balanced exactly, so the equal-weight mixture and calibration budgets are feasible. Balance is verified only in the evaluation/contract layer; it is not passed to the zero-label estimator except as the pre-frozen dataset-level assumption.

## 6. Parent-mean reproduction gate

The rebuild must reproduce, from the exact trial covariances:

- all `(54, 2, 20, 20)` full AIRM marginal means;
- all `(54, 2, 2, 20, 20)` full AIRM class means;
- all full `U` matrices;

against the immutable parent archive. Reproduction uses the parent mean tolerance `1e-9`, maximum 100 iterations, SPD and Karcher gates, and a strict relative-Frobenius comparison. A failure is `UNASSESSED_TRIAL_LEVEL_OBJECT_INSUFFICIENT`, not permission to alter preprocessing.

## 7. Resource contract

- Existing raw source tree: 108 files, approximately 60 GiB.
- New derived cache: approximately 35--120 MiB compressed plus metadata; not committed.
- New compact scientific artifacts: anticipated below 50 MiB.
- Peak resident memory: anticipated below 4 GiB with one subject/session streamed at a time.
- Exact source hash plus covariance rebuild: anticipated 30--120 minutes depending on disk and the two canonical downloads.
- Mode/refit, 1,999 nulls, 10,000 subject bootstraps, and calibration: anticipated 20--90 minutes.
- Available volume space at audit: approximately 57.7 GiB, sufficient because raw sources are not duplicated and only one temporary download exists at a time.

## 8. Audit decision

The immutable trial covariance object is absent, but an exactly reproducible, hash-locked raw source and the exact parent preprocessing implementation are available. The trial-level gate is therefore **PASS WITH EXACT REBUILD REQUIRED**. No alternative preprocessing or cached representation is authorized.
