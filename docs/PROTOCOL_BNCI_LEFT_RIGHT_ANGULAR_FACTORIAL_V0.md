# Protocol: BNCI2014_001 Left/Right-only Angular Factorial Diagnostic V0

Status: **FROZEN BEFORE ACCESS TO THE NEW LEFT/RIGHT STATISTIC**. This is a minimal retrospective diagnostic of immutable squared angular-cost artifacts. It defines no new representation, preprocessing, optimizer, factorial contrast, or threshold.

## 1. Immutable lineage and source artifacts

The exact parent is `pilot/local-movement-component-decomposition-v0` at `edc1d344cb0657f2f2d87b2992049bceec4705d2`. Its protocol freeze is `95c330de9596fa4c4eb4ee377d5af8d99896f4c3`, scientific result is `0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`, and terminal is `BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS`. The immutable four-class angular result is `T_J=0.19240885452534362`, subject-break `p=0.001`, and class-break `p=0.0105`.

Only these parent artifacts may be read:

- `arrays/component_cost_matrices.npz`, SHA256 `51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091`;
- `arrays/split_half_component_matrices.npz`, SHA256 `cdf618662c1ac5eb3b8fa9b65ad9fde921b45d011c967e87f6ec9a2ab1775307`;
- `tables/c_ang_matrix.csv`, SHA256 `f6c06c3f44807207d7baf0d84226859380e008b1d27a47572f9b94d0dc6735bd`;
- finalized parent `protocol/artifact_manifest.csv`, SHA256 `c3aea494fedee8af1ee42a5c26ff94fc5bfae5e5289549bc878c3941da43df79`.

The three numerical/readable artifacts must also be byte-identical to the parent scientific-result commit and match its manifest entries. The finalized manifest differs only through post-result reporting finalization and is validated separately. Exact `c_ang=c_full-c_len` is an integrity check only. The NPZ is authoritative; parsing the readable decimal CSV must reproduce it within absolute tolerance 1e-15, preserving any sub-ULP text-round-trip difference as a diagnostic rather than replacing NPZ values.

No EEG, covariance, AIRM mean, anti-development, movement tuple, M1 reference, or quotient optimizer may be fitted or recomputed.

## 2. Frozen cell subset

The parent cell order is subjects 1 through 9, each crossed with `[left_hand, right_hand, feet, tongue]`. Select exactly parent indices `[0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33]`, producing subjects 1 through 9 crossed with `[left_hand,right_hand]`. Rows remain session-0 anchors and columns session-1 targets.

Feet and tongue are excluded from the observed statistic, both nulls, and split-half statistics. No other class pair is calculated and no pair search occurs.

## 3. Existing relation-cell statistics at K=2

For every subject and hand anchor, `a` is same-subject/same-class cost; `b` is same-subject/opposite-hand cost; `c` is the mean over eight other subjects for the same class; and `d` is the mean over eight other subjects for the opposite hand. Smaller squared angular cost means greater similarity.

Retain exactly the existing BNCI contrasts:

- `S_sc=c_sc-a_sc`;
- `C_sc=b_sc-a_sc`;
- `J_sc=b_sc+c_sc-a_sc-d_sc`.

The alternative balanced factorial S/C definitions are forbidden. Average Left/Right within subject to form `S_s`, `C_s`, and `J_s`, then average the nine subjects to form `T_subject`, `T_class`, and `T_J`. Means are mandatory; no median, pair weighting, or pair-level population inference is allowed.

Before use at K=2, one generalized K-class implementation must reproduce the frozen K=4 angular values `T_subject=0.3091561771980925`, `T_class=0.39309843397343514`, and `T_J=0.19240885452534362` within absolute/relative tolerance 1e-14.

## 4. Frozen null mappings and endpoint

Use exactly 1,999 draws and one-sided greater-than-or-equal plus-one p-values `(1+count(null>=observed))/2000`, with alpha 0.05.

Subject-break uses `np.random.default_rng(np.random.SeedSequence([20260810,1102]))`. Independently for Left and Right in every draw, apply an ordinary random permutation of all nine complete session-1 subject tuples. Fixed points are allowed.

Class-break uses `np.random.default_rng(np.random.SeedSequence([20260810,1101]))`. Independently within every session-1 subject in every draw, apply an ordinary random permutation of the complete Left/Right class tuples.

The primary endpoint is `T_J_ang_BNCI_LR`. It is supported if and only if it is strictly positive and both its subject-break and class-break p-values are below 0.05. `T_subject` with its subject-break p-value and `T_class` with its class-break p-value are supporting and cannot rescue J.

## 5. Frozen split-half diagnostic

Use the two saved split-half `c_ang_matrix` arrays only. Extract the identical 18 cells and compute the same K=2 `T_J` for Half A and Half B without optimizer fitting or half-level p-values. Define `split_half_sign_stable=(T_J_A>0) AND (T_J_B>0)`. Split halves cannot rescue a failed primary null test.

## 6. Integrity gates

Before interpretation require:

1. exact canonical 18-cell subjects/classes;
2. exact array slicing from parent `c_ang`;
3. zero Feet/Tongue entries in observed and null objects;
4. all subject-break mappings preserve class;
5. all class-break mappings preserve subject;
6. exact K=4 frozen-statistic regression;
7. exact split-half slicing;
8. parent artifact hashes unchanged before/after execution.

Any genuine failure returns `UNASSESSED_BNCI_LR_ANGULAR_DIAGNOSTIC_FAILURE` and stops interpretation.

## 7. Terminal and claims

Use exactly one:

- `BNCI_LR_ANGULAR_INTERACTION_SUPPORTED_AND_STABLE` when inferentially supported and both halves are positive;
- `BNCI_LR_ANGULAR_INTERACTION_SUPPORTED_BUT_SPLIT_UNSTABLE` when inferentially supported but a half is not positive;
- `BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED` when the primary support rule fails;
- `UNASSESSED_BNCI_LR_ANGULAR_DIAGNOSTIC_FAILURE` only for technical/integrity failure.

If unsupported, report: “The previously supported BNCI four-class angular/joint interaction was not supported when the same frozen analysis was restricted to Left versus Right.” Do not claim absence without equivalence testing. Do not alter or reinterpret the parent four-class terminal, discuss OpenBMI as a formal replication, or make physiological, motor-strategy, neural-direction, or anatomical claims.

After protocol freeze, no subset, source artifact, contrast, aggregation, mapping, seed, draw count, alpha, stability rule, terminal, or claim rule may change. No rescue analysis is allowed.
