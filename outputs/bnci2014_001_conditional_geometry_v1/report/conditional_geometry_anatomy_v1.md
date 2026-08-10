# BNCI2014_001 Conditional Geometry Anatomy v1

## Scientific question

This preregistered anatomy asks whether four true motor-imagery class prototypes have a relative WHOLE-covariance geometry that is reliable across runs, shared across subjects and sessions under fixed semantic names, and identifiable when an oracle supplies the four target components.

## Why this follows V1/V2/Trajectory v0

V1 diagnosed information in WHOLE versus WINDOW5 representations; V2 audited marginal centering geometry; Trajectory v0 tested local temporal order. This experiment instead examines class-prototype relational anatomy using WHOLE covariances only. It does not repeat WINDOW5 or trajectory analysis.

## Frozen protocol

Protocol version 1.0, seed 20260809, AIRM primary and LE robustness. Discovery is session `0train`; confirmation is locked session `1test`. Splits are A=runs 0–2, B=runs 3–5 and F=runs 0–5. Four classes follow the fixed order `left_hand, right_hand, feet, tongue`. The unlock status was `CONFIRMATORY_UNLOCKED` with designation `STRICT_CONFIRMATORY`.

## Data and numerical gates

Validated hard-gate rows: 6770; rows carrying `FAIL`: 0. Global failure classification: **none**. A single required failure makes every scientific chain UNASSESSED; no available-case substitute is used.

## Exact AIRM objects D and G

D contains the six pairwise AIRM distances among class Fréchet means. G is the 4×4 marginal-anchor tangent Gram matrix and was checked by both the direct linear-solve expression and the whitened-identity expression. AIRM marginal centering is an isometry: it cannot create this within-subject class geometry. See [D heatmaps](../figures/figure_5_D_heatmaps.png) and [G heatmaps](../figures/figure_6_G_heatmaps.png).

## Discovery reliability

| geometry | object | observed | null median | effect | p | B |
| --- | --- | --- | --- | --- | --- | --- |
| AIRM | D | 0.9923 | 0.9895 | 0.0028 | 0.2005 | 1999 |
| AIRM | G | 0.9432 | 0.9052 | 0.0380 | 0.0800 | 1999 |
| LE | D | 0.9912 | 0.9863 | 0.0048 | 0.1190 | 1999 |
| LE | G | 0.9457 | 0.8797 | 0.0660 | 0.0095 | 1999 |

[Subject scores](../figures/figure_1_within_subject_reliability.png) and [label-null distribution](../figures/figure_2_reliability_label_null.png).

## Confirmatory reliability

| geometry | object | observed | null median | effect | p | B |
| --- | --- | --- | --- | --- | --- | --- |
| AIRM | D | 0.9957 | 0.9867 | 0.0090 | 0.0020 | 1999 |
| AIRM | G | 0.9730 | 0.8824 | 0.0906 | 0.0005 | 1999 |
| LE | D | 0.9941 | 0.9827 | 0.0114 | 0.0010 | 1999 |
| LE | G | 0.9663 | 0.8543 | 0.1119 | 0.0005 | 1999 |

The confirmatory p-values use the independently shuffled `1test` labels within subject×session×run and the frozen plus-one rule.

## Same-subject vs unrelated reference

| phase | geometry | object | same | unrelated median | min | max | !9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discovery | AIRM | D | 0.9923 | 0.9590 | 0.9212 | 0.9865 | 133496 |
| discovery | AIRM | G | 0.9432 | 0.8005 | 0.6454 | 0.9305 | 133496 |
| discovery | LE | D | 0.9912 | 0.9545 | 0.9133 | 0.9864 | 133496 |
| discovery | LE | G | 0.9457 | 0.7860 | 0.6056 | 0.9260 | 133496 |
| confirmatory | AIRM | D | 0.9957 | 0.9547 | 0.9176 | 0.9834 | 133496 |
| confirmatory | AIRM | G | 0.9730 | 0.7673 | 0.5967 | 0.8939 | 133496 |
| confirmatory | LE | D | 0.9941 | 0.9459 | 0.9039 | 0.9806 | 133496 |
| confirmatory | LE | G | 0.9663 | 0.7374 | 0.5786 | 0.8976 | 133496 |

This exhaustive derangement reference is descriptive and never votes. [Figure 3](../figures/figure_3_same_vs_unrelated.png).

## Discovery cross-subject shared geometry

| geometry | object | observed | null median | effect | p | B |
| --- | --- | --- | --- | --- | --- | --- |
| AIRM | D | 0.9795 | 0.9640 | 0.0156 | 0.0051 | 100000 |
| AIRM | G | 0.8735 | 0.7702 | 0.1033 | 0.0042 | 100000 |
| LE | D | 0.9783 | 0.9577 | 0.0206 | 0.0030 | 100000 |
| LE | G | 0.8669 | 0.7415 | 0.1254 | 0.0037 | 100000 |

Each discovery score averages source-A→target-B and source-B→target-A LOSO comparisons. Templates normalize the sum of unit subject shapes.

## Locked confirmatory shared geometry

| geometry | object | observed | null median | effect | p | B |
| --- | --- | --- | --- | --- | --- | --- |
| AIRM | D | 0.9679 | 0.9641 | 0.0039 | 0.3203 | 100000 |
| AIRM | G | 0.8539 | 0.7526 | 0.1013 | 0.0180 | 100000 |
| LE | D | 0.9623 | 0.9563 | 0.0060 | 0.2637 | 100000 |
| LE | G | 0.8531 | 0.7232 | 0.1299 | 0.0044 | 100000 |

The confirmatory template is the fixed discovery-F LOSO template; confirmatory data never updates it. [Subject scores](../figures/figure_4_shared_template_similarity.png).

## Oracle semantic-permutation identifiability

| geometry | object | observed | null median | effect | p | B |
| --- | --- | --- | --- | --- | --- | --- |
| AIRM | D | 0.9130 | 0.4783 | 0.4348 | 0.0025 | 1000000 |
| AIRM | G | 0.9130 | 0.4783 | 0.4348 | 0.0025 | 1000000 |
| LE | D | 0.9130 | 0.4783 | 0.4348 | 0.0026 | 1000000 |
| LE | G | 0.9130 | 0.4783 | 0.4348 | 0.0026 | 1000000 |

Stage P supplies the target's four true components and hides only their names. It is therefore not clustering, unlabeled component recovery, adaptation, or a deployable score. Identity receives the conservative worst rank within 1e-12. [Candidate scores](../figures/figure_7_oracle_permutation_scores.png); [rank and margin](../figures/figure_8_oracle_rank_margin.png).

## D-chain

| geometry | stage | discovery effect | confirmatory effect | confirmatory p | status |
| --- | --- | --- | --- | --- | --- |
| AIRM | R | 0.0028 | 0.0090 | 0.0020 | PASS |
| AIRM | S | 0.0156 | 0.0039 | 0.3203 | FAIL |
| AIRM | P | 0.4348 | 0.4348 | 0.0025 | DESCRIPTIVE_ONLY |
| LE | R | 0.0048 | 0.0114 | 0.0010 | PASS |
| LE | S | 0.0206 | 0.0060 | 0.2637 | FAIL |
| LE | P | 0.4348 | 0.4348 | 0.0026 | DESCRIPTIVE_ONLY |

## G-chain

| geometry | stage | discovery effect | confirmatory effect | confirmatory p | status |
| --- | --- | --- | --- | --- | --- |
| AIRM | R | 0.0380 | 0.0906 | 0.0005 | PASS |
| AIRM | S | 0.1033 | 0.1013 | 0.0180 | PASS |
| AIRM | P | 0.3913 | 0.4348 | 0.0025 | PASS |
| LE | R | 0.0660 | 0.1119 | 0.0005 | PASS |
| LE | S | 0.1254 | 0.1299 | 0.0044 | PASS |
| LE | P | 0.4348 | 0.4348 | 0.0026 | PASS |

## LE robustness

Frozen LE label: **AIRM+LE CONSISTENT**. LE is secondary and cannot rescue or change the AIRM terminal decision. Paired deltas below use each subject's null-referenced effect (`observed − subject-null median`), never the raw score. Intervals are frozen 20,000-resample subject-bootstrap intervals and do not vote.

| object | stage | confirm−discovery effect Δ [CI] | discovery AIRM−LE effect Δ [CI] | confirmatory AIRM−LE effect Δ [CI] |
| --- | --- | --- | --- | --- |
| D | R | 0.0064 [-0.0017, 0.0155] | -0.0025 [-0.0049, -0.0002] | -0.0032 [-0.0053, 0.0016] |
| D | S | 0.0075 [-0.0123, 0.0215] | -0.0033 [-0.0064, 0.0014] | -0.0043 [-0.0056, 0.0006] |
| D | P | -0.0435 [-0.0870, 0.0435] | 0.0000 [-0.1304, 0.0000] | 0.0000 [-0.0870, 0.0435] |
| G | R | 0.0280 [-0.0176, 0.1207] | -0.0168 [-0.0395, -0.0007] | -0.0286 [-0.0364, 0.0219] |
| G | S | 0.0778 [-0.1142, 0.1274] | -0.0053 [-0.0375, 0.0084] | -0.0213 [-0.0283, 0.0034] |
| G | P | -0.0435 [-0.0870, 0.0000] | 0.0000 [-0.1304, 0.0000] | 0.0000 [-0.1304, 0.0000] |

[AIRM/LE stage effects](../figures/figure_10_airm_le_stage_effects.png); [paired subject effects and influence](../figures/figure_9_subject_forest_influence.png).

## Terminal frozen decision

**STOP_TANGENT_ONLY**

The label follows only the frozen AIRM D/G R→S→P chains and the explicit global failure gate. No absolute cosine cutoff or post-result rule was used.

## What is actually justified

The anchored tangent G chain passed while the exact metric-shape D chain did not. The frozen terminal rule is STOP_TANGENT_ONLY; this discrepancy is not evidence for a new method.

[Subject effect deltas and cached-score influence](../figures/figure_9_subject_forest_influence.png) are descriptive and do not vote. The influence statistic remains the raw cached subject-score `T_leave-one-subject-out − T_full`; paired deltas instead use null-referenced subject effects.

## What is NOT justified

D/G are not full conditional distributions. This experiment did not show that all subject variation was removed. Oracle P is not unlabeled target-component recovery, pseudo-labeling, conditional alignment, or domain-adaptation success. No target-label-free conditional identifiability claim is made. There is no new WINDOW5, temporal-order, trajectory, neural-model, neuroscience-mechanism, or cross-dataset conclusion.

All figure artifacts:

- [figure_1_within_subject_reliability PNG](../figures/figure_1_within_subject_reliability.png); [PDF](../figures/figure_1_within_subject_reliability.pdf); [source CSV](../figures/figure_1_within_subject_reliability.csv)
- [figure_2_reliability_label_null PNG](../figures/figure_2_reliability_label_null.png); [PDF](../figures/figure_2_reliability_label_null.pdf); [source CSV](../figures/figure_2_reliability_label_null.csv)
- [figure_3_same_vs_unrelated PNG](../figures/figure_3_same_vs_unrelated.png); [PDF](../figures/figure_3_same_vs_unrelated.pdf); [source CSV](../figures/figure_3_same_vs_unrelated.csv)
- [figure_4_shared_template_similarity PNG](../figures/figure_4_shared_template_similarity.png); [PDF](../figures/figure_4_shared_template_similarity.pdf); [source CSV](../figures/figure_4_shared_template_similarity.csv)
- [figure_5_D_heatmaps PNG](../figures/figure_5_D_heatmaps.png); [PDF](../figures/figure_5_D_heatmaps.pdf); [source CSV](../figures/figure_5_D_heatmaps.csv)
- [figure_6_G_heatmaps PNG](../figures/figure_6_G_heatmaps.png); [PDF](../figures/figure_6_G_heatmaps.pdf); [source CSV](../figures/figure_6_G_heatmaps.csv)
- [figure_7_oracle_permutation_scores PNG](../figures/figure_7_oracle_permutation_scores.png); [PDF](../figures/figure_7_oracle_permutation_scores.pdf); [source CSV](../figures/figure_7_oracle_permutation_scores.csv)
- [figure_8_oracle_rank_margin PNG](../figures/figure_8_oracle_rank_margin.png); [PDF](../figures/figure_8_oracle_rank_margin.pdf); [source CSV](../figures/figure_8_oracle_rank_margin.csv)
- [figure_9_subject_forest_influence PNG](../figures/figure_9_subject_forest_influence.png); [PDF](../figures/figure_9_subject_forest_influence.pdf); [source CSV](../figures/figure_9_subject_forest_influence.csv)
- [figure_10_airm_le_stage_effects PNG](../figures/figure_10_airm_le_stage_effects.png); [PDF](../figures/figure_10_airm_le_stage_effects.pdf); [source CSV](../figures/figure_10_airm_le_stage_effects.csv)

## One next question only

Which representation or mechanistic anchor could replace the failed WHOLE-covariance relational anchor?
