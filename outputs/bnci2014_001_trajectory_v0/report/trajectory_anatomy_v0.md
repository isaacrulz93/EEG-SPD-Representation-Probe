# BNCI2014_001 Trajectory Anatomy v0

## 1. Scientific question

Does the ordered five-state local covariance path contain class information beyond an unordered set of the same states, and how much subject structure remains? This is representation anatomy, not a classifier-development study.

## 2. Why V1 did not test this question

V1's WHOLE representation compressed each 1,000-sample trial to one covariance. Its WINDOW5 probe treated the five local covariances as independent views and averaged held-out class probabilities; it did not encode a trial as an unordered finite SPD metric space or as an ordered path. V1 therefore could not isolate temporal-order information, and the present analysis does not reinterpret V1 performance as trajectory evidence.

## 3. Frozen protocol

Protocol 0.0; SHA-256 `16062edc068a8287bee1abc97d00ab9d6b15f9fcb29f76225e391594474ed75e`; config SHA-256 `b75890b1f4f8215561501c50ca4049b833efccee7bfce8559a548b59587c99db`; seed 20260809. Only session `0train` was admissible: 9 subjects, 4 fixed classes, 2,592 trials, 22 EEG channels, 8–32 Hz, cue-relative 0–3.996 s, 250 Hz, 1,000 samples, OAS float64. Each trial was split into five ordered, non-overlapping 200-sample windows. AIRM was primary and LE secondary. No loading, clipping, tuning, result-selected embedding, or session `1test` access was permitted.

## 4. Geometry correctness

Persisted `trajectory_geometry_gate.json` validation: `PASS`. Required recorded gate rows passed: 119476/119476. Covariance rows: 12960/12,960. Geometry-correctness rows: 106502. Terminal gate status: `PASS`.

- None recorded.

## 5. Five-state AIRM geometry

AIRM trial-feature rows: 2592/2,592. Each trial used the full pairwise five-state distance matrix, intrinsic path quantities, an AIRM local barycenter, and fixed-time endpoint-geodesic deviations d(C_j,G(j/4)); deviations were not minimized over the geodesic set.

[Scalar-by-class figure](../figures/figure_5_scalars_by_class.png) and [scalar-by-subject figure](../figures/figure_6_scalars_by_subject.png).

## 6. BAG vs PATH definition

PATH_D10 retains the fixed chronological upper-triangle distance order. BAG_CANON_D10 is the lexicographically minimal upper triangle over all 120 state permutations and removes ordering; BAG_SORTED_D10 is secondary. BAG invariance is a hard gate, not an order-null classifier vote.

[Class LOSO figure](../figures/figure_1_class_loso_ba.png).

## 7. Intrinsic path quantities

- total_path_length: n=2592, mean=16.177072, SD=1.990081, median=15.793231.
- endpoint_distance: n=2592, mean=4.394223, SD=0.834740, median=4.204246.
- efficiency: n=2592, mean=0.271771, SD=0.039153, median=0.265459.

The eleven-scalar probe was frozen before results. Individual steps, turns, and deviations are descriptive only and were not selected post hoc.

## 8. Class LOSO results

Balanced accuracy (BA) is primary. Summaries use only PASS rows descriptively and show the required denominator 9; an incomplete denominator never receives a verdict.
Across all fitted-condition tables, convergence warnings were recorded in 0/7251 rows; any such required row is FAILED.

| geometry | representation | pass_n | required_n | mean_ba | sd_ba_ddof1 | median_ba | min_ba | max_ba |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIRM | BAG_CANON_D10 | 9.0000 | 9.0000 | 0.2654 | 0.0267 | 0.2639 | 0.2222 | 0.3056 |
| AIRM | BAG_SORTED_D10 | 9.0000 | 9.0000 | 0.2604 | 0.0231 | 0.2569 | 0.2326 | 0.2986 |
| AIRM | PATH_D10 | 9.0000 | 9.0000 | 0.2616 | 0.0101 | 0.2604 | 0.2465 | 0.2812 |
| AIRM | SCALARS_11 | 9.0000 | 9.0000 | 0.2635 | 0.0159 | 0.2639 | 0.2431 | 0.2951 |
| LE | BAG_CANON_D10 | 9.0000 | 9.0000 | 0.2596 | 0.0251 | 0.2569 | 0.2222 | 0.2986 |
| LE | PATH_D10 | 9.0000 | 9.0000 | 0.2608 | 0.0169 | 0.2674 | 0.2361 | 0.2778 |
| LE | SCALARS_11 | 9.0000 | 9.0000 | 0.2666 | 0.0185 | 0.2674 | 0.2361 | 0.3056 |

## 9. Order-shuffle falsification

Each of 199 replicates independently applied a nonidentity S5 permutation to every trial. The statistic was median subject BA and the one-sided Monte Carlo p-value used the fixed plus-one rule.

| geometry | representation | observed_median_subject_ba | null_median | effect | p_value | exceedance_count | null_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AIRM | PATH_D10 | 0.2604 | 0.2604 | 0.0000 | 0.5150 | 102.0000 | 199.0000 |
| LE | PATH_D10 | 0.2674 | 0.2569 | 0.0104 | 0.2250 | 44.0000 | 199.0000 |

[Order-null figure](../figures/figure_2_order_shuffle_null.png).

## 10. Label-destruction null

Labels were permuted within subject×run using identical permutations for AIRM PATH_D10 and BAG_CANON_D10. Features and identities remained fixed.

| geometry | representation | observed_median_subject_ba | null_median | effect | p_value | exceedance_count | null_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AIRM | PATH_D10 | 0.2604 | 0.2500 | 0.0104 | 0.1550 | 30.0000 | 199.0000 |
| AIRM | BAG_CANON_D10 | 0.2639 | 0.2500 | 0.0139 | 0.1350 | 26.0000 | 199.0000 |

[Label-null figure](../figures/figure_3_label_destruction_null.png).

## 11. Subject-information results

The subject-ID probe used train-only scaling and disjoint run halves; chance was exactly 1/9.

| representation | A_TO_B_ba | B_TO_A_ba | average_ba | chance |
| --- | --- | --- | --- | --- |
| BAG_CANON_D10 | 0.3789 | 0.3434 | 0.3611 | 0.1111 |
| PATH_D10 | 0.3573 | 0.3326 | 0.3449 | 0.1111 |
| SCALARS_11 | 0.4028 | 0.3603 | 0.3816 | 0.1111 |

[Subject-probe figure](../figures/figure_7_subject_probe.png).

## 12. Class vs subject vs interaction effects

Balanced sums of squares were decomposed for all 11 frozen scalars without p-values. Every nondegenerate scalar required relative SS closure ≤1e-10.

| geometry | scalar | eta2_subject | eta2_class | eta2_interaction | eta2_residual |
| --- | --- | --- | --- | --- | --- |
| AIRM | total_path_length | 0.6211 | 0.0078 | 0.0107 | 0.3604 |
| AIRM | endpoint_distance | 0.3960 | 0.0070 | 0.0125 | 0.5845 |
| AIRM | efficiency | 0.0381 | 0.0016 | 0.0176 | 0.9426 |
| AIRM | excess | 0.4289 | 0.0046 | 0.0143 | 0.5521 |
| AIRM | mean_turn | 0.0409 | 0.0033 | 0.0286 | 0.9272 |
| AIRM | max_turn | 0.0152 | 0.0003 | 0.0099 | 0.9746 |
| AIRM | mean_geodesic_deviation | 0.5539 | 0.0047 | 0.0244 | 0.4170 |
| AIRM | max_geodesic_deviation | 0.4021 | 0.0043 | 0.0362 | 0.5574 |
| AIRM | frechet_variance | 0.6524 | 0.0046 | 0.0165 | 0.3266 |
| AIRM | frechet_radius_mean | 0.6598 | 0.0063 | 0.0146 | 0.3192 |
| AIRM | diameter | 0.4615 | 0.0054 | 0.0217 | 0.5113 |
| LE | total_path_length | 0.5174 | 0.0055 | 0.0141 | 0.4630 |
| LE | endpoint_distance | 0.2956 | 0.0068 | 0.0142 | 0.6834 |
| LE | efficiency | 0.0329 | 0.0028 | 0.0193 | 0.9450 |
| LE | excess | 0.3161 | 0.0024 | 0.0181 | 0.6634 |
| LE | mean_turn | 0.0833 | 0.0016 | 0.0228 | 0.8923 |
| LE | max_turn | 0.0285 | 0.0011 | 0.0085 | 0.9619 |
| LE | mean_geodesic_deviation | 0.4668 | 0.0040 | 0.0306 | 0.4986 |
| LE | max_geodesic_deviation | 0.3292 | 0.0041 | 0.0423 | 0.6243 |
| LE | frechet_variance | 0.5538 | 0.0037 | 0.0216 | 0.4209 |
| LE | frechet_radius_mean | 0.5750 | 0.0051 | 0.0181 | 0.4019 |
| LE | diameter | 0.3776 | 0.0050 | 0.0245 | 0.5929 |

[All-scalar eta-squared figure](../figures/figure_4_scalar_eta2.png).

## 13. LOCAL_BARYCENTER / WHOLE contextual controls

LOCAL_BARYCENTER MDM PASS denominator 9/9, mean BA 0.302469. WHOLE-1000 MDM PASS denominator 9/9, mean BA 0.314815. WHOLE uses 1,000 samples per covariance while each local covariance uses 200, so this is estimator-regime-confounded context and not an unconfounded method comparison.

## 14. AIRM vs LE robustness

The robustness table contains 90 paired rows; 11 rows were not marked as agreement. LE is secondary and cannot rescue an AIRM failure or vote in the terminal verdict.

Descriptive robustness label for the repeated LOSO/order-evidence pattern: **AIRM+LE CONSISTENT** (AIRM order evidence=False; LE order evidence=False). This label does not extend to an LE label-destruction hypothesis and does not vote in the terminal verdict.

[AIRM/LE robustness figure](../figures/figure_8_airm_le_robustness.png).

## 15. Frozen verdict

Terminal verdict: **STOP_LOCAL_TRAJECTORY_V0**.

Failure status: **none**.

- H_PATH_CLASS: False; effect=0.010417, p=0.155000.
- H_BAG_CLASS: False; effect=0.013889, p=0.135000.
- H_ORDER: False; order effect=0.000000, p=0.515000, median subject PATH−BAG=-0.010417.

The ordered decision rule in protocol Section 15.2 was applied to full-precision values. Displays were rounded only after the decision.

## 16. What is actually justified

The frozen null comparisons did not establish local PATH or BAG class-label signal in this session-0 pilot.

## 17. What is NOT justified

This pilot does not justify conditional geometry or conditional alignment, domain adaptation, pseudo-labels, target-label-free conditional identifiability, a neural/sequence/trajectory model, a new distribution classifier, session-1 generalization, classifier/SOTA claims, post-hoc scalar selection, or an unconfounded LOCAL_BARYCENTER-versus-WHOLE comparison.

## 18. Single recommended next step

Return to the frozen Conditional-Geometry Anatomy preregistration.
