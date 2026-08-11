# Local AIRM Metric Interaction V0

The earlier trajectory audit asked whether distance-only local geometry contains class information within a subject and across sessions. This experiment asks the narrower structural question: is the exact same subject–class pairing more reproducible across sessions than expected from separate subject and class correspondence effects?

## 1. Branch

`pilot/local-metric-interaction-v0`

## 2. Scientific base SHA

`355fe0b55b1ef692f7b4ddd16d19b7ccc30e72e1`

## 3. Protocol-freeze SHA

`12a4d3ebd29a3a95e7fbc1bbc3a6d70b770532dc`

## 4. Final scientific-result SHA

`PENDING_FINAL_SCIENTIFIC_RESULT_COMMIT`

## 5. Frozen representation reproduction

PASS. The immutable two-session trajectory cache contained 5184 trials. Frozen cache, distance-matrix, and PATH_D10 hashes matched; the maximum absolute difference between saved PATH_D10 and the fixed upper triangle reconstructed from the saved distance matrices was 0.0e+00.

## 6. Mathematical audit

PASS. Exact enumeration of all 120 vertex-induced S5 edge permutations was symmetric, permutation-invariant, and satisfied the finite-group orbit-metric triangle inequality. The slow exhaustive and vectorized implementations agreed at the frozen 1e-12 tolerance. Common orthogonal and well-conditioned nonorthogonal congruence tests passed. The trial-pair raw/size/normalized squared-distance identity passed, and the inversion fixture demonstrated why this is only a pseudometric on original SPD configurations.

## 7. Degeneracy audit

PASS: 0 trials had zero edge-RMS metric size.

## 8–11. Primary RAW interaction

- T_J: 0.0070155240
- p_J_classbreak: 0.003500
- p_J_subjectbreak: 0.199500
- conservative p_J=max: 0.199500

## 12. All nine subject-level J_s values

- Subject 1: 0.02487021
- Subject 2: -0.00263181
- Subject 3: 0.00441348
- Subject 4: -0.00130769
- Subject 5: -0.00066687
- Subject 6: 0.00556584
- Subject 7: -0.00242959
- Subject 8: 0.01239025
- Subject 9: 0.02293588

## 13. All 36 subject×class J_sc values

- Subject 1, left_hand: 0.03310272
- Subject 1, right_hand: 0.00394314
- Subject 1, feet: 0.04873449
- Subject 1, tongue: 0.01370050
- Subject 2, left_hand: -0.02695064
- Subject 2, right_hand: 0.03729458
- Subject 2, feet: -0.01685572
- Subject 2, tongue: -0.00401547
- Subject 3, left_hand: 0.05274932
- Subject 3, right_hand: 0.00491292
- Subject 3, feet: -0.02600759
- Subject 3, tongue: -0.01400072
- Subject 4, left_hand: 0.01817306
- Subject 4, right_hand: -0.00022279
- Subject 4, feet: -0.03448334
- Subject 4, tongue: 0.01130230
- Subject 5, left_hand: -0.02696318
- Subject 5, right_hand: -0.04955609
- Subject 5, feet: 0.03223100
- Subject 5, tongue: 0.04162080
- Subject 6, left_hand: 0.11632652
- Subject 6, right_hand: 0.05699911
- Subject 6, feet: -0.04442737
- Subject 6, tongue: -0.10663489
- Subject 7, left_hand: 0.03535263
- Subject 7, right_hand: 0.00814920
- Subject 7, feet: -0.01890311
- Subject 7, tongue: -0.03431707
- Subject 8, left_hand: 0.05562367
- Subject 8, right_hand: 0.02060398
- Subject 8, feet: 0.00477820
- Subject 8, tongue: -0.03144484
- Subject 9, left_hand: 0.02591021
- Subject 9, right_hand: 0.08257264
- Subject 9, feet: -0.01981903
- Subject 9, tongue: 0.00307970

## 14. Supporting subject specificity

T_S=0.2611978434; subject-break p=0.000500. This is supporting anatomy and does not establish interaction by itself.

## 15. Supporting class specificity

T_C=0.0102179039; class-break p=0.000500. This is supporting anatomy and does not establish interaction by itself.

## 16. Edge-RMS metric-size control

T_J_size=0.0082690033; class-break p=0.003500; subject-break p=0.195000; conservative p=0.195000.

## 17. Size-normalized relative-pattern control

T_J_norm=0.0003612545; class-break p=0.041000; subject-break p=0.057500; conservative p=0.057500.

## 18. Mechanism tag

`MECHANISM_UNRESOLVED`

This is a nonterminal mechanism control. It is not a causal mediation result, and raw J is not the sum of size and normalized J.

## 19. Secondary cross-session class decoding

- Mean subject BA: 0.275463
- Median subject BA: 0.272569
- Null median: 0.250000
- One-sided plus-one p: 0.007500
- Chance reference: 0.25

- Subject 1: 0.288194 (0→1 0.302083; 1→0 0.274306)
- Subject 2: 0.255208 (0→1 0.260417; 1→0 0.250000)
- Subject 3: 0.253472 (0→1 0.284722; 1→0 0.222222)
- Subject 4: 0.236111 (0→1 0.229167; 1→0 0.243056)
- Subject 5: 0.234375 (0→1 0.270833; 1→0 0.197917)
- Subject 6: 0.272569 (0→1 0.288194; 1→0 0.256944)
- Subject 7: 0.279514 (0→1 0.267361; 1→0 0.291667)
- Subject 8: 0.321181 (0→1 0.340278; 1→0 0.302083)
- Subject 9: 0.338542 (0→1 0.357639; 1→0 0.319444)

This nearest-class-medoid result is secondary and did not affect the primary terminal.

## 20. Within-cell reliability diagnostics

The diagnostic was non-gating and used no post-hoc threshold. Median summaries over the 36 subject×class cells in each session were:

```
         median_within_cell_delta_raw  iqr_within_cell_delta_raw  medoid_objective
session
0train                       0.457777                   0.312753          0.338292
1test                        0.481608                   0.322237          0.342869
```

All 72 cell-level diagnostics, including IQR and medoid objective, are in the machine-readable table. Any broad within-cell variability must be considered when interpreting a negative result.

## 21. Terminal scientific decision

`STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0`

The terminal is based only on raw T_J and the two preregistered correspondence-breaking nulls. The experiment does not identify a full SPD configuration, pose, temporal trajectory, physiology, source-space organization, or a global subject transformation.

## 22. Total runtime

18.34 seconds.

## 23. Relevant tests

Pre-freeze local-metric mathematical, synthetic, null, and reproduction tests passed.

## 24. Full repository tests

Full pre-freeze repository suite passed; final verification pending result artifact commit.

## 25. Git status

`scientific outputs pending commit`

## 26. Post-result immutability

Confirmed: no scientific definition changed after the first real-data Stage-P1 statistic was observed.
