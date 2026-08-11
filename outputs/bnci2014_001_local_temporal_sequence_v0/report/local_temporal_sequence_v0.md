# Local Temporal Sequence Correspondence V0

## Outcome

Terminal: `GO_REPRODUCIBLE_SUBJECT_CLASS_TEMPORAL_SEQUENCE`.

This experiment directly compares correct chronological local-state matching with completely wrong temporal matchings of the same two ordered mean covariance sequences. It is not an order-shuffle classifier, GPA, pose, or quotient-shape analysis.

## 1–4. Branch and immutable lineage

- Branch: `pilot/local-temporal-sequence-correspondence-v0`
- Local metric final: `796f04e7970972175a660a521caff47c83e0295f`
- GPA V1 final: `122eacff868aa8f656ad6360716c1816f453979f`
- GPA outer-convergence audit final: `347d61f17793d636653b614ec2104baa61ac7a4b`
- Protocol-freeze SHA: `70981aa89ddbadceca42f354c3c51d05bf6dbf0c`
- Final scientific-result SHA: `43e926073fab0ba76fd5baa881804538f0d7beee`

## 5. Frozen input reproduction

PASS. The immutable uncentered WINDOW5 bank contained 5184 trials. Session state and existing AIRM-distance hashes matched. The maximum absolute frozen AIRM-distance reproduction difference was 0.000e+00 at the frozen `1e-12` tolerance. No trial-local centering was applied.

## 6. Per-window AIRM mean numerical status

PASS. All 360 full-cell means, 720 split-half means, and the one common visualization reference were finite SPD, emitted no warnings, and passed the frozen normalized Karcher residual gate `<=1e-7`. The maximum normalized residual was 8.300e-11.

## 7. Split-half temporal reliability

This was a non-gating diagnostic with no post-hoc threshold. Across all 72 cells, same-position Half-A/Half-B AIRM distances had mean 0.7784915776, median 0.7486105354, Q25 0.6764371882, and Q75 0.8707846704. Position-wise and complete 5×5 cell matrices are saved in the tables and arrays.

## 8–9. Primary temporal correspondence

- T_temporal: 0.2072760401
- p_temporal: 0.000500 (1,999 temporal-label draws, one-sided plus-one)

All nine A_s values:

- Subject 1: 0.2314851404
- Subject 2: 0.1651408554
- Subject 3: 0.3626261006
- Subject 4: 0.2430582275
- Subject 5: 0.0380165017
- Subject 6: 0.2311403754
- Subject 7: 0.1128454078
- Subject 8: 0.1584942483
- Subject 9: 0.3226775037

## 10. Identity-rank summary

For the 36 same-subject/same-class comparisons, identity rank among all 120 permutations had mean 3.056, median 2.000, range 1–19, and rank 1 in 17/36 cells. Category-level rank summaries are saved machine-readably. Identity rank is descriptive, not the primary statistic.

## 11–12. Subject specificity

- T_subject: 0.1766280571
- p_subject: 0.000500 (1,999 subject-break draws)

All nine S_s values:

- Subject 1: 0.1960207307
- Subject 2: 0.1253087749
- Subject 3: 0.3139929719
- Subject 4: 0.2162127567
- Subject 5: 0.0317833980
- Subject 6: 0.1925274526
- Subject 7: 0.0898058162
- Subject 8: 0.1339851205
- Subject 9: 0.2900154927

## 13–14. Class specificity

- T_class: 0.1188494432
- p_class: 0.000500 (1,999 class-break draws)

All nine C_s values:

- Subject 1: 0.1975255141
- Subject 2: 0.0222672760
- Subject 3: 0.1907681439
- Subject 4: 0.0800959469
- Subject 5: 0.0174821727
- Subject 6: 0.0625107197
- Subject 7: 0.0456350129
- Subject 8: 0.1190022344
- Subject 9: 0.3343579684

## 15–17. Secondary explicit subject×class interaction

- T_J: 0.1049833990
- p_J_subjectbreak: 0.000500
- p_J_classbreak: 0.000500
- Both-null interaction criterion passed: TRUE

All nine J_s values:

- Subject 1: 0.1702285997
- Subject 2: 0.0151773197
- Subject 3: 0.1723013360
- Subject 4: 0.0776582382
- Subject 5: 0.0149982849
- Subject 6: 0.0514779990
- Subject 7: 0.0342027352
- Subject 8: 0.0991066558
- Subject 9: 0.3096994222

## 18. Predeclared raw D_id comparisons

- S1 Left vs S1 Left: D_id=0.7352528615; median derangement=0.8861860742; G=0.1509332127; identity rank=1
- S1 Left vs S2 Left: D_id=2.8426401240; median derangement=2.8170212625; G=-0.0256188615; identity rank=107
- S1 Left vs S1 Feet: D_id=1.8632683639; median derangement=1.8399288759; G=-0.0233394881; identity rank=100
- S1 Left vs S2 Feet: D_id=2.6225400098; median derangement=2.5777134617; G=-0.0448265482; identity rank=117

`D_id` is descriptive because generic proximity affects it. `G`, the identity advantage over the 44 complete derangements, is the primary structural quantity.

## 19. Terminal decision

`GO_REPRODUCIBLE_SUBJECT_CLASS_TEMPORAL_SEQUENCE`

The secondary J result is separate and did not change this primary terminal.

## 20. Runtime

24.44 seconds total scientific execution.

## 21. Tests

- Focused temporal suite: 14/14 passed before protocol freeze and after the scientific run
- Full repository suite: 244/244 passed before protocol freeze and after the scientific run

## 22. Git status

`clean after the final report/provenance commit`

## 23. Post-result immutability

Confirmed: no scientific definition changed after the first real temporal statistic was observed. No rescue analysis or rerun with altered settings was performed.

## Claim boundary

The analysis concerns mean covariance sequences, chronological local-state correspondence, temporal correspondence, and temporal sequence specificity. It does not identify a continuous physiological trajectory, causal temporal dynamics, subject-specific pose, mean shape, GPA consensus, a neural state sequence, or a source-space trajectory. A subject×class interaction is claimed only if the separately reported secondary J passes both correspondence-breaking nulls.
