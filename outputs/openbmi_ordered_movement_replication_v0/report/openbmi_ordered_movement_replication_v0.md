# OpenBMI Ordered Movement External Replication V0

**PRIMARY STATUS: UNASSESSED.** The active runtime used MNE 1.10.1, whereas the frozen donor preprocessing contract requires MNE 1.12.1 for exact sampling handling. The provisional endpoint below is retained only for audit and is not interpreted as a negative replication. No rerun was performed after endpoint access.

## Immutable lineage

- Branch: `replication/openbmi-ordered-movement-v0`
- BNCI component parent: `edc1d344cb0657f2f2d87b2992049bceec4705d2`
- BNCI component protocol freeze: `95c330de9596fa4c4eb4ee377d5af8d99896f4c3`
- BNCI component scientific result: `0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`
- BNCI terminal: `BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS`
- OpenBMI donor branch/head: `pilot/subject-class-interaction-v0` / `272d775678644aad062df424a70586d4b42de652`
- Protocol freeze SHA: `19d103666624d9f961839f3099c70e6c2c8b6b0c`
- Scientific result SHA: `67a78bb1e461c48e33180c65d8da85cc9285d2cf`

## Contract and numerical status

The donor Lee2019-MI source/count contract was reproduced for 54 subjects, both sessions, left/right hand, 50 trials per cell, and the frozen ordered 20-channel montage. The intended computation applied continuous 8–30 Hz filtering, 100 Hz resampling, the half-open 1.0–3.5 s epoch, no baseline, and OAS covariance estimation. The 250-sample epoch has an exact, prespecified partition into five non-overlapping 50-sample (0.5 s) bins. All 108 raw source hashes match the donor manifest.

The raw-source, count, channel, epoch-length, and five-bin gates passed, but the exact runtime-version gate failed: installed MNE was 1.10.1 instead of the donor's required 1.12.1. Because resampling is part of the frozen preprocessing contract, this overrides the otherwise valid downstream numerical calculations and prohibits scientific interpretation.

- AIRM mean and anti-development gates: PASS
- Full/split quotient determinant-sector certification: PASS
- Minimum raw c_ang: 0.28878006923219091
- Minimum raw c_ori: 0.51617579980187056
- Maximum decomposition reconstruction error: 4.4408920985006262e-16
- Maximum selected full-fit gradient norm: 9.9980311902336491e-07

## Quarantined provisional endpoint (not a replication result)

- T_J_ang: -0.01582673234
- Subject-break p: 0.661
- Class-break p: 0.663

All 54 subject J values:

- S01: -0.1617535898
- S02: 0.03256548622
- S03: 0.02000788351
- S04: -0.04710241794
- S05: 0.1117023562
- S06: 0.02334797527
- S07: 0.008782569671
- S08: -0.4049729646
- S09: 0.0710014
- S10: 0.07085430901
- S11: 0.01672258725
- S12: -0.03928539965
- S13: -0.03291874729
- S14: -0.07911481746
- S15: 0.00618231114
- S16: -0.4801639649
- S17: 0.04800476746
- S18: 0.1609753454
- S19: 0.5234197365
- S20: -0.5356427443
- S21: -0.1582078844
- S22: 0.4777994801
- S23: 0.3530770892
- S24: 0.0254814246
- S25: -0.2265020484
- S26: -0.05555367451
- S27: -0.04720047609
- S28: -0.02131605594
- S29: 0.2212053939
- S30: 0.2057553657
- S31: 0.02122982688
- S32: -0.02735177188
- S33: -0.3496049954
- S34: -0.1665968735
- S35: -0.01291625316
- S36: -0.06691226793
- S37: 0.02421162635
- S38: -0.2015016653
- S39: -0.0404446368
- S40: 0.3297835394
- S41: -0.08532013394
- S42: 0.03972117811
- S43: 0.2603321767
- S44: -0.115594352
- S45: 0.0915847748
- S46: 0.1477544931
- S47: -0.0405863467
- S48: -0.06233458433
- S49: -0.09090513217
- S50: 0.1140673432
- S51: -0.6917055641
- S52: 0.154483398
- S53: -0.4798377827
- S54: 0.3066497613

## Prespecified secondary results

- sensor: T_subject=0.1102173947 (p=0.001), T_class=-0.02202242214 (p=0.71), T_J=-0.01845998926 (p_subject=0.6745, p_class=0.6705)
- full: T_subject=0.09596224787 (p=0.001), T_class=-0.02505344519 (p=0.7525), T_J=-0.02264710437 (p_subject=0.728, p_class=0.722)
- len: T_subject=0.0005892988689 (p=0.4595), T_class=-0.004558221341 (p=0.803), T_J=-0.006820372027 (p_subject=0.874, p_class=0.8825)
- ang: T_subject=0.095372949 (p=0.001), T_class=-0.02049522385 (p=0.712), T_J=-0.01582673234 (p_subject=0.661, p_class=0.663)
- ori: T_subject=0.01425514679 (p=0.063), T_class=0.003031023047 (p=0.417), T_J=0.004187115102 (p_subject=0.377, p_class=0.375)

Raw five-state temporal correspondence versus all 44 complete derangements:

- Mean identity advantage: 0.01906937863
- Fraction identity better than median derangement: 0.712963
- Median identity rank among all 120 permutations: 29

Split-half angular stability (odd/even acquisition positions; no half-level p-value):

- Half A T_J_ang: 0.07647586722
  - S01: 0.06307974613
  - S02: -0.1057408293
  - S03: 0.3083233781
  - S04: -0.1560825289
  - S05: -0.07762824197
  - S06: -0.4802153307
  - S07: -0.6489415612
  - S08: -0.5440651963
  - S09: 0.0398348996
  - S10: -0.1186818369
  - S11: -0.1175044154
  - S12: -0.3129383348
  - S13: 0.6838311997
  - S14: 0.1170359989
  - S15: 0.4888467601
  - S16: -0.1762237538
  - S17: 0.1552829055
  - S18: 0.4502887729
  - S19: 2.114165087
  - S20: -0.1814618146
  - S21: 0.2921399496
  - S22: 1.413514694
  - S23: 0.8520632944
  - S24: -0.09056720412
  - S25: -0.09469878555
  - S26: 0.2658758736
  - S27: -0.1104939504
  - S28: -0.1343766392
  - S29: -0.1143827209
  - S30: 0.2563924382
  - S31: -0.895236242
  - S32: 0.1339376247
  - S33: 0.8811663796
  - S34: -0.07413546646
  - S35: 0.0784765193
  - S36: 0.2451353892
  - S37: 0.2824004003
  - S38: 0.6130038638
  - S39: 0.2330006817
  - S40: 0.115429762
  - S41: 0.09796574941
  - S42: 0.08352508564
  - S43: 0.274240683
  - S44: -1.498601579
  - S45: 0.722242009
  - S46: 0.0667866304
  - S47: 0.8688305084
  - S48: 0.05438449848
  - S49: -1.832195529
  - S50: 0.4234372194
  - S51: -0.3717045033
  - S52: 0.1927407725
  - S53: -0.7789635971
  - S54: 0.1771581172
- Half B T_J_ang: -0.04424524153
  - S01: -0.3486160129
  - S02: 0.1060066181
  - S03: 0.1385213385
  - S04: 0.01804020414
  - S05: -0.1064570531
  - S06: -0.6030858983
  - S07: 0.3350081882
  - S08: -0.3901085564
  - S09: 0.2894905975
  - S10: 0.08837647266
  - S11: -0.05313780417
  - S12: -0.3219578121
  - S13: -0.3039619297
  - S14: -0.32768573
  - S15: -0.4075184644
  - S16: -1.435876374
  - S17: 0.08186408733
  - S18: -0.09865854498
  - S19: 0.06031518484
  - S20: -0.3620813856
  - S21: -0.165171325
  - S22: 0.2598806483
  - S23: 0.06465226348
  - S24: 0.05772914137
  - S25: -0.05692073378
  - S26: -0.4779779
  - S27: -0.2813427272
  - S28: 0.3836589855
  - S29: 0.6159582357
  - S30: 0.2819621699
  - S31: 0.6137857183
  - S32: -0.06026348862
  - S33: -0.7886880949
  - S34: -0.4116493437
  - S35: -0.0004984405428
  - S36: -0.08932489301
  - S37: -0.2875136649
  - S38: -0.5857663928
  - S39: 0.3022952415
  - S40: 0.03326730136
  - S41: -0.4049326253
  - S42: 0.07942914352
  - S43: 1.098082365
  - S44: -0.1731824836
  - S45: 0.03689306249
  - S46: 0.9904426581
  - S47: -0.31659425
  - S48: 0.9794589483
  - S49: 0.02079264389
  - S50: 0.1188483295
  - S51: -0.8787240047
  - S52: 0.1586515856
  - S53: -0.4648613992
  - S54: 0.5999031576

## Terminal

`UNASSESSED_OPENBMI_DATA_CONTRACT_FAILURE`

No replication or non-replication claim is made. Had the runtime contract passed, this would have been a two-class external structural replication, not a reproduction of BNCI's four-class combinatorial structure or physical bin duration. The quarantined object is the discrete anti-development of the ordered window-wise mean covariance movement at a fixed 5 × 0.5 s OpenBMI discretization. No physiological, continuous-dynamic, source-space, motor-strategy, or stable-pose claim is made.

## Runtime, tests, and immutability

- Scientific execution runtime: 3750.778 seconds
- Data reproduction/preparation runtime: 1191.377 seconds
- Total measured preparation plus scientific runtime: 4942.155 seconds
- Focused and full test results: recorded in `provenance/test_results.json`
- No scientific setting was changed and no rescue rerun was performed after first result access.
- Git status at final handoff: recorded after result finalization.
