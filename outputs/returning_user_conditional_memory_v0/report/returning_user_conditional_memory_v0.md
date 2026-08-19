# Returning-User Low-Rank Conditional Memory Downstream Pilot V0

**Terminal: `STOP_NO_CROSS_SESSION_DOWNSTREAM_UTILITY`**

## Frozen deployment scenario

Stieger2021 session 2 labeled enrollment → session 3 zero-label deployment, task 3 and four literal classes, is the voting primary. OpenBMI official source session 1 → 2 is the external binary replication. The evaluation is offline batch zero-recalibration after one labeled enrollment session.

## Cache and leakage gates

All 278 tracked PR #16--#19 artifacts retained canonical snapshot `7de0093b94ed622f87326e5b5814f8f8833674e26bfeebb848fd0422e1305b3d`. Stieger 124 tangent records and OpenBMI `(54,2,100,210)` tangents passed frozen hashes. No raw data were downloaded or rebuilt. Outer target deployment labels were sealed until predictions were saved.

## Stieger chronological primary

LRCM subject-mean balanced accuracy was 0.331556, macro-F1 0.309233, NLL 1.630403, and ECE10 0.164078. Selected ranks were `[1, 3, 3, 3, 3, 8]` (median 3.0).

Primary comparisons (positive values favor LRCM):

- vs `IDENTITY_RESIDUAL_CARRY`: mean ΔBA -0.019793, 95% CI [-0.037910, -0.002384], raw p=0.9825, Holm p=1, win rate 0.435.
- vs `PAST_PROTOTYPE_DIRECT`: mean ΔBA -0.014781, 95% CI [-0.032661, 0.002307], raw p=0.9445, Holm p=1, win rate 0.452.
- vs `POPULATION_ONLY`: mean ΔBA -0.005468, 95% CI [-0.016582, 0.005159], raw p=0.837, Holm p=1, win rate 0.484.

Memory null p-values: subject-memory permutation 0.829; enrollment-class permutation 0.035; random rank-matched subspace 1; unpaired source sessions 0.8335.

Low-rank audit: LRCM−full-ridge mean ΔBA -0.008213, CI [-0.019026, 0.002982]; LRCM−enrollment-PCA mean ΔBA -0.001957, CI [-0.014257, 0.009881], p=0.648.

## OpenBMI external replication

Chronological LRCM subject-mean balanced accuracy was 0.691481. Against identity residual carry, mean ΔBA was 0.043704, CI [0.023699, 0.064444], p=0.0005. Selected ranks were `[3, 5, 3, 13, 5, 2]`.

## Calibration-equivalent K-shot curve

- stieger: LRCM BA 0.331556; smallest direct current-session K reaching it = 8.0; bracket [4, 8.0].
- openbmi: LRCM BA 0.691481; no tested direct current-session budget through K=16 per class reached it; lower bracket K=16 with no upper bracket in the frozen grid.

## Validation

Focused tests: 33 passed. Full repository suite: 472 passed, 1 skipped, with two pre-existing Stieger interpolation warnings. `git diff --check`, all 278 frozen parent-artifact hashes, both trial-cache contracts, and every result-manifest hash passed final validation.

## Interpretation boundary

This pilot does not establish new-user zero calibration, online causal adaptation, full conditional recovery, unlabeled semantic identification, physiology, source anatomy, universal subject coordinates, pseudo-label validity, TTA, ASD generalization, or clinical efficacy.

## Exact next question or statement

Stable low-dimensional subject×class interaction does not provide incremental cross-session decoding utility beyond direct enrollment-session residual transfer under the tested returning-user protocol.
