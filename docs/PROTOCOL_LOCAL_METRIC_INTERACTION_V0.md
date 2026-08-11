# Protocol: Local AIRM Metric Interaction V0

Status: pre-data frozen protocol, 2026-08-11 (Asia/Seoul).

## Scientific question and scope

This structural anatomy experiment asks whether the unordered five-point internal AIRM distance geometry of one MI EEG trial contains a cross-session-reproducible subject-specific class interaction. It tests whether the exact same subject–class pairing is more reproducible than expected from separable subject and class correspondence effects.

This is not a domain-adaptation, classifier-selection, pose, trajectory-order, Procrustes, optimal-transport, or neural-network study. Absolute SPD position, common (GL(22)) congruence, common orthogonal pose, and window identity/order are deliberately discarded. A future separately preregistered Procrustes-pose anatomy study may test whether full locally centered SPD configurations contain population-shared or subject-specific pose information beyond the distance geometry studied here.

## Frozen provenance and EEG representation

The branch is `pilot/local-metric-interaction-v0`. The scientific base is commit `355fe0b55b1ef692f7b4ddd16d19b7ccc30e72e1`, the finalized trajectory-within-subject audit. No infrastructure newer than that scientific commit is copied. The existing immutable trajectory V1 loader and cache are reused without changing the scientific representation; the new implementation only consumes its saved AIRM five-by-five distance matrices and exact PATH_D10 upper triangles.

The frozen data are BNCI2014_001 subjects 1–9, sessions `0train` and `1test`, runs 0–5, and classes in the order `left_hand`, `right_hand`, `feet`, `tongue`. The 22 ordered EEG channels, 8–32 Hz filtering, cue-relative 0–3.996 s interval, 250 Hz sampling, 1,000 samples, OAS covariance, float64 covariance geometry, no resampling, no baseline, no extra diagonal loading, no eigenvalue clipping, and deterministic symmetrization are unchanged. Each trial is five non-overlapping 200-sample windows, each represented by one SPD(22) covariance.

The input gate requires 5,184 trials and the pinned combined-cache SHA-256 `5accf2f3e6becce187b18d30a1ea1741ae0d0faafc15a1f8ac16632a5a71628d`. AIRM distance-matrix and PATH_D10 content hashes must be `681d8a075eff1218e5e2b2d0e292631ead67badaccc00ec075ba428c9d5aed64` and `8179f7654d6a1c89065aca12e99029e6a7476d332fdf62767e83c0f0966008f9`. The upper triangle reconstructed from every saved distance matrix must equal the saved PATH_D10 array exactly (maximum absolute difference zero). Any failure terminates as `UNASSESSED_TECHNICAL_FAILURE_REPRODUCTION`.

## Exact unordered metric geometry

For one trial, (D[i,j]=d_{\mathrm{AIRM}}(C_i,C_j)). The ten upper-triangle edges are fixed as `(12,13,14,15,23,24,25,34,35,45)`. Every vertex permutation in (S_5) induces one edge-coordinate permutation; all and only those 120 actions are enumerated. Arbitrary (S_{10}) edge permutations, approximations, greedy matching, optimal transport, learned correspondence, and regularization are forbidden.

The primary trial-pair distance is

\[
\delta_{\mathrm{raw}}(A,B)=
\min_{\pi\in S_5}\sqrt{\frac{1}{10}\lVert x_A-P_\pi x_B\rVert_2^2}.
\]

Its scientific name is *unlabeled local AIRM metric-configuration distance* or *unlabeled internal AIRM distance-geometry distance*. It is a metric on distance-matrix (S_5) orbit space and only a pseudometric when pulled back to original SPD configurations. It is not a complete SPD shape, full intrinsic shape, or pose distance.

Define edge-RMS metric size (s(B)=\lVert x_B\rVert_2/\sqrt{10}), normalized edges (\hat x_B=x_B/s(B)),

\[
\delta_{\mathrm{size}}(A,B)=|s(A)-s(B)|,
\qquad
\delta_{\mathrm{norm}}(A,B)=
\min_{\pi\in S_5}\sqrt{\frac{1}{10}\lVert\hat x_A-P_\pi\hat x_B\rVert_2^2}.
\]

No log-size is primary. If any (s(B)=0), the trial is marked `DEGENERATE_METRIC_CONFIGURATION`; no epsilon is inserted and inference stops for review. The exact identity

\[
\delta_{\mathrm{raw}}^2=(s_A-s_B)^2+s_As_B\delta_{\mathrm{norm}}^2
\]

is a trial-pair squared-distance identity only. No additive or causal decomposition of the group interaction (J) is allowed.

The vectorized matcher precomputes all 120 induced edge permutations, uses float64, and chunks trial banks at 128. It must agree with the slow exhaustive reference within absolute and relative (10^{-12}).

## Cross-session cell summaries

Each session has 36 subject×class cells with 72 trials. For (a=(s,c)), (b=(t,k)),

\[
M_{01}[a,b]=\operatorname{median}_{i\in 0:a,\,j\in 1:b}\delta(i,j).
\]

Trial pairs are not inferential samples; this median is only a robust cell summary. The primary structural matrix is session-role symmetrized:

\[
M[a,b]=\tfrac12\{M_{01}[a,b]+M_{01}[b,a]\}.
\]

All 36×36 entries of (M_{01}) and (M) are retained. The same construction is applied independently to raw, size, and normalized distances.

## Primary and supporting contrasts

For anchor cell ((s,c)):

- (a_{sc}=M[(s,c),(s,c)]);
- (b_{sc}) is the mean over the same subject and the other three classes;
- (c_{sc}) is the mean over the same class and the other eight subjects;
- (d_{sc}) is the mean over the other eight subjects and other three classes.

The primary interaction is

\[
J_{sc}=b_{sc}+c_{sc}-a_{sc}-d_{sc}
=(b_{sc}-a_{sc})-(d_{sc}-c_{sc}).
\]

Positive (J_{sc}) means that the same-class advantage is stronger within the same subject than across other subjects. Subject summaries and the group statistic are

\[
J_s=\tfrac14\sum_cJ_{sc},\qquad T_J=\tfrac19\sum_sJ_s.
\]

The subject is the scientific group unit. Edges, trials, trial pairs, cells, and subject pairs are not independent group samples.

Supporting anatomy is (S_{sc}=c_{sc}-a_{sc}), (S_s=\operatorname{mean}_cS_{sc}), (T_S=\operatorname{mean}_sS_s), and (C_{sc}=b_{sc}-a_{sc}), (C_s=\operatorname{mean}_cC_{sc}), (T_C=\operatorname{mean}_sC_s). Positive supporting effects alone do not establish interaction.

## Frozen correspondence-breaking nulls

There are exactly 1,999 draws with master seed 20260810. The class-break stream tag is 1101. In every draw, the four complete session-1 class cells are independently permuted within each subject; subject identity and every trial inside a cell are preserved. The subject-break stream tag is 1102. In every draw, the nine complete session-1 subject cells are independently permuted within each class; class identity and every trial inside a cell are preserved. A mapping is applied to (M_{01}) columns before session-role symmetrization and recomputation of all contrasts.

For either null, the one-sided greater-or-equal plus-one p-value is

\[
p=(1+\#\{T_{\mathrm{null}}\ge T_{\mathrm{obs}}\})/2000.
\]

The primary conservative p-value is the maximum of the two p-values. Raw interaction support requires all three conditions: (T_J>0), (p_{J,\mathrm{classbreak}}<0.05), and (p_{J,\mathrm{subjectbreak}}<0.05). Equality to 0.05 is not a pass. The class-break mappings also provide the supporting (T_C) null, and subject-break mappings the supporting (T_S) null. Identical precomputed mappings are reused for raw, size, and normalized analyses.

Synthetic cell-distance fixtures A–D (no effects, subject only, class only, and additive subject+class without interaction) must produce (J\approx0) and not be declared significant. A planted interaction fixture must produce (J>0). These checks are pre-data hard gates.

## Frozen stage sequence and decisions

Stage P1 computes the raw matrix, all 36 (J_{sc}), all nine (J_s), (T_J), both primary nulls, and supporting (S/C) anatomy. Once P1 is calculated, every scientific definition in this document is immutable.

The raw terminal is exactly:

- `GO_STABLE_SUBJECT_CLASS_LOCAL_METRIC_INTERACTION` if (T_J>0) and both primary p-values are strictly below 0.05;
- `STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0` otherwise;
- `UNASSESSED_TECHNICAL_FAILURE` for a scientific-blocking numerical or implementation failure.

Stage P2 repeats the entire matrix, contrast, and same-null calculation with δsize. Stage P3 repeats it with δnorm. Neither can change the raw terminal. The nonterminal mechanism tag is:

- `BOTH_SIZE_AND_RELATIVE_PATTERN_SUPPORTED` if both controls have positive (T_J) and pass both nulls;
- `RELATIVE_PATTERN_SUPPORTED` if only normalized passes;
- `METRIC_SIZE_SUPPORTED_RELATIVE_PATTERN_NOT_ESTABLISHED` if only size passes;
- `MECHANISM_UNRESOLVED` otherwise.

No causal mediation or (J_{raw}=J_{size}+J_{norm}) claim is allowed.

## Non-gating reliability diagnostic

For every subject/session/class, report 72-trial count, median within-cell raw distance, 25th and 75th percentiles, IQR, and the minimum median-distance medoid objective. The medoid excludes self-distance; ties use the lowest global sample index. This diagnostic is descriptive and cannot change the terminal decision. No reliability threshold may be invented after observing BNCI.

## Secondary cross-session decoder

Stage D uses δraw only. For each subject, session, and class, choose the class trial minimizing median distance to the other same-class training trials. Ties use lowest global sample index. Each test trial is assigned to its nearest one of four medoids; class ties use frozen class order. Both directions (`0train` to `1test`, and reverse) are run. Their balanced accuracies are averaged within subject; all nine subject values plus group mean and median are reported with chance 0.25.

The descriptive decoder null uses 1,999 deterministic draws, master seed 20260810, stream tag hexadecimal `0x4C4D494445434F44`. In each replicate, training labels are independently shuffled within each subject×session×run, preserving exactly 12 labels per class in every run. The classifier is fully recomputed. The null statistic is the median of the nine subject-average balanced accuracies, with a one-sided greater-or-equal plus-one p-value. Decoder results cannot change the primary structural terminal. No classifier tuning or representation selection is permitted.

## Output, immutability, and claims

All new artifacts are confined to `outputs/bnci2014_001_local_metric_interaction_v0/`. Each output family carries protocol SHA, scientific source SHA, config SHA, and implementation-source hash through its provenance/manifest. Existing trajectory, interaction, common-action, and representation-probe outputs are immutable.

If raw (J) passes, the maximum permitted claim is that the unlabeled five-point AIRM distance geometry exhibits a cross-session-reproducible subject-specific class interaction, or that exact subject–class pairing is more reproducible than expected from separate subject and class correspondence effects. If normalized (J) also passes, one may say the interaction persists after removing overall edge-RMS metric size and therefore includes relative internal edge-length pattern information.

It is forbidden to claim identification of the full SPD configuration, a complete intrinsic shape, pose, temporal trajectory, causal eigenvectors, physiology, source-space organization, a recovered global subject transform, or complete determination of SPD configurations by pairwise distances. This branch develops no adaptation method.
