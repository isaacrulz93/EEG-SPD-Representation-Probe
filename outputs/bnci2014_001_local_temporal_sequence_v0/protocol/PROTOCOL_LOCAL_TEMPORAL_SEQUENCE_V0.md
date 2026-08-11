# Protocol: Local Temporal Sequence Correspondence V0

Status: pre-result frozen protocol, 2026-08-11 (Asia/Seoul).

## Scientific scope and lineage

This experiment asks whether the five experimentally fixed, consecutive local covariance states in a BNCI2014_001 motor-imagery trial have reproducible chronological correspondence across sessions. For two ordered cell-level mean covariance sequences, it tests whether matching temporal positions 1→1, …, 5→5 is better than deliberately incorrect temporal correspondences. It is not a classifier or adaptation experiment and does not study GPA consensus, quotient mean shape, pose, Procrustes alignment, or unordered five-point geometry.

The branch is `pilot/local-temporal-sequence-correspondence-v0`. Its immutable scientific lineage is:

- local AIRM metric Stage 1 final: `796f04e7970972175a660a521caff47c83e0295f`;
- local GPA V1 final: `122eacff868aa8f656ad6360716c1816f453979f`;
- GPA outer-convergence audit final and branch parent: `347d61f17793d636653b614ec2104baa61ac7a4b`.

All artifacts are confined to `outputs/bnci2014_001_local_temporal_sequence_v0/`. Earlier outputs and branches are immutable.

## Frozen input and reproduction gate

The frozen representation contains BNCI2014_001 subjects 1–9; sessions `0train` and `1test`; classes in the order `left_hand`, `right_hand`, `feet`, `tongue`; six runs per session; and exactly 72 trials per subject×session×class (12 per run). It uses the fixed 22-channel order, 250 Hz sampling, 8–32 Hz filtering, cue-relative 0–3.996 s epochs, 1,000 samples per trial, five consecutive non-overlapping 200-sample windows, pyRiemann OAS covariance, float64, no extra regularization or eigenvalue clipping, and AIRM geometry.

The original uncentered five-window covariance objects are loaded from `cache/bnci2014_001/covariances.npz` and `cache/bnci2014_001_trajectory_within_subject_v1/session1_window_covariances.npz`. Their file hashes and state-content hashes must match the immutable loader contract: session-0 state SHA-256 `c75044f48552f12ad088306b505b074e930f396fdcb544307fff394717e2ca86`, session-1 state SHA-256 `1afc8cd52d82310a05857d1ffa67859427c4c9aa1302897a140ebda64d0442f8`, and combined frozen AIRM-distance content SHA-256 `681d8a075eff1218e5e2b2d0e292631ead67badaccc00ec075ba428c9d5aed64`. Recomputed within-trial AIRM distances must agree with the saved matrices within an absolute maximum of `1e-12`; the existing exact hash is also recorded. Metadata counts and run labels must pass their frozen contracts. Failure terminates as `UNASSESSED_TECHNICAL_FAILURE_REPRODUCTION` before scientific output.

## No trial-local centering

All primary and secondary calculations use the original frozen covariance matrices. Trial-local Fréchet centering, whitening by a trial-specific mean, GPA, orthogonal action fitting, and point permutation during averaging are forbidden. The temporal indices have fixed experimental meaning. Generic cross-cell closeness is controlled inside each sequence pair by comparing the identity temporal matching with completely wrong matchings of the same two sequences.

## Ordered cell mean covariance sequences

For every subject `s`, session `q`, class `c`, and temporal position `i`, fit the AIRM Fréchet mean across the 72 original window covariance matrices:

\[
M[s,q,c,i]=\operatorname{FM}_{\mathrm{AIRM}}\{C[t,i]:t\in(s,q,c)\}.
\]

This produces exactly 72 ordered five-state mean covariance sequences. No point permutation or nuisance action is optimized. The implementation uses pyRiemann `mean_riemann` with float64, `init=None`, tolerance `1e-9`, and at most 100 iterations. Every input and output must be finite SPD, no solver warning is permitted, and the normalized Karcher residual must be at most `1e-7`. Required distances must be finite and nonnegative. A failure of these frozen numerical gates terminates as `UNASSESSED_NUMERICAL_FAILURE`; no tolerance rescue is allowed.

## Run-blocked split-half reliability diagnostic

For every cell and temporal position, independently fit means from Half A runs `{0,2,4}` and Half B runs `{1,3,5}`, exactly 36 trials per half. Save both ordered half sequences, the five same-position AIRM distances, and the complete 5×5 Half-A-to-Half-B cross-position AIRM distance matrix. Report position-wise and cell/session summaries. This diagnostic is non-gating; no post-hoc reliability threshold may be introduced.

## Cross-time distance matrices and matching costs

Index each session by the same canonical 36 subject×class cells, ordered by subject and then frozen class order. For session-0 cell `a=(s,c)` and session-1 cell `b=(t,k)`, save

\[
K_{ab}[i,j]=d_{\mathrm{AIRM}}(M[s,0,c,i],M[t,1,k,j])
\]

for all `36×36×5×5` entries. The raw identity discrepancy is

\[
D_{id}(a,b)=\sqrt{\frac15\sum_i K_{ab}[i,i]^2}.
\]

All 120 permutations of `(0,1,2,3,4)` are generated in Python lexicographic order. A permutation array `pi` means that row position `i` is matched to column position `pi[i]`. The identity is `(0,1,2,3,4)`. The wrong-order comparator is exactly the 44 permutations satisfying `pi[i] != i` for every position. For any permutation,

\[
D_\pi(a,b)=\sqrt{\frac15\sum_i K_{ab}[i,\pi(i)]^2}.
\]

The primary temporal correspondence gain and descriptive relative gain are

\[
G(a,b)=\operatorname{median}_{\pi\in\mathrm{Derangements}(S_5)}D_\pi(a,b)-D_{id}(a,b),
\qquad
G_{rel}(a,b)=G(a,b)/\operatorname{median}_{\pi\in\mathrm{Derangements}(S_5)}D_\pi(a,b).
\]

The median derangement denominator must be strictly positive. Identity rank is descriptive and is defined as one plus the number of the 120 costs strictly smaller than `D_id`; ties share the best applicable competition rank. The exact identity tie count is also saved. Identity rank is not the primary statistic. `D_id` is descriptive because generic proximity can affect it; `G` is primary because it contrasts correct and wrong temporal matchings for the same pair of mean sequences.

## Primary temporal correspondence test

For subject `s` and class `c`, let `A_sc=G((s,c),(s,c))`, `A_s=mean_c A_sc`, and `T_temporal=mean_s A_s`. The primary question is whether same-subject/same-class mean covariance sequences repeat their chronological correspondence across sessions.

The temporal-label null has exactly 1,999 draws. It uses NumPy `default_rng(SeedSequence([20260810,1201]))`. Independently for every session-1 subject×class cell in each draw, one of the 120 frozen permutations is sampled uniformly and the whole cell's five temporal labels are relabeled accordingly. Covariance matrices, subject identity, class identity, and all five states remain together. `T_temporal` is recomputed from the same `G` definition, including the 44 derangements relative to the relabeled indices. The one-sided plus-one p-value is `(1 + count(T_null >= T_observed))/2000`. Temporal correspondence passes only if `T_temporal > 0` and `p_temporal < 0.05`.

## Subject and class temporal-sequence specificity

For each anchor `(s,c)`, define:

- `a_sc = G((s,c),(s,c))`;
- `b_sc = mean_{k != c} G((s,c),(s,k))` (same subject, different class);
- `c_sc = mean_{t != s} G((s,c),(t,c))` (different subject, same class);
- `d_sc = mean_{t != s,k != c} G((s,c),(t,k))`.

Subject specificity is `S_sc=a_sc-c_sc`, `S_s=mean_c S_sc`, and `T_subject=mean_s S_s`. Its 1,999-draw subject-break null uses the exact prior frozen mapping stream `default_rng(SeedSequence([20260810,1102]))`: independently within each class, session-1 subject identities are permuted while complete five-state sequences move together and temporal indices and class are preserved. The one-sided plus-one p-value is `p_subject`; support requires `T_subject > 0` and `p_subject < 0.05`.

Class specificity is `C_sc=a_sc-b_sc`, `C_s=mean_c C_sc`, and `T_class=mean_s C_s`. Its 1,999-draw class-break null uses the exact prior frozen mapping stream `default_rng(SeedSequence([20260810,1101]))`: independently within each session-1 subject, the four complete class sequences are permuted, preserving subject and chronological indices. The one-sided plus-one p-value is `p_class`; support requires `T_class > 0` and `p_class < 0.05`.

## Secondary explicit subject×class interaction

The secondary contrast is `J_sc=a_sc-b_sc-c_sc+d_sc`, `J_s=mean_c J_sc`, and `T_J=mean_s J_s`. The same subject-break and class-break mappings are applied to session-1 cell columns of `G`, and all contrasts are recomputed. A subject×class interaction claim requires `T_J > 0`, `p_J_subjectbreak < 0.05`, and `p_J_classbreak < 0.05`. Subject and class specificity alone never establish interaction.

## Visualizations and descriptive comparisons

For the common mean-sequence chart, collect all `72×5=360` full-cell AIRM means and fit one global AIRM reference mean with the same numerical gates. Map every covariance to the whitened AIRM tangent coordinate `log(R^-1/2 M R^-1/2)`, apply Frobenius-isometric upper-triangle `svec`, and fit one deterministic global two-component PCA using scikit-learn's full SVD. The scientific calculations remain in AIRM space. Connect positions 1→2→3→4→5. Fixed, outcome-independent views are Subject 1 Left, Subject 1 Feet, Subject 2 Left, and Subject 2 Feet, each with both sessions, plus a nine-subject small-multiple figure containing all four classes and both sessions. No subject or cell is selected after viewing results.

Create 5×5 heatmaps for those same four fixed cross-session same-cell comparisons. Also average `K` elementwise for the four predeclared groups: same subject/same class, different subject/same class, same subject/different class, and different subject/different class. Smaller diagonals are descriptive only.

Save the full `D_id` matrix and explicitly report the four anchored examples: S1 Left versus S1 Left, S1 Left versus S2 Left, S1 Left versus S1 Feet, and S1 Left versus S2 Feet.

## Terminal decisions

Use exactly one primary terminal:

- `GO_REPRODUCIBLE_SUBJECT_CLASS_TEMPORAL_SEQUENCE` when temporal correspondence, subject specificity, and class specificity all pass their frozen positive-effect and p-value criteria;
- `GO_SHARED_TEMPORAL_SEQUENCE_WITH_PARTIAL_SPECIFICITY` when temporal correspondence passes and exactly one of subject/class specificity passes;
- `GO_SHARED_TEMPORAL_SEQUENCE_ONLY` when temporal correspondence passes and neither specificity passes;
- `STOP_NO_REPRODUCIBLE_TEMPORAL_SEQUENCE_V0` when the primary temporal-correspondence test fails;
- `UNASSESSED_NUMERICAL_FAILURE` when required AIRM means or distances fail the frozen numerical gates;
- `UNASSESSED_TECHNICAL_FAILURE_REPRODUCTION` for frozen-input reproduction failure;
- `UNASSESSED_TECHNICAL_FAILURE` for another blocking failure.

The secondary `J` is reported separately and cannot change the primary terminal.

## Output contract and post-result immutability

Save the 72 ordered full mean sequences, all split-half mean sequences, mean diagnostics, all `K` matrices, all matching-cost summaries, `D_id`, median derangement cost, `G`, `G_rel`, identity rank/ties, subject/class/interaction tables, all three null families, common PCA coordinates/reference/components, fixed and group heatmaps, figures, runtime, environment, config, provenance, hashes, and a manifest.

After the first real temporal statistic is observed, window boundaries, temporal indexing, AIRM mean settings, no-centering decision, permutation and derangement sets, gain definition, aggregation, null mappings/seeds/count, alpha, visualization selection, and terminal logic are immutable. There is no automatic rescue analysis.

If the temporal test passes, the allowed claim is: “The five chronological local covariance states show reproducible cross-session temporal correspondence.” If subject specificity passes, one may add that correspondence is stronger within the same subject than across subjects. If class specificity passes, one may add that correspondence is stronger for the same MI class than other classes within subject. The preferred nouns are *mean covariance sequence*, *chronological local-state correspondence*, *temporal correspondence*, and *temporal sequence specificity*.

It is forbidden to claim a continuous physiological trajectory, causal temporal dynamics, subject-specific pose, mean shape, GPA consensus, neural state sequence, source-space trajectory, or a subject×class interaction unless secondary `J` passes both frozen nulls. This experiment is not a rerun of the earlier order-shuffle classifier.
