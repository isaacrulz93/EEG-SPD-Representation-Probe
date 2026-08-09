# Conditional-Geometry Anatomy v1 — Frozen Protocol

Protocol version: **1.0**  
Freeze date: **2026-08-09 (Asia/Seoul)**  
Base commit: **f76785d4c06ddc3cf2f1f7a6310d4fb153a19ba9**  
Branch: **pilot/conditional-geometry-anatomy-v1**  
Primary geometry: **Affine-Invariant Riemannian Metric (AIRM)**  
Secondary geometry: **Log-Euclidean (LE), robustness only**  
Master seed: **20260809**

This document is frozen before any Conditional-Geometry v1 scientific object is
computed. Its SHA-256 is stored in the companion YAML and copied beside outputs.
Results, thresholds, nulls, seeds, class order, or interpretation rules are not
changed after this commit.

## 1. Question and scope

For each subject and session, do the four semantic motor-imagery class prototypes
form a relative geometry that is reliable across independent runs, shared across
subjects and sessions with the same semantic labels, and informative enough to
identify the semantic permutation when the four true target components are given
by an oracle?

This is anatomy, not a domain-adaptation method. True class labels are used to
measure class prototypes. It does not perform clustering, target-component
recovery, pseudo-labeling, conditional alignment, model development, WINDOW5 or
trajectory analysis.

AIRM marginal centering is an isometry. It cannot create the within-subject class
geometry measured here. Centered matrix entries are never compared across
subjects. The objects compared across subjects are invariant relational shapes.

## 2. Repository and output isolation

The branch starts exactly at the Geometry V2 final commit above. V1, V2,
Trajectory v0, `main`, and their outputs are read-only. All new output lives in:

    outputs/bnci2014_001_conditional_geometry_v1/

with subdirectories `protocol/`, `objects/`, `tables/`, `nulls/`, `figures/`, and
`report/`. Regenerable covariance and checkpoint arrays live only in ignored
`cache/bnci2014_001_conditional_geometry_v1/`.

## 3. Session lock

- Discovery session: `0train`.
- Confirmatory session: `1test`.
- Discovery-only scripts have no API that accepts `1test` and must not resolve,
  hash, open or preprocess A??E raw files.
- Before confirmatory unlock, no `1test` marginal/class mean, D, G, shape vector,
  semantic-permutation score, table, or figure may be computed.
- Discovery objects, all discovery R/S/P statistics, protocol/config/code hashes,
  output hashes, HEAD and clean working-tree status are committed first.
- `confirmatory_unlock.json` is then created from that clean commit. Confirmatory
  code validates the manifest and every locked discovery hash before any `1test`
  path is resolved.
- Any post-unlock source/config/protocol change invalidates the unlock.

History audit found prior raw session-inventory access but no direct `1test`
conditional geometry object or semantic-permutation analysis. The replication
designation is therefore **STRICT_CONFIRMATORY**, with
`prior_non_anatomy_session1_access=true` and
`prior_session1_conditional_object_analysis=false` recorded transparently.

## 4. Frozen data and preprocessing

Dataset is MOABB `BNCI2014_001`: subjects 1–9 and fixed class order
`left_hand`, `right_hand`, `feet`, `tongue`. Only the 22 frozen EEG channels are
used; EOG is excluded. The cue-relative epoch is 0.000–3.996 s, 1,000 samples at
250 Hz, band-pass 8–32 Hz, no baseline, no resampling. Each trial yields one
WHOLE OAS covariance in float64. There is no extra diagonal loading, eigenvalue
clipping, or trial rejection.

Each session must contain exactly 2,592 trials: 288 per subject, 72 per
subject/class, six runs `0`–`5`, 48 per subject/run and 12 per
subject/run/class. Any mismatch is a DATA CONTRACT FAILURE; rows are never
dropped or relabeled to fit remembered counts.

Discovery uses the hash-pinned V1 WHOLE covariance array read-only. Only a
covariance-identity failure may trigger WHOLE-only OAS recomputation from the
hash-pinned V1 prepared epochs and metadata into the v1 cache namespace.
Pre-unlock raw/MOABB fallback is forbidden. Confirmatory WHOLE preparation is
allowed only after unlock and writes only to the new ignored cache.

## 5. Splits

Within each session and subject:

- `A`: runs 0,1,2; 144 trials, 36 per class.
- `B`: runs 3,4,5; 144 trials, 36 per class.
- `F`: runs 0–5; 288 trials, 72 per class.

Canonical iteration is session, numeric subject, split A/B/F, fixed class order,
numeric run, trial UID. Appearance order in metadata never defines semantics.

## 6. AIRM objects

For subject s, session q and split h, the label-free marginal mean is

\[
M_{s,q,h}=\arg\min_M\sum_i d_{AI}(M,C_i)^2.
\]

For class c, the anatomy class mean is

\[
M_{s,q,h,c}=\arg\min_M\sum_{i:y_i=c}d_{AI}(M,C_i)^2.
\]

The exact metric object is

\[
D_{s,q,h}(c,c')=d_{AI}(M_{s,q,h,c},M_{s,q,h,c'}).
\]

Let

\[
L_c=\log(M^{-1/2}M_cM^{-1/2}),\qquad
U_c=M^{1/2}L_cM^{1/2}.
\]

The marginal-anchor star Gram object is

\[
G(c,c')=\mathrm{Tr}(M^{-1}U_cM^{-1}U_{c'})
       =\mathrm{Tr}(L_cL_{c'}).
\]

The direct expression is evaluated with linear solves, never an explicit
inverse. D asks whether exact class-prototype metric shape is shared; G asks
whether marginal-center anchored radii and directions are also shared.

## 7. LE robustness objects

LE means are `exp(mean(log C))`. Define `H_c=log M_c-log M`:

\[
D_{LE}(c,c')=\|\log M_c-\log M_{c'}\|_F,
\qquad G_{LE}(c,c')=\mathrm{Tr}(H_cH_{c'}).
\]

LE centering is log translation, not general affine congruence whitening. LE
must satisfy `D²(c,c') = Gcc + Gc'c' - 2Gcc'` within the frozen tolerance. LE
repeats R/S/P but never rescues or changes the AIRM terminal decision.

## 8. Shapes, permutation action and degeneracy

The D vector is the fixed upper triangle
`[D12,D13,D14,D23,D24,D34]` in class order and is normalized by its Euclidean
norm. G uses Frobenius-isometric symmetric `svec` in 10 dimensions (diagonal
unchanged, off-diagonal multiplied by sqrt(2)) and is normalized by its
Euclidean norm.

For each unnormalized vector v, the degeneracy threshold is

    100 * eps_float64 * max(1, max(abs(v))).

Norm at or below this value is `DEGENERATE_CLASS_GEOMETRY`; the subject is not
deleted and the terminal result is `UNASSESSED_DEGENERATE_GEOMETRY`.

Class permutations enumerate all 24 S4 tuples in Python lexicographic order,
identity first. The action is `O_pi = P_pi O P_pi^T`, implemented as the same
row/column reindexing for D and G. Shape vectors are reconstructed and
renormalized after the action.

## 9. Numerical hard gates

All scientific nulls wait for the relevant session gates:

1. Exact counts, finite covariances, relative symmetry error <=1e-12, strictly
   positive minimum eigenvalue and condition number <=1e12.
2. `pyriemann.geometry.mean.mean_riemann`, `tol=1e-9`, `maxiter=100`,
   `init=None`; every marginal/class mean is SPD and has independently computed
   whitened Karcher residual <=1e-7. Any convergence warning fails the row.
3. Direct class-mean D versus D after subject marginal congruence centering:
   relative Frobenius error <=1e-10.
4. Direct tangent G versus whitened-identity G: relative Frobenius error
   <=1e-10.
5. A deterministic orthogonal Q from the frozen gauge RNG conjugates centered
   class means; D and G relative errors are each <=1e-10.
6. D symmetric, diagonal zero and nonnegative; G symmetric; shape norms finite.
7. D/G class-permutation equivariance <=1e-10.
8. LE official/custom mean error and LE D–G identity relative error <=1e-10.

No clipping, silent repair, silent exclusion or available-case substitute is
allowed.

## 10. Stage R — split-half reliability

For object O in {D,G},

\[
r^O_{s,q}=\cos(z^O_{s,q,A},z^O_{s,q,B}),\qquad
T^O_{rel,q}=\operatorname{median}_s r^O_{s,q}.
\]

The label-destruction null has B=1,999. Within every
subject×session×run, the existing balanced label multiset is independently
shuffled. Covariances and counts remain fixed. A/B class means and both D/G
objects are recomputed for every replicate; the same label plan is used for
AIRM and LE. Null subject scores and group medians are stored by fixed replicate
index. The one-sided plus-one p-value and effect are

\[
p=(1+\#\{T_b\ge T_{obs}\})/(B+1),\qquad
E=T_{obs}-\operatorname{median}_b T_b.
\]

## 11. Unrelated-subject reference

All 133,496 fixed-point-free derangements of nine subjects are enumerated. For
each, the median cosine between subject s A and deranged subject sigma(s) B is
computed. This reference is descriptive only and never enters a gate or
p-value.

## 12. Stage S — shared semantic geometry

Templates always use **normalize(sum of already unit-normalized subject shape
vectors)**. They are not vectorizations of an averaged D/G matrix.

Discovery LOSO excludes target s. Source-A template is compared with target-B,
source-B with target-A, and the two cosines are averaged to subject score
`u_s,D`. The group statistic is its subject median.

Confirmatory uses the fixed discovery-F LOSO template (still excluding the
same target subject's discovery object) and averages its cosine with target
confirmatory A and B. Confirmatory data never updates the template.

The semantic null has B=100,000. Each draw assigns every source subject an
independent uniform permutation from all 24 S4 mappings, including identity.
The same subject permutation applies to that subject's discovery A/B/F,
AIRM/LE and D/G objects in that draw. LOSO target objects remain in true fixed
class order. Object permutations only are needed; Fréchet means are not
recomputed. Discovery and confirmatory use the same indexed permutation plan.
Effect and plus-one p-value use the formulas in Stage R.

## 13. Stage P — oracle semantic-name identifiability

This stage assumes that the target's four true class components are already
given. Only their class names are hidden. It is not clustering, component
recovery, unlabeled adaptation or a DA score.

Discovery P, required by the discovery-effect gate, is frozen as follows. For
each target subject and each of 24 candidate target permutations, compare
permuted target-B with the discovery source-A LOSO template and permuted
target-A with the source-B LOSO template; average the two candidate scores
before ranking.

Confirmatory P compares each permutation of target confirmatory-F with the
fixed discovery-F LOSO template. Identity is the true semantic mapping.

Tie tolerance is absolute 1e-12. Identity receives a conservative worst rank
within tolerance:

    R_id = 1 + count_nonidentity(score >= score_id - 1e-12).

Normalized rank is `Q=(24-R)/23`; 1 is best and 0 worst. For descriptive best
and second-best permutations, scores sort descending then permutation tuples
lexicographically. Top1 requires rank exactly 1. Margin is identity score minus
the maximum nonidentity score.

The oracle-rank null has B=1,000,000. For every draw and subject, the candidate
designated “true” is independently uniform over all 24 fixed candidate scores.
No SPD object is recomputed. Discovery and confirmatory use the same indexed
random-candidate plan. Group statistic is median subject Q; effect and plus-one
p-value follow Stage R.

## 14. RNG and resumability

All stochastic work uses
`Generator(PCG64DXSM(SeedSequence([20260809,family_tag,phase_tag,replicate])))`.
`default_rng` is forbidden. Integer family tags are:

- label destruction 1101
- semantic permutation 1201
- oracle rank 1301
- subject bootstrap 1401
- orthogonal gauge 1501

Phase tags are discovery=0, confirmatory=1 and paired/common=2. Label nulls use
phase-specific streams. Semantic and oracle plans use the common stream and are
reused across discovery/confirmatory. Canonical subject/run ordering determines
draw order inside a replicate.

Every checkpoint contains protocol/config/code hashes, family/phase/count,
replicate-indexed statistics, a completed bitmap and a payload hash. Resume
rejects any mismatch, preserves fixed indices and is bitwise invariant to chunk
size or interruption. Partial checkpoint files are ignored by Git.

## 15. Subject-level descriptive summaries

The independent unit is subject (n=9), never trials, halves or permutations.
For every stage/object/geometry/phase, save all subject scores, subject null
percentiles, discovery-confirmatory deltas and AIRM-LE paired effect deltas.

Bootstrap B=20,000 resamples nine cached subject scores with replacement and
reports the median's 2.5/97.5 percentile interval using
`np.quantile(method="linear")`. Influence is `T_leave_one_subject_out-T_full`
on the cached subject scores, without refitting LOSO templates. Both are
descriptive and do not vote.

## 16. Fixed-sequence chains and stage PASS

D and G are independent families with alpha 0.025 each. For each R, S or P,
the underlying stage criterion requires all four:

1. discovery effect > 0;
2. confirmatory effect > 0;
3. confirmatory one-sided p <= 0.025;
4. every numerical/data gate passed.

Within each chain, R is interpreted first, then S only if R passed, then P only
if R and S passed. All statistics are still saved, but a downstream stage after
failure is labeled `DESCRIPTIVE_ONLY`, never PASS. A chain passes only when
R→S→P all pass in sequence.

## 17. Terminal AIRM decision and LE label

Only AIRM chain status votes:

- D PASS and G PASS: `GO_STRONG`.
- D PASS and G FAIL: `GO_METRIC_ONLY`.
- D FAIL and G PASS: `STOP_TANGENT_ONLY`.
- D FAIL and G FAIL: `STOP_NO_SHARED_GEOMETRY`.
- numerical failure: `UNASSESSED_NUMERICAL_FAILURE`.
- data failure: `UNASSESSED_DATA_CONTRACT_FAILURE`.
- degeneracy: `UNASSESSED_DEGENERATE_GEOMETRY`.

LE robustness label compares the final D/G chain pass pair:

- identical AIRM and LE pairs: `AIRM+LE CONSISTENT`;
- any AIRM PASS without an opposite LE-only PASS: `AIRM-SPECIFIC`;
- both AIRM chains FAIL and any LE chain PASS: `LE-ONLY — DOES NOT RESCUE AIRM FAILURE`;
- every other crossed pattern: `AIRM/LE DISCORDANT`.

No absolute cosine cutoff and no post-result threshold are used.

## 18. Runtime measurement and implementation

Before the official 1,999-replicate label null, time a small synthetic or
discovery-only dry run and record its sample size/runtime. Scientific
definitions may not be approximated. Exact acceleration may use deterministic
batch AIRM means cross-checked against pyRiemann, multiprocessing, cached
subject/run arrays, checkpoint/resume and object-level vectorization.

Semantic and oracle nulls operate only on cached 4×4 objects. General large
arrays/caches stay ignored. Required compact compressed null group-statistic
NPZ files may be narrowly unignored and tracked when safely below GitHub limits;
otherwise deterministic seed/hash manifests and compact summaries are tracked.

## 19. Required outputs

Root-level provenance: `manifest.json`, `git_provenance.json`,
`environment.json`, `confirmatory_unlock.json`.

Required geometry tables: `dataset_contract.csv`, `covariance_sanity.csv`,
`airm_mean_convergence.csv`, `le_mean_correctness.csv`,
`centering_isometry_gate.csv`, `orthogonal_gauge_gate.csv`,
`degenerate_geometry_audit.csv`.

Required objects: `airm_marginal_means.npz`, `airm_class_means.npz`,
`le_marginal_means.npz`, `le_class_means.npz`, `D_matrices.npz`,
`G_matrices.npz`, `D_shape_vectors.csv`, `G_shape_vectors.csv`,
`absolute_geometry_scales.csv`, `radius_angle_summary.csv`.

Required R/reference outputs: `within_subject_reliability.csv`,
`label_destruction_subject_summary.csv`,
`label_destruction_group_summary.csv`,
`nulls/label_destruction_group_statistics.npz`,
`unrelated_subject_derangement_summary.csv`.

Required S outputs: `loso_templates.csv`,
`cross_subject_shared_geometry.csv`,
`semantic_permutation_null_summary.csv`,
`nulls/semantic_permutation_group_statistics.npz`.

Required P outputs: `oracle_permutation_all_24_scores.csv`,
`oracle_permutation_subject_summary.csv`,
`oracle_permutation_group_summary.csv`, `nulls/oracle_rank_null.npz`.

Decision outputs: `discovery_confirmatory_comparison.csv`,
`airm_le_robustness.csv`, `hypothesis_chain_status.csv`,
`confirmatory_decision.json`. Additional tracked audit tables include
`subject_bootstrap_summary.csv` and `leave_one_subject_out_influence.csv`.

Discovery-locked snapshots live under `objects/discovery/`,
`tables/discovery/` and `nulls/discovery/`; confirmatory snapshots use matching
`confirmatory/` paths. Final required combined artifacts never overwrite a
locked discovery snapshot.

## 20. Figures and report

Exactly ten required stems each have PNG, PDF and source CSV:

1. `figure_1_within_subject_reliability`
2. `figure_2_reliability_label_null`
3. `figure_3_same_vs_unrelated`
4. `figure_4_shared_template_similarity`
5. `figure_5_D_heatmaps`
6. `figure_6_G_heatmaps`
7. `figure_7_oracle_permutation_scores`
8. `figure_8_oracle_rank_margin`
9. `figure_9_subject_forest_influence`
10. `figure_10_airm_le_stage_effects`

No t-SNE, UMAP, centered-entry heatmap, cherry-picked subject, classifier
accuracy plot, or result-selected panel is allowed.

The report is
`report/conditional_geometry_anatomy_v1.md`, titled
`# BNCI2014_001 Conditional Geometry Anatomy v1`, with exactly these level-two
sections in order:

1. Scientific question
2. Why this follows V1/V2/Trajectory v0
3. Frozen protocol
4. Data and numerical gates
5. Exact AIRM objects D and G
6. Discovery reliability
7. Confirmatory reliability
8. Same-subject vs unrelated reference
9. Discovery cross-subject shared geometry
10. Locked confirmatory shared geometry
11. Oracle semantic-permutation identifiability
12. D-chain
13. G-chain
14. LE robustness
15. Terminal frozen decision
16. What is actually justified
17. What is NOT justified
18. One next question only

It explicitly states that centering did not
create class geometry; D/G are not full conditional distributions; total
subject variation was not shown removed; oracle P is not unlabeled recovery or
DA success; there is no new WINDOW5/trajectory, neuroscience-mechanism or
cross-dataset claim.

The report ends with one question only. A GO decision asks about unlabeled
component-recovery identifiability. A STOP decision asks which representation or
mechanistic anchor could replace the failed WHOLE-covariance relational anchor.

## 21. Required tests and publication

Tests cover exact splits/counts, discovery session barrier, unlock validation,
mean residuals, D/G symmetry, centering and gauge invariance, direct/whitened G,
LE D–G identity, permutation equivariance, shape normalization/degeneracy,
LOSO exclusion, same subject semantic permutation across A/B/F, exactly 24 S4
mappings, conservative rank ties, plus-one p-values, fixed-sequence gates,
checkpoint resume/replay, and all inherited V1/V2 tests.

Milestone commits are logical and non-rewritten. Before push, run the complete
test suite, verify the working tree, tracked sizes, ignored raw/cache arrays and
commit history. Push only `pilot/conditional-geometry-anatomy-v1`; do not merge
main and do not force-push.
