# BNCI2014_001 Angular Relation Anatomy V0

## Verdict and provenance

`COMPLETED_FROZEN_BNCI_ANGULAR_RELATION_ANATOMY_V0`

- Branch: `pilot/bnci-angular-relation-anatomy-v0`
- Exact base: `edc1d344cb0657f2f2d87b2992049bceec4705d2`
- Parent protocol freeze: `95c330de9596fa4c4eb4ee377d5af8d99896f4c3`
- Parent scientific result: `0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`
- Protocol freeze SHA: `93ba2834eda20d1baa03dac2d26b6285062a8b1d`
- Scientific result SHA: `PENDING_SCIENTIFIC_RESULT_COMMIT`
- Frozen matrix SHA-256: `51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091`

Only the frozen 36 x 36 cross-session squared `c_ang` matrix was read. No EEG,
covariance mean, anti-development, ordered movement object, or quotient
optimizer was fitted or recomputed. The matrix is directed from session 0 rows
to session 1 columns for formal S/C/J calculations.

## Immutable formal inference

The parent four-class angular result remains unchanged: `T_subject=0.3091561772`,
`T_class=0.393098434`, and `T_J=0.1924088545`.
Its previously frozen null p-values and positive subject x class conclusion are
not rerun or redefined here.

## Primary formal retrospective decomposition

| pair | T_S | T_C | T_J |
| --- | ---: | ---: | ---: |
| LR | 0.3648215448 | 0.1565649309 | 0.1191292918 |
| LF | 0.4079388071 | 0.2599722447 | 0.01651258238 |
| LT | 0.3419988384 | 1.098902167 | 0.514204525 |
| RF | 0.276313516 | 0.1575289066 | 0.1277244434 |
| RT | 0.2103735473 | 0.4041010383 | 0.1625736042 |
| FT | 0.2534908096 | 0.2815213159 | 0.2143086803 |

These are supporting algebraic components, not six competing primary tests;
no pairwise p-values were computed. Ranked descriptively by `T_J`: LT=0.514204525, FT=0.2143086803, RT=0.1625736042, RF=0.1277244434, LR=0.1191292918, LF=0.01651258238.

The exact mean aggregation gate passed. Across every subject and group statistic,
the four-class value equals the arithmetic mean of the six binary-pair values.
Maximum absolute reconstruction error: `2.2204460492503131e-16`.
Group signed residuals were `S=5.5511151231257827e-17`,
`C=0`, and
`J=2.7755575615628914e-17`. Thus the frozen four-class
interaction is not a Left/Right-only phenomenon; its exact pairwise anatomy is
reported without selecting a replacement endpoint.

Subject concentration is descriptive: the largest subject-level mean angular
`J_s` is S9=1.431861256; the
remaining eight subjects average `0.03747730436`. This
does not change the subject-as-population-unit inference already frozen in the
parent.

## Descriptive subject-fixed class relation anatomy (G_s)

`G_s` symmetrizes the two directed cross-session costs for each within-subject
class pair. The raw six-entry class-relation profiles have mean pairwise
centered correlation `0.257141696` and
mean leave-one-subject-out commonality `0.4173744227`.
After subtracting the relevant self-instability baselines, these are
`0.2141898909` and
`0.4038530357`. Pairwise Euclidean distances and all
individual commonality values are saved in the tables.

The across-subject mean raw profile ranks class costs as
LR=0.5086492324, RF=0.5166058397, LF=0.6191039119, FT=0.6257990225, RT=0.7413313791, LT=1.436187242. The baseline-adjusted rank is
LR=0.1565649309, RF=0.1575289066, LF=0.2599722447, FT=0.2815213159, RT=0.4041010383, LT=1.098902167. This describes the observed breadth of a
common class hierarchy and the relative Left/Right versus hand/nonhand
organization; it is not a new inferential claim.

## Descriptive class-fixed subject relation anatomy (H_c)

The four 36-entry subject-relation profiles have mean pairwise centered
correlation `0.2385514522` and mean
leave-one-class-out commonality `0.4940394293`. Their
baseline-adjusted counterparts are
`0.1504772222` and
`0.4090114797`. These values quantify how similarly
the frozen classes arrange subject-to-subject costs. They do not establish a
reusable transformation or generative subject factor.

## Integrated anatomy

The formal scalar contrasts and descriptive matrices are complementary. `S`
summarizes subject correspondence, `C` class correspondence, and `J` their
non-additive correspondence contrast. `G_s` displays which class relations are
close or far within each subject; `H_c` displays which subject relations are
close or far within each class. They anatomize but do not replace S/C/J.

The exact pair decomposition, the observed spread of `J_s,p`, the `G_s`
commonality, and the `H_c` commonality should be read together: shared class
ordering or shared subject ordering can coexist with subject x class-specific
deformation. Coarse hand-versus-nonhand organization is assessed descriptively
from LR versus LF/LT/RF/RT/FT costs and contrasts, not elevated into a new
primary test. Static-versus-movement implications are not identified by this
frozen movement matrix alone.

At the current resolution, the anatomy is:

- **Common class structure:** partial and heterogeneous, not broad and uniform.
  Raw and adjusted `G_s` correlations are positive on average but well below
  one. The mean profile chiefly separates tongue from left hand (`LT` is the
  largest raw and adjusted relation), whereas `LR` and `RF` are the two
  smallest adjusted relations. This is a shared tendency, not a universal
  subject-independent class hierarchy.
- **Common subject structure:** also partial and class-dependent. The `H_c`
  profile correlations and leave-one-class-out correlations are positive on
  average but well below one, supporting descriptive reuse of some subject
  ordering without establishing a reusable subject transformation.
- **Subject x class-specific deformation:** visibly substantial. Pair `T_J`
  values are uneven, and S9 has four-class
  `J_s=1.431861256` versus `0.03747730436`
  averaged over the other subjects. The parent inferential interaction is
  therefore accompanied by pronounced subject/pair concentration, not a
  uniform offset across all subjects and classes.
- **Coarse hand-versus-nonhand organization:** insufficient as a complete
  description. The four hand/nonhand `T_J` components average
  `0.2052537888`, but they range from LF `0.01651258238` to LT
  `0.514204525`; the nonhand FT component `0.2143086803` exceeds
  LR `0.1191292918`. The dominant feature is more specifically the LT
  relation and its subject concentration, not a consistent binary hand/nonhand
  split.

## Result-to-claim decision table

| Observed pattern | What may be said | What may not be said |
| --- | --- | --- |
| Strong G_s similarity across subjects | Evidence of a common class-relation profile | A universal intrinsic class shape or reusable transform |
| Weak G_s but strong H_c similarity | Subject relations look more reusable across classes than class relations across subjects | A generative subject factor |
| Both strong | Both relation profiles show common descriptive organization | S/C/J are redundant or a separable generative model is proven |
| Both weak | Relation profiles are heterogeneous in both views | Subject/class correspondence is absent without reference to frozen formal tests |
| LR weak but hand-vs-nonhand pairs strong | Coarse hand/nonhand organization may dominate this descriptive resolution | LR represents the complete four-class structure |
| J concentrated in LF/LT/RF/RT rather than LR | The exact four-class J is chiefly carried by those pair components | The best pair becomes a new primary endpoint |
| J concentrated in only a few subjects | The group mean has substantial subject-level concentration | Matrix entries are independent replicates or an effect is population-wide |
| Static versus movement contrast | May be discussed only with already-frozen, directly comparable prior evidence | This movement-only anatomy establishes a static/movement distinction |

## Overclaim risks

1. `G_s` and `H_c` are relation matrices or distance profiles, not strongly
   intrinsic manifold shapes.
2. Baseline-adjusted matrices are descriptive, not inferential primaries.
3. Pair outputs must not redefine the parent study conclusion; no pairwise
   p-values were used here.
4. Common relation geometry is not a reusable transformation.
5. S/C/J alone do not identify a generative subject/class decomposition.
6. No single class pair, especially LR, is the whole four-class story.
7. Upstream artifacts must not be refit or silently modified.
8. Trial pairs and matrix entries are not independent inferential units;
   subjects remain the population unit.

## Validation and immutability

- Frozen input reproduction: PASS
- Parent four-class scalar reproduction: PASS
- Six-pair reconstruction: PASS
- Parent hashes unchanged after execution: true
- Focused pre-result tests: `10 passed in 0.04s`
- Focused post-result tests: `10 passed in 0.07s`
- Full repository tests: `1 failed, 280 passed, 4 skipped (unrelated missing ignored legacy cache: combined_trajectory_features.npz)`
- Scientific settings changed after protocol freeze: false
- Post-result changes: presentation only (shared heatmap colorbar layout and
  expanded interpretation of already-frozen tables); documented in
  `provenance/post_result_presentation_note.json`
- Runtime: `0.858027` seconds
