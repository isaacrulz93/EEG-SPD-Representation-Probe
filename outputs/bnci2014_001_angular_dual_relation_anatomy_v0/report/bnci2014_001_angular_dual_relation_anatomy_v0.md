# BNCI Angular Six-Pair and Dual Relation Anatomy V0

## Verdict and provenance

`COMPLETED_BNCI_ANGULAR_DUAL_RELATION_ANATOMY_V0`

- Status: retrospective anatomy, not prospective or confirmatory
- Branch: `audit/bnci-angular-dual-relation-anatomy-v0`
- Exact parent: `edc1d344cb0657f2f2d87b2992049bceec4705d2`
- Parent protocol freeze: `95c330de9596fa4c4eb4ee377d5af8d99896f4c3`
- Parent scientific result: `0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`
- Protocol freeze SHA: `54154436aeabd818cacaa8c6973400b93de7a9ea`
- Scientific result SHA: `PENDING_SCIENTIFIC_RESULT_COMMIT`
- Frozen D SHA-256: `51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091`

Only the frozen directed 36 by 36 squared angular cost matrix was used. No EEG,
covariance mean, anti-development, movement tuple, AIRM mean, or quotient
optimizer was fitted or recomputed.

## Immutable formal prior evidence

The frozen four-class result remains `T_subject=0.3091561772`,
`T_class=0.393098434`, and
`T_J=0.1924088545` with subject-break `p=0.001` and
class-break `p=0.0105`. This inference is immutable and is not rerun or
redefined by the descriptive anatomy.

## Formal retrospective six-pair decomposition

| pair | T_S | T_C | T_J | exact fraction of four-class J |
| --- | ---: | ---: | ---: | ---: |
| LR | 0.3648215448 | 0.1565649309 | 0.1191292918 | 0.1031911032 |
| LF | 0.4079388071 | 0.2599722447 | 0.01651258238 | 0.01430338053 |
| LT | 0.3419988384 | 1.098902167 | 0.514204525 | 0.4454096168 |
| RF | 0.276313516 | 0.1575289066 | 0.1277244434 | 0.1106363181 |
| RT | 0.2103735473 | 0.4041010383 | 0.1625736042 | 0.1408230446 |
| FT | 0.2534908096 | 0.2815213159 | 0.2143086803 | 0.1856365367 |

Fractions are `(T_J,p/6)/T_J,4c` and sum to one; they are additive accounting,
not post-hoc inferential weights. No pairwise p-values were computed.

The hard mean-aggregation gate passed for every subject and group statistic.
Maximum absolute reconstruction error was
`2.2204460492503131e-16`; group signed errors were
`S=5.5511151231257827e-17`,
`C=0`, and
`J=2.7755575615628914e-17`. The reconstructed four-class
`T_J=0.1924088545` matches the frozen parent.

## Supporting retrospective Left/Right result

PR #12 remains supporting evidence: `T_subject=0.3648215448, p=0.0005`,
`T_class=0.1565649309, p=0.158`, and `T_J=0.1191292918` with subject-break
`p=0.09` and class-break `p=0.3065`. Its terminal remains
`BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED`. The six-pair anatomy does not turn
LR into a new primary endpoint.

## Q1 — Within-subject class relations

Across-subject mean raw G relations, ordered small to large, are LR=0.5086492324, RF=0.5166058397, LF=0.6191039119, FT=0.6257990225, RT=0.7413313791, LT=1.436187242.
The adjusted ordering is LR=0.1565649309, RF=0.1575289066, LF=0.2599722447, FT=0.2815213159, RT=0.4041010383, LT=1.098902167. The most frequent raw smallest pair
counts are FT=2/9, LR=5/9, LT=1/9, RF=1/9; adjusted counts are FT=1/9, LF=1/9, LR=4/9, LT=1/9, RF=1/9, RT=1/9. Full
per-subject orderings are saved in the profile tables. These patterns show both
shared tendencies and subject-specific deviations; G is a relation matrix, not
an intrinsic metric geometry.

## Q2 — Common class-relation backbone

Raw G profiles have mean pairwise Pearson correlation
`0.257141696` (median
`0.5162184207`, range
`-0.7611850307` to
`0.9115560978`) and mean leave-one-subject-out
correlation `0.4173744227`. Adjusted values are
`0.2141898909` and
`0.4038530357`. Centered cosine gives the
same values up to maximum recorded numerical error
`3.33e-16`. Thus a partial class
backbone is descriptively present, but it is heterogeneous rather than a
universal ordering.

## Q3/Q4 — Subject relations within and across classes

Raw H profiles have mean class-to-class Pearson correlation
`0.2385514522` and leave-one-class-out mean
`0.4940394293`; adjusted values are
`0.1504772222` and
`0.4090114797`. Raw class-specific leave-out
correlations are F=0.8402448145, L=0.8094886482, R=0.3661260782, T=-0.03970182362. Subject-pair ordering is therefore partly reused but
class-dependent. This is descriptive relational reuse, not evidence for a
single reusable subject transformation.

## Subject-level interaction anatomy

Largest positive pair contribution by subject: S1:RT=0.5968464099; S2:RF=0.07843794203; S3:RT=0.5368750406; S4:RT=0.1612115631; S5:FT=0.01125145884; S6:FT=0.2407772573; S7:FT=0.06989083906; S8:RT=0.190996591; S9:LT=4.949301756.
The largest four-class subject value is S9=
`1.431861256`; the other eight
average `0.03747730436`. LR is
positive in `4/9` subjects and
has median `J_s,LR=-0.02004534409`. The 9 by 6
heatmap shows that the group result is substantially concentrated and that
different subjects contribute through different pairs. No subject is removed,
and leave-one-subject values are descriptive only.

## Coarse effector-boundary anatomy

The prespecified descriptive contrast is `K_J=0.03853480269`. The mean
cross hand/nonhand pair `T_J` is `0.2052537888`
versus `0.1667189861` for LR/FT. This does not make
feet and tongue biologically homogeneous and has no new p-value. Pair values
show that a simple coarse hierarchy is not a complete account of the frozen
interaction.

## Pattern assessment

- **Pattern A, common class backbone:** partially present, with substantial
  subject heterogeneity.
- **Pattern B, common subject backbone:** partially present and class-dependent;
  it does not identify a reusable transformation.
- **Pattern C, heterogeneous subject-by-class deformation:** strongly visible
  in the spread of `J_s,p` and subject influence.
- **Pattern D, coarse effector hierarchy:** `K_J` is
  `positive`. Pair heterogeneity determines whether that
  coarse balance is a useful partial summary; it cannot be a complete
  explanation by itself.

Multiple patterns coexist; none is forced into a replacement primary result.
The descriptive G/H deviations are not themselves formal interaction tests.

## Formal versus descriptive hierarchy

1. **Immutable formal prior evidence:** the frozen four-class angular J and its
   two parent null tests remain the inferential result.
2. **Formal retrospective decomposition:** the six binary-pair statistics are
   an exact algebraic partition of the frozen mean-aggregated S/C/J values.
3. **Supporting retrospective result:** PR #12 supplies the separately frozen
   Left/Right-only audit; it is not promoted by this analysis.
4. **Descriptive anatomy:** G, delta-G, H, delta-H, profile correlations,
   subject-by-pair contributions, influence summaries, and K_J explain the
   relation structure without replacing the formal S/C/J test.

## Result-to-claim guide

| Descriptive outcome | Permitted interpretation | Not established |
| --- | --- | --- |
| Strong G similarity across subjects | A common class-relation backbone is descriptively present. | A universal intrinsic class shape or generative class factor. |
| Weak G but strong H similarity | Subject-pair ordering is more reusable across classes than class-pair ordering is across subjects. | One reusable subject transformation. |
| Both G and H strong | Common class and subject relational backbones may coexist. | An additive or causal subject/class decomposition. |
| Both G and H weak | The relation profiles are heterogeneous at this resolution. | Absence of formal S/C/J effects or equivalence to zero. |
| LR weak but hand/nonhand pairs strong | The four-class result may be concentrated across coarse effector boundaries. | Feet and tongue as a biologically homogeneous class. |
| J concentrated in LF/LT/RF/RT rather than LR | Cross-effector class-pair contributions explain more of the additive four-class J than LR. | A newly selected pairwise primary endpoint. |
| J concentrated in a few subjects | The group mean has high descriptive subject influence and should be reported with its distribution. | Permission to drop subjects or treat entries as independent observations. |
| Static versus movement patterns differ | Only already-frozen results may be compared, with their distinct objects kept explicit. | Interchangeability of static placement and ordered movement anatomy. |

## Overclaim risks

- G and H are relation matrices/profiles, not strongly intrinsic manifold
  shapes.
- Baseline-adjusted matrices remain descriptive and are not inferential
  primaries.
- Pairwise values or p-values cannot redefine the frozen main conclusion.
- Common relation geometry does not identify a reusable transformation.
- S/C/J alone does not identify a generative subject/class decomposition.
- LR is one pair and cannot stand in for the entire four-class structure.
- No upstream artifact may be refitted or silently modified.
- Subjects, not trial pairs or matrix entries, are the population units.

## Validation

- Parent artifact and K=4 reproduction: PASS
- Six-pair subject/group reconstruction: PASS
- Symmetric A and G/H index/diagonal gates: PASS
- Parent artifacts unchanged: true
- Focused pre-result tests: `13 passed in 0.08s`
- Focused post-result tests: `16 passed in 0.17s`
- Full repository tests: `286 passed, 4 skipped, 1 unrelated failure: missing ignored cache/bnci2014_001_trajectory_within_subject_v1/combined_trajectory_features.npz`
- Scientific setting changed after protocol freeze: false
- Runtime: `1.397791` seconds

The outputs concern frozen window-wise mean-covariance movement relation costs.
They do not establish physiology, cortical direction, causal motor strategy,
intrinsic manifold shape, or a generative subject/class factor.
