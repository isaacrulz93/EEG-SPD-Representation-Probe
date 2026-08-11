# Local Mean Covariance Movement V0

## Provenance and frozen scope

- Branch: `pilot/local-mean-movement-antidevelopment-v0`
- Finalized temporal parent HEAD: `6d5ad6a0bdd4f2d19bfee8ce6fcbb97a5c499a5d`
- Temporal protocol freeze: `70981aa89ddbadceca42f354c3c51d05bf6dbf0c`
- Temporal scientific result: `43e926073fab0ba76fd5baa881804538f0d7beee`
- Protocol freeze SHA: `e24312147ef3020854ef6f6cd174071d1c6ead02`
- Final result SHA: `c3f1d5ff9cf23db2007bbf839cf4b266e2cb8960`
- Scientific object: window-wise mean covariance movement, not trial-level or continuous velocity.

The full and split-half mean artifacts were loaded unchanged from the finalized temporal result. Full SHA-256 `e03b94daef3eb37f9209ee7a7482ea575b1eb353804505a3e91013339da1913f`; split SHA-256 `355f098de7ff3dcf274e5a62cf6d92022bc1b1f6ed4301a2dc53f9a19f3cd868`. All 1080 saved mean matrices matched the frozen result exactly; maximum absolute difference `0.000e+00`.

## Anti-development mathematical gates

All 216 full/split mean sequences and 864 transitions passed. Maximum norm-identity error: `2.132e-14`; maximum edgewise transport relative error: `1.646e-14`; maximum Z symmetry relative error: `0.000e+00`.

The synthetic d=22 common-congruence check used one common O across all four steps and had maximum relative error `2.141e-14`. The known-Q quotient distance was `2.318e-15`.

## Quotient optimizer status

Status: **PASS**. TrustRegions used 6 deterministic starts per fit with equal coverage of both determinant sectors. Certified primary fits: 1296; certified split-half fits: 72. Maximum selected projected gradient norm: `9.992e-07`. Synthetic forward/reverse equality: `True`; gradient finite-difference absolute error: `1.678e-10`.

Fitted Q matrices are nuisance quotient variables and are not interpreted scientifically.

## Primary common-O quotient movement results

- `T_subject = 0.2582696475`, `p_subject = 0.000500`
- All S_s: S1=0.1976201130, S2=0.5142251516, S3=0.2524125903, S4=0.1839125372, S5=0.2750450150, S6=0.1718587516, S7=0.1727196869, S8=0.2161668621, S9=0.3404661199
- `T_class = 0.2391984646`, `p_class = 0.000500`
- All C_s: S1=0.3725037618, S2=0.0499278497, S3=0.3022706854, S4=0.1474162343, S5=0.0618030161, S6=0.0631618084, S7=0.0906667223, S8=0.2188641967, S9=0.8461719069
- `T_J = 0.1540437935`, `p_J_subjectbreak = 0.000500`, `p_J_classbreak = 0.000500`
- All J_s: S1=0.2025244431, S2=0.0426492226, S3=0.1844273212, S4=0.1093751927, S5=0.0086829373, S6=0.0108868152, S7=0.0310933890, S8=0.0950342778, S9=0.7017205424

All p-values are one-sided plus-one values from exactly 1,999 inherited whole-cell relabelings.

## Magnitude-only ordered speed control

- `T_subject = 0.1872503513`, `p_subject = 0.000500`
- `T_class = 0.1257760146`, `p_class = 0.000500`
- `T_J = 0.1057261433`, `p_J_subjectbreak = 0.000500`, `p_J_classbreak = 0.000500`
- S_s: S1=0.1410818782, S2=0.1866366498, S3=0.2009522233, S4=0.0814168259, S5=0.2738946693, S6=0.1835364036, S7=0.1852317639, S8=0.1654152649, S9=0.2670874830
- C_s: S1=0.1126049050, S2=0.0067052738, S3=0.1678624767, S4=0.2111842195, S5=0.0196696706, S6=0.0749557481, S7=0.0357439492, S8=0.0762930670, S9=0.4269648219
- J_s: S1=0.0540398923, S2=0.0160269933, S3=0.1309227675, S4=0.1733212519, S5=0.0146002841, S6=0.0622274428, S7=0.0207247981, S8=0.0877881535, S9=0.3918837066

## Direct montage-registered control

- `T_subject = 0.3361539037`, `p_subject = 0.000500`
- `T_class = 0.2846113113`, `p_class = 0.000500`
- `T_J = 0.1958728574`, `p_J_subjectbreak = 0.000500`, `p_J_classbreak = 0.000500`
- S_s: S1=0.2832488599, S2=0.5643206594, S3=0.3558090122, S4=0.3160156863, S5=0.3269000336, S6=0.2146194041, S7=0.2659117747, S8=0.3122072950, S9=0.3863524082
- C_s: S1=0.4281698316, S2=0.0850252453, S3=0.3147567188, S4=0.2077284722, S5=0.0804791311, S6=0.0772338531, S7=0.1472997283, S8=0.3272787242, S9=0.8935300969
- J_s: S1=0.2677692258, S2=0.0672440962, S3=0.1989455474, S4=0.1749617998, S5=0.0261954418, S6=0.0188736930, S7=0.0807846902, S8=0.1997412571, S9=0.7283399653

The magnitude-only control also passes; ordered displacement magnitude contributes information, without establishing a unique directional contribution.

## Split-half movement reliability (non-gating)

All 72 distances are reported below. Mean `0.7408522314`, median `0.7185224191`, range `[0.5084253272, 1.2539514752]`. No threshold was applied.

| subject | session | class | d_mov |
| --- | --- | --- | --- |
| 1 | 0train | left_hand | 0.6128323906 |
| 1 | 0train | right_hand | 0.6495480047 |
| 1 | 0train | feet | 0.6436983502 |
| 1 | 0train | tongue | 0.6525854664 |
| 2 | 0train | left_hand | 0.6355722805 |
| 2 | 0train | right_hand | 0.7100534963 |
| 2 | 0train | feet | 0.7543131635 |
| 2 | 0train | tongue | 0.8495871515 |
| 3 | 0train | left_hand | 0.7288553070 |
| 3 | 0train | right_hand | 0.7012798044 |
| 3 | 0train | feet | 0.6992683056 |
| 3 | 0train | tongue | 0.6971926218 |
| 4 | 0train | left_hand | 0.7552088678 |
| 4 | 0train | right_hand | 0.6238714940 |
| 4 | 0train | feet | 0.7007817024 |
| 4 | 0train | tongue | 0.7317107256 |
| 5 | 0train | left_hand | 0.5733726323 |
| 5 | 0train | right_hand | 0.5773214171 |
| 5 | 0train | feet | 0.5242660604 |
| 5 | 0train | tongue | 0.6953200647 |
| 6 | 0train | left_hand | 0.7640159703 |
| 6 | 0train | right_hand | 0.7246384011 |
| 6 | 0train | feet | 0.8487115687 |
| 6 | 0train | tongue | 0.8029614559 |
| 7 | 0train | left_hand | 0.6418116971 |
| 7 | 0train | right_hand | 0.6524640722 |
| 7 | 0train | feet | 0.6943737043 |
| 7 | 0train | tongue | 0.7371554589 |
| 8 | 0train | left_hand | 0.7122647671 |
| 8 | 0train | right_hand | 0.8075844082 |
| 8 | 0train | feet | 0.6359630228 |
| 8 | 0train | tongue | 0.7048261030 |
| 9 | 0train | left_hand | 1.2539514752 |
| 9 | 0train | right_hand | 0.8543346414 |
| 9 | 0train | feet | 0.9076377586 |
| 9 | 0train | tongue | 0.7379439598 |
| 1 | 1test | left_hand | 0.6835579815 |
| 1 | 1test | right_hand | 0.6953255276 |
| 1 | 1test | feet | 0.6956063353 |
| 1 | 1test | tongue | 0.6512233877 |
| 2 | 1test | left_hand | 0.5824342597 |
| 2 | 1test | right_hand | 0.9035521719 |
| 2 | 1test | feet | 0.6133321896 |
| 2 | 1test | tongue | 0.5084253272 |
| 3 | 1test | left_hand | 0.6935619815 |
| 3 | 1test | right_hand | 0.7146652739 |
| 3 | 1test | feet | 0.7624371572 |
| 3 | 1test | tongue | 0.7741380912 |
| 4 | 1test | left_hand | 0.8913979480 |
| 4 | 1test | right_hand | 0.7079923113 |
| 4 | 1test | feet | 0.7259911696 |
| 4 | 1test | tongue | 0.8962550094 |
| 5 | 1test | left_hand | 0.5126494525 |
| 5 | 1test | right_hand | 0.5781987074 |
| 5 | 1test | feet | 0.7223795643 |
| 5 | 1test | tongue | 0.5364831186 |
| 6 | 1test | left_hand | 0.8689776878 |
| 6 | 1test | right_hand | 0.7011770074 |
| 6 | 1test | feet | 0.8144599598 |
| 6 | 1test | tongue | 0.8669133173 |
| 7 | 1test | left_hand | 0.6616866349 |
| 7 | 1test | right_hand | 0.7261434458 |
| 7 | 1test | feet | 0.7251008676 |
| 7 | 1test | tongue | 0.8920785222 |
| 8 | 1test | left_hand | 0.8552203139 |
| 8 | 1test | right_hand | 0.7796865757 |
| 8 | 1test | feet | 0.8932198594 |
| 8 | 1test | tongue | 0.7969715350 |
| 9 | 1test | left_hand | 1.1111975898 |
| 9 | 1test | right_hand | 0.7514341754 |
| 9 | 1test | feet | 1.1572218652 |
| 9 | 1test | tongue | 0.8929865649 |

## Fixed illustrative comparisons

| comparison | d_mov | d_len | d_direct |
| --- | --- | --- | --- |
| S1 Left vs S1 Left | 0.4688124283 | 0.0862818653 | 0.7110064619 |
| S1 Left vs S2 Left | 0.9234394514 | 0.1425491774 | 1.1520136646 |
| S1 Left vs S1 Feet | 0.9059359350 | 0.2601690850 | 1.2293988184 |
| S1 Left vs S2 Feet | 1.2345196531 | 0.4555207223 | 1.4981469186 |

## Terminal and interpretation

`GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT`

After removing the initial SPD location and one common sequence-level orthogonal gauge, the ordered displacement pattern of the window-wise mean covariance trajectory is cross-session reproducible and subject×class-specific.

This result concerns an ordered discrete AIRM anti-development of a window-wise mean covariance trajectory. It does not establish individual-trial neural velocity, continuous-time dynamics, causal or physiological state transitions, source-space dynamics, physical sensor orientation, an absolute subject pose, biological privilege of AIRM, or completeness of five windows.

The prior unordered metric analysis supported subject and class specificity but not its explicit interaction. The ordered raw mean-sequence analysis supported temporal correspondence, subject specificity, class specificity, and interaction. This experiment replaces absolute point placement with ordered adjacent relative movement; it is not a rescue of the unordered result.

## Runtime, tests, and immutability

- Total scientific runtime: `150.983` seconds
- Quotient matrix runtime: `138.492` seconds
- Split-half quotient runtime: `7.351` seconds
- Focused tests: `15 passed in 1.95s`
- Full repository tests: `255 passed, 4 skipped in 27.69s`
- Git status at scientific execution start: clean; HEAD equaled the protocol-freeze SHA.
- Git status immediately before report-only result finalization: clean at the scientific-result commit.
- Final handoff status is verified after committing this report.
- No scientific setting changed after the first movement result was observed. Finalization is limited to inserting the result commit SHA, post-result provenance, and refreshed artifact hashes.

Descriptive PCA uses one global Frobenius-isometric svec basis over all 72×4 full movement points. No inference was performed in PCA space.
