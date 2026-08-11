# Local Ordered Movement Component Decomposition V0

## Identity and frozen lineage

- Branch: `pilot/local-movement-component-decomposition-v0`
- Exact authoritative parent: `12c19f38266bc76875cffae056e7f9403df299c1`
- Movement V0 protocol freeze: `e24312147ef3020854ef6f6cd174071d1c6ead02`
- Movement V0 scientific result: `c3f1d5ff9cf23db2007bbf839cf4b266e2cb8960`
- Component protocol freeze: `95c330de9596fa4c4eb4ee377d5af8d99896f4c3`
- Scientific result SHA: `0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`
- Parent terminal: `GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT`

The finalized Movement V0 anti-developments, root-distance matrices, and null mappings were reused exactly. No AIRM means, raw covariances, anti-developments, M1 references, or full-data quotient matrices were refit.

## Frozen-input reproduction

Status: **PASS**. Byte and exact-array equality were verified against `c3f1d5ff9cf23db2007bbf839cf4b266e2cb8960` for all required parent artifacts. Saved null mappings exactly matched the regenerated `SeedSequence([20260810,1102])` and `SeedSequence([20260810,1101])` streams.

## Numerical decomposition gates

All 1,296 pairs passed raw nonnegativity and exact reconstruction at `atol=rtol=1e-8`. Raw minima: c_len=0.00053545571295, c_ang=0.166563296852, c_ori=0.210085583818, split_A_c_len=0.000675270515951, split_A_c_ang=0.267519412136, split_A_c_ori=0.471089876295, split_B_c_len=0.00101151212987, split_B_c_ang=0.265016979312, split_B_c_ori=0.415177577792. No meaningful negative value was clipped. The maximum recorded reconstruction/reproduction absolute error was `2.88657986403e-15`.

The saved root distances reproduced their squared costs (`c_full=d_mov²`, `c_len=d_len²`, `c_sensor=d_direct²`), while `c_len` and `c_sensor` were also independently rebuilt from the frozen Z tuples. Every anchor, subject, group statistic, and indexed null draw passed `full=len+ang` and `sensor=len+ang+ori` reconstruction.

## Squared-cost inference

| component | T_subject | p_subject | T_class | p_class | T_J | p_J_subject | p_J_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| len | 0.2009845035 | 0.0005 | 0.1366499427 | 0.0005 | 0.1234538087 | 0.0035 | 0.0005 |
| ang | 0.3091561772 | 0.0005 | 0.393098434 | 0.0005 | 0.1924088545 | 0.001 | 0.0105 |
| ori | 0.2744207204 | 0.0005 | 0.2094600677 | 0.0005 | 0.1513437529 | 0.0005 | 0.0005 |
| full | 0.5101406807 | 0.0005 | 0.5297483767 | 0.0005 | 0.3158626632 | 0.0005 | 0.0005 |
| sensor | 0.7845614012 | 0.0005 | 0.7392084443 | 0.0005 | 0.4672064161 | 0.0005 | 0.0005 |

The primary test is the `ang` row. It is the length-weighted, common-O(22)-invariant directional/joint-matrix component after exact removal of ordered speed. The `ori` row is secondary common-frame/simultaneous-conjugation-sensitive localization evidence.

### Subject-level J values

- `J_s_len`: S1=0.0261708254, S2=0.0487648659, S3=0.0790681967, S4=0.1307172059, S5=-0.0029331512, S6=0.0301508605, S7=0.0011740877, S8=0.0391723386, S9=0.7587990489
- `J_s_ang`: S1=0.2422969576, S2=-0.0240685391, S3=0.1463734737, S4=0.0142566235, S5=-0.0521695755, S6=-0.0528123935, S7=0.0027626534, S8=0.0231792348, S9=1.4318612558
- `J_s_ori`: S1=0.2350744759, S2=0.0499030799, S3=0.0802242847, S4=0.1676937544, S5=0.0353349675, S6=0.0141895623, S7=0.0881336438, S8=0.2564240632, S9=0.4351159445

### Exact T_J reconstruction

- `T_J_full - T_J_len - T_J_ang = 5.5511151231257827e-17`
- `T_J_sensor - T_J_len - T_J_ang - T_J_ori = 1.1102230246251565e-16`

Subject and class component summaries, including all nine `S_s` and `C_s` values, are saved in `tables/component_subject_stats.csv`; anchor-level values are in `tables/component_relation_cells.csv`.

## Split-half angular stability

- Half A cross-session `T_J_ang=0.2403025884`; `J_s_ang`: S1=0.1277624424; S2=-0.0553793144; S3=0.1207564857; S4=0.0717970458; S5=-0.0381375536; S6=0.0100265093; S7=-0.0378157851; S8=0.1475193048; S9=1.8161941609
- Half B cross-session `T_J_ang=0.1691101140`; `J_s_ang`: S1=0.2979587814; S2=-0.0436750664; S3=0.1499638782; S4=-0.0381164197; S5=-0.0517922058; S6=-0.0568832325; S7=0.0172555475; S8=0.0280455119; S9=1.2192342311
- Prespecified sign stability: `True`

Each replicate used its matching independent frozen half in both sessions and the exact frozen Movement V0 optimizer. No half-specific p-value was required.

## Fixed illustrative decompositions

| comparison | c_len | c_ang | c_ori | c_full | c_sensor | fraction_len | fraction_ang | fraction_ori |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 Left0 -> S1 Left1 | 0.007444560287 | 0.2123405327 | 0.2857450959 | 0.219785093 | 0.5055301889 | 0.01472624277 | 0.4200353161 | 0.5652384411 |
| S1 Left0 -> S2 Left1 | 0.02032026797 | 0.8324201524 | 0.474395063 | 0.8527404203 | 1.327135483 | 0.01531137417 | 0.6272307257 | 0.3574579001 |
| S1 Left0 -> S1 Feet1 | 0.06768795277 | 0.7530319655 | 0.6907015363 | 0.8207199183 | 1.511421455 | 0.04478430061 | 0.498227654 | 0.4569880454 |
| S1 Left0 -> S2 Feet1 | 0.2074991285 | 1.316539646 | 0.7204054158 | 1.524038774 | 2.24444419 | 0.09245011723 | 0.5865771363 | 0.3209727464 |

Fractions are descriptive and were computed only where `c_sensor>1e-12`.

## Step-level descriptive localization

Step-norm distributions over all 72 frozen sequences:

| transition | value | count | mean | std | minimum | q25 | median | q75 | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1->2 | step_norm | 72 | 1.38883004 | 0.6367169573 | 0.586092256 | 0.9752965791 | 1.231029133 | 1.548532153 | 3.992466419 |
| 2->3 | step_norm | 72 | 0.7133011495 | 0.1768280882 | 0.4368918951 | 0.5882994041 | 0.6789875715 | 0.7795702687 | 1.389686873 |
| 3->4 | step_norm | 72 | 0.6972680616 | 0.2557939899 | 0.4453968073 | 0.5617459799 | 0.6392551917 | 0.7820007497 | 2.363032254 |
| 4->5 | step_norm | 72 | 0.63260418 | 0.2470937399 | 0.4555728996 | 0.5185047107 | 0.5776515848 | 0.652997658 | 2.154495759 |

Per-step `c_len` contributions over all 1,296 cross-session pairs:

| transition | value | count | mean | std | minimum | q25 | median | q75 | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1->2 | c_len_step | 1296 | 0.2015006975 | 0.3709163442 | 9.194012029e-08 | 0.01156666184 | 0.05250189309 | 0.219789476 | 2.866296509 |
| 2->3 | c_len_step | 1296 | 0.01564799563 | 0.02867468041 | 3.987659336e-09 | 0.001036208218 | 0.004714820104 | 0.01640560497 | 0.2269545675 |
| 3->4 | c_len_step | 1296 | 0.03373971357 | 0.12343232 | 1.544866664e-09 | 0.001289823818 | 0.005651886211 | 0.02033770925 | 0.9193314263 |
| 4->5 | c_len_step | 1296 | 0.03019626267 | 0.1085077595 | 8.266510054e-09 | 0.000524358599 | 0.002377078956 | 0.008774221672 | 0.7215847208 |

Per-step angular contributions were evaluated only for the four fixed comparisons after deterministic reproduction of their frozen optimal objectives:

| comparison | transition | c_len_step | c_ang_step | c_full_step |
| --- | --- | --- | --- | --- |
| S1 Left0 -> S1 Left1 | 1->2 | 0.00144239884 | 0.04397928451 | 0.04542168335 |
| S1 Left0 -> S1 Left1 | 2->3 | 0.005288988186 | 0.06549369392 | 0.07078268211 |
| S1 Left0 -> S1 Left1 | 3->4 | 1.09353943e-06 | 0.05641664767 | 0.05641774121 |
| S1 Left0 -> S1 Left1 | 4->5 | 0.0007120797213 | 0.04645090658 | 0.0471629863 |
| S1 Left0 -> S2 Left1 | 1->2 | 0.01239137215 | 0.7049368846 | 0.7173282567 |
| S1 Left0 -> S2 Left1 | 2->3 | 0.002595380636 | 0.03418838637 | 0.036783767 |
| S1 Left0 -> S2 Left1 | 3->4 | 0.004338710974 | 0.05048143756 | 0.05482014853 |
| S1 Left0 -> S2 Left1 | 4->5 | 0.0009948042025 | 0.04281344386 | 0.04380824806 |
| S1 Left0 -> S1 Feet1 | 1->2 | 0.00968847797 | 0.5198887141 | 0.5295771921 |
| S1 Left0 -> S1 Feet1 | 2->3 | 0.004192836966 | 0.06168827256 | 0.06588110953 |
| S1 Left0 -> S1 Feet1 | 3->4 | 4.761595962e-05 | 0.06917828174 | 0.0692258977 |
| S1 Left0 -> S1 Feet1 | 4->5 | 0.05375902187 | 0.1022766971 | 0.1560357189 |
| S1 Left0 -> S2 Feet1 | 1->2 | 0.2065299303 | 1.044323044 | 1.250852975 |
| S1 Left0 -> S2 Feet1 | 2->3 | 1.897964348e-05 | 0.05843864517 | 0.05845762481 |
| S1 Left0 -> S2 Feet1 | 3->4 | 0.0001345722899 | 0.1494585811 | 0.1495931534 |
| S1 Left0 -> S2 Feet1 | 4->5 | 0.0008156462481 | 0.06431937479 | 0.06513502104 |

These are descriptive diagnostics, not post-hoc transition tests.

## Terminal

`BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS`

The experiment concerns BNCI2014_001 window-wise mean covariance movement, its discrete anti-development, and the fixed 5 × 0.8-s discretization. Neither `c_ang` nor `c_ori` is interpreted as neural, physiological, source-space, anatomical, or subject pose information.

## Runtime, tests, and immutability

- Total scientific runtime: `267.895 s`
- Split-half quotient runtime: `259.715 s`
- Focused tests before execution: `16 passed in 1.97s`
- Full repository before execution: `270 passed, 4 skipped, 1 failed in 27.72s; sole failure was tests/test_local_metric_data_v0.py because the isolated worktree lacks cache/bnci2014_001_trajectory_within_subject_v1/combined_trajectory_features.npz`
- Focused tests after execution: `16 passed in 2.04s`
- Full repository after execution: `270 passed, 4 skipped, 1 failed in 28.09s; sole failure was tests/test_local_metric_data_v0.py because the isolated worktree lacks cache/bnci2014_001_trajectory_within_subject_v1/combined_trajectory_features.npz`
- Git status: clean at scientific-run start and required clean before finalization; final hand-off status is reported with the committed result.
- No scientific setting changed after first component-statistic access: `true`.

The full pair costs, nulls, relation tables, split-half matrices, reconstruction diagnostics, figures, provenance, and hashes are retained under the frozen output namespace.
