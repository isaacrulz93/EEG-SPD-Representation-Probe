# D/G Discrepancy Decomposition v1

## Status and scope

This is a retrospective/post hoc mechanism anatomy, not a new method and not external confirmation. Session `0train` is retrospective mechanism set A; session `1test` is locked internal replication set B. The original `STOP_TANGENT_ONLY` decision and all Conditional Geometry Anatomy v1 artifacts remain untouched.

D-/G+ must not be reinterpreted as tangent superiority unless matched-object evidence supports that interpretation. This diagnostic introduces no classifier, alignment loss, pseudo-label, neural network, trajectory, HGD, or WINDOW5 analysis. Oracle semantic-name scoring remains descriptive; oracle component recovery is not solved.

Base commit: `74bbae09fd9880c5809b61769d8f6e27b9b994bf`. Diagnostic implementation commit: `76c81f25c0340a2ddd3eadf1503db6896452b832`.

## Algebraic controls

All stored-object identity gates passed. LE satisfied `D_exact²=D_tan²` and `K_exact=G0=G` within 1e-10. AIRM curvature residuals satisfied the exponential-metric-increasing inequality within the frozen tolerance. K_exact was retained signed and was never PSD-clipped.

## Stage R mechanism prerequisite

Observed A/B reliability was computed for all five objects. Exact v1 D/G label-null summaries were reproduced. The immutable source does not store trial covariances/metadata or per-replicate fitted D/G objects, so the derived-object label nulls cannot be exactly computed from the authorized source. They are explicitly marked `NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE`; no proxy or approximation was substituted, and Stage R does not vote in the mechanism label.

| session | geometry | object | observed | null_median | effect | p_value | replicates | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0train | AIRM | D_exact | 0.992253 | 0.989483 | 0.0027699 | 0.2005 | 1999 | EXACT_V1_REGRESSION |
| 0train | AIRM | D_tan | 0.992246 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 0train | AIRM | K_exact | 0.943214 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 0train | AIRM | G0 | 0.943223 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 0train | AIRM | G | 0.943223 | 0.905212 | 0.0380107 | 0.08 | 1999 | EXACT_V1_REGRESSION |
| 0train | LE | D_exact | 0.991171 | 0.986343 | 0.00482766 | 0.119 | 1999 | EXACT_V1_REGRESSION |
| 0train | LE | D_tan | 0.991171 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 0train | LE | K_exact | 0.945734 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 0train | LE | G0 | 0.945734 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 0train | LE | G | 0.945734 | 0.879744 | 0.0659893 | 0.0095 | 1999 | EXACT_V1_REGRESSION |
| 1test | AIRM | D_exact | 0.995706 | 0.986721 | 0.00898584 | 0.002 | 1999 | EXACT_V1_REGRESSION |
| 1test | AIRM | D_tan | 0.995702 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 1test | AIRM | K_exact | 0.973035 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 1test | AIRM | G0 | 0.973024 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 1test | AIRM | G | 0.973024 | 0.88238 | 0.0906444 | 0.0005 | 1999 | EXACT_V1_REGRESSION |
| 1test | LE | D_exact | 0.994122 | 0.982696 | 0.0114253 | 0.001 | 1999 | EXACT_V1_REGRESSION |
| 1test | LE | D_tan | 0.994122 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 1test | LE | K_exact | 0.96628 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 1test | LE | G0 | 0.96628 | NA | NA | NA | 0 | NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE |
| 1test | LE | G | 0.96628 | 0.854339 | 0.111941 | 0.0005 | 1999 | EXACT_V1_REGRESSION |

## Stage S null-referenced effects

| session | geometry | object | observed | null_median | effect | p_value | exceedances | replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0train | AIRM | D_exact | 0.97952 | 0.963953 | 0.0155672 | 0.00512995 | 512 | 100000 |
| 0train | AIRM | D_tan | 0.979513 | 0.963925 | 0.0155882 | 0.00509995 | 509 | 100000 |
| 0train | AIRM | K_exact | 0.873479 | 0.770336 | 0.103144 | 0.00417996 | 417 | 100000 |
| 0train | AIRM | G0 | 0.873476 | 0.770215 | 0.10326 | 0.00415996 | 415 | 100000 |
| 0train | AIRM | G | 0.873475 | 0.770215 | 0.10326 | 0.00415996 | 415 | 100000 |
| 0train | LE | D_exact | 0.978339 | 0.957696 | 0.0206431 | 0.00297997 | 297 | 100000 |
| 0train | LE | D_tan | 0.978339 | 0.957696 | 0.0206431 | 0.00297997 | 297 | 100000 |
| 0train | LE | K_exact | 0.866875 | 0.741495 | 0.12538 | 0.00366996 | 366 | 100000 |
| 0train | LE | G0 | 0.866875 | 0.741495 | 0.12538 | 0.00366996 | 366 | 100000 |
| 0train | LE | G | 0.866875 | 0.741495 | 0.12538 | 0.00366996 | 366 | 100000 |
| 1test | AIRM | D_exact | 0.967946 | 0.964071 | 0.00387566 | 0.320287 | 32028 | 100000 |
| 1test | AIRM | D_tan | 0.967923 | 0.964043 | 0.00387992 | 0.320257 | 32025 | 100000 |
| 1test | AIRM | K_exact | 0.853998 | 0.752732 | 0.101266 | 0.0179898 | 1798 | 100000 |
| 1test | AIRM | G0 | 0.853932 | 0.752596 | 0.101336 | 0.0179798 | 1797 | 100000 |
| 1test | AIRM | G | 0.853931 | 0.752596 | 0.101335 | 0.0179798 | 1797 | 100000 |
| 1test | LE | D_exact | 0.962312 | 0.956343 | 0.00596912 | 0.263737 | 26373 | 100000 |
| 1test | LE | D_tan | 0.962312 | 0.956343 | 0.00596912 | 0.263737 | 26373 | 100000 |
| 1test | LE | K_exact | 0.853088 | 0.723234 | 0.129854 | 0.00441996 | 441 | 100000 |
| 1test | LE | G0 | 0.853088 | 0.723234 | 0.129854 | 0.00441996 | 441 | 100000 |
| 1test | LE | G | 0.853088 | 0.723234 | 0.129854 | 0.00441996 | 441 | 100000 |

## Mechanism contrasts

| session | geometry | Delta_curvature | Delta_anchor | Delta_encoding_exact | Delta_encoding_tangent |
| --- | --- | --- | --- | --- | --- |
| 0train | AIRM | 2.10679e-05 | -2.15199e-07 | 0.0875765 | 0.0876723 |
| 0train | LE | -1.11022e-16 | -1.11022e-16 | 0.104737 | 0.104737 |
| 1test | AIRM | 4.26584e-06 | -4.9714e-07 | 0.09739 | 0.0974557 |
| 1test | LE | 1.11022e-16 | -1.11022e-16 | 0.123885 | 0.123885 |

## Provisional decision

- Session 0 label: **PROVISIONAL_MIXED**
- Session 1 label: **PROVISIONAL_MIXED**
- Overall: **PROVISIONAL_MIXED**

The matched encoding contrasts are large and direction-replicated in both AIRM and the exact-information LE control, whereas the anchor contrasts are nonpositive and numerically negligible. AIRM curvature contrasts are also positive in both sessions but are orders of magnitude smaller than the encoding contrasts; because the frozen rules forbid adding a post hoc materiality cutoff, the replicated encoding-plus-curvature sign pattern receives the mixed label.

The label is provisional and uses algebraic matching plus replicated contrast direction, not a new post hoc absolute cutoff and not a single p-value. It is not a scientific GO before external confirmation.

## Descriptive distortion summaries

AIRM curvature (F split, median across subjects):

| session | median_C | max_C |
| --- | --- | --- |
| 0train | 0.00037726 | 0.000870441 |
| 1test | 0.000274815 | 0.000550017 |

Anchor fraction (F split, median across subjects):

| session | geometry | anchor_fraction |
| --- | --- | --- |
| 0train | AIRM | 1.69933e-06 |
| 0train | LE | 8.82032e-18 |
| 1test | AIRM | 1.82988e-06 |
| 1test | LE | -1.71673e-18 |

## Limits

The analysis decomposes the stored representation/statistic discrepancy only. It does not establish a biological mechanism, a deployable adaptation method, or unlabeled component recovery. WINDOW5 remains outside this diagnostic.
