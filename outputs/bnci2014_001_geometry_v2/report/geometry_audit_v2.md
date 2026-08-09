# BNCI2014_001 WHOLE-SPD Geometry Audit V2

## Motivation

This audit asks whether V1's subject-marginal centering effect is robust to covariance geometry and to a held-out-run target-center protocol. It evaluates WHOLE covariance only; it proposes no classifier or alignment method.

## Frozen protocol

Protocol version `2.0`, seed `20260809`, config SHA-256 `ddb96fc11a77a8640996755b25031d8c2c4ed0b7d20c4e59b6549b29303c6914`. BNCI2014_001 session `0train` contains 2592 validated WHOLE trials (9 subjects; [288] evaluation trials per subject), 22 EEG channels, and four classes. Each LOSO target has source count(s) [2304]. Frozen preprocessing is [8.0, 32.0] Hz, 250.0 Hz, 1000 samples, OAS covariance, float64 covariance geometry, no scaler/PCA/tuning. T1 fits each centered target mean on all 288 target covariates and evaluates those same trials. T2 fits on three runs and evaluates the disjoint other three, with A/B reversed splits; primary T2 rows are the preregistered `AGGREGATE` rows.

## Geometry definitions

G0 `RAW` uses each covariance unchanged. G1 `LE` subtracts the subject Log-Euclidean mean in log coordinates and is V1-svec equivalent. G2 `AIRM` uses the affine-invariant Fréchet mean and congruence whitening. G3 `EA` uses arithmetic-mean congruence only and is not interpreted as a Riemannian method. The common primary decoder is unscaled log-svec multinomial logistic regression (`C=1`, `lbfgs`, `max_iter=5000`, `tol=1e-4`).

## Geometry correctness

The classification hard gate was all-pass: 2331/2331 required rows passed and 0 failed. The correctness table contains 2358 total checks. Classification/reporting would hard-stop for a missing, malformed, provenance-mismatched, or non-PASS gate. [Correctness table](../tables/geometry_correctness.csv).

## LE vs AIRM Fréchet means

| T1 normalized center quantity | Mean | SD | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| dLE(LE,AIRM) / LE dispersion | 0.1053 | 0.0167 | 0.0986 | 0.0867 | 0.1320 |
| dAI(LE,AIRM) / AIRM dispersion | 0.0906 | 0.0136 | 0.0894 | 0.0751 | 0.1120 |
| Mean transformed-coordinate L2 difference | 0.6917 | 0.1061 | 0.6817 | 0.5671 | 0.8894 |

These describe measured mean differences; they do not identify one geometry as correct. [Figure 4](../figures/figure_4_le_vs_airm_centers.png) and [source CSV](../figures/figure_4_le_vs_airm_centers.csv).

## V1 leakage audit

| Condition | Pooled BA | Pooled accuracy | Pooled macro-F1 | Center/eval overlap | Accuracy - 0.6119 |
| --- | --- | --- | --- | --- | --- |
| v1_all_sample | 0.6119 | 0.6119 | 0.6118 | 2592 | -0.0000 |
| fold_safe | 0.6069 | 0.6069 | 0.6067 | 0 | -0.0050 |

The observed fold-safe minus all-sample differences were -0.0050 BA and -0.0050 accuracy. The published V1 value `0.6119` is an audit benchmark, not a threshold or forced reproduction. All-sample centering has evaluation-covariate overlap; fold-safe centering does not. [Figure 5](../figures/figure_5_v1_leakage_audit.png) and [source CSV](../figures/figure_5_v1_leakage_audit.csv).

## LOSO transductive results

| Geometry | Native metric | PASS subjects | BA mean ± SD | Accuracy mean ± SD | Macro-F1 mean ± SD | BA median [min, max] |
| --- | --- | --- | --- | --- | --- | --- |
| AIRM | euclidean_log_svec | 9/9 | 0.5386 ± 0.1568 | 0.5386 ± 0.1568 | 0.5365 ± 0.1554 | 0.5000 [0.3438, 0.7326] |
| EA | euclidean_log_svec | 9/9 | 0.5340 ± 0.1551 | 0.5340 ± 0.1551 | 0.5299 ± 0.1563 | 0.5035 [0.3368, 0.7292] |
| LE | euclidean_log_svec | 9/9 | 0.5374 ± 0.1557 | 0.5374 ± 0.1557 | 0.5356 ± 0.1554 | 0.5035 [0.3472, 0.7535] |
| RAW | euclidean_log_svec | 8/9 | 0.3494 ± 0.0959 | 0.3494 ± 0.0959 | 0.2766 ± 0.1252 | 0.3247 [0.2465, 0.4861] |

Aggregates use PASS target-subject rows only, show the denominator, and use `ddof=1` for SD. T1 centered target means use the same 288 unlabeled target covariates that are evaluated; this is transductive label-free target centering, not inductive evaluation. [Figure 1](../figures/figure_1_loso_ba_by_subject.png), [paired deltas](../figures/figure_2_paired_delta_vs_raw.png).

Primary logistic failures: 4 row(s).

| Subject | Geometry | Protocol | Split |
| --- | --- | --- | --- |
| 4 | RAW | T1 | ALL |
| 4 | RAW | T2 | A |
| 4 | RAW | T2 | AGGREGATE |
| 4 | RAW | T2 | B |

Unique captured warning(s):

- `["lbfgs failed to converge after 5000 iteration(s) (status=1):\nSTOP: TOTAL NO. OF ITERATIONS REACHED LIMIT\n\nIncrease the number of iterations to improve the convergence (max_iter=5000).\nYou might also want to scale the data as shown in:\n    https://scikit-learn.org/stable/modules/preprocessing.html\nPlease also refer to the documentation for alternative solver options:\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"]`

PASS-only descriptive aggregates above are incomplete and do not support a frozen verdict.

## Calibration-to-held-out-run results

| Geometry | Native metric | PASS subjects | BA mean ± SD | Accuracy mean ± SD | Macro-F1 mean ± SD | BA median [min, max] |
| --- | --- | --- | --- | --- | --- | --- |
| AIRM | euclidean_log_svec | 9/9 | 0.5320 ± 0.1509 | 0.5320 ± 0.1509 | 0.5300 ± 0.1499 | 0.4965 [0.3576, 0.7326] |
| EA | euclidean_log_svec | 9/9 | 0.5216 ± 0.1546 | 0.5216 ± 0.1546 | 0.5164 ± 0.1568 | 0.4931 [0.3368, 0.7222] |
| LE | euclidean_log_svec | 9/9 | 0.5293 ± 0.1586 | 0.5293 ± 0.1586 | 0.5275 ± 0.1581 | 0.5069 [0.3403, 0.7361] |
| RAW | euclidean_log_svec | 8/9 | 0.3494 ± 0.0959 | 0.3494 ± 0.0959 | 0.2766 ± 0.1252 | 0.3247 [0.2465, 0.4861] |

Each subject-level primary row pools the two deterministic A/B held-out-run evaluations. Target center-fit and evaluation trial UIDs are disjoint within each split; source data and decoder are identical to T1. [Figure 3](../figures/figure_3_t1_vs_t2_ba.png) and [source CSV](../figures/figure_3_t1_vs_t2_ba.csv).

PASS-only descriptive aggregates are shown with denominators; no 8/9 available-case verdict is permitted.

## Metric-native MDM sanity check

T1 metric-native MDM:

| Geometry | Native metric | PASS subjects | BA mean ± SD | Accuracy mean ± SD | Macro-F1 mean ± SD |
| --- | --- | --- | --- | --- | --- |
| RAW | riemann | 9/9 | 0.3148 ± 0.0978 | 0.3148 ± 0.0978 | 0.1945 ± 0.1364 |
| RAW | logeuclid | 9/9 | 0.3071 ± 0.0872 | 0.3071 ± 0.0872 | 0.1857 ± 0.1240 |
| LE | logeuclid | 9/9 | 0.4603 ± 0.1323 | 0.4603 ± 0.1323 | 0.4462 ± 0.1220 |
| AIRM | riemann | 9/9 | 0.4865 ± 0.1453 | 0.4865 ± 0.1453 | 0.4761 ± 0.1371 |
| EA | riemann | 9/9 | 0.4780 ± 0.1568 | 0.4780 ± 0.1568 | 0.4646 ± 0.1533 |

T2 metric-native MDM:

| Geometry | Native metric | PASS subjects | BA mean ± SD | Accuracy mean ± SD | Macro-F1 mean ± SD |
| --- | --- | --- | --- | --- | --- |
| RAW | riemann | 9/9 | 0.3148 ± 0.0978 | 0.3148 ± 0.0978 | 0.1945 ± 0.1364 |
| RAW | logeuclid | 9/9 | 0.3071 ± 0.0872 | 0.3071 ± 0.0872 | 0.1857 ± 0.1240 |
| LE | logeuclid | 9/9 | 0.4518 ± 0.1243 | 0.4518 ± 0.1243 | 0.4354 ± 0.1134 |
| AIRM | riemann | 9/9 | 0.4803 ± 0.1381 | 0.4803 ± 0.1381 | 0.4681 ± 0.1295 |
| EA | riemann | 9/9 | 0.4753 ± 0.1491 | 0.4753 ± 0.1491 | 0.4620 ± 0.1460 |

Secondary MDM inventory: 0 explicit FAILED or missing logical rows (none). MDM aggregates use PASS rows only and display their denominator. MDM is secondary and does not vote in Q1–Q3. RAW is intentionally evaluated under both `riemann` and `logeuclid`; EA remains an arithmetic-control transformation.

No explicit FAILED or missing secondary MDM rows were recorded.

## Marginal domain diagnostics

| Geometry | Reference metric | Mean source-target mean distance | SD | Mean absolute dispersion difference | Subject silhouette |
| --- | --- | --- | --- | --- | --- |
| AIRM | riemann | 0.0000 | 0.0000 | 0.2563 | -0.0601 |
| EA | arithmetic_frobenius | 0.0000 | 0.0000 | 0.2427 | -0.0555 |
| LE | logeuclid | 0.0000 | 0.0000 | 0.2144 | -0.0601 |
| RAW | logeuclid | 3.3747 | 1.5312 | 1.8520 | 0.2469 |
| RAW | riemann | 3.5835 | 1.5017 | 1.8475 | 0.2469 |

These T1 diagnostics use no class labels. They quantify marginal domain location/dispersion and subject structure; they do not establish conditional class alignment. [Domain table](../tables/domain_shift_diagnostics.csv).

## Frozen decision-rule verdicts

Paired-delta aggregates were not computed because a primary technical failure prohibits available-case verdicts.

| Question | Verdict | Measured operands |
| --- | --- | --- |
| Q1 | UNASSESSED — TECHNICAL FAILURE | {"available_case_verdict_prohibited":true,"calculation_status":"NOT_COMPUTED","failed_rows":[{"geometry":"RAW","protocol":"T1","source_table":"loso_logistic_transductive.csv","split":"ALL","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"A","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"AGGREGATE","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"B","subject":4}],"primary_failed_row_count":4,"reason":"one_or_more_primary_logistic_rows_failed","unique_warning_message_count":1} |
| Q2 | UNASSESSED — TECHNICAL FAILURE | {"available_case_verdict_prohibited":true,"calculation_status":"NOT_COMPUTED","failed_rows":[{"geometry":"RAW","protocol":"T1","source_table":"loso_logistic_transductive.csv","split":"ALL","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"A","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"AGGREGATE","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"B","subject":4}],"primary_failed_row_count":4,"reason":"one_or_more_primary_logistic_rows_failed","unique_warning_message_count":1} |
| Q3 | UNASSESSED — TECHNICAL FAILURE | {"available_case_verdict_prohibited":true,"calculation_status":"NOT_COMPUTED","failed_rows":[{"geometry":"RAW","protocol":"T1","source_table":"loso_logistic_transductive.csv","split":"ALL","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"A","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"AGGREGATE","subject":4},{"geometry":"RAW","protocol":"T2","source_table":"loso_logistic_calibration.csv","split":"B","subject":4}],"primary_failed_row_count":4,"reason":"one_or_more_primary_logistic_rows_failed","unique_warning_message_count":1} |

The frozen Q1–Q3 formulas were not evaluated. All three verdicts are the configured technical-failure verdict; failed rows were not imputed and no available-case threshold was used. [Machine-readable summary](../tables/geometry_v2_summary.csv).

## What is actually justified

No primary geometry-effect or protocol-sensitivity conclusion is justified: the complete primary grid contains at least one FAILED row.

PASS-only scores are descriptive diagnostics with explicit denominators, not an 8/9 substitute for the frozen nine-subject verdict.

## What is NOT justified

This audit does not justify a conditional-alignment method, a distribution classifier, a neural architecture, a temporal or trajectory model, or any WINDOW5 conclusion. It does not show that target-label-free conditional structure is identifiable, and it does not demonstrate domain-adaptation improvement. T1 is transductive, T2 uses fixed half-run calibration, MDM is secondary, and all results come from one dataset/session and one frozen preprocessing pipeline.

## Single recommended next experiment

Run exactly one next experiment: **preregistered numerical-convergence audit of the fixed unscaled logistic decoder**. Keep the present data, preprocessing, geometry definitions, decoder, seed, and reporting thresholds fixed; preregister that experiment in a new output namespace before reading its results.

### Figure and source index

- [figure_1_loso_ba_by_subject.png](../figures/figure_1_loso_ba_by_subject.png) ([source CSV](../figures/figure_1_loso_ba_by_subject.csv))
- [figure_2_paired_delta_vs_raw.png](../figures/figure_2_paired_delta_vs_raw.png) ([source CSV](../figures/figure_2_paired_delta_vs_raw.csv))
- [figure_3_t1_vs_t2_ba.png](../figures/figure_3_t1_vs_t2_ba.png) ([source CSV](../figures/figure_3_t1_vs_t2_ba.csv))
- [figure_4_le_vs_airm_centers.png](../figures/figure_4_le_vs_airm_centers.png) ([source CSV](../figures/figure_4_le_vs_airm_centers.csv))
- [figure_5_v1_leakage_audit.png](../figures/figure_5_v1_leakage_audit.png) ([source CSV](../figures/figure_5_v1_leakage_audit.csv))
