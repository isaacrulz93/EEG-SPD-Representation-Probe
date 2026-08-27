# Cross-Session Subject Location → Conditional Configuration Prediction V0

Terminal: **STOP_LOCATION_DOES_NOT_PREDICT_CONDITIONAL_CONFIGURATION**

This deterministic held-out-subject analysis uses only the locked PR #19 Stieger2021 task-3 full-split compact geometry. No raw EEG was opened, no classifier or neural network was trained, and no TTA was performed.

## Exact objects

- Subject reference: `R_s = FM_AIRM(M_s^(2), M_s^(3))`.
- Fold source reference: `M_0 = FM_AIRM({R_s : s in S_train})` with one vote per source subject.
- Label-free input: `q_s^(u) = svec(log(M_0^(-1/2) M_s^(u) M_0^(-1/2)))`.
- Equal-class centered configuration: `d_s,c^(v) = z_s,c^(v) - (1/4) sum_j z_s,j^(v)`.
- Source-centered target: `Delta_s,c^(v) = d_s,c^(v) - mean_source d_s,c^(v)`.

## Primary held-out results

| scope | R2_cond | bootstrap_95_CI | mean_error_gain | positive | sign_flip_p |
|---|---|---|---|---|---|
| FORWARD | -0.004976 | [-0.011017, -0.000610] | -0.00816993 | 16/62 | 0.031790 |
| REVERSE | -0.000122 | [-0.001533, 0.001345] | -0.000154727 | 13/62 | 0.869830 |
| POOLED | -0.002862 | [-0.006014, -0.000365] | -0.00416233 | 17/62 | 0.034020 |

The pooled leave-one-subject R² range is [-0.003547, -0.002329]. The subject-direction positive-gain count is 29/124.

## Selected source-only models

| direction | fold | selected_rank | selected_ridge_multiplier | ridge_lambda | output_numerical_rank |
|---|---|---|---|---|---|
| FORWARD | 1 | 1 | 1000.0 | 10854.420264016626 | 51 |
| FORWARD | 2 | 0 | 1000.0 | 11147.177490535216 | 51 |
| FORWARD | 3 | 1 | 10.0 | 116.16912557884255 | 51 |
| FORWARD | 4 | 2 | 100.0 | 1168.6474822308094 | 50 |
| FORWARD | 5 | 0 | 1000.0 | 11792.777728207398 | 50 |
| FORWARD | 6 | 3 | 100.0 | 1211.2064962491008 | 51 |
| REVERSE | 1 | 0 | 1000.0 | 12368.938175757141 | 51 |
| REVERSE | 2 | 0 | 1000.0 | 13453.347094625326 | 51 |
| REVERSE | 3 | 0 | 1000.0 | 13307.541270643931 | 51 |
| REVERSE | 4 | 1 | 100.0 | 1282.1587052788523 | 50 |
| REVERSE | 5 | 0 | 1000.0 | 13359.593228750642 | 50 |
| REVERSE | 6 | 2 | 100.0 | 1410.6878982392072 | 51 |

## Required nulls

Source-pairing null:

| scope | observed_r2 | null_mean_r2 | null_95_low | null_95_high | null_p_value | replicates |
|---|---|---|---|---|---|---|
| FORWARD | -0.0049764953154465275 | -0.0006477174872424407 | -0.011919690668057936 | 0.008147312294045117 | 0.797 | 1999 |
| REVERSE | -0.0001221172692338346 | -8.454313385436205e-05 | -0.0009437831504604421 | 0.0007240054685703597 | 0.5445 | 1999 |
| POOLED | -0.0028619480409137754 | -0.0004024010362559892 | -0.006835360958875469 | 0.004554762492574349 | 0.796 | 1999 |

Held-out target-location derangement null:

| scope | observed_r2 | null_mean_r2 | null_95_low | null_95_high | null_p_value | replicates |
|---|---|---|---|---|---|---|
| FORWARD | -0.0049764953154465275 | 0.0005449674431832318 | -0.003643246936971123 | 0.0044764538648484 | 0.9955 | 1999 |
| REVERSE | -0.0001221172692338346 | -0.0004985580099375388 | -0.0022539002840937018 | 0.00132090586047095 | 0.3465 | 1999 |
| POOLED | -0.0028619480409137754 | 9.041201264285944e-05 | -0.002324926058552412 | 0.002363943658905232 | 0.9885 | 1999 |

## Non-voting controls

- Location-norm-only: {"FORWARD": {"mean_error_gain": -4.887530789673071e-05, "positive_gain_count": 15, "r2_cond": -2.977108237001147e-05, "subject_count": 62}, "REVERSE": {"mean_error_gain": -0.00019736036127566834, "positive_gain_count": 9, "r2_cond": -0.00015576554160068312, "subject_count": 62}}
- Same-session: {"SAME_SESSION_2": {"mean_error_gain": -0.015033893711088441, "positive_gain_count": 7, "r2_cond": -0.011865415026290016, "subject_count": 62}, "SAME_SESSION_3": {"mean_error_gain": -0.036582459735201855, "positive_gain_count": 22, "r2_cond": -0.02228322375735514, "subject_count": 62}}
- Post-hoc all-subject reference was never used for prediction or nulls. Maximum normalized fold-reference shift: 0.125648.
- Descriptive q subject identification: session2→3 0.371, session3→2 0.355. This is not evidence for conditional prediction.
- q repeatability vs paired prediction gain Spearman rho: 0.185471 (p=0.148955).

## Terminal gates

| gate | passed |
|---|---|
| pooled_r2_at_least_0_05 | False |
| forward_r2_positive | False |
| reverse_r2_positive | False |
| pooled_bootstrap_lower_positive | False |
| source_pairing_null_p_at_most_0_05 | False |
| subject_gain_sign_flip_p_at_most_0_05 | True |
| positive_subject_direction_fraction_at_least_0_60 | False |
| leave_one_subject_minimum_positive | False |
| all_engineering_gates | True |

## Exact next question

> What target-observable statistic other than global subject location is required to identify unseen-subject class-conditional deformation?

The next question is recorded only. No follow-up architecture or experiment was executed.
