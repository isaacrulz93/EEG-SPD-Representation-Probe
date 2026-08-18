# Source-Referenced Conditional Residual V1

## Scope

Retrospective OpenBMI-only mechanistic follow-up stacked on PR #17 head `8346a3e0f731c80668bd7147a2fe0fd12da6b914`. PR #16/#17 artifacts were hash-verified and remained unchanged. This work does not recompute their terminals.

## Algebraic beta identity

- Decision: `BETA_REFERENCE_IDENTITY_VERIFIED`
- Maximum absolute error: `1.110223025e-16`
- Mean gamma: `0.2029814499`; session means `[0.20456627723612172, 0.20139662254542548]`
- Sign-change proportion after subtracting gamma: `0.5833333333`

## Source trial/prototype correction

- Decision: `SOURCE_REFERENCE_COORDINATE_CORRECTION_SUPPORTED`
- Mean MAE improvement over uncentered trial delta: `0.2043876827`
- 95% subject-bootstrap CI: `[0.2013922037672018, 0.20726531314569807]`
- Paired sign-flip p: `0.0005`
- Session improvements: `[0.2059938507427202, 0.2027815145998034]`
- Corrected full-trial metrics: `{"beta_sign_accuracy": 1.0, "calibration_intercept": 1.0194407474214646e-05, "calibration_slope": 0.9721793493023979, "mae": 0.004949059957768716, "normalized_mae": 0.03066159519008633, "pearson": 0.9999619178262147, "projected_prototype_reconstruction_error": 0.004949059957768716, "signed_r2": 0.9991045351871157, "spearman": 0.9998761515523927}`
- Gamma range: `[0.1795095403, 0.2134846572]`; mean-correction range: `[-0.007003985043, -0.005875998396]`

## Explicit source semantic ordering

- Decision: `SOURCE_SEMANTIC_ORDERING_SUPPORTED_RETROSPECTIVELY`
- Pooled/session target ordering accuracy: `0.9259259259` / `[0.9629629629629629, 0.8888888888888888]`
- 95% subject-bootstrap CI: `[0.8703703703703703, 0.9722222222222222]`
- Exact binomial p: `1.176941767e-21`
- Leave-one-training-subject source order stable: `True`
- Violating subjects: `[11, 15, 16, 27, 34, 35, 48]`

This is retrospective evidence under an explicit source-ordering assumption. The zero-label signed coordinate remains `NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION` without that assumption.

## Zero-label under the source-order assumption

- Label: `ZERO_LABEL_UNDER_SOURCE_ORDER_ASSUMPTION`
- Decision: `SOURCE_REFERENCED_ZERO_LABEL_RECOVERY_SUPPORTED`
- Metrics: `{"beta_sign_accuracy": 0.8425925925925926, "calibration_intercept": -0.0012852161502194676, "calibration_slope": 0.7066312213174429, "mae": 0.10064330353913416, "normalized_mae": 0.6235293688179897, "pearson": 0.8455615145332542, "projected_prototype_reconstruction_error": 0.10064330353913416, "signed_r2": 0.5917391091511364, "spearman": 0.6726475412995007}`
- MAE improvement over beta=0: `0.06076577929`, CI `[0.01436984229473751, 0.11243570888478602]`, p `0.0075`
- Beta-sign-accuracy CI: `[0.7592592592592593, 0.9166666666666666]`

## Corrected minimal anchor

- Decision: `SOURCE_REFERENCED_MINIMAL_ANCHOR_EFFICIENT`
- Selected budget: `2`
- Frozen budgets/subsamples: `[0,2,4,8,16,32]`, 200 inherited subsamples per positive budget

|   budget |   proposed_mae |   direct_mae |   mean_improvement_direct_minus_proposed |   improvement_session0 |   improvement_session1 |   improvement_ci_low |   improvement_ci_high |   paired_sign_flip_p |   beta_sign_accuracy |   beta_sign_accuracy_ci_low |   beta_sign_accuracy_ci_high |   semantic_sign_accuracy |   semantic_sign_accuracy_ci_low |   semantic_sign_accuracy_ci_high |   proposed_pearson |   proposed_spearman |   proposed_signed_r2 |   direct_pearson |   direct_spearman |   direct_signed_r2 | leave_one_subject_improvement_positive   | eligible   | passes   |
|---------:|---------------:|-------------:|-----------------------------------------:|-----------------------:|-----------------------:|---------------------:|----------------------:|---------------------:|---------------------:|----------------------------:|-----------------------------:|-------------------------:|--------------------------------:|---------------------------------:|-------------------:|--------------------:|---------------------:|-----------------:|------------------:|-------------------:|:-----------------------------------------|:-----------|:---------|
|        2 |       0.123658 |    0.182072  |                                0.0584145 |              0.0548089 |              0.0620201 |            0.03864   |            0.077214   |               0.0005 |             0.865185 |                    0.803332 |                     0.920186 |                 0.750046 |                        0.708519 |                         0.79139  |           0.966195 |            0.810039 |             0.848921 |         0.997587 |          0.990978 |           0.99382  | True                                     | True       | True     |
|        4 |       0.114977 |    0.131381  |                                0.0164048 |              0.0136647 |              0.0191449 |           -0.0050283 |            0.0361508  |               0.0515 |             0.872037 |                    0.807175 |                     0.927826 |                 0.79412  |                        0.753656 |                         0.834678 |           0.962198 |            0.804931 |             0.851245 |         0.997837 |          0.993598 |           0.994438 | True                                     | True       | False    |
|        8 |       0.108961 |    0.0911426 |                               -0.0178185 |             -0.0228822 |             -0.0127548 |           -0.0399282 |            0.00284683 |               0.9425 |             0.873148 |                    0.806851 |                     0.932779 |                 0.843148 |                        0.804907 |                         0.880834 |           0.953003 |            0.808915 |             0.840445 |         0.999085 |          0.996809 |           0.996836 | False                                    | True       | False    |
|       16 |       0.104769 |    0.0619547 |                               -0.0428145 |             -0.0482195 |             -0.0374095 |           -0.0656542 |           -0.0225929  |               1      |             0.868333 |                    0.797638 |                     0.931806 |                 0.892315 |                        0.858148 |                         0.924398 |           0.933578 |            0.771648 |             0.800904 |         0.999555 |          0.9986   |           0.998404 | False                                    | False      | False    |
|       32 |       0.102271 |    0.0393677 |                               -0.0629038 |             -0.067672  |             -0.0581356 |           -0.0858914 |           -0.0428387  |               1      |             0.86463  |                    0.791434 |                     0.930187 |                 0.931157 |                        0.901481 |                         0.95764  |           0.918599 |            0.765319 |             0.761247 |         0.999792 |          0.999543 |           0.998692 | False                                    | False      | False    |

The direct baseline is fair: every direct calibration contrast uses the same source correction and gamma subtraction. Semantic-delta sign and beta sign are reported separately.

## Non-voting oracle bottleneck decomposition

| oracle                                                 | budget       |        mae |   normalized_mae |   pearson |   spearman |   signed_r2 |   beta_sign_accuracy |   calibration_slope |   calibration_intercept |   projected_prototype_reconstruction_error |
|:-------------------------------------------------------|:-------------|-----------:|-----------------:|----------:|-----------:|------------:|---------------------:|--------------------:|------------------------:|-------------------------------------------:|
| A_ORACLE_SIGN_ESTIMATED_MAGNITUDE                      | FULL_OR_ZERO | 0.0998287  |      0.618483    |  0.861143 |   0.71162  |    0.606917 |             0.851852 |            0.702395 |             0.00464675  |                                 0.0998287  |
| C_ORACLE_DELTA_SOURCE_REFERENCE                        | FULL_OR_ZERO | 5.4692e-18 |      3.38841e-17 |  1        |   1        |    1        |             1        |            1        |             6.00926e-18 |                                 5.4692e-18 |
| E_FULL_TRIAL_DELTA_SOURCE_REFERENCE                    | FULL_OR_ZERO | 0.00494906 |      0.0306616   |  0.999962 |   0.999876 |    0.999105 |             1        |            0.972179 |             1.01944e-05 |                                 0.00494906 |
| B_ESTIMATED_SIGN_ORACLE_ABSOLUTE_DELTA                 | 2            | 0.0473042  |      0.29307     |  0.989818 |   0.985195 |    0.939075 |             0.898148 |            0.942996 |             0.0399742   |                                 0.0473042  |
| D_ESTIMATED_DELTA_ORACLE_GAMMA_NO_CURVATURE_CORRECTION | 2            | 0.0643298  |      0.398551    |  0.966124 |   0.809315 |    0.862681 |             0.916667 |            0.868229 |             0.0420578   |                                 0.0643298  |
| B_ESTIMATED_SIGN_ORACLE_ABSOLUTE_DELTA                 | 4            | 0.0347905  |      0.215542    |  0.99298  |   0.988758 |    0.9638   |             0.935185 |            0.931581 |             0.0265055   |                                 0.0347905  |
| D_ESTIMATED_DELTA_ORACLE_GAMMA_NO_CURVATURE_CORRECTION | 4            | 0.0650339  |      0.402914    |  0.962138 |   0.804873 |    0.862143 |             0.898148 |            0.839514 |             0.0313174   |                                 0.0650339  |
| B_ESTIMATED_SIGN_ORACLE_ABSOLUTE_DELTA                 | 8            | 0.0230761  |      0.142967    |  0.996438 |   0.993293 |    0.983335 |             0.972222 |            0.94214  |             0.0156359   |                                 0.0230761  |
| D_ESTIMATED_DELTA_ORACLE_GAMMA_NO_CURVATURE_CORRECTION | 8            | 0.068846   |      0.426531    |  0.952948 |   0.808915 |    0.848891 |             0.907407 |            0.822628 |             0.0229221   |                                 0.068846   |
| B_ESTIMATED_SIGN_ORACLE_ABSOLUTE_DELTA                 | 16           | 0.0141208  |      0.0874843   |  0.998454 |   0.996475 |    0.993505 |             1        |            0.95609  |             0.0074374   |                                 0.0141208  |
| D_ESTIMATED_DELTA_ORACLE_GAMMA_NO_CURVATURE_CORRECTION | 16           | 0.0764908  |      0.473894    |  0.933528 |   0.771409 |    0.806855 |             0.907407 |            0.79636  |             0.0144461   |                                 0.0764908  |
| B_ESTIMATED_SIGN_ORACLE_ABSOLUTE_DELTA                 | 32           | 0.00847422 |      0.0525015   |  0.999403 |   0.997961 |    0.997322 |             1        |            0.965162 |             0.00272738  |                                 0.00847422 |
| D_ESTIMATED_DELTA_ORACLE_GAMMA_NO_CURVATURE_CORRECTION | 32           | 0.0820583  |      0.508387    |  0.918559 |   0.765319 |    0.766059 |             0.87963  |            0.772522 |             0.0104361   |                                 0.0820583  |

## Decisions

{
  "beta_identity": "BETA_REFERENCE_IDENTITY_VERIFIED",
  "coordinate_correction": "SOURCE_REFERENCE_COORDINATE_CORRECTION_SUPPORTED",
  "minimal_anchor": "SOURCE_REFERENCED_MINIMAL_ANCHOR_EFFICIENT",
  "source_ordering": "SOURCE_SEMANTIC_ORDERING_SUPPORTED_RETROSPECTIVELY",
  "zero_label_under_assumption": "SOURCE_REFERENCED_ZERO_LABEL_RECOVERY_SUPPORTED"
}

## Boundaries

This work does not establish a full conditional distribution, physiology, source anatomy, causality, a universal individual coordinate, downstream classification benefit, pseudo-label validity, TTA recoverability, a multiclass solution, or an ASD biomarker. No other dataset or classifier was run.

## Exact next scientific question

Can source-referenced conditional residualization and semantic ordering prospectively replicate across repeated sessions and multiple classes in Stieger2021 under a pre-frozen multiclass permutation protocol?
