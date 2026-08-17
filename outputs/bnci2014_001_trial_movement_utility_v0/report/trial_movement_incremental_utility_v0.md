# Trial Movement Incremental Utility Audit V0

Terminal decision: **STOP_NO_TRIAL_MOVEMENT_INCREMENTAL_UTILITY**.

Primary protocol was V2's complementary held-out-run T2 centering. All movement features were computed independently per trial before classifier fitting. Target labels entered only metric evaluation.

## T2 condition means

- STATIC: 0.497685
- MOV_LEN: 0.262346
- MOV_GRAM: 0.260417
- MOV_SENSOR: 0.310185
- STATIC_PLUS_LEN: 0.500000
- STATIC_PLUS_GRAM: 0.500386
- STATIC_PLUS_SENSOR: 0.460262

## Primary subject-level deltas

- DELTA_LEN: mean 0.002315, median 0.003472, positive subjects 6/9, raw p 0.285156, Holm p 0.855469.
- DELTA_GRAM: mean 0.002701, median 0.000000, positive subjects 4/9, raw p 0.335938, Holm p 0.855469.
- DELTA_SENSOR: mean -0.037423, median -0.038194, positive subjects 0/9, raw p 1.000000, Holm p 1.000000.

## Interpretation boundary

The result is limited to frozen trial-level features, a fixed linear decoder, BNCI2014_001 session 0train, and the frozen LOSO protocol. MOV_GRAM is not a complete quotient geometry. A negative result does not prove that every nonlinear network fails; a positive result does not establish mechanism or causality.

## Baseline reproduction note

V2 AIRM T2 split/centering direction was reproduced and exact subject differences were saved. Exact decoder reproduction is not claimed because this audit prespecified source-only StandardScaler and tighter logistic stopping parameters, whereas V2 did not scale.
