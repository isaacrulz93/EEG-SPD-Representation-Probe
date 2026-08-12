# BNCI2014_001 Left/Right-only Angular Factorial Diagnostic V0

## Frozen lineage

- Branch: `audit/bnci-left-right-angular-factorial-v0`
- Parent HEAD: `edc1d344cb0657f2f2d87b2992049bceec4705d2`
- Parent protocol freeze: `95c330de9596fa4c4eb4ee377d5af8d99896f4c3`
- Parent scientific result: `0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`
- Parent terminal: `BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS`
- Protocol freeze SHA: `659a9018adea03db5aa8879c7aff218622428dfb`
- Scientific result SHA: `19bc935620aae0659077cd620f31e4309a130a40`

The diagnostic used only the frozen parent squared angular-cost arrays. No raw EEG, covariance mean, anti-development, movement tuple, or quotient optimizer was fitted or recomputed. Parent artifacts reproduced exactly and remained unchanged.

## Plain-language answers

1. Cross-session subject correspondence within hand class was supported (T=0.3648215448, p=0.0005).
2. Left/Right class correspondence within subject was not supported (T=0.1565649309, p=0.158).
3. The previously supported BNCI four-class angular/joint interaction was not supported when the same frozen analysis was restricted to Left versus Right. T_J=0.1191292918, subject-break p=0.09, class-break p=0.3065.
4. The interaction was split-half sign-stable: Half A T_J=0.1563572692; Half B T_J=0.05966883127.
5. The immutable four-class angular result was T_J=0.1924088545, p_subjectbreak=0.001, and p_classbreak=0.0105. This diagnostic changes neither that result nor its terminal.

## Exact binary inference

- T_subject: 0.36482154482234475
- p_subject: 0.00050000000000000001
- T_class: 0.15656493093976115
- p_class: 0.158
- T_J: 0.11912929182411669
- p_J_subjectbreak: 0.089999999999999997
- p_J_classbreak: 0.30649999999999999

Subject summaries:

- S1: S_s=0.2069231837, C_s=0.04307447881, J_s=0.03207943465
- S2: S_s=1.257724884, C_s=0.0218269439, J_s=-0.02004534409
- S3: S_s=0.2798165142, C_s=0.03587185158, J_s=0.04307461871
- S4: S_s=0.4501390004, C_s=0.02437961157, J_s=0.0304965065
- S5: S_s=0.1088565463, C_s=0.006364050388, J_s=-0.04009680228
- S6: S_s=0.09435475902, C_s=-0.02856677098, J_s=-0.1081762289
- S7: S_s=0.1786047034, C_s=-0.05167536183, J_s=-0.04526238241
- S8: S_s=0.2595293431, C_s=-0.01210790847, J_s=-0.06503441463
- S9: S_s=0.4474449698, C_s=1.369917483, J_s=1.245128239

## Integrity and split-half checks

- Canonical 18-cell order: PASS
- Feet/Tongue excluded from all observed and null statistics: PASS
- Frozen NPZ/CSV and parent-result byte reproduction: PASS
- Generalized K=4 regression against frozen angular statistics: PASS
- 1,999 subject-break mappings preserve class: PASS
- 1,999 class-break mappings preserve subject: PASS
- Half A T_J: 0.15635726917541753
- Half B T_J: 0.059668831272278275
- Split-half sign stable: true

## Terminal

`BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED`

This is a retrospective diagnostic of frozen window-wise mean covariance movement costs. It does not establish absence through equivalence testing and makes no physiological, motor-strategy, neural-direction, or anatomical claim.

## Runtime, tests, and immutability

- Scientific runtime: 0.596582 seconds
- Focused pre-result tests: `14 passed in 0.18s`
- Focused post-result tests: `14 passed in 0.14s`
- Full repository tests: `1 failed, 284 passed, 4 skipped (unrelated missing ignored legacy cache: combined_trajectory_features.npz)`
- Scientific setting changed after protocol freeze: false
- Parent artifact changed: false
