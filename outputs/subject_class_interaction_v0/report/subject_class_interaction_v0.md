# Subject Class Interaction V0

## 1. Scientific question

This frozen premise-falsification experiment asks whether a cross-session-reproducible subject×class interaction remains after marginal subject/session location, the population class template, and the class-independent subject residual are removed from covariance representations.

## 2. Why this follows prior anatomy

Prior discrepancy decomposition did not distinguish a class-dependent individual interaction from a generic subject residual. This analysis separates those alternatives without a classifier, TTA, neural network, new loss, low-rank fit, or mixed-effects fit.

## 3. Definitions in plain language

Each subject/session marginal covariance mean is moved to identity. Class means in that common tangent coordinate system are marginally recentered class effects. The frozen population class template and then the class-weighted residual are subtracted.

## 4. U / R / Z distinction

U is the marginally recentered class effect. R is U minus the population class template. Z is R minus its class-weighted mean. Z is the primary subject×class interaction object; R is a descriptive subject-residual control.

## 5. Dataset roles

BNCI2014_001 is retrospective development only. OpenBMI/Lee2019-MI is the prospective external replication. The OpenBMI manifest was frozen and unlocked only after all three BNCI primary directions were strictly positive.

## 6. Numerical and data gates

BNCI contained 5,184 expected trials from 9 subjects. OpenBMI contained 10,800 expected trials from 54 subjects, two sessions, two balanced classes, and the frozen ordered 20-channel motor-cortex subset. All 108 OpenBMI source records have SHA-256 provenance; 106 complete MNE-cache files were reused and the two incomplete subject-5 files were downloaded afresh. Covariance, finite, symmetry, SPD, AIRM convergence, marginal-identity, class-weight, and weighted-Z-zero gates passed.

## 7. BNCI development

The primary AIRM/session-specific/sensor/Z effects were strictly positive for all stages: R (T_obs=0.747637, null median=-0.00106258, E=0.748699, p=0.0005); I (T_obs=0.773041, null median=-0.183146, E=0.956187, p=7.49081e-06); C (T_obs=0.773041, null median=-0.000723119, E=0.773764, p=0.0005).

## 8. OpenBMI prospective external replication

OpenBMI used the frozen preprocessing and no result-selected change. The primary chain used 1,999 label-destruction nulls for R, 100,000 deterministic random derangements for I, and 1,999 label-destruction nulls for C.

## 9. Measurement reliability

Stage R measures the median subject reliability between independently refitted within-session acquisition-order halves. OpenBMI: T_obs=0.437948, null median=-7.4506e-05, E=0.438023, p=0.0005. Gate R: **PASS**.

## 10. Same-subject cross-session reproducibility

Stage I measures the median diagonal of the session-0 by session-1 subject similarity matrix against fixed-point-free subject mappings. OpenBMI: T_obs=0.33967, null median=0.0240364, E=0.315633, p=9.9999e-06. Gate I: **PASS**.

## 11. True-class dependence

Stage C compares the true-label same-subject statistic with label-destruction refits. OpenBMI: T_obs=0.33967, null median=0.000632531, E=0.339037, p=0.0005. Gate C: **PASS**.

## 12. R-versus-Z control

The AIRM/session-specific/sensor R control passed all three descriptive stage criteria: **True**. It does not vote on or rescue the primary Z decision.

## 13. Gauge-sensitive versus spectrum control

The sensor Z chain passed R/I/C. The orthogonally invariant spectrum Z control passed all three criteria: **False**. Its Stage C result was T_obs=0.913906, null median=0.88305, E=0.0308561, p=0.108; therefore the evidence is sensor-space-specific under the frozen outcome logic.

## 14. Template and geometry sensitivities

The pooled-session AIRM sensor Z sensitivity passed all stages: **True**. The session-specific log-Euclidean sensor Z robustness chain passed all stages: **True**. Neither secondary analysis can rescue the primary chain.

## 15. Frozen terminal decision

**GO_SENSOR_SPACE_ONLY**. The primary OpenBMI sensor Z gates R/I/C passed, while the spectrum control did not support all three stages. No absolute cosine threshold or result-selected protocol modification was used.

## 16. What is justified

The frozen analysis supports a stable OpenBMI subject×class interaction in the montage-registered sensor representation after the specified marginal, population-class, and class-independent residual removals. The result replicated the three BNCI development directions.

## 17. What is NOT justified

It does not establish physiology, personality, a neural trait, source anatomy, a biomarker, unlabeled recoverability, identifiable TTA parameters, intrinsic Riemannian random effects, the full conditional distribution, or a causal brain mechanism.

## 18. Exactly one next structural question

Is the stable subject×class interaction low-dimensional and structured across the population?

## 19. Direction killed if STOP

No primary sensor-space direction was killed. The spectrum control failed Stage C, selecting the frozen sensor-space-only GO rather than a spectrum-supported GO.
