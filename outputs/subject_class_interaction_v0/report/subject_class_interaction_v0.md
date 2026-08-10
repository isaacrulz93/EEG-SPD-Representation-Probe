# Subject Class Interaction V0

## 1. Scientific question

This premise-falsification experiment asks whether a cross-session-reproducible subject×class interaction in marginally recentered covariance representation remains after marginal subject/session location, the session-specific population class template, and a class-independent subject residual are removed.

## 2. Why this follows prior anatomy

Prior discrepancy decomposition left open whether stable residual structure was class-dependent or merely a global subject residual. This pilot separates those alternatives without a classifier, TTA, neural network, new loss, low-rank factorization, or mixed-effects fit.

## 3. Definitions in plain language

For each subject/session, the marginal covariance mean is moved to identity. Class means in that common tangent coordinate system are called marginally recentered class effects. A target-excluding population class template is then subtracted, followed by the weighted class-independent residual.

## 4. U / R / Z distinction

U is the marginally recentered class effect. R is U minus the session-specific LOSO population class template. Z is R minus its class-weighted mean. Z, not U or R, is the primary subject×class interaction object.

## 5. Dataset roles

BNCI2014_001 is retrospective development only. OpenBMI/Lee2019-MI is the prospective external replication and remains scientifically locked unless all three BNCI primary effect directions are positive.

## 6. Numerical/data gates

All 5,184 expected BNCI trials, 9 subjects, two sessions, four classes, six runs, 22 ordered channels, and frozen file/content hashes matched. Covariance SPD/finite/symmetry checks, AIRM convergence with Karcher residual ≤1e-7, marginal-to-identity checks, U/Z symmetry, class weights, and weighted-Z-zero checks passed.

## 7. BNCI development

The primary chain is AIRM, session-specific LOSO population templates, the montage-registered sensor signature, and Z. BNCI is descriptive/developmental rather than strict confirmation.

## 8. OpenBMI external replication if unlocked

OpenBMI status: **eligible for manifest resolution but not yet unlocked**. No OpenBMI scientific score, similarity matrix, or figure was produced.

## 9. Measurement reliability

Stage R first computes each subject's sessionwise cosine between independently refitted run-halves, averages the two session cosines per subject, and takes the subject median. Its null permutes labels within subject/session 1,999 times, preserves counts, and refits every label-dependent mean and interaction. Primary result: T_obs=0.747637, null median=-0.00106258, E=0.748699, p=0.0005.

## 10. Same-subject cross-session reproducibility

Stage I takes the median diagonal of the session-0 by session-1 subject similarity matrix. Its unrelated-subject null exhaustively evaluates all 133,496 fixed-point-free mappings. Primary result: T_obs=0.773041, null median=-0.183146, E=0.956187, p=7.49081e-06.

## 11. Class-destruction control

Stage C compares the true-label same-subject statistic with 1,999 independent within-subject/session label destructions, each followed by the full refit. Primary result: T_obs=0.773041, null median=-0.000723119, E=0.773764, p=0.0005.

## 12. R-versus-Z control

The R signature retains class-independent residual structure and has no primary vote. Its three frozen descriptive stage criteria were all supported. A stable R cannot rescue a failed Z chain.

## 13. Gauge-sensitive versus spectrum control

The sensor signature is montage-registered and coordinate-dependent. The ascending-eigenvalue spectrum is invariant to orthogonal conjugation and is secondary only. Its three descriptive criteria were all supported.

## 14. Descriptive energy anatomy

The saved DESCRIPTIVE ENERGY FRACTIONS report squared population-class, class-independent Rbar, Z, split-half discrepancy, and cross-session discrepancy energies. Only the algebraically orthogonal Rbar-versus-Z residual fractions are normalized; they are not identified variance components.

## 15. Frozen terminal decision

**OPENBMI_UNLOCK_ELIGIBLE_NOT_TERMINAL**. The decision uses no absolute cosine threshold and no result-selected protocol modification.

## 16. What is justified

The BNCI result justifies only the frozen retrospective direction screen for a cross-session-reproducible subject×class interaction in marginally recentered covariance representation.

## 17. What is NOT justified

It does not establish physiology, personality, a neural trait, source anatomy, a biomarker, unlabeled recoverability, identifiable TTA parameters, low dimensionality, intrinsic Riemannian random effects, full conditional distributions, or a causal brain mechanism.

## 18. Exactly one next structural question if GO

Deferred until prospective OpenBMI replication reaches a frozen GO.

## 19. What direction is killed if STOP

The first nonpositive BNCI primary effect direction is **none in BNCI direction screen**. No BNCI direction was killed; the next allowed action is metadata-only OpenBMI protocol resolution and manifest freeze.
