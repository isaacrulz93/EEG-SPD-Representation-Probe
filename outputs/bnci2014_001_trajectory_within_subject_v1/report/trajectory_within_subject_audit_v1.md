# BNCI2014_001 Trajectory Within-Subject Audit v1

## 1. What this audit asks

Trajectory Anatomy v0 asked whether class trajectory structure transferred across different subjects. This v1 audit asks whether class structure exists inside each individual subject and whether it transfers across sessions for that same subject.

This is a retrospective interpretation audit of the unchanged five-window AIRM representation. It does not introduce a new method, tune a classifier, change a window, or use the WHOLE subject×class interaction result.

## 2. Data and reproduction gate

Both BNCI2014_001 sessions were used: 9 subjects, 4 balanced classes, and 2,592 trials per session (5,184 total). Session 1 used the exact frozen v0 channel order, 8–32 Hz filtering, cue-relative 0–3.996 s, 250 Hz, five non-overlapping 200-sample windows, and float64 OAS covariance without added regularization or eigenvalue clipping.

Hard gate 0 passed. Session-0 trial identities were exact, all frozen v0 geometry gates passed, and recomputed PATH_D10, BAG_CANON_D10, and SCALARS_11 each had maximum absolute difference 0.0 from the frozen v0 reference. Session-1 numerical and geometry gates also passed.

## 3. Stage W — class information inside a subject/session

Stage W trained on runs 0–2 and tested on runs 3–5, then reversed the direction, separately for each subject and session. The subject statistic averages four scores; the group statistic is the median across nine subjects.

PATH observed T_W = 0.307292. The 1,999-replicate label-null median was 0.250000, giving effect 0.057292 and one-sided plus-one p = 0.000500. Stage W **PASS**.

Descriptive session medians for PATH were 0.291667 in session 0 and 0.295139 in session 1. Eight of nine PATH subject statistics were strictly above the 0.25 chance reference; this count is descriptive and not the test.

## 4. Stage X — same-subject transfer across sessions

Stage X trained on all six runs of one session and tested on the other session, then reversed direction, always within the same subject.

PATH observed T_X = 0.288194. The shared 1,999-replicate label-null median was 0.250000, giving effect 0.038194 and p = 0.000500. Stage X **PASS**. All nine PATH subject statistics were strictly above 0.25 descriptively.

## 5. Stage O — does chronological order add stable information?

Stage O kept the same five local SPD states and all pairwise distances but independently replaced state identity with a nonidentity S5 permutation for every trial. The same frozen Stage-X pipeline was refit for 1,999 replicates.

Observed T_O = 0.288194; order-null median = 0.272569; effect = 0.015625; p = 0.072000 (143 null statistics at least as large as observed). Stage O **FAIL** because p > 0.05.

Therefore the audit does not support chronological ordering as a required contributor to the stable same-subject signal.

## 6. Mandatory BAG_CANON comparator

BAG_CANON removes state labels/order while retaining the same local pairwise-distance configuration.

- W: observed 0.270833, null median 0.250000, effect 0.020833, p = 0.009500 — **PASS**.
- X: observed 0.282986, null median 0.250000, effect 0.032986, p = 0.000500 — **PASS**.

BAG's W/X passes together with the Stage-O failure favor the interpretation that stable subject-specific local relative geometry exists, while chronological order is not established as essential. Raw PATH-minus-BAG accuracy differences are not treated as a superiority test.

## 7. SCALARS_11 descriptive context

SCALARS_11 was evaluated without feature selection and has no terminal vote. Its observed median subject BA was 0.282986 for W and 0.281250 for X. The descriptive above-chance counts were 5/9 and 7/9, respectively.

## 8. Relation to the old cross-subject result

The old v0 AIRM PATH cross-subject mean BA was approximately 0.2616 and its label-destruction p-value was 0.155; it did not establish a population-shared class trajectory. The present positive W/X results are not a contradiction: they show that class-discriminative local covariance geometry can be subject-specific and reproducible for the same subject while failing to form one shared cross-subject representation.

## 9. Frozen terminal decision

**GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY**

PATH passed within-subject/within-session decoding and same-subject cross-session transfer. The chronological-order falsification did not pass. The unordered BAG comparator passed W and X.

## 10. What is supported

Under the frozen five-window AIRM representation, class-discriminative local covariance geometry is subject-specific and cross-session reproducible. The evidence is strongest for local relative geometry that does not require chronological state labels.

## 11. What is not supported

This audit does not establish brain physiology, individual motor strategy, causal neural dynamics, a cause of WHOLE-Z or domain-adaptation behavior, target-unlabeled identifiability, or a benefit from training a personalized model. It does not establish chronological order as necessary.

## 12. Reproducibility and validation

Label and order nulls each completed 1,999/1,999 deterministic replicates with resumable checkpoints and frozen SeedSequence derivation. Label realizations were shared between PATH and BAG. Required classifier fits had zero convergence failures. Final test result: 178 passed in 18.40s.

Compact null artifact SHA-256 values:

- label: `49813d09ba698c110d9a54b1a65f8cf66cf3ef06128d5066f91cfc547413ccc0`
- order: `2ab0612565990e5e7a0e27d2b84c126966a7b933a13d055e5de6f041f100d441`
