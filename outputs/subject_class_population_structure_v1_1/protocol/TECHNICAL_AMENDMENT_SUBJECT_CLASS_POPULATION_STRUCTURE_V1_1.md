# Technical Amendment: Subject-Class Population Structure V1.1

Status: **TECHNICAL RECOVERY ONLY — SCIENTIFIC CONTRACT UNCHANGED**

Date: 2026-08-18 (Asia/Seoul)

## Immutable V1 history

V1 protocol freeze `bebc2149b79ac9026703e743f2129f65d792441e`, failed-result commit `6e5cbd1bd9d8356396f89f0421609fda0a43fb96`, and every artifact below `outputs/subject_class_population_structure_v1/` are immutable inputs to this amendment. V1 remains `UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE`.

The V1 real-data runner completed the reliability prerequisite and entered the non-voting ordered-`Z` spectrum diagnostic. Its inner rank-selection fit raised `NumericalContractError: projected training-score scale is degenerate`. No primary statistic, null, effective rank, or BNCI result was interpreted or reported.

## Unchanged scientific contract

V1.1 asks exactly the V1 question: whether the stable montage-registered mean-level subject×class interaction has a population-shared low-dimensional linear structure that generalizes to held-out subjects and independent sessions.

The V1 frozen config is the sole source of every scientific setting. V1.1 changes no hypothesis, parent object, subject/session/class/channel order, feature, normalization, fold, inner fold, rank grid, one-standard-error rule, low-rank cap, bootstrap, null count, null mapping/seed namespace, terminal gate, threshold, or preprocessing rule. A machine-checked equivalence table must contain no scientific `changed=true` row.

No hidden or partial value from the failed V1 process, traceback, shell history, log, or temporary state may be recovered. V1.1 recomputes the deterministic pipeline from the immutable parent objects only after this amendment is frozen.

## Component roles and exception boundaries

### Voting / required

- reliability prerequisite
- OpenBMI sensor-space primary observed analysis
- subject-pairing null
- class-semantics null
- equal-rank random-subspace control
- full-space sensor baseline
- V1 frozen terminal gates

Any nonfinite value, degenerate primary projected scale, invalid fold, zero primary feature norm, decomposition failure, hash mismatch, or leakage violation in a voting component immediately produces `UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE`.

### Non-voting / optional diagnostic

- magnitude sensitivity
- same-session PCA
- ordered-`Z` eigenspectrum
- generalized-eigen signature
- PR #4 action-overlap diagnostic
- BNCI secondary four-class analysis
- selected-mode split-half and display-only diagnostics

Every non-voting component executes inside its own exception boundary. A numerical degeneracy is recorded as `CONTROL_UNASSESSED_NUMERICAL_DEGENERACY`; a data-contract or execution failure receives a corresponding `CONTROL_UNASSESSED_*` status. Such a status cannot abort, rescue, or overturn a completed OpenBMI terminal. BNCI remains scheduled after the OpenBMI terminal and is not silently skipped merely because it is secondary.

No epsilon, jitter, singular-value clipping, coordinate deletion, fold deletion, rank-grid reduction, regularization, or new threshold is permitted to force a diagnostic to complete. The ordered-spectrum failure is retained and characterized without a workaround.

## Frozen execution order

1. immutable V1 and parent hashes / scientific-equivalence audit
2. OpenBMI split-half reliability
3. OpenBMI sensor primary observed analysis and full-space baseline
4. subject-pairing, class-semantics, and equal-rank random-subspace pipelines
5. V1 terminal decision
6. independently isolated non-voting OpenBMI controls
7. independently isolated BNCI diagnostic and action-overlap diagnostic
8. terminal-aware report and immutable manifest

The report next-question rule is terminal-specific: `GO_*` may use the positive question, `STOP_*` may use the negative question, and `UNASSESSED_*` must state: “The population-structure hypothesis remains unassessed; the technical/data-contract blocker must be resolved before a scientific next question is selected.”

## Only permitted changes

The scientific-equivalence table permits `changed=true` only for orchestration-level secondary failure isolation and the new V1.1 artifact namespace. These changes do not vote in the terminal.
