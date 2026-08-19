# Selective Conditional Memory Feasibility Audit V0

Terminal: `STOP_RELIABILITY_GATE_CANNOT_SELECT_MEMORY`

The frozen PR #20 terminal remains `STOP_NO_CROSS_SESSION_DOWNSTREAM_UTILITY` and is not rescued or reinterpreted.

## Oracle selective-memory ceiling

Stieger oracle mean BA: 0.3702701153; gain over identity: 0.0189214300; 95% CI [0.011256125114624696, 0.027645365775671386]. Population was selected for 41.935% of subjects.

## Deployable enrollment-only gate

Stieger mean BA: selective 0.3463520798, identity 0.3513486852, population 0.3370236467, global kappa 0.3561468691.
- Selective − IDENTITY_RESIDUAL_CARRY: mean -0.0049966054, 95% CI [-0.019001484377989084, 0.008662797592002126], raw p=0.753000, Holm p=1.000000.
- Selective − POPULATION_ONLY: mean 0.0093284332, 95% CI [0.002775831934893896, 0.01630764442893051], raw p=0.004000, Holm p=0.012000.
- Selective − GLOBAL_KAPPA: mean -0.0097947893, 95% CI [-0.02132023249923105, 0.00035225670726083655], raw p=0.961000, Holm p=1.000000.

## Required nulls

- ENROLLMENT_SUBJECT_MEMORY_PERMUTATION: p=0.000500.
- RELIABILITY_FEATURE_PERMUTATION: p=0.687500.
- ENROLLMENT_CLASS_SEMANTICS_PERMUTATION: p=0.000500.
- UNPAIRED_SOURCE_SESSION_GATE_TRAINING: p=0.015000.

## External replication

OpenBMI selective − identity: 0.0312962963, 95% CI [0.014999999999999998, 0.04777777777777778], p=0.000500.

## Boundary

No neural network, residual decoder, low-rank rescue, raw preprocessing, or deployment-label gate input was used. A STOP terminates this persistent conditional-memory architecture line in the present lineage.

Next statement: Persistent conditional-memory SPDNet development terminates under this lineage; no low-rank-map rescue is authorized.
