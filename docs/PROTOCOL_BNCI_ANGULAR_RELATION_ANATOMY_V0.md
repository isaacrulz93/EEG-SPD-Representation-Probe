# Protocol: BNCI2014_001 Angular Relation Anatomy V0

## Status and lineage

This is a minimal retrospective anatomy analysis of one immutable squared-cost
matrix. The authoritative parent is branch
`pilot/local-movement-component-decomposition-v0` at exact HEAD
`edc1d344cb0657f2f2d87b2992049bceec4705d2`; its protocol freeze is
`95c330de9596fa4c4eb4ee377d5af8d99896f4c3`, scientific result is
`0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`, and terminal is
`BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS`.

The sole numerical input is key `c_ang_matrix` in
`outputs/bnci2014_001_local_movement_component_decomposition_v0/arrays/component_cost_matrices.npz`,
whose frozen SHA-256 is
`51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091`.
The matrix is 36 by 36, with session 0 row anchors and session 1 column
targets, ordered by subjects 1 through 9 and then
`left_hand,right_hand,feet,tongue`. It is generally not symmetric.

No EEG, covariance mean, anti-development, ordered movement object, or
quotient optimizer may be fitted or recomputed. Failure to recover this exact
input gives `UNASSESSED_MISSING_FROZEN_ANGULAR_MATRIX`.

## Required pre-analysis reproduction

Before any new pair or relation-anatomy statistic is accessed, the input hash,
keys, dtype, shape, finite status, subjects, and classes must reproduce. The
frozen parent angular statistics must reproduce at absolute and relative
tolerance `1e-12`:

- `T_subject=0.3091561771980925`
- `T_class=0.39309843397343514`
- `T_J=0.19240885452534362`

Failure stops the analysis.

## Formal six-pair decomposition

The primary formal retrospective output uses unordered binary class pairs
`LR,LF,LT,RF,RT,FT`. Each pair restricts the original directed matrix without
symmetrization. For each subject and each of its two anchor classes, the frozen
definitions are retained:

- `a`: same subject, same class;
- `b`: same subject, opposite pair class;
- `c`: mean different subject, same class;
- `d`: mean different subject, opposite pair class;
- `S=c-a`, `C=b-a`, and `J=b+c-a-d`.

The two anchors are averaged within subject. Subjects are then averaged using
the arithmetic mean. No median, weighting, independent matrix-entry analysis,
or replacement factorial contrast is allowed. Pairwise p-values are not
computed: pair results are supporting decomposition outputs and cannot replace
or redefine the immutable four-class inference.

The following exact identities are hard gates at `atol=rtol=1e-12`:

`S_s^4 = mean_p S_s,p`, `C_s^4 = mean_p C_s,p`, and
`J_s^4 = mean_p J_s,p` for every subject, and the corresponding group
identities `T_S^4=mean_p T_S,p`, `T_C^4=mean_p T_C,p`, and
`T_J^4=mean_p T_J,p`. Failure stops scientific interpretation.

## Descriptive dual relation matrices

For each subject, the class relation matrix is

`G_s(c,k)=0.5*(D[(s,c),(s,k)]+D[(s,k),(s,c)])`.

For each class, the subject relation matrix is

`H_c(s,t)=0.5*(D[(s,c),(t,c)]+D[(t,c),(s,c)])`.

Baseline-adjusted matrices subtract the mean of the two corresponding diagonal
entries from every pair: `Delta_G_s(c,k)=G_s(c,k)-0.5*(G_s(c,c)+G_s(k,k))`
and analogously for `Delta_H_c`. The six upper off-diagonal entries of each
`G_s` (ordered `LR,LF,LT,RF,RT,FT`) and the 36 upper off-diagonal entries of
each `H_c` form the profiles.

Raw profile similarity is Euclidean distance. Pattern similarity is centered
Pearson correlation. Leave-one-out commonality is the centered correlation of
each profile with the arithmetic mean profile of all other subjects or
classes. Calculations are repeated for baseline-adjusted profiles. Pairwise
distance/correlation and leave-one-out values are summarized descriptively;
no strength threshold or inferential label is imposed. Optional permutation
nulls are not run in V0.

## Interpretation hierarchy

The immutable parent S/C/J statistics remain the formal inference. The exact
six-pair decomposition is the primary retrospective anatomy and explains its
linear class-pair composition. `G_s`, `H_c`, their adjusted forms, and their
commonality summaries are descriptive. They show which relations are close or
far but do not replace S/C/J.

`G_s` and `H_c` are relation matrices, relation geometry, or distance profiles;
they are not asserted to be intrinsic manifold shapes. Baseline-adjusted
matrices are not inferential primaries. Common relation geometry is not a
reusable transformation, and no generative subject/class decomposition follows
from S/C/J alone. Neither the largest class pair nor Left/Right may be treated
as the whole four-class result. Trial pairs and matrix entries are not
independent population units.

The report will separately label: immutable formal inference; primary formal
six-pair decomposition; descriptive matrix anatomy; and optional diagnostics
(none here). It will include a result-to-claim decision table covering strong
or weak `G_s` and `H_c` similarity, both combinations, Left/Right versus
hand/nonhand concentration, pair concentration, and subject concentration.
Static-versus-movement claims require already-frozen external evidence and are
not made by this analysis alone.

## Immutability and outputs

After the protocol-freeze commit, no input, statistic, aggregation, tolerance,
profile definition, output hierarchy, or interpretation rule may change. The
run will save exact pair tables, reconstruction diagnostics, matrices,
profiles, similarities, heatmaps, provenance, and a report under
`outputs/bnci2014_001_angular_relation_anatomy_v0/`. Parent hashes are checked
before and after execution. There is no rescue analysis.
