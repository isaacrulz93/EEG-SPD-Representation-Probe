# Protocol: BNCI Angular Dual Relation Anatomy V0

## Retrospective status and immutable lineage

This is a retrospective anatomy analysis, not a prospective or confirmatory
experiment. The authoritative parent is
`pilot/local-movement-component-decomposition-v0` at exact HEAD
`edc1d344cb0657f2f2d87b2992049bceec4705d2`. Its protocol freeze is
`95c330de9596fa4c4eb4ee377d5af8d99896f4c3`, scientific result is
`0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca`, and frozen angular interaction is
`T_J=0.19240885452534362`, subject-break `p=0.001`, class-break `p=0.0105`.

PR #12 (`audit/bnci-left-right-angular-factorial-v0`, final HEAD
`7e692749e8b14ab1792d7175443ab476676fcb14`) is supporting retrospective
lineage. It reported `T_subject=0.36482154482234475, p=0.0005`,
`T_class=0.15656493093976115, p=0.158`, and `T_J=0.11912929182411669` with
subject-break `p=0.09`, class-break `p=0.3065`, terminal
`BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED`.

## Sole frozen input and stop rules

The sole scientific input is key `c_ang_matrix` in
`outputs/bnci2014_001_local_movement_component_decomposition_v0/arrays/component_cost_matrices.npz`,
SHA-256 `51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091`.
It is the directed 36 by 36 session-0-row to session-1-column squared angular
cost matrix for subjects 1 through 9 and classes
`left_hand,right_hand,feet,tongue`.

No EEG preprocessing, covariance mean, anti-development, movement tuple, AIRM
mean, or quotient optimizer may be invoked. The input is not reconstructed via
fitting. Before new anatomy, its bytes, schema, ordering, exact `c_full-c_len`
identity, and frozen four-class `T_subject`, `T_class`, and `T_J` must reproduce
at `atol=rtol=1e-12`. Failure yields
`UNASSESSED_PARENT_ANGULAR_REPRODUCTION_FAILURE` and stops. Parent hashes are
checked before and after execution.

## Part A: exact six-pair formal retrospective decomposition

Use the six unordered pairs `LR,LF,LT,RF,RT,FT`. Each binary subset retains all
nine subjects in subject-major, pair-class-minor order and applies the same
frozen definitions as PR #12: `S_sc=c_sc-a_sc`, `C_sc=b_sc-a_sc`, and
`J_sc=b_sc+c_sc-a_sc-d_sc`. Average the two anchor classes within subject and
then take the arithmetic mean across the nine subjects. Median aggregation is
forbidden.

For every subject, require the four-class `S_s,C_s,J_s` to equal the arithmetic
mean of the six corresponding pair values. Require the same identities for
group `T_S,T_C,T_J`. The maximum absolute reconstruction error must not exceed
`1e-12`, and reconstructed `T_J` must match the frozen value. Failure yields
`UNASSESSED_SIX_PAIR_RECONSTRUCTION_FAILURE` and stops.

Pair p-values are not computed in V0. Pair magnitudes, signs, subject
distributions, and additive contributions are supporting decomposition
outputs; no pair becomes a post-hoc primary. For nonzero four-class `T_J`, the
reported exact additive fraction is `(T_J,p/6)/T_J,4c`, which sums to one across
the six pairs and can be negative.

## Part B: descriptive dual relation anatomy

For descriptive anatomy only define `A=(D+D^T)/2`. Original directed `D`, never
`A`, remains the input to formal S/C/J reconstruction.

For each subject, `G_s(c,k)=A[(s,c),(s,k)]`. Its upper off-diagonal profile is
ordered `LR,LF,LT,RF,RT,FT`. Define
`Delta_G_s(c,k)=G_s(c,k)-0.5*(G_s(c,c)+G_s(k,k))`. This is excess cross-class
angular cost beyond average same-class cross-session instability, not a metric,
unbiased latent distance, or primary test.

For each class, `H_c(s,t)=A[(s,c),(t,c)]`. Define
`Delta_H_c(s,t)=H_c(s,t)-0.5*(H_c(s,s)+H_c(t,t))`. This is excess subject-pair
separation beyond their self-instability for that class, descriptively only.
The diagonal identity `G_s(c,c)=H_c(s,s)` is a hard extraction gate.

For raw and adjusted G profiles compute pairwise Pearson correlation, centered
cosine similarity, Euclidean profile distance, and each subject's correlation
and centered cosine to the mean of the other eight profiles. For raw and
adjusted H profiles, do the same across the four classes and correlate each
class to the mean of the other three. Pearson correlation and centered cosine
are algebraically identical for nondegenerate centered profiles; both are saved
as requested and their numerical agreement is checked. These are descriptive
commonality summaries, not new primary endpoints.

## Subject-pair anatomy and coarse contrast

Save all `J_s,p` in a 9 by 6 subject/pair view without dropping any subject.
Also save the descriptive leave-one-subject group mean and its change from the
full mean; no subject-deleted result is primary.

Predefine the descriptive contrast
`K_J=mean(T_J,LF,T_J,LT,T_J,RF,T_J,RT)-mean(T_J,LR,T_J,FT)`. It measures coarse
effector-boundary concentration only. Feet and tongue are not asserted to be a
biologically homogeneous group; no p-value is assigned.

## Figures and outputs

Produce: raw and adjusted 9 by 6 G-profile heatmaps; grouped six-pair
`T_S,T_C,T_J` bars with no significance encoding; a 9 by 6 `J_s,p` heatmap;
four comparable-scale H heatmaps; and four comparable-scale adjusted-H
heatmaps. Save exact decomposition, reconstruction, raw/adjusted G and H,
pairwise/leave-out commonality, coarse contrast, subject influence, arrays,
figures, provenance, and report under
`outputs/bnci2014_001_angular_dual_relation_anatomy_v0/`.

## Interpretation hierarchy and restrictions

The report separates: immutable formal prior evidence; formal retrospective
six-pair algebra; supporting PR #12 evidence; and descriptive dual anatomy.
It assesses, without forcing one winner, common class backbone, common subject
backbone, heterogeneous subject-by-class deformation, and coarse effector
hierarchy.

`G_s` and `H_c` are relation matrices/profiles, not intrinsic metric geometry.
Common subject ordering is not proof of a reusable `Q_s`. Descriptive G/H
deviation is not itself a formal interaction. No physiological neural movement,
cortical direction, causal strategy, or generative subject/class factor is
claimed. LR does not stand for all MI classes. An extreme subject neither
invalidates nor establishes the population result. Trial pairs and matrix
entries are not independent inferential units.

After the protocol-freeze commit, inputs, definitions, aggregation, pair set,
tolerance, profiles, contrast, inference hierarchy, and claim restrictions are
immutable. No rescue analysis is permitted.
