# D/G Discrepancy Decomposition v1 — Frozen Protocol

Protocol version: **1.0**  
Freeze date: **2026-08-10 (Asia/Seoul)**  
Base commit: **74bbae09fd9880c5809b61769d8f6e27b9b994bf**  
Branch: **pilot/dg-discrepancy-decomposition-v1**  
Master seed: **20260809** (reused from Conditional Geometry Anatomy v1)

This protocol is frozen and committed before any new scientific derived object is
computed. Conditional Geometry Anatomy v1 and every file below
`outputs/bnci2014_001_conditional_geometry_v1/` are read-only.

## 1. Scope and scientific status

This is a retrospective, post-result mechanism anatomy of the already observed
D-/G+ Stage-S discrepancy. Session `0train` is the **retrospective mechanism set
A** and session `1test` is the **locked internal replication set B**. Neither is
new confirmatory evidence. This diagnostic does not modify or reinterpret the
frozen `STOP_TANGENT_ONLY` result and cannot produce a scientific GO.

The only question is whether the discrepancy is attributable to (M1) AIRM
curvature/nonlinearity, (M2) marginal-anchor placement, (M3) deterministic
distance-versus-Gram encoding/statistic sensitivity, a mixture, or no coherent
decomposition. No method, classifier, alignment loss, pseudo-label, WINDOW5,
trajectory, neural network, HGD, new covariance, new Fréchet mean, or commutator
analysis is permitted. Oracle P remains descriptive and is not component
recovery.

## 2. Frozen sources and integrity

The source Conditional Geometry v1 protocol and config are:

- `docs/PROTOCOL_CONDITIONAL_GEOMETRY_V1.md`, SHA-256
  `ff7143dca8408a73233321fc7219358e48f2336bad0cb63f4e76f8988e77ecce`.
- `configs/bnci2014_001_conditional_geometry_v1.yaml`, SHA-256
  `ca0178be9c5fd168509f48974d2b3e1bbc91b2ba4e8e9e320299fd98dcf765e8`.

The exact session snapshots are:

| session role | artifact | SHA-256 |
| --- | --- | --- |
| set A / `0train` | `objects/discovery/D_matrices.npz` | `0053a7413d804cc88fd3ac586a94d237e9f260cf79e6ada123c648857881a3d8` |
| set A / `0train` | `objects/discovery/G_matrices.npz` | `54e08e56c969d5475106e4f6e027dbb518dd1dc9449b49322f1317de96c65ca9` |
| set B / `1test` | `objects/confirmatory/D_matrices.npz` | `a3376f93fc862ce36af7ee9103044bfe57e2b8f7f456b96fc8584dc1f226e4c0` |
| set B / `1test` | `objects/confirmatory/G_matrices.npz` | `b2aaa5b0516a86429684cbab81d2b3cbcc9869247f9530a14e9afbc63aa630fc` |

Each archive must have axes `(geometry, subject, split, class, class)` with
geometries `[AIRM, LE]`, subjects `1..9`, splits `[A,B,F]`, and classes
`[left_hand,right_hand,feet,tongue]`. Root D/G archives must byte-match the set-B
hashes. The v1 manifest/config hashes must agree with the source files. Failure
of any check stops all derived computation.

Before the freeze, the existing source output contained 139 files, 36,630,557
bytes, with aggregate SHA-256
`51332ac66d3235e80ed43df4e80cdf752d2d37f81b2ae71c19c7caa3b3e7e8ba`,
where the aggregate is SHA-256 of canonical compact JSON over sorted records
`{path,bytes,sha256}` relative to the source output. The identical aggregate is
required after the diagnostic.

## 3. Class weights

For every subject, session, and split, actual class counts are read from the v1
`dataset_contract.csv`, not assumed. With `n_c` the validated count,

\[
\pi_c=n_c/\sum_c n_c,\qquad H_\pi=I-\mathbf 1\pi^T.
\]

All rows must contain every class with positive count and sum to one. The BNCI
balanced expectation `pi=(.25,.25,.25,.25)` is a checked fact, never hard-coded
as the calculation input.

## 4. Deterministic derived objects

For each subject/session/split/geometry, `D_exact` and original `G` are the
stored v1 matrices without refitting.

Define for each pair

\[
q_{cc'}=G_{cc}+G_{c'c'}-2G_{cc'},\qquad D_{tan}(c,c')=\sqrt{q_{cc'}}.
\]

The diagonal is exactly zero. Let
`scale=max(1,abs(Gcc),abs(Gc'c'),abs(2Gcc'))`. If
`q < -1e-10*scale`, the row is a numerical failure. Values in
`[-1e-10*scale,0)` alone are set to zero, with the cleanup magnitude recorded.

The anchor-removed tangent Gram is

\[
G_0=H_\pi G H_\pi^T.
\]

Anchor energy and fraction are

\[
E_{anchor}=\pi^TG\pi,\qquad F_{anchor}=E_{anchor}/\operatorname{tr}(G),
\]

with the fraction defined only when the trace exceeds the float64 machine-safe
threshold `100*eps*max(1,max(abs(G)))`.

The pure exact-distance re-encoding is

\[
K_{exact}=-\tfrac12H_\pi(D_{exact}^{\circ2})H_\pi^T.
\]

`K_exact` is retained as a signed symmetric matrix. It is never eigen-clipped,
PSD-projected, whitened, or otherwise repaired.

Curvature distortion is

\[
C=D_{exact}^{\circ2}-D_{tan}^{\circ2},\qquad
C_{rel,cc'}=C_{cc'}/\max(D_{exact,cc'}^2,safe),
\]

where `safe=eps*max(1,max(D_exact**2))`. Upper-triangle mean, median, maximum,
mean relative distortion, and maximum relative distortion are descriptive.

## 5. Vectorization and normalization

Distance objects `D_exact` and `D_tan` use the raw fixed upper triangle
`[12,13,14,23,24,34]`. Gram objects `G`, `G0`, and `K_exact` use the exact v1
Frobenius-isometric symmetric svec: diagonal entries unchanged and off-diagonal
entries multiplied by `sqrt(2)`, in upper-triangle row order
`[11,12,13,14,22,23,24,33,34,44]`. Every vector is divided by its own L2 norm.
The degeneracy threshold remains
`100*eps_float64*max(1,max(abs(raw_vector)))`; a norm at or below it fails the
experiment. No scaler, PCA, centering in feature space, or whitening is allowed.

Class permutation is the same simultaneous row/column reindexing `P O P^T` used
by v1. Objects are re-vectorized and normalized after permutation.

## 6. Numerical hard gates

All required rows must pass:

1. `D_tan**2` reconstructs the quadratic expression from G.
2. LE `relative_frobenius_error(D_exact**2,D_tan**2) <= 1e-10`.
3. `G0 == H_pi G H_pi.T` by independent reconstruction.
4. `relative_frobenius_error(D_tan(G),D_tan(G0)) <= 1e-10`.
5. In validated balanced LE, `relative_frobenius_error(G,G0) <= 1e-10`.
6. LE `relative_frobenius_error(K_exact,G0) <= 1e-10`.
7. D_tan, G0, and K_exact permutation equivariance errors are `<=1e-10`.
8. Positive common scaling leaves every corresponding unit shape unchanged
   within `1e-10` relative error.
9. An audit records that no K_exact PSD projection/clipping code path exists and
   the stored object equals the direct signed formula.
10. AIRM `C >= -1e-10*max(1,abs(D_exact**2),abs(D_tan**2))` elementwise.
11. Stored original D/G shapes and Stage-S results reproduce v1 within `1e-12`
    absolute tolerance and `1e-10` relative tolerance.
12. The complete source-output aggregate remains unchanged.
13. No existing protocol, config, or output is overwritten.

Relative Frobenius error is `||A-B||F/max(||A||F,||B||F,eps)`; when both norms
are zero it is zero. Failures stop interpretation and are printed as hard-gate
failures, never silently excluded.

## 7. Stage R reliability

For each of `D_exact,D_tan,K_exact,G0,G`, observed subject reliability is the
A/B cosine and the group statistic is the subject median. The strict v1 label
destruction null has B=1,999, shuffles labels within
subject×session×run while preserving the multiset, and uses the same indexed
`PCG64DXSM SeedSequence([20260809,1101,phase,replicate])` plan. Each shuffled
label fit must first produce D/G, after which D_tan/G0/K_exact are derived by
the frozen formulas. Original D/G observed and null results must regress to v1.
Stage R is only a mechanism prerequisite, not a new confirmation.

If the immutable v1 source does not contain the trial-level covariance/metadata
inputs or per-replicate D/G needed for this exact refit, the implementation must
not invent or approximate a derived-object label null. It must report the
derived Stage-R null fields as `NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE`, retain the
exact published D/G Stage-R regression, and make no R-based mechanism claim.
This fail-closed storage contingency does not relax any Stage-S requirement.

## 8. Stage S shared semantic geometry

Stage S exactly reuses v1 LOSO statistics. Set A uses source-A→target-B and
source-B→target-A templates and averages the two scores. Set B uses the fixed
set-A F LOSO template against target set-B A/B and averages. Each template is
`normalize(sum(unit source-subject shapes))`, excluding the target subject.

The semantic null uses B=100,000 and the v1 indexed plan
`PCG64DXSM SeedSequence([20260809,1201,2,replicate])`. Each source subject draws
an independent uniform S4 permutation including identity. Replicate `b` uses
the same subject permutations for both sessions, all geometries, all splits,
and all five objects. Stored outputs include observed subject scores, group
median, null subject scores, null group distribution, null median, effect
`E_S=observed-null_median`, plus-one one-sided p, and plus-one subject null
percentiles.

Primary descriptive contrasts are

\[
\Delta_{curvature}=E_S(D_{tan})-E_S(D_{exact}),
\]
\[
\Delta_{anchor}=E_S(G)-E_S(G0),
\]
\[
\Delta_{encoding,exact}=E_S(K_{exact})-E_S(D_{exact}),
\]
\[
\Delta_{encoding,tangent}=E_S(G0)-E_S(D_{tan}).
\]

LE curvature and anchor contrasts are numerical-zero controls. Effects, not raw
cosines, are primary.

## 9. Oracle P

The same 24 lexicographic S4 candidate scores are computed descriptively for
every object. Set A uses the bidirectional cross-half score; set B uses the fixed
set-A F LOSO template and set-B F candidates. Identity gets conservative worst
rank within `1e-12`; normalized rank is `(24-rank)/23`, top1 requires rank one,
and margin is identity minus the best nonidentity. P never votes in the
mechanism decision.

## 10. Deterministic provisional interpretation

No new absolute cutoff is introduced. Labels use algebraic matching and the
sign/replication pattern across set A and set B:

- `PROVISIONAL_ENCODING_SUPPORTED`: K_exact consistently follows G/G0 while
  D_exact differs, and/or G0 consistently differs from D_tan despite matched
  information, especially with the same LE pattern.
- `PROVISIONAL_CURVATURE_SUPPORTED`: in both sessions AIRM D_tan has a positive
  effect increment over D_exact, D_tan and G0 give compatible matched-information
  inference, and LE curvature is numerical zero.
- `PROVISIONAL_ANCHOR_SUPPORTED`: in both sessions AIRM G has a positive effect
  increment over G0 and LE G/G0 is the numerical-zero control.
- `PROVISIONAL_MIXED`: at least two of curvature, anchor, and encoding have
  coherent, nonzero, direction-replicated evidence.
- `PROVISIONAL_UNRESOLVED`: directions do not replicate or no coherent matched
  decomposition exists.

Session labels apply the same logic within each session; the overall label
requires compatible directions across sessions. P-values are displayed but a
single p-value never determines a label. These labels are not scientific GO
before external confirmation.

## 11. Output contract

All new artifacts live only in
`outputs/bnci2014_001_dg_decomposition_v1/{protocol,objects,tables,nulls,figures,report}`.
Objects are compressed NPZ archives with explicit session, geometry, subject,
split, and class axes. The required tables are:

`source_artifact_audit.csv`, `derived_object_identity_gates.csv`,
`D_exact_shapes.csv`, `D_tan_shapes.csv`, `G_shapes.csv`, `G0_shapes.csv`,
`K_exact_shapes.csv`, `anchor_offset_summary.csv`,
`curvature_distortion_summary.csv`, `reliability_summary.csv`,
`shared_semantic_summary.csv`, `subject_semantic_effects.csv`,
`mechanism_contrasts.csv`, `oracle_descriptive_summary.csv`, and
`session_replication_summary.csv`.

Null archives retain replicate indices and common-plan subject/group arrays.
Figures 1–8 each have PNG, PDF, and source CSV. The report is
`report/dg_discrepancy_decomposition_v1.md`. A manifest records all new artifact
hashes without including itself. The report explicitly states the retrospective
status, lack of a new method or external confirmation, unchanged
`STOP_TANGENT_ONLY`, the prohibition on interpreting D-/G+ as tangent
superiority absent matched-object evidence, unsolved oracle component recovery,
and exclusion of WINDOW5.

The run ends after tests, source-output invariance verification, report creation,
and a clean commit. It prints exactly the required provenance/gate/session-label
block followed by the frozen numerical summary.
