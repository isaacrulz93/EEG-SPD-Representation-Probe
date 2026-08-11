# Local Metric Interaction V0 — Pre-data Mathematical Audit

Status: pre-data audit. No new BNCI cell distance, interaction statistic, or decoding score was used in this audit.

## 1. Mathematical object and exact finite quotient

A trial supplies five SPD matrices (B=(C_1,\ldots,C_5)). Its internal AIRM distance matrix has entries (D_B[i,j]=d_{\mathrm{AIRM}}(C_i,C_j)), and its ten unique edges are stored in the fixed order

\[
(12,13,14,15,23,24,25,34,35,45).
\]

Only vertex labels are discarded. Every π in (S_5) induces one edge-coordinate permutation (P_\pi); exhaustive construction yields exactly 120 distinct coordinate permutations. Arbitrary elements of (S_{10}) are not admissible.

The raw distance is

\[
\delta_{\mathrm{raw}}([x],[y])=
\min_{\pi\in S_5}\frac{\lVert x-P_\pi y\rVert_2}{\sqrt{10}}.
\]

The implementation enumerates all 120 actions. A transparent loop is retained as the reference; the production implementation batches all actions and uses the identity ‖x−y‖²=‖x‖²+‖y‖²−2xᵀy. Random known-answer banks agree within absolute and relative tolerance (10^{-12}).

## 2. Metric proof on distance-matrix orbit space

Permutation invariance follows because relabeling one input merely reindexes the finite minimization set. Symmetry follows from orthogonality of permutation matrices and closure under inverses:

\[
\lVert x-P_\pi y\rVert
=\lVert y-P_{\pi^{-1}}x\rVert.
\]

For identity of indiscernibles, the minimum is zero exactly when (x=P_\pi y) for some vertex permutation. This says exactly that the two five-by-five distance matrices are related by vertex relabeling; it does not say that their original SPD points are congruent.

For the triangle inequality, let (g,h\in S_5) minimize the finite orbit distances from (x) to (y), and (y) to (z). Since the action is isometric,

\[
\begin{aligned}
\min_k\lVert x-kz\rVert
&\le \lVert x-ghz\rVert\\
&\le \lVert x-gy\rVert+\lVert gy-ghz\rVert\\
&=\lVert x-gy\rVert+\lVert y-hz\rVert.
\end{aligned}
\]

Division by √10 preserves the inequality. Thus δraw is a metric on the finite (S_5) orbit space of ten-edge distance matrices. Numerical stress tests over random triples passed.

## 3. Pullback caveat and counterexample

When pulled back from distance matrices to original SPD configurations, δraw is generally only a pseudometric. Equal pairwise distance geometry need not imply an orthogonal relation (B_i=Q A_{\pi(i)}Q^T), nor does this audit claim to classify all SPD isometries.

The known-answer fixture uses scalar SPD matrices

\[
A_i=e^{t_i}I,\qquad B_i=e^{-t_i}I,
\]

with (t=(0,0.2,0.55,0.95,1.35)). AIRM gives (d(A_i,A_j)=\sqrt d\lvert t_i-t_j\rvert=d(B_i,B_j)), so the ten-edge geometries are identical. Yet every orthogonal conjugation leaves each scalar matrix unchanged, and the two scalar sets are not equal under any vertex relabeling. The exhaustive fixture therefore has zero quotient distance and strictly positive minimum orthogonal-conjugation mismatch. This is a counterexample to full-configuration completeness, not a classification theorem.

## 4. Common-congruence invariance

For every invertible (A), AIRM is congruence invariant:

\[
d_{\mathrm{AIRM}}(AC_iA^T,AC_jA^T)=d_{\mathrm{AIRM}}(C_i,C_j).
\]

The test suite verifies this for random orthogonal transforms and well-conditioned nonorthogonal transforms. Therefore all ten internal distances, and hence the quotient representation, are invariant to one common (GL(d)) congruence applied to the five states. The representation is not claimed to quotient *exactly* by (GL(d)): distinct (GL(d)) orbits may still share the same distance matrix. A common orthogonal action on all five states is deliberately invisible here.

## 5. Edge-RMS metric size and exact identity

Define the edge-RMS metric size

\[
s(B)=\frac{\lVert x_B\rVert_2}{\sqrt{10}}.
\]

For nondegenerate configurations, \hat x=x/s. The normalized quotient metric and size discrepancy are

\[
\delta_{\mathrm{norm}}(A,B)=\min_\pi
\frac{\lVert\hat x_A-P_\pi\hat x_B\rVert_2}{\sqrt{10}},
\qquad
\delta_{\mathrm{size}}(A,B)=\lvert s(A)-s(B)\rvert.
\]

For any fixed π,

\[
\frac{\lVert x_A-P_\pi x_B\rVert^2}{10}
=s_A^2+s_B^2-2s_As_B\frac{\hat x_A^T P_\pi\hat x_B}{10}.
\]

The same π maximizes the raw and normalized dot products because (s_As_B>0). Taking the optimum and rearranging gives the exact identity

\[
\delta_{\mathrm{raw}}^2
=(s_A-s_B)^2+s_As_B\delta_{\mathrm{norm}}^2.
\]

One hundred random configuration pairs satisfied the identity to numerical precision. It holds at the trial-pair squared-distance level only. It does not imply (J_{\mathrm{raw}}=J_{\mathrm{size}}+J_{\mathrm{norm}}); size and normalized analyses are mechanism controls, not an additive or causal decomposition of the group contrast.

## 6. Degenerate configurations

If (s(B)=0), all ten pairwise distances vanish. Normalization is undefined. The implementation raises `DEGENERATE_METRIC_CONFIGURATION`; it never inserts an epsilon. The real-input gate must list every such trial and stop before scientific inference if any exist.

## 7. Synthetic interaction and null audit

Known-answer 36-by-36 cell-distance fixtures use

\[
M[(s,c),(t,k)]
=m+u\mathbf 1(s\ne t)+v\mathbf 1(c\ne k)
-w\mathbf 1(s=t,c=k).
\]

Five preregistered cases were checked:

| Fixture | Parameters | Expected result | Verified |
|---|---:|---:|---:|
| A: no effects | (u=v=w=0) | (T_J=0) | yes |
| B: subject main only | (u>0,v=w=0) | (T_J=0) | yes |
| C: class main only | (v>0,u=w=0) | (T_J=0) | yes |
| D: additive subject + class | (u,v>0,w=0) | (T_J=0) | yes |
| E: positive interaction | (u,v,w>0) | (T_J=w>0) | yes |

The exact 1,999-draw class-break and subject-break generators are deterministic, move whole cells, preserve their designated blocks, and use independent frozen streams. In the additive subject+class fixture, both one-sided primary p-values are non-significant and the conservative decision does not declare interaction. In the positive known-answer fixture, both correspondence-breaking nulls reject at the Monte Carlo floor (1/2000). Supporting (S) and (C) recover the planted main effects but are not substituted for (J).

## 8. Audit conclusion

All mathematical, known-answer, null-calibration, degeneracy, and brute-force-versus-vectorized tests passed before any new BNCI scientific statistic was viewed. The accepted scientific object is an unlabeled internal AIRM distance-matrix orbit metric. It deliberately excludes absolute SPD position, common congruence pose, and temporal window identity.
