# Signed Coordinate Non-Identifiability

For one unseen subject/session, the zero-label observation is the binary mixture

\[
P_s^U = \pi_L P_{s,L}+\pi_RP_{s,R}.
\]

Let `tau` exchange the semantic class names while leaving their component distributions and weights otherwise unchanged. Then

\[
\tau(P_s^U)=\pi_RP_{s,R}+\pi_LP_{s,L}=P_s^U
\]

for balanced OpenBMI classes, and more generally the mixture remains the same unordered weighted measure when labels and their weights are permuted together. Every statistic measurable from the pooled unlabeled trials is therefore invariant to `tau`.

The signed interaction contrast is antisymmetric:

\[
X^{\mathrm{raw}}_{s,q}=\operatorname{svec}\!\left(\frac{Z_R-Z_L}{2}\right)
\quad\Longrightarrow\quad
\tau(X^{\mathrm{raw}}_{s,q})=-X^{\mathrm{raw}}_{s,q}.
\]

For a frozen fold/session mode `b`, both signed coordinates change sign,

\[
\tau(\alpha_{s,q})=-\alpha_{s,q},\qquad
\tau(\beta_{s,q})=-\beta_{s,q},
\]

while `|beta|`, `beta^2`, an unordered separation axis, total projected variance, and between-component energy are unchanged.

Consequently no zero-label rule can distinguish the two signed semantic worlds without an additional semantic anchor or an independently justified structural assumption that breaks the permutation. A signed zero-label predictor would merely encode an arbitrary external orientation and is not an identifiability test.

The frozen signed zero-label decision is therefore:

```text
NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION
```

unless the executable class-swap symmetry contract itself fails, in which case the scientific status is `UNASSESSED_SYMMETRY_CONTRACT_FAILURE`.
