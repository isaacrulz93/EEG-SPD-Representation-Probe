# Mathematical Specification: Subject Location → Conditional Configuration V0

Let `M_s^(v)` and `C_s,c^(v)` be the locked full-split marginal and class AIRM
means for subject s, session v, and class c. For one outer source set S,

`R_s = argmin_R [d_AIRM²(R,M_s^(2)) + d_AIRM²(R,M_s^(3))]`,

`M_0 = argmin_M sum_{s in S} d_AIRM²(M,R_s)`.

All tangent coordinates use the congruence chart at M_0. The label-free input
is

`q_s^(u) = svec(log(M_0^(-1/2) M_s^(u) M_0^(-1/2))) in R^210`.

For each output class,

`z_s,c^(v) = svec(log(M_0^(-1/2) C_s,c^(v) M_0^(-1/2)))`,

`d_s,c^(v) = z_s,c^(v) - (1/4) sum_j z_s,j^(v)`,

`d_bar_c^(v) = |S|^(-1) sum_{s in S} d_s,c^(v)`,

`Delta_s,c^(v) = d_s,c^(v) - d_bar_c^(v)`.

Therefore `sum_c d_s,c = sum_c Delta_s,c = 0`. Equal-class centering, rather
than prevalence-weighted centering, prevents the marginal subject location
from being copied into the prediction target.

For centered source matrices Q and D, let `D = U S V^T`; for rank r use
`Y_r = D V_r`. With `K = Q Q^T` and ridge lambda, solve

`alpha = (K + lambda I)^(-1) Y_r`

by a stable linear solve. A centered query q has score

`y_hat = (q-q_bar) Q^T alpha`,

and `Delta_hat = d_mean + y_hat V_r^T`. Rank zero is specially defined as the
exact zero-residual population predictor, not an intercept model.

The primary error is `SSE_location = ||Delta-Delta_hat||²`, the baseline is
`SSE_zero = ||Delta||²`, and

`R²_cond = 1 - sum SSE_location / sum SSE_zero`.

The source-pairing null permutes source rows of D relative to Q, holds the
source-selected rank and ridge fixed, refits, and evaluates the correctly
paired held-out q. The target-pairing null deranges already frozen held-out
predictions within each fold. Neither null reopens model selection.

All AIRM means use float64, tolerance 1e-9, at most 100 iterations, Karcher
residual at most 1e-7, and condition number at most 1e12. Failure is closed;
no Log-Euclidean or identity fallback is allowed.
