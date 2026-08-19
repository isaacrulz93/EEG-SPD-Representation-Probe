# LRCM Mathematical Specification V0

`LRCM` means **Low-Rank Conditional Memory**. For source-only centered session
signatures `A` and `B`, define `K=A A^T + lambda I` and solve `K T=B`. Then
`W=A^T T`; no explicit inverse is formed. If the thin SVD of `A W` has right
vectors `V`, the rank-R prediction can be evaluated without materializing a
large truncated matrix:

```text
G = A.T @ T @ V[:, :R]
a_target = (x_enrollment - mean_enrollment) @ G
x_deployment_hat = mean_deployment + a_target @ V[:, :R].T
```

The stored returning-user memory is `a_target`, an R-vector. With float64 its
payload is `8R` bytes, excluding population-shared model parameters and metadata.
The full-ridge prediction is `mean_deployment + (x-mean_enrollment) @ A.T @ T`.

Enrollment PCA transfer differs mechanistically: it obtains `V_E` from `A`
alone, maps `A @ V_E` to `B` with ridge, and therefore never uses cross-session
pairing to choose the input subspace. Identity residual carry is
`mean_deployment + x_target_enrollment - mean_enrollment`.

For a class-centered prototype matrix `P0`, `H.T @ (H @ P0) = P0` because
`H.T H` is the projector orthogonal to the all-ones class vector. No unit norm is
applied, so prototype displacement magnitude is retained.

