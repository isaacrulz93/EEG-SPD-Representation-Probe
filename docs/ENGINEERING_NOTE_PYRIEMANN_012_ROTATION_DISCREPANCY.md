# pyRiemann 0.12 Euclidean `TLRotate` discrepancy

Status: permanent implementation-history note for the common-subject-action
audit. This note was written before loading any real BNCI common-action object
or computing any common-action scientific score.

## Scientific boundary

The audit's action is the 22-dimensional sensor-space congruence

\[
U \mapsto Q U Q^T, \qquad Q \in O(22),
\]

and its fitting loss in identity-tangent coordinates is

\[
L(Q)=\sum_c w_c\lVert U_c-QB_cQ^T\rVert_F^2.
\]

This is an identity-tangent sensor-space orthogonal-conjugation objective. It
is not the AIRM RPA loss, and it is not an arbitrary rotation of an `svec(U)`
vector space.

## Installed pyRiemann source audit

- Package: `pyriemann==0.12`
- `TLRotate` source:
  `.venv/lib/python3.12/site-packages/pyriemann/transfer/_estimators.py`
  (SHA-256 `945ea0aa34c81afeaca4240ed456d7e6e6a25520616af969f04c4bb1d55cf9a4`)
- Manifold rotation source:
  `.venv/lib/python3.12/site-packages/pyriemann/optimization/grassmann.py`
  (SHA-256 `9cab908bf7a02108eb1370e3b5e1813a9ba568b780210e176bb05499c581fdfa`)
- Inspected symbols: `TLRotate.fit`, `TLRotate._fit_manifold`,
  `TLRotate._fit_tangentspace`, `_get_rotation_manifold`, `_warm_start`,
  `_loss`, `_grad`, `_project`, `_retract`, `_run_minimization`, and
  `_get_rotation_tangentspace`.

For the 3-D manifold path, the implemented action is `Q @ Y @ Q.T`. The code
explicitly starts and compares the positive- and negative-determinant sectors,
projects the Euclidean update onto the tangent of `O(d)`, uses a sign-corrected
QR retraction and Armijo-style backtracking, and stops on step norm or maximum
iterations. The 2-D tangent-space path instead fits a vector-space rotation;
that path is not a permissible substitute for the sensor-space action in this
audit.

## Euclidean loss/gradient inconsistency

In `pyriemann.optimization.grassmann` 0.12:

- `_loss(..., metric="euclid")` returns a weighted sum of
  `distance(..., metric="euclid")`. The distance call is unsquared by default.
- `_grad(..., metric="euclid")` is the analytic gradient of a weighted sum of
  **squared** Frobenius distances.
- `_get_rotation_manifold` documents a squared-distance objective.

Thus the runtime Euclidean loss, analytic gradient, and documented objective
are not mutually consistent. The package is not modified, and its apparent
runtime behavior is not emulated.

In a deterministic exact synthetic congruence fixture (`d=6`, known
negative-determinant action), the installed pyRiemann path had squared loss
`6.55809e-2`, fit-set relative prediction error `1.05590e-1`, and held-out
relative error `1.49073e-1`. The independent squared-loss solver had squared
loss `3.98e-29`, fit-set relative error `2.60e-15`, and held-out relative error
`2.11e-15`. This is why pyRiemann 0.12 Euclidean `TLRotate` is retained only as
implementation history, not as a numerical oracle.

## Original RPA author implementation

The original author repository was inspected at commit
`cfcddb3d31b482941a23353dfbe46dffb118d02d`, file
[`rpa/helpers/transfer_learning/manopt.py`](https://github.com/plcrodrigues/RPA/blob/cfcddb3d31b482941a23353dfbe46dffb118d02d/rpa/helpers/transfer_learning/manopt.py)
(raw-file SHA-256
`d10659e696ed8f14d5a7753096e0e8901c410ee2456ce4ae0b68d48544134d0c`).
Its Euclidean pair cost is explicitly
`norm(M - Q @ Mtilde @ Q.T)**2`, and it uses Pymanopt's historical
`Rotations(n)` plus steepest descent. This supports the historical form of the
squared orthogonal-conjugation objective. It does not make the present
generalized-source/LOCO tangent-effect experiment an RPA replication.

The historical `Rotations(n)` search is determinant-positive. The present
scientific model remains full `O(d)` and therefore uses modern Pymanopt
`Stiefel(d, d)`, with deterministic starts in both determinant components.

## Standard implementation selected for this audit

- Package: `pymanopt==2.2.1`
- Manifold: `Stiefel(d, d, retraction="polar")`; for square matrices this is
  exactly `O(d)`.
- Optimizer: Pymanopt nonlinear Riemannian conjugate gradient with the
  Hestenes-Stiefel update and Pymanopt backtracking line search.
- Cost: the exact squared Frobenius objective stated above.
- Euclidean gradient: analytic, checked against finite differences and
  Autograd; Pymanopt performs the tangent projection and polar retraction.
- Independent check: the pre-existing projected-Armijo solver is retained only
  as a validation implementation.

Installed Pymanopt source hashes used in the audit:

- `manifolds/stiefel.py`:
  `56085dccb036838804ef6a16028fea1d5e9838076c1da2f8a819b42279a0e87d`
- `optimizers/conjugate_gradient.py`:
  `4df13f407bf7e80037f6cf6a28dd8b672a2430ce046b7003a9e391226707de07`
- `optimizers/line_search.py`:
  `b243ebbd8128bf38f154603ef32dfc19dbf80e4fa80ebb3bd7a5d491ef3409e8`
- `core/problem.py`:
  `26bcfadcc75d2694738ff71d75dae332a4e62c4c199e45602060193f765262dd`
