# Beta Reference Identity

For outer fold `f`, session `q`, training subjects `T`, and frozen rank-1 mode `b[f,q]`, define

`d_proto[s,q] = b[f,q]^T svec(0.5 (U[s,q,R] - U[s,q,L]))`

and

`gamma[f,q] = b[f,q]^T svec(0.5 (mean_T U[:,q,R] - mean_T U[:,q,L]))`.

With balanced binary classes, the class-independent residual cancels from the class contrast. The outer-fold population template is the training mean for a held-out subject, so

`0.5 (Z[s,q,R] - Z[s,q,L]) = 0.5 (U[s,q,R] - U[s,q,L]) - 0.5 (mu[T,q,R] - mu[T,q,L])`.

Frobenius-isometric vectorization and linear projection therefore imply the exact contract

`beta[s,q] = d_proto[s,q] - gamma[f,q]`.

This equality is checked against the immutable PR #17 `beta` for all 108 held-out subject/session observations. It is an algebraic gate, not evidence selected by a statistical threshold. A failure terminates the experiment as `UNASSESSED_BETA_REFERENCE_IDENTITY_FAILURE`.

The trial projection contrast

`delta_trial = 0.5 (mean(y|R) - mean(y|L))`

estimates the total prototype coordinate `d_proto`, not the residual coordinate `beta`. Arithmetic means of individual trial log coordinates do not have to equal the log of an AIRM class mean. The predeclared source-only additive correction estimates that trial/prototype curvature difference using outer-training subjects only:

`correction[f,q] = mean_{r in T} (d_proto[r,q] - delta_trial[r,q;f])`.

The primary full-trial prediction is consequently

`beta_ref_full = delta_trial_full + correction - gamma`.

No target observation participates in `gamma`, correction, source-only affine calibration, or semantic source order.
