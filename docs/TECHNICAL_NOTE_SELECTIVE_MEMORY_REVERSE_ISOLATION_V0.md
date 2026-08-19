# Technical note — non-voting reverse isolation

After protocol freeze, the Stieger chronological voting analysis completed and
the automatically chained descriptive reverse analysis reached the frozen
L-BFGS-B 500-iteration limit in one inner fit. No voting chronological
optimization failed. The reverse direction is explicitly non-voting.

This recovery changes only orchestration: a numerical exception in a reverse
diagnostic is stored as `CONTROL_UNASSESSED_OPTIMIZATION_FAILURE`, while the
chronological voting pipeline remains available. The optimizer, iteration
limit, feature definitions, folds, L2 grid, scaling, inference, nulls, and
decisions are unchanged. Voting chronological numerical failures still
propagate and fail closed. The failed reverse optimization is not regularized,
restarted with new settings, or removed from history.

Before the frozen 1,999-replicate nulls, the same scalar per-subject
cross-entropy and analytic gradient were additionally expressed as one batched
array operation. A regression test requires the scalar and vectorized fitted
parameters and objectives to agree within strict floating-point tolerance. This
changes neither the objective nor any scientific setting; it only removes
Python-loop overhead inside the null pipeline.
