# Selective Memory Gate Specification V0

The deployment prototype is constrained to the line segment between the frozen
population-only and identity-residual-carry prototypes:

```text
population: mu_0[c] = zbar_D^U + gamma_D[c]
identity:   mu_1[c] = zbar_D^U + gamma_D[c] + r_E[c]
selective:  mu_k[c] = mu_0[c] + kappa_s r_E[c]
```

The only learned subject variation is `kappa_s=sigmoid(w^T h_s+b)`. Its
24-dimensional input is constructed exclusively from enrollment split-half
reliability, magnitude, variance, count, and prototype-distance summaries.
Thus `kappa=0` is exactly population only and `kappa=1` is exactly the frozen
identity residual carry baseline. No decoder, latent transfer, deployment
feature, or neural representation is present.

Training uses only outer-source paired sessions. Evaluation labels of an outer
target never enter the gate, while target-label oracle kappa is stored as a
separate ceiling. This makes the experiment a direct test of whether enrollment
reliability selects reuse strength—not whether an unconstrained predictor can
learn a new cross-session residual map.
