# Audit: Subject Location → Conditional Configuration V0

## Parent and provenance

The implementation branch is rooted at the exact draft PR #19 commit
`6abb73d82a0f616e0ca9d3eaa44e23d911a2123f`. The locked Stieger2021 geometry
object has SHA-256
`bce785c13d3e851fb73e5554a5efc990592e832946b53b64fcdc08a708c83515`.
The inherited fold object's canonical SHA-256 is
`a3bf9afddb83ab0c0f192b7e337a44dabe24790a2bb083316aa4d18c0347610d`,
and the streamed manifest's canonical SHA-256 is
`60aa67ccf0eaee9ef2618e4c775eec5f8f4c60bb40d1c8eb0c3b244dbe53f465`.

PR #20 and PR #21 are negative scientific context only. Their model code is
neither inspected nor imported. This pilot is not a persistent-memory rescue
and contains no low-rank-map SPDNet.

## Locked object contract

The compact NPZ contains 35 keys. Metadata keys are `subjects`, `groups`,
`class_names`, `sessions`, and `channels`. Each of the six epoch/split cells
(`primary` or `pretarget`, crossed with `F`, `A`, or `B`) contains `U`,
`proportions`, `counts`, `class_means`, and `marginal`.

Only these full-split objects are used:

| Key | Shape | Meaning |
|---|---:|---|
| `primary__F__marginal` | 62×2×20×20 | subject/session marginal AIRM means |
| `primary__F__class_means` | 62×2×4×20×20 | subject/session/class AIRM means |
| `primary__F__proportions` | 62×2×4 | class proportions |
| `primary__F__counts` | 62×2×4 | class counts |

Subjects are ordered 1 through 62. Sessions are ordered 2 then 3. Classes are
ordered `right_hand`, `left_hand`, `both_hand`, `rest`. All marginal and class
mean matrices are finite SPD(20); every required subject/session/class cell is
present and has a positive trial count. The tangent coordinate dimension is
20×21/2 = 210.

The six outer test folds have sizes 10, 10, 10, 11, 11, and 10 and partition
all 62 subjects exactly once. Each outer source population has five inherited
inner validation folds that partition that source population exactly once.

## Coordinate convention

The inherited `svec` uses the upper triangle in NumPy row-major order.
Diagonal entries are unchanged and off-diagonal entries are multiplied by
sqrt(2), so Euclidean dot products equal Frobenius inner products of symmetric
matrices. No coordinate standardization is used.

## Safety conclusions

- The required 62×2 marginal and 62×2×4 conditional means are reconstructible
  exactly from the locked compact object; raw EEG is unnecessary and forbidden.
- Fold references, conditional population means, output bases, and ridge models
  can all be fit source-only.
- A dedicated label-free packet and a separately hashed conditional-outcome
  vault can enforce prediction-before-evaluation.
- A dual ridge solve is appropriate because each source fold has at most 52
  subjects while the input has 210 coordinates.
- No classifier, neural network, subject identity feature, TTA, or target-fitted
  component is required.

Failure of any exact hash, schema, SPD, fold, AIRM, zero-sum, or vault gate is
an `UNASSESSED_*` condition; there is no raw-data fallback.
