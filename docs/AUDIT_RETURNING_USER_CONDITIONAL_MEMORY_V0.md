# Audit — Returning-User Conditional Memory V0

## Lineage and immutability

The isolated branch starts exactly at PR #19 head
`6abb73d82a0f616e0ca9d3eaa44e23d911a2123f`. The audit snapshots every tracked
file below the PR #16--#19 output namespaces before real downstream access. The
snapshot is written to the new protocol namespace and verified again after the
scientific result. Parent directories are never write targets.

## Stieger2021 cache contract

The committed manifest is
`outputs/stieger2021_multiclass_confirmation_v0/objects/trial_tangent_cache_manifest.json`.
Its file SHA-256 is `c3951e33ff8d83e2a25cc7a8d33ed0eaac528ab5f724b5014019b1bd2ce45463`
and canonical content SHA-256 is
`9eaeda751afd6529b45a11c7e50c59d7631985f559c54eff318d0aacb2d11f06`.
It declares 124 records: subjects 1--62 crossed with sessions 2 and 3. Each
ignored cache file contains scalar subject/session identity, `targetnumber`,
`acquisition_index`, and finite float64 `primary_trial_svec` with 210 columns.
Labels are stripped into a separate evaluator before prediction. Literal label
values 1--4 map to right hand, left hand, both hands, and rest. The existing
tangents were produced from the frozen session-wise all-trial marginal. No raw
Stieger data are read, downloaded, or regenerated.

The exact inherited fold file SHA-256 is
`f5df81c88f64dd4125db98edae6f60437a450a3ec5cd2b2c5a0e137d8ab41c34`
with canonical membership hash
`a3bf9afddb83ab0c0f192b7e337a44dabe24790a2bb083316aa4d18c0347610d`.

## OpenBMI cache and chronology contract

The committed covariance manifest SHA-256 is
`d1cda390e44f503eebad73b679f85da4bab0e266ee37bebaca83d342f68a384a`.
The combined covariance cache, metadata, tangent cache, and analysis-core SHA-256
values are respectively `8fc43ff837c9ab1651f639c30a2aa4a09e760a6a2192b4c5407848fd9da04b28`,
`f89329519b900da5585abdd495aa601dddabc030567883137088ad674bf83ea1`,
`36b56818cdfca24212666e5181cca120c20a0ccc553f51cde9009663fbb1be39`, and
`5714d959e6696fd99f403b6c9c425c4ef8be64e9a2b9485a3d072ead554d26e8`.
The feature tensor is finite float64 `(54,2,100,210)`; class labels are 50/50 and
trial IDs are `(54,2,100)`. Parent metadata explicitly maps array sessions
`"0","1"` to source session IDs `1,2`; therefore chronology is session 1 to 2,
not an inference from array indexing. The committed PR #16 fold membership is
used byte-for-byte through its frozen CSV.

## Leakage boundary

The implementation loads tangent features, identities, and enrollment labels
into fitting objects. Deployment labels and label-derived metrics live in a
separate evaluation object. A label-access sentinel is asserted around every
outer prediction builder. Neither target deployment labels nor target scores can
enter source means, maps, rank/ridge/temperature selection, prototype prediction,
null fitting, or stopping. K-shot and full-oracle labels are evaluation-only
baselines and never vote on LRCM fitting.

## Resource estimate and gate

The existing external cache footprint is approximately 435 MiB for Stieger and
1.2 GiB for OpenBMI; symbolic links add no duplicate payload. Anticipated peak
RAM is below 4 GiB. Observed/nested fitting is expected to take tens of minutes;
the deterministic 1,999-replicate controls and 200-draw K-shot curves may take
several hours on one worker. The program records actual runtimes.

Before protocol freeze, the executable audit must reproduce all hashes, counts,
shapes, finite checks, class balance, identities, fold coverage, and parent
immutability. Any mismatch yields
`UNASSESSED_REQUIRED_TRIAL_CACHE_MISSING_OR_INVALID`; no alternate preprocessing
or raw reconstruction is authorized.
