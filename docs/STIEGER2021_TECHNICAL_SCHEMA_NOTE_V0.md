# Stieger2021 Technical Schema Note V0

After the scientific freeze and while the first source file was still an unopened partial download, comparison against the installed official MOABB loader revealed that `noisechan` is stored at `BCI.chaninfo.noisechan` in the loader contract rather than only at `BCI.noisechan`. No MAT had been parsed and no EEG statistic, covariance, eligibility result, or outcome had been accessed.

The implementation was corrected to accept the official nested path while preserving the frozen one-based interpolation rule. `positionsrecorded` receives the equivalent top-level/nested schema fallback. The interrupted partial source is preserved and the downloader now requires an HTTP 206 content-range response before appending, so it is neither silently deleted nor restarted. These are parser/provenance corrections only; sessions, task, classes, channels, epochs, preprocessing, covariance, eligibility, folds, ranks, nulls, estimators, and decisions are unchanged.
