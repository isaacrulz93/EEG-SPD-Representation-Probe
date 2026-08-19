# Stieger2021 Technical Schema Note V0

After the scientific freeze and while the first source file was still an unopened partial download, comparison against the installed official MOABB loader revealed that `noisechan` is stored at `BCI.chaninfo.noisechan` in the loader contract rather than only at `BCI.noisechan`. No MAT had been parsed and no EEG statistic, covariance, eligibility result, or outcome had been accessed.

The implementation was corrected to accept the official nested path while preserving the frozen one-based interpolation rule. `positionsrecorded` receives the equivalent top-level/nested schema fallback. The interrupted partial source is preserved and the downloader now requires an HTTP 206 content-range response before appending, so it is neither silently deleted nor restarted. These are parser/provenance corrections only; sessions, task, classes, channels, epochs, preprocessing, covariance, eligibility, folds, ranks, nulls, estimators, and decisions are unchanged.

After the first source passed its official checksum and was opened, the allowed metadata audit showed that its exact time vector is stored in milliseconds (`-2000, ..., 0, ..., 9040`) at 1000 Hz, while the protocol epochs are specified in seconds. The parser now recognizes only the two explicit sampling-consistent representations (seconds with step `1/SRATE`, or milliseconds with step `1000/SRATE`) and converts the latter to seconds for cropping while retaining the exact raw vector and unit in the compact object. The same metadata audit showed recorded electrode objects with literal `label/X/Y/Z` fields; these are now parsed and preferred as required by the frozen channel contract. No covariance, eligibility decision, population SVD, interaction, semantic match, or recovery statistic had been computed when these schema corrections were made.

A completed retained raw file is now locally re-hashed and reused without reopening the network response. This closes a resume-only orchestration bug that otherwise could append a second response to a file already at its official byte size. It does not change any preprocessing or scientific array.

After the first compact object validated the direct parser, measured single-stream throughput implied an avoidable 14–20-hour transfer. The downloader now uses at most four range connections to one source file when `aria2c` is available, with exactly one raw source file in flight. It still resumes the retained partial, verifies official bytes and MD5 plus local SHA-256 before parsing, and retains any failed partial. This is the bounded one-file streaming option permitted by the frozen storage contract; it does not alter source selection or any scientific setting.

Figshare redirects to short-lived signed S3 URLs. A bounded transfer can make progress and then receive HTTP exit 22 after that redirect expires. The runner therefore re-invokes the original official Figshare URL for a fresh redirect while resuming the same partial. It permits at most 64 invocations and fails after three consecutive no-progress attempts. Final parsing remains blocked until exact byte count and official MD5 pass. After five source sessions demonstrated stable range-resume integrity, the per-file bound was raised from four to eight connections to reduce transfer wall time; exactly one raw file remains in flight, and final compact bytes are checksum-defined rather than transport-defined.

Before cohort lock or any population-statistic access, static review found a
serialization-only defect in `_save_population_result`: an unused block copied
from the scatter-summary path referenced undefined local names. The block did
not contribute to any statistic and would only have raised `NameError` before
the observed summary was written. It was removed without changing model
fitting, scores, folds, ranks, nulls, thresholds, or terminal logic.

A missing reporting-only calculation of predicted scatter ranks was placed in
the scatter summary where it belongs. During that review, an attempted
finite-sample decomposition assertion briefly changed scatter covariance to
`ddof=0`. Rechecking the immutable YAML showed that `scatter.covariance_ddof`
was frozen literally as `1`; before cohort lock or any population/scatter
statistic access, a subsequent commit restored `ddof=1` and removed the extra
assertion, which was not a declared Stieger gate. The pushed correction history
is retained rather than rewritten. The final executable contract therefore
uses the originally frozen `ddof=1`; no estimator, direction, gate, null, or
threshold changed.

Before cohort lock, source-session warnings exposed a second official-schema
distinction. The dataset paper states that `electrodes` contains generic 10-10
coordinates even when `positionsrecorded` is false. The initial direct parser
treated every populated `electrodes` table as individually recorded and thus
bypassed the frozen `standard_1005` fallback. Streaming was stopped with its
current raw partial retained. The parser now uses file coordinates only when
the official `positionsrecorded` flag is true; otherwise it uses the frozen
standard montage. The literal file coordinate table, its hash, the unit
decision, and the interpolation-position source are retained in compact
provenance. Explicitly recorded coordinates pass a strict meter/centimeter/
millimeter scale gate; ambiguous scale fails closed. All 31 pre-lock compact
objects produced by the earlier path were moved intact to an ignored quarantine
and are regenerated from the same hash-locked source files. No cohort, fold,
population, interaction, semantic, or recovery statistic had been computed.

The first corrected fallback parse then failed closed because MNE's
`standard_1005` names do not include the Neuroscan labels `CB1` and `CB2`.
Compumedics/Curry's published release notes explicitly define the corresponding
locations as `CB1 = I1` and `CB2 = I2`. The fallback montage therefore performs
only that vendor-defined deterministic rename before complete-montage
interpolation. The retained checksum-complete raw source is reused; no EEG
selection or scientific parameter is changed.
