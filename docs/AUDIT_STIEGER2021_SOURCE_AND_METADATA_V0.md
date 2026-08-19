# Audit — Stieger2021 Source and Metadata V0

Audit date: 2026-08-18 Asia/Seoul. This audit is metadata-only and precedes the scientific protocol freeze. No Stieger EEG sample or scientific array has been opened.

## Lineage and immutability

- Exact parent: PR #18, branch `pilot/source-referenced-conditional-residual-v1`, head `19a3ad1cdc6b57c89526e618779cd24b7db8c99c`.
- New branch: `pilot/stieger2021-multiclass-prospective-confirmation-v0` in an isolated worktree.
- Parent manifest SHA-256 values are frozen in the config and checked before every stage. The PR #16, #17, and #18 output directories are read-only inputs to provenance validation and never regenerated.

## Official source manifest

The Figshare API article 13123148 version 1 reports 598 files. Filename-only selection for subjects 1–62 and sessions 2/3 returns exactly 124 unique files, one per subject/session pair, with no missing pair. Their official reported total is 77,689,711,027 bytes (72.3542 GiB); individual files range from about 0.48 to 0.92 GB. Each entry provides a Figshare file ID, download URL, byte size, and MD5 checksum. The complete canonical selected manifest is generated and hashed by the freeze script before any download.

Available workspace capacity at audit was approximately 56 GiB, so retaining all selected raw files concurrently is not possible. One-file streaming is required. This does not change the scientific contract.

## Schema and direct-parser requirement

The published dataset and official MOABB loader document the `BCI` struct with continuous/trial data, time, position, sample rate, `TrialData`, metadata, and channel information. The ordinary MOABB event representation maps target labels but omits `TrialData.tasknumber`; it is therefore insufficient as the sole primary input. The frozen direct parser retains task context and the sealed outcome fields.

The published task contract is task 1 = left/right, task 2 = up/down (both-hand/rest), task 3 = 2D with all four targets. The primary must use task 3 alone to prevent class/task block confounding.

## Access boundary and risks

Before freeze, allowed access is limited to API metadata, checksums, file names/sizes, published descriptions, and loader source. Prohibited access includes downloading/opening a MAT, inspecting EEG values, generating covariances/prototypes, evaluating eligibility from EEG, or calculating scientific statistics.

Primary leakage risks and controls:

- outcome/performance fields: retained sealed, never used in processing;
- acquisition order: retained for deterministic split halves but removed from scientific target features;
- task context: task 3 only in primary;
- target labels: used for class prototypes/source training and sealed evaluation only where declared; never used in pooled target scatter or mixture fitting;
- target fold: excluded from templates, centering, modes, rank selection, source correction, and null calibration;
- pre-target class structure: explicit fail-closed confound gate;
- channel failures: frozen interpolation/ineligibility rules, no adaptive subset;
- raw deletion: blocked until all hash, serialization, reread, and metadata gates pass.

## Anticipated resources

Streaming transfers 72.3542 GiB sequentially. Peak raw disk should remain below the largest source file plus compact buffers (target below 3 GiB). Primary/pre-target task-3 covariance cache is expected to be far below raw size because every trial becomes one or two 20-by-20 float64 matrices plus metadata. Filtering dominates preprocessing; on one worker the anticipated wall time is roughly 20–45 hours depending on network and MAT decompression. The 1,999-replicate nested nulls are expected to require several additional hours using sample-space SVD. Exact observed runtime and peak storage will be recorded at cohort lock.

## Gate

Metadata/source contract: `PASS_METADATA_ONLY_AWAITING_PROTOCOL_FREEZE`. Scientific cohort status and every EEG result remain unaccessed.
