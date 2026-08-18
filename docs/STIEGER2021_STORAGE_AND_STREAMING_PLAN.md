# Stieger2021 Storage and Streaming Plan

The selected 124 official files total 77,689,711,027 bytes, exceeding current free workspace capacity. Processing is bounded to one raw MAT at a time.

For each manifest entry the runner creates an explicit temporary file, streams and hashes it, verifies official MD5 and reported bytes, parses/validates subject-session identity, serializes compact task-specific arrays, rereads and validates them, records the compact SHA-256 and metadata summary, then removes the raw file. Deletion is never attempted if any preceding gate fails. Resume state is per-file and hash-validated; a valid compact object is not silently regenerated.

Ignored cache contains trial covariance/tangent arrays and sealed evaluation metadata. The committed source manifest contains complete regeneration provenance and hashes. A repository-size check decides whether any compact scientific NPZ is committed; raw MAT and continuous EEG are always excluded.

Peak expected storage is the largest raw file, decompressed in-memory trial data, and one compact session object. The implementation uses one worker and one download at a time. The cohort lock records actual bytes transferred, raw hashes, compact hashes, failures, eligible subjects, exclusions, and exact folds.
