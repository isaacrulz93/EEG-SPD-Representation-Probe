# Stieger2021 Prospective Confirmation Feasibility

Status: **METADATA-ONLY; NO RAW DOWNLOAD, PREPROCESSING, COVARIANCE, OR SCIENTIFIC STATISTIC**

Audit date: 2026-08-18 (Asia/Seoul)

This document was prepared independently of the Phase-1 scientific outcome. A positive OpenBMI result cannot change the Stieger class set, channel rule, session-pair rule, preprocessing, exclusions, statistic, or terminal logic. Any future confirmation requires a new clean branch and a separately committed protocol before scientific access.

## Official loader and source contract

The repository environment pins MOABB 1.5.0. Its installed `moabb.datasets.stieger2021.Stieger2021` source was read as text without instantiating the loader. The pinned constructor declares:

- subject IDs `1..62`;
- maximum `sessions_per_subject=11`, while the dataset description states that subjects completed 7–11 sessions;
- event mapping `right_hand=1`, `left_hand=2`, `both_hand=3`, `rest=4`;
- default cue-relative interval `[0,3]` seconds;
- imagery paradigm;
- Figshare item `13123148` and paper DOI `10.1038/s41597-021-00883-1`;
- raw sampling rate read from each MATLAB container (official metadata: 1000 Hz);
- standard 10-05 EEG channel-name selection from the recorded montage;
- one returned MNE Raw run per subject/session.

The current official MOABB documentation, checked on 2026-08-18, describes 62 subjects, 7 or 11 sessions, four classes, 3 s nominal trials, 1000 Hz, visual feedback, and approximately 250,000 trials. The current loader API additionally exposes `fix_bads`, subject filtering, trial-length metadata, and interval suggestions. These newer conveniences are not silently assumed by this repository's pinned 1.5.0 environment. Sources: [MOABB Stieger2021 documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Stieger2021.html), [official MOABB loader](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/stieger2021.py), [official Figshare record](https://figshare.com/articles/dataset/Human_EEG_Dataset_for_Brain-Computer_Interface_and_Meditation/13123148).

## Session, task, and class availability

Participants performed 7–11 longitudinal BCI sessions. Each session nominally contained LR, UD, and 2D cursor-control tasks, with two blocks per task and 450 total trials per day. The four loader labels are common target semantics across subjects and repeated sessions, so repeated-session subject pairing and a within-subject condition `c` are available in principle.

The design targets left, right, both-hands/up, and rest/down directions. Nominal target scheduling is intended to balance task directions, but the usable four-class counts are not asserted to be exactly equal: artifact flags and variable trial lengths cause the pinned loader to reject trials, and no raw/TrialData metadata was accessed here. A future protocol must freeze either exact-count inclusion or class-proportion weighting before loading scientific arrays. It may not choose a favorable subset after inspecting population-mode results.

The maximum BaseDataset session count is not evidence that every subject has 11 valid files. Exact per-subject session IDs and the common repeated-session pairing set remain **UNASSESSED_METADATA_REQUIRING_FUTURE_DOWNLOAD**. The future protocol must resolve this from file metadata only, save the eligibility table, and freeze a session-pair rule before covariance/statistic access.

## Feedback and learning difference

Stieger is an online closed-loop learning dataset. Participants controlled a continuously updated cursor with visual feedback, and performance/neural control could change across 7–11 training days. OpenBMI V0 used an offline training phase and BNCI used conventional cue-based MI sessions. Consequently:

- repeated sessions are scientifically useful but are not exchangeable replicates of an unchanged acquisition state;
- a stable mode could reflect longitudinal task learning, feedback adaptation, intervention, or recording stability in addition to subject×class prototype structure;
- first/last, adjacent, or all-pairs session analyses answer different questions;
- intervention group and session time must not be used to select a favorable pairing after OpenBMI results.

A future confirmation must predeclare whether it tests early-session replication, adjacent-session reproducibility, or a longitudinal extension. Only the first two-session choice most closely matches the present two-view model, and even that is not an exact protocol replication because feedback and learning differ.

## Channel and preprocessing compatibility

The loader produces standard-named scalp EEG channels and MNE Raw objects, so the repository's continuous filtering, epoching, OAS covariance, AIRM mean, identity-tangent `U`, and Full/A/B object code is technically reusable. Compatibility is conditional on a separately frozen contract for:

- an exact ordered channel intersection present for every included subject/session;
- bad-channel handling (the current official loader can interpolate; the pinned loader contract differs);
- cue-relative interval under variable trial duration;
- continuous-filter boundary handling and sampling-rate policy;
- artifact rejection already encoded in TrialData;
- acquisition-order split halves within subject/session/class;
- exact session eligibility and class counts.

No channel subset, bad-channel policy, interval, or session pair is selected in this branch. Using the OpenBMI 20-channel montage by convenience would require an authoritative overlap and a new pre-result freeze; using all Stieger channels would change ambient sensor geometry and comparison scope.

## Storage and preprocessing estimate

The official Figshare record reports a 351.01 GB download. The current volume had approximately 58 GiB free during this audit, so a full raw download is not feasible under the present storage budget and was not attempted.

At approximately 250,000 trials, a 3 s, 62-channel, 1000 Hz float64 epoch bank would be roughly 372 GB before overhead. A full 62×62 float64 covariance bank would be roughly 7.7 GB; a fixed 20-channel covariance bank would be roughly 0.8 GB. Raw filtering, MATLAB parsing, MNE object construction, artifact handling, and repeated covariance estimation would require an estimated 1–3 wall-clock days with four workers plus at least 400–500 GB of safe working space. These are planning estimates, not measured runtimes.

The future workflow should use subject-wise streaming and compact class/marginal means, but may not discard provenance or alter the scientific trial set to save space. A complete raw/source hash manifest is required before statistic access.

## Feasibility decision

`FEASIBLE_IN_PRINCIPLE_BUT_BLOCKED_BY_STORAGE_AND_UNFROZEN_LONGITUDINAL_CONTRACT`

The dataset has the necessary repeated subjects and within-subject class conditions. The current branch does not have enough free storage and has not resolved an outcome-independent session/channel/bad-trial contract. No confirmation claim, real statistic, or preprocessing artifact exists.
