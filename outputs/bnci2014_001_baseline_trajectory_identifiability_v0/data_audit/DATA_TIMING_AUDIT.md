# BNCI2014_001 Data Timing Audit

DATA gate: **PASS**

- Dataset: `moabb.datasets.BNCI2014_001` under MOABB 1.5.0
- Raw inventory: 18 MAT files, 9 subjects, 2 sessions, 6 runs/session.
- Stim event: trial start; raw annotation: cue onset exactly +2.0 s / +500 samples later.
- Baseline C0: `[cue-250, cue)`, exactly 250 samples.
- Post-cue: `[cue, cue+1000)`, exactly 1000 samples; five non-overlapping 200-sample windows.
- Channels: 22 ordered EEG channels; EOG1-3 and STI excluded from covariance.
- Filtering: continuous run, MNE IIR 8-32 Hz, default order-4 Butterworth SOS, zero-phase forward-backward.
- Leakage note: zero-phase filtering intentionally crosses the cue boundary. Padding occurs only at continuous-run edges, never at trial/window boundaries.
- Covariance: pyRiemann OAS, float64, numerical symmetrization, no trace normalization, no added jitter.
- Trial balance: 5184 total; 12 per subject/session/run/class.
- Old WINDOW5 reproduction: PASS, max absolute difference `1.827e-17`.

## SPD numerical gate

- C0: n=5184, min eigenvalue=6.996925e-14, max symmetry error=0.000e+00.
- C1..C5: n=25920, min eigenvalue=8.112918e-14, max symmetry error=0.000e+00.
- Cfull: n=5184, min eigenvalue=4.046877e-14, max symmetry error=0.000e+00.

The full event-level timing table is `data_audit/event_timing.csv`; run/filter details are in `data_audit/filter_contract.csv`. No label enters filtering, slicing, or covariance estimation.
