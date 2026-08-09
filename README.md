# EEG-SPD-Representation-Probe

An independent, diagnostic-only study of what is hidden when one
BNCI2014_001 motor-imagery trial is compressed into one channel covariance
matrix. The project compares one whole-trial covariance (`WHOLE`) with five
ordered, non-overlapping local covariances (`WINDOW5`) before and after simple
subject-marginal centering in Log-Euclidean coordinates.

This is not a model-development repository. It contains no neural network,
sequence model, attention mechanism, domain adaptation, pseudo-labeling,
optimal-transport loss, or hyperparameter sweep. t-SNE is used only for
visualization; every quantitative diagnostic is computed in the original
253-dimensional `svec(log(C))` space.

## Frozen primary protocol

- Dataset: MOABB `BNCI2014_001`, all 9 subjects, chronological session 1
  (`0train`) only, four motor-imagery classes.
- Channels: the 22 EEG channels explicitly listed in the YAML config; EOG is
  excluded and channel names/types are validated after loading.
- Epoch: 0.000--3.996 s relative to the dataset event (physical source
  interval 2.000--5.996 s), 1,000 samples at 250 Hz. The MNE-inclusive final
  sample at 6.000 s is excluded up front so the common analyzed interval is
  exactly divisible into five equal windows.
- Filter: 8--32 Hz (the fixed MOABB `MotorImagery` default); no resampling and
  no baseline.
- Covariance: OAS shrinkage for both WHOLE and WINDOW5, followed only by
  numerical symmetrization. The implementation is pyRiemann
  `Covariances(estimator="oas")`, whose OAS backend is sklearn. No extra
  eigenvalue clipping or tuned ridge.
- WINDOW5: five ordered, non-overlapping 200-sample windows. Exact division is
  required; unexpected remainders cause an error rather than silent deletion.
- Coordinates: symmetric eigendecomposition matrix logarithm followed by
  Frobenius-isometric `svec` (22 diagonal terms unchanged, off-diagonals
  multiplied by `sqrt(2)`), yielding 253 dimensions. No `StandardScaler`.
- Centering: one mean per subject. WINDOW5 uses all trials and all windows
  together; labels and window indices are not used.
- Visualization: one fixed PCA(40) + t-SNE fit per representation/state,
  seed 20260809. WINDOW5 panels reuse one global embedding.
- Linear information probe: fixed `C=1` multinomial logistic regression,
  5-fold subject-by-class-stratified group splits, grouped by trial, without
  tuning or scaling. WINDOW5's primary accuracy averages held-out probabilities
  over the five windows of each trial; individual-window accuracy is secondary.
- Example trajectories: subjects 1, 2, and 3, selecting the smallest `trial_id`
  for every subject/class. RAW and CENTERED use exactly the same trials.

The protocol above and all thresholds in
[`configs/bnci2014_001_5window.yaml`](configs/bnci2014_001_5window.yaml) were
fixed before examining representation results.

## Environment

The project uses a Python 3.12 virtual environment because the machine's
Homebrew Python 3.14 installation is outside the conservative MOABB support
path. GPU and PyTorch are not required.

```bash
cd ~/EEG-SPD-Representation-Probe
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Exact runtime package and platform metadata are captured by the preparation
script in `outputs/bnci2014_001/tables/environment.json`. The completed run used
macOS 26.5.2 arm64, CPython 3.12.8, NumPy 2.5.1, SciPy 1.18.0,
scikit-learn 1.9.0, matplotlib 3.11.1, pandas 3.0.5, MOABB 1.5.0, and
pyRiemann 0.12. PyTorch was not installed and CUDA was unavailable; the full
pilot ran on CPU.

## Reproduce

Run each frozen stage:

```bash
.venv/bin/python scripts/01_prepare_bnci.py --config configs/bnci2014_001_5window.yaml
.venv/bin/python scripts/02_build_covariances.py --config configs/bnci2014_001_5window.yaml
.venv/bin/python scripts/03_make_embeddings.py --config configs/bnci2014_001_5window.yaml
.venv/bin/python scripts/04_compute_diagnostics.py --config configs/bnci2014_001_5window.yaml
.venv/bin/python scripts/05_make_report.py --config configs/bnci2014_001_5window.yaml
```

Or run the same stages in order:

```bash
.venv/bin/python scripts/run_all.py --config configs/bnci2014_001_5window.yaml
```

Run the validation suite with:

```bash
.venv/bin/python -m pytest -q
```

MOABB downloads and regenerable numerical arrays stay under `cache/` and are
ignored by Git. Small metadata CSV files, diagnostic tables, PNG figures, and
the Markdown report are tracked.

## Verified dataset metadata

These values come from the loaded MOABB epochs and runtime dataset inventory,
not from a hard-coded summary block:

| Item | Observed value |
| --- | --- |
| Dataset | `BNCI2014_001` |
| Subjects | 9 (`1` through `9`) |
| Sessions available | 2 (`0train`, `1test`) |
| Primary session analyzed | `0train` only |
| Classes | 4 (`left_hand`, `right_hand`, `feet`, `tongue`) |
| EEG channels | 22; all output types were EEG and output EOG count was 0 |
| Sampling frequency | 250 Hz |
| Dataset event interval | 2.000--6.000 s |
| Analyzed source interval | 2.000--5.996 s |
| Relative epoch | 0.000--3.996 s |
| Samples per trial | 1,000 |
| Loaded shape | `(2592, 22, 1000)` |
| Trials | 288 per subject, 72 per subject/class, 2,592 total |
| Cached signal units | volts (MOABB converts source MAT microvolts to volts) |
| WHOLE covariances | 2,592 matrices of shape `(22, 22)` |
| WINDOW5 covariances | 12,960 matrices of shape `(22, 22)` |

The exact channel order is `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz,
C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`. Full counts are in
[`dataset_metadata.json`](outputs/bnci2014_001/tables/dataset_metadata.json).

## Primary result snapshot

- All 2,592 WHOLE and 12,960 WINDOW5 matrices were SPD; no NaN, Inf, or
  silently dropped covariance occurred.
- RAW class linear-probe accuracy was 0.5660 for WHOLE and 0.5880 for WINDOW5
  at trial level. CENTERED values were 0.6119 and 0.6211. These are information
  probes, not classifier-performance claims.
- RAW subject-probe accuracy was 0.9996 for WHOLE and 1.0000 for WINDOW5; after
  subject marginal centering it was 0.0123 and 0.0162.
- H1--H6 verdicts were respectively `MIXED`, `SUPPORTED`, `SUPPORTED`, `MIXED`,
  `NOT SUPPORTED`, and `NOT SUPPORTED`.

The interpretation, threshold votes, limitations, and the single recommended
next experiment are in the
[`representation_probe_report.md`](outputs/bnci2014_001/report/representation_probe_report.md).

## Outputs

- Final report: [`representation_probe_report.md`](outputs/bnci2014_001/report/representation_probe_report.md)
- Covariance checks: [`covariance_sanity.csv`](outputs/bnci2014_001/tables/covariance_sanity.csv)
- Separability: [`separability_metrics.csv`](outputs/bnci2014_001/tables/separability_metrics.csv)
- Linear probes: [`linear_probe_metrics.csv`](outputs/bnci2014_001/tables/linear_probe_metrics.csv)
- Figures: [`outputs/bnci2014_001/figures/`](outputs/bnci2014_001/figures/)

## Geometry audit V2 (branch-only pilot)

The `pilot/geometry-audit-v2` branch adds a preregistered WHOLE-covariance
audit without changing the V1 experiment or its outputs. It compares RAW,
Log-Euclidean Fréchet centering, AIRM Fréchet congruence centering, and an
arithmetic-mean congruence control under leakage-safe LOSO evaluation. The
primary protocol is transductive label-free target centering; complementary
run halves provide the calibration-to-held-out-run secondary protocol.

The frozen definitions, tolerances, leakage barriers, table schemas, and
decision rules are in
[`PROTOCOL_GEOMETRY_V2.md`](docs/PROTOCOL_GEOMETRY_V2.md). Run the stages in
order:

```bash
.venv/bin/python scripts/10_geometry_correctness_v2.py --config configs/bnci2014_001_geometry_v2.yaml
.venv/bin/python scripts/11_loso_alignment_v2.py --config configs/bnci2014_001_geometry_v2.yaml
.venv/bin/python scripts/12_v1_leakage_audit.py --config configs/bnci2014_001_geometry_v2.yaml
.venv/bin/python scripts/13_geometry_report_v2.py --config configs/bnci2014_001_geometry_v2.yaml
```

Or execute the same fixed sequence with:

```bash
.venv/bin/python scripts/run_geometry_v2.py --config configs/bnci2014_001_geometry_v2.yaml
```

Stage 10 is a hard gate. A failed geometry check returns nonzero, and the
one-command runner stops before fitting any classifier. V2 writes only below
`outputs/bnci2014_001_geometry_v2/`; the V1 cache is validated by fixed hashes
and reused read-only when it matches the frozen preprocessing contract.
