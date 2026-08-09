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
- Filter: 8--30 Hz through MOABB `MotorImagery`; no resampling and no baseline.
- Covariance: OAS shrinkage for both WHOLE and WINDOW5, followed only by
  numerical symmetrization. No extra eigenvalue clipping or tuned ridge.
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
  5-fold stratified group splits, grouped by trial, without tuning or scaling.

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
script in `outputs/bnci2014_001/tables/environment.json`.

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

MOABB downloads and regenerable numerical arrays stay under `cache/` and are
ignored by Git. Small metadata CSV files, diagnostic tables, PNG figures, and
the Markdown report are tracked.

## Verified dataset metadata

The final observed subject/session/class/channel/trial counts are written here
after the primary run and are also available in
`outputs/bnci2014_001/tables/dataset_metadata.json`.

## Outputs

- Final report: `outputs/bnci2014_001/report/representation_probe_report.md`
- Covariance checks: `outputs/bnci2014_001/tables/covariance_sanity.csv`
- Separability: `outputs/bnci2014_001/tables/separability_metrics.csv`
- Linear probes: `outputs/bnci2014_001/tables/linear_probe_metrics.csv`
- Figures: `outputs/bnci2014_001/figures/`

