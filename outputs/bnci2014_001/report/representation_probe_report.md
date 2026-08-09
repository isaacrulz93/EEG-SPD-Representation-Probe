# BNCI2014_001 SPD Representation Probe

## 1. Question

This frozen pilot asks what is hidden when one motor-imagery trial is compressed into one 22 × 22 covariance matrix. It compares that WHOLE state with five ordered, non-overlapping local covariance states from exactly the same preprocessed trial interval. The two questions are whether local class/subject/temporal structure becomes visible and whether class-related temporal structure remains after label-free subject marginal centering. It does not propose or evaluate a new model.

## 2. Frozen protocol

The primary analysis loaded only session `0train` from the 2 available `BNCI2014_001` sessions (0train, 1test), using all 9 subjects, four classes (feet, left_hand, right_hand, tongue), and 22 EEG channels. The output contained 0 EOG channels. The configured epoch was 0.000–3.996 s relative to the motor-imagery cue; its observed source-time interval was 2.000–5.996 s within the dataset event interval 2.000–6.000 s (1000 samples at 250.0 Hz). Cached EEG amplitudes were in volts (source MAT units: microvolts).

A single pipeline applied an 8–32 Hz band-pass, no baseline correction, and no resampling. WHOLE used all 1000 samples. WINDOW5 divided those samples into five consecutive 200-sample blocks, with no overlap and `require_exact_division` remainder handling. Both used OAS covariance (pyriemann.estimation.Covariances(estimator='oas'); pyRiemann delegates to sklearn.covariance.oas), deterministic symmetrization, and no extra regularization.

Each SPD matrix was mapped by a symmetric eigendecomposition to `log(C)` and Frobenius-isometric `svec` coordinates (22 × 23 / 2 = 253 dimensions; diagonal unchanged, off-diagonal multiplied by sqrt(2)). No StandardScaler was used. CENTERED coordinates subtract one mean per subject: all trials for WHOLE and all trial × window samples together for WINDOW5. Neither class labels nor window indices enter centering.

Visualization used one fixed PCA(40) + t-SNE fit for each of the four representation/state combinations (seed 20260809, perplexity 30.0); every WINDOW5 panel reuses its representation's global coordinates. Quantitative diagnostics use the original 253-D coordinates, not t-SNE. The linear information probe was fixed multinomial logistic regression (`C=1.0`, 5 stratified group folds, no scaling or tuning), with every trial's five windows kept in one fold.

Runtime record: macOS-26.5.2-arm64-arm-64bit; Python 3.12.8; NumPy 2.5.1; scikit-learn 1.9.0; MOABB 1.5.0; pyRiemann 0.12.

## 3. Data sanity

Observed epoch array shape was **2592 × 22 × 1000** (trials × EEG channels × time). There were 2592 trials: 288 per subject, 72 per subject/class, and feet=648, left_hand=648, right_hand=648, tongue=648. WHOLE produced 2592 matrices of shape 22 × 22; WINDOW5 produced 12960 matrices of the same shape.

| Representation | count | SPD | non-SPD | NaN | Inf | min eig | max eig | max condition | max symmetry error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WHOLE | 2592 | 2592 | 0 | 0 | 0 | 4.047e-14 | 3.246e-09 | 7.539e+03 | 0.000e+00 |
| WINDOW5 | 12960 | 12960 | 0 | 0 | 0 | 8.113e-14 | 6.546e-09 | 1.978e+03 | 0.000e+00 |

No covariance was silently removed. `is_spd` requires finite entries, positive minimum eigenvalue, and symmetry within the pipeline tolerance. After centering, the maximum absolute subject-mean coordinate was 2.919e-14 for WHOLE and 1.066e-13 for WINDOW5. The saved coordinate dimension was 253, and StandardScaler applied was `False`. Detailed row-level checks are in [`covariance_sanity.csv`](../tables/covariance_sanity.csv).

## 4. Raw representation

In original Log-Euclidean space, the RAW WINDOW5-minus-WHOLE class-silhouette difference was +0.0012, and the grouped class-probe accuracy difference was +0.0220. These are descriptive representation diagnostics, not claims of classifier performance. H1 is **MIXED** under the frozen rules in Section 10.

WHOLE RAW figures: [1A class](../figures/figure_1a_whole_raw_class.png), [1B subject](../figures/figure_1b_whole_raw_subject.png).

WINDOW5 RAW figures: [2A class](../figures/figure_2a_window5_raw_class.png), [2B subject](../figures/figure_2b_window5_raw_subject.png), [2C window](../figures/figure_2c_window5_raw_window.png), [2D class by window](../figures/figure_2d_window5_raw_class_panels.png).

The plots are visualization only. Cluster appearance in t-SNE is not counted as evidence for a verdict.

## 5. Effect of subject centering

| Representation | subject sil. RAW | subject sil. CENTERED | reduction | subject probe RAW | subject probe CENTERED | reduction |
| --- | --- | --- | --- | --- | --- | --- |
| WHOLE | 0.2469 | -0.0601 | 0.3071 | 0.9996 | 0.0123 | 0.9873 |
| WINDOW5 | 0.0588 | -0.0555 | 0.1143 | 1.0000 | 0.0162 | 0.9838 |

H2 (strong RAW subject structure) is **SUPPORTED** and H3 (reduction after subject centering) is **SUPPORTED**. Centering is a diagnostic transform estimated from every sample of each subject in the primary session; therefore these numbers must not be read as held-out domain-adaptation performance.

Centered figures: [3A WHOLE class](../figures/figure_3a_whole_centered_class.png), [3B WHOLE subject](../figures/figure_3b_whole_centered_subject.png), [4A WINDOW5 class](../figures/figure_4a_window5_centered_class.png), [4B WINDOW5 subject](../figures/figure_4b_window5_centered_subject.png), [4C WINDOW5 window](../figures/figure_4c_window5_centered_window.png), [4D class by window](../figures/figure_4d_window5_centered_class_panels.png).

## 6. Temporal-window diagnostics

| Window | RAW class silhouette | CENTERED class silhouette | CENTERED - RAW |
| --- | --- | --- | --- |
| 1 | -0.0146 | -0.0074 | 0.0073 |
| 2 | -0.0327 | -0.0112 | 0.0215 |
| 3 | -0.0335 | -0.0141 | 0.0194 |
| 4 | -0.0256 | -0.0152 | 0.0104 |
| 5 | -0.0246 | -0.0158 | 0.0088 |

Secondary pooled OOF class point accuracy by window:

| Window | RAW point accuracy | CENTERED point accuracy | CENTERED - RAW |
| --- | --- | --- | --- |
| 1 | 0.3615 | 0.3738 | 0.0123 |
| 2 | 0.5108 | 0.5482 | 0.0374 |
| 3 | 0.4842 | 0.5143 | 0.0301 |
| 4 | 0.4360 | 0.4479 | 0.0120 |
| 5 | 0.3854 | 0.4035 | 0.0181 |

These per-window point accuracies disclose where the fixed linear probe could access class information. They are secondary diagnostics and are **not** part of the frozen H5 verdict, which uses the predeclared per-window class-silhouette range threshold.

The RAW per-window class-silhouette range was 0.0189; the CENTERED range was 0.0085. Windows satisfying both a positive centered silhouette of at least 0.0200 and retention within that tolerance of their RAW value: **none**. Thus H4 is **MIXED** and H5 is **NOT SUPPORTED**.

The panel comparison uses one global embedding per state and is not an independent fit per window. The silhouette table above, not visual panel separation, supplies the quantitative evidence.

## 7. Transition diagnostics

Consecutive displacement magnitudes were computed as `||z_(w+1) - z_w||₂` for pairs 1→2 through 4→5. Subject centering subtracts a constant within each subject, so it cannot change these within-trial displacement vectors except for floating-point noise; the measured maximum coordinate-wise RAW-versus-CENTERED displacement difference was 2.220e-16. The exported overall mean was 3.4145. The class eta-squared was 0.0055; the class mean-range/grand-mean fraction was 0.0244. Pairwise class mean-vector cosine similarities ranged from -0.4391 to 0.9649.

Class mean transition magnitudes:

| Class | Mean magnitude |
| --- | --- |
| feet | 3.4136 |
| left_hand | 3.3847 |
| right_hand | 3.3918 |
| tongue | 3.4680 |

Mean magnitude by window pair:

| Window pair | Mean magnitude |
| --- | --- |
| 1->2 | 3.5093 |
| 2->3 | 3.3806 |
| 3->4 | 3.3693 |
| 4->5 | 3.3987 |

Subject mean transition magnitudes:

| Subject | Mean magnitude |
| --- | --- |
| 1 | 3.2397 |
| 2 | 3.4455 |
| 3 | 3.2083 |
| 4 | 3.2132 |
| 5 | 3.0510 |
| 6 | 3.7334 |
| 7 | 3.3335 |
| 8 | 3.3508 |
| 9 | 4.1552 |

The complete descriptive outputs are [`transition_class_summary.csv`](../tables/transition_class_summary.csv), [`transition_subject_summary.csv`](../tables/transition_subject_summary.csv), [`transition_pair_summary.csv`](../tables/transition_pair_summary.csv), [`transition_cosine_similarity.csv`](../tables/transition_cosine_similarity.csv), and [`transition_effects.csv`](../tables/transition_effects.csv).

H6 is **NOT SUPPORTED**. This is a low-cost descriptive effect-size diagnostic; it is neither a trajectory classifier nor proof that order is causally necessary.

Trajectory figures use the predeclared smallest-trial-ID rule for subjects 1, 2, 3: [5A RAW](../figures/figure_5a_window5_raw_example_trajectories.png), [5B CENTERED](../figures/figure_5b_window5_centered_example_trajectories.png).

## 8. Quantitative separability

| Representation | class silhouette | subject silhouette | window silhouette | class distance ratio | subject distance ratio |
| --- | --- | --- | --- | --- | --- |
| WHOLE RAW | -0.0282 | 0.2469 | N/A | 1.0102 | 2.0571 |
| WINDOW5 RAW | -0.0270 | 0.0588 | -0.0183 | 1.0085 | 1.5074 |
| WHOLE CENTERED | -0.0068 | -0.0601 | N/A | 1.0433 | 0.9983 |
| WINDOW5 CENTERED | -0.0149 | -0.0555 | -0.0177 | 1.0184 | 0.9997 |

Silhouettes and between/within distance ratios were computed in the original 253-D log-svec space. WHOLE global silhouettes used 2592 trial points; WINDOW5 global silhouettes used 5000 points from 1000 deterministically subject/class-balanced whole trials. Per-window class silhouettes used all trials at each window. A distance ratio above 1 means average between-group distance exceeds average within-group distance. The full deterministic table is [`separability_metrics.csv`](../tables/separability_metrics.csv). Cross-representation silhouette deltas compare one point per WHOLE trial with repeated local states per WINDOW5 trial and different deterministic sample counts; they are descriptive, do not adjust for repeated measurements, and do not test statistical significance.

## 9. Linear information probes

| Representation | Target | Primary accuracy | Fold SD | Chance | Folds | Aggregation | Point accuracy | Convergence warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WHOLE RAW | class | 0.5660 | 0.0122 | 0.2500 | 5 | single_covariance | 0.5660 | no |
| WHOLE RAW | subject | 0.9996 | 0.0008 | 0.1111 | 5 | single_covariance | 0.9996 | no |
| WINDOW5 RAW | class | 0.5880 | 0.0152 | 0.2500 | 5 | trial_mean_probability | 0.4356 | no |
| WINDOW5 RAW | subject | 1.0000 | 0.0000 | 0.1111 | 5 | trial_mean_probability | 0.9964 | no |
| WHOLE CENTERED | class | 0.6119 | 0.0144 | 0.2500 | 5 | single_covariance | 0.6119 | no |
| WHOLE CENTERED | subject | 0.0123 | 0.0060 | 0.1111 | 5 | single_covariance | 0.0123 | no |
| WINDOW5 CENTERED | class | 0.6211 | 0.0166 | 0.2500 | 5 | trial_mean_probability | 0.4576 | no |
| WINDOW5 CENTERED | subject | 0.0162 | 0.0101 | 0.1111 | 5 | trial_mean_probability | 0.0420 | no |

These values measure linearly accessible information only. Trial-grouped folds prevent the five windows from one trial crossing train/test boundaries. WHOLE accuracy is point/trial accuracy; WINDOW5 primary accuracy averages held-out probabilities across the five windows before assigning one prediction per trial, while its point accuracy is shown separately. Probes with a convergence warning: **none**. This within-primary-session diagnostic does not test new sessions or unseen subjects and was not tuned. Full fold/summary output is in [`linear_probe_metrics.csv`](../tables/linear_probe_metrics.csv).

## 10. Hypothesis verdicts

| Hypothesis | Verdict | Claim |
| --- | --- | --- |
| H1 | MIXED | WHOLE보다 WINDOW5에서 class structure가 더 명확하다. |
| H2 | SUPPORTED | RAW representation에서 subject structure가 강하다. |
| H3 | SUPPORTED | subject centering 후 subject structure가 감소한다. |
| H4 | MIXED | centering 후에도 특정 WINDOW5 구간의 class structure가 유지/강화된다. |
| H5 | NOT SUPPORTED | class information은 window에 따라 비균일하다. |
| H6 | NOT SUPPORTED | transition statistics는 class별로 체계적으로 다르다. |

The decision rules were fixed from the YAML thresholds before reading results: H1 requires at least two of RAW class-silhouette, class-probe, and (when exported) relative class-distance-ratio deltas to reach their thresholds, with no threshold-sized contradiction. H2 requires at least two of subject silhouette, above-chance subject probe, and (when exported) subject distance ratio to pass in both RAW representations. H3 requires at least two threshold-sized reductions in both representations, with no threshold-sized increase. H4 requires at least one retained positive window and a CENTERED WINDOW5 class-probe margin over chance. H5 requires the per-window silhouette range threshold in both RAW and CENTERED. H6 requires both transition effect thresholds. Partial or conflicting evidence is MIXED; zero passing evidence is NOT SUPPORTED.

| Hypothesis | Evidence | Measured value | Frozen rule | Vote |
| --- | --- | --- | --- | --- |
| H1 | RAW class silhouette: WINDOW5 - WHOLE | 0.0012 | >= +0.0200 support; <= -0.0200 contradict | 0 |
| H1 | RAW class-probe accuracy: WINDOW5 - WHOLE | 0.0220 | >= +0.0200 support; <= -0.0200 contradict | + |
| H1 | RAW class distance ratio: relative WINDOW5 - WHOLE | -0.0017 | >= +0.0500 support; <= -0.0500 contradict | 0 |
| H2 | WHOLE RAW subject silhouette | 0.2469 | >= 0.0200 | + |
| H2 | WHOLE RAW subject-probe margin over chance | 0.8885 | >= 0.0500 | + |
| H2 | WHOLE RAW subject distance-ratio excess over 1 | 1.0571 | >= 0.0500 | + |
| H2 | WINDOW5 RAW subject silhouette | 0.0588 | >= 0.0200 | + |
| H2 | WINDOW5 RAW subject-probe margin over chance | 0.8889 | >= 0.0500 | + |
| H2 | WINDOW5 RAW subject distance-ratio excess over 1 | 0.5074 | >= 0.0500 | + |
| H3 | WHOLE subject silhouette reduction | 0.3071 | >= +0.0200 support; <= -0.0200 contradict | + |
| H3 | WHOLE subject-probe accuracy reduction | 0.9873 | >= +0.0200 support; <= -0.0200 contradict | + |
| H3 | WHOLE subject distance-ratio relative reduction | 0.5147 | >= +0.0500 support; <= -0.0500 contradict | + |
| H3 | WINDOW5 subject silhouette reduction | 0.1143 | >= +0.0200 support; <= -0.0200 contradict | + |
| H3 | WINDOW5 subject-probe accuracy reduction | 0.9838 | >= +0.0200 support; <= -0.0200 contradict | + |
| H3 | WINDOW5 subject distance-ratio relative reduction | 0.3368 | >= +0.0500 support; <= -0.0500 contradict | + |
| H4 | Number of retained positive temporal windows | 0.0000 | >= 1.0000 | 0 |
| H4 | CENTERED WINDOW5 class-probe margin over chance | 0.3711 | >= 0.0500 | + |
| H5 | RAW per-window class-silhouette range | 0.0189 | >= 0.0200 | 0 |
| H5 | CENTERED per-window class-silhouette range | 0.0085 | >= 0.0200 | 0 |
| H6 | Class effect on transition magnitude (eta-squared) | 0.0055 | >= 0.0100 | 0 |
| H6 | Range of class mean transition magnitudes / grand mean | 0.0244 | >= 0.0500 | 0 |

Thresholds: `metric_silhouette_delta=0.02`, `probe_accuracy_delta=0.02`, `above_chance_margin=0.05`, `window_silhouette_range=0.02`, `distance_ratio_relative_delta=0.05`, `transition_eta_squared=0.01`, `transition_mean_range_fraction=0.05`.

## 11. What we learned

| Possibility | Assessment | Meaning in this pilot |
| --- | --- | --- |
| A | MIXED | Windowed covariance states add class information beyond WHOLE. |
| B | NOT SUPPORTED | Temporal order/transition may carry class-dependent information. |
| C | MIXED | No major difference from a single covariance was detected by these diagnostics. |

1. WINDOW5 changed RAW class silhouette by +0.0012 and class-probe accuracy by +0.0220 relative to WHOLE; the predeclared H1 verdict is MIXED.

2. Subject centering changed subject-probe accuracy from 0.9996 to 0.0123 for WHOLE and from 1.0000 to 0.0162 for WINDOW5; H3 is SUPPORTED.

3. Window nonuniformity is NOT SUPPORTED (silhouette ranges 0.0189 RAW, 0.0085 CENTERED), while class-dependent transition evidence is NOT SUPPORTED (eta-squared 0.0055, range fraction 0.0244).

These alternatives are kept separate: evidence for local states (A) does not by itself establish that order matters (B), and absence of either under these fixed diagnostics supports the no-major-difference reading (C).

## 12. What we should NOT conclude yet

- This pilot does not establish that a distribution classifier is needed.

- It does not establish that a trajectory or sequence model is needed.

- It does not show that domain adaptation will improve, or evaluate cross-session/cross-subject adaptation.

- It does not support a claim that pseudo-label-free conditional alignment has been solved or is feasible.

- It contains no neural model, SOTA comparison, hyperparameter sweep, inferential test, or correction for multiple diagnostics. Negative and mixed verdicts are retained as such.

## 13. Recommended next experiment

Run exactly one locked replication on BNCI2014_001 session 2: change only the session selector and output/cache namespace, then execute the identical 22-channel, 8–32 Hz, 0–3.996 s, OAS WHOLE/WINDOW5, log-svec, subject-centering, embedding, diagnostic, and verdict pipeline with the same seed and thresholds. Do not use session-1 results to alter preprocessing, windows, probes, or verdict rules. The purpose is solely to test whether the session-1 conclusions reproduce.
