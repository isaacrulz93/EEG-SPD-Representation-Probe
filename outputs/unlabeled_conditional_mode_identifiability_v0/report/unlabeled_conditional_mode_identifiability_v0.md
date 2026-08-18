# Unlabeled Conditional-Mode Identifiability V0

## Frozen parent

PR #16 at `9dee7642ac573f37756b8427a75864a50c32044e` remains `GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION`. This work neither recomputes nor reinterprets that terminal, and all PR #16 artifact hashes remain unchanged.

## Trial-level object gate

`PASS_EXACT_PARENT_PREPROCESSING_REBUILT`. The exact parent preprocessing was rebuilt from the hash-locked source manifest; no alternative filtering, epoch, channel, covariance estimator, regularization, reference, or trial rule was used.

## Mode identity and stability

Minimum parent-direction absolute cosine `1.0000000000`; median fold-pair `0.9838015899`; median cross-view `0.9083782808`; median leave-one-training-subject refit `0.9993453890`. This audit is descriptive and montage-coordinate only.

## Separate frozen decisions

1. Signed zero-label identification: **`NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION`**.
2. Trial/prototype bridge: **`TRIAL_TO_PROTOTYPE_MODE_COMPATIBLE`**. Session metrics: `[{"intercept_beta_on_trial_delta": -0.20632563089728756, "normalized_absolute_error": 1.3809568266075005, "pearson": 0.9992410163481643, "session": "0", "sign_agreement": 0.4074074074074074, "slope_beta_on_trial_delta": 0.9786300170500102, "spearman": 0.9977892128835524}, {"intercept_beta_on_trial_delta": -0.20385931238301674, "normalized_absolute_error": 1.2216248825887595, "pearson": 0.998609868073819, "session": "1", "sign_agreement": 0.42592592592592593, "slope_beta_on_trial_delta": 0.9798727123418258, "spearman": 0.9921478940346863}]`; subject-permutation p `0.0005`; random-direction p `0.0005`.
3. Unsigned zero-label recovery: **`UNSIGNED_CONDITIONAL_ENERGY_RECOVERY_SUPPORTED`**. pooled Spearman `0.6692120795`, sessions `[0.624496828637982, 0.7273685450913854]`, 95% CI `[0.45985203150114196, 0.8220680467789596]`, subject-permutation p `0.0005`, random-direction p `0.0005`.
4. Minimal semantic anchoring: **`MINIMAL_SEMANTIC_ANCHOR_NOT_EFFICIENT`**. selected budget `None`; zero-label status `NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION`.

## Interpretation boundary

The zero-label signed coordinate is non-identifiable under class permutation. Any supported unsigned result concerns projected between-class energy only. Any anchoring result concerns semantic orientation of the frozen one-dimensional mode only. This does not establish a full conditional distribution, physiology, source anatomy, causal mechanism, classification benefit, pseudo-label quality, TTA recoverability, ASD biomarker, or clinical utility.

## Dataset boundary

Only retrospective OpenBMI / Lee2019-MI was executed. BNCI, Stieger2021, HGD, all ASD datasets, downstream classifiers, domain adaptation, TTA, neural networks, and pseudo-labeling were not run.

## Exact next scientific question

What additional structural assumption or semantic target supervision is required when pooled unlabeled dispersion does not recover the stable conditional mode efficiently?
