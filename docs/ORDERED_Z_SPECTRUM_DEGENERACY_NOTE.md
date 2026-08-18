# Ordered-Z Spectrum Degeneracy Note

Status: **CONTROL_UNASSESSED_NUMERICAL_DEGENERACY — NON-VOTING DIAGNOSTIC ONLY**

This note characterizes the V1/V1.1 ordered-`Z` eigenspectrum failure without changing epsilon, adding jitter, clipping singular values, dropping folds or coordinates, reducing the rank grid, or applying any regularization/workaround. It does not vote in, rescue, or overturn the OpenBMI sensor terminal.

## Location and numerical contract

The first affected fit is `inner` outer fold `0` inner fold `0`, requested rank `13`. Centered feature ranks were `10` and `10`; cross-covariance numerical rank was `10`. Minimum projected-score standard deviations were `5.9828862429994448e-17` and `8.7003708273149514e-17` against the frozen threshold `9.9301366129890925e-16`. Exact zero: `False`.

First-fit cross-covariance singular values: `0.098999973236953698, 0.05236531062008086, 0.023900135391945417, 0.0079068344131941297, 0.006078800403491953, 0.0031636540322715536, 0.0014501295400526841, 0.00072000754167120121, 0.00045951844092219758, 8.6084646163138917e-05, 5.4883457949387479e-18, 1.8433169998434021e-18, 8.323759193724572e-19, 5.4926854111129253e-19, 3.5217748248484794e-19, 1.6717405217866394e-19, 4.447200739787604e-20, 3.491894735475569e-20, 9.239501095410762e-21, 1.4192656101168535e-21`.

## Binary algebra diagnostic

Binary `Z_R=-Z_L` relative residual was `1.4206060237405877e-19`; the reversed ordered-eigenvalue opposition residual was `2.9135016481658324e-16`, and the spectrum-signature palindrome residual was `1.3877787807814457e-16`. The frozen diagnostic classifies the degeneracy as following from binary `Z` algebra and ordered-eigen symmetry: `True`.

All affected-fold ranks, singular values, projected-score standard deviations, epsilon-relative ratios, and subject memberships are stored in `outputs/subject_class_population_structure_v1_1/controls/ordered_z_spectrum_degeneracy.json`.
