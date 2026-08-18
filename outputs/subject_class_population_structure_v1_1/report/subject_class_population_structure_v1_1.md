# Cross-Session Population Structure of Subject-Class Interaction V1.1

## Outcome

The technical-amendment terminal is **GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION**. V1 remains immutable at `UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE`; V1.1 recomputed the frozen pipeline from parent objects and did not recover any hidden V1 value.

OpenBMI reliability was `True`: session 0 `T=0.431295`, `p=0.000500`; session 1 `T=0.372795`, `p=0.000500`.

The held-out primary statistic is `0.413639` (95% subject-bootstrap CI `0.201758` to `0.537580`), with direction medians `0.403899` and `0.404057`. Selected ranks are `[1, 1, 1, 1, 1, 1]` (median `1.0`, frequency `{'1': 6}`). Subject-pairing, class-semantics, and equal-rank random-subspace p-values are `0.000500`, `0.001000`, and `0.000500`. The full-space baseline is `0.262257` with pairing p `0.000500`. Leave-one-subject sign stability is `True`.

## V1/V1.1 equivalence

All scientific rows in `protocol/v1_v1_1_scientific_contract_equivalence.csv` are unchanged. The only `changed=true` row is secondary failure isolation. Parent hashes, folds, ranks, features, normalization, low-rank cap, 1,999-replicate nulls, seeds/namespaces, and terminal function are identical to V1.

## Non-voting controls

- `magnitude_sensitivity`: `CONTROL_COMPLETED`
- `same_session_pca`: `CONTROL_COMPLETED`
- `ordered_z_eigenspectrum`: `CONTROL_UNASSESSED_NUMERICAL_DEGENERACY`
- `generalized_eigen_signature`: `CONTROL_COMPLETED`
- `selected_mode_split_half`: `CONTROL_COMPLETED`
- `action_overlap`: `CONTROL_UNASSESSED_DATA_CONTRACT_FAILURE`

The ordered-spectrum failure is retained without workaround and characterized in `docs/ORDERED_Z_SPECTRUM_DEGENERACY_NOTE.md`. No non-voting status changed the OpenBMI terminal.

## BNCI secondary diagnostic

Executed: `True`; status `CONTROL_COMPLETED`; T=`0.627880`, pairing p=`0.050500`, class p=`0.002000`, ranks=`[1, 1, 1, 1, 1, 1, 1, 1, 1]`, influence sign=`True`, action overlap=`CONTROL_COMPLETED`.

BNCI is explicitly secondary and cannot rescue or overturn OpenBMI. Its basis is not claimed to equal the OpenBMI basis.

## Interpretation boundary

This terminal answers only whether the stable montage-registered mean-level subject×class interaction has a held-out-subject, cross-session, population-shared low-dimensional linear structure under the V1 gates. It does not establish a full conditional distribution, dispersion structure, physiology, source anatomy, causality, unlabeled target identifiability, ASD biomarker, clinical diagnosis, TTA recoverability, globally identifiable `Q_s`, or cross-dataset equality of modes. No classifier, network, adapter, loss, or TTA method is proposed.

## Next question

Can an unseen subject's coordinates in the stable interaction subspace be identified from unlabeled marginal EEG without reliable pseudo-labels?
