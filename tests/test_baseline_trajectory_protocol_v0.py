import numpy as np
import pandas as pd

from src.baseline_trajectory_v0.identifiability import source_transform
from src.baseline_trajectory_v0.task import split_manifest


def fixture():
    rows, labels = [], []
    for subject in range(1, 10):
        for session in ("0train", "1test"):
            for trial in range(40):
                rows.append({"subject": subject, "session": session, "trial_uid": f"{subject}-{session}-{trial}"})
                labels.append(trial % 4)
    return pd.DataFrame(rows), np.asarray(labels)


def test_split_contracts_and_both_session_directions():
    metadata, y = fixture(); seen = set()
    for protocol, unit, split, train, test in split_manifest(metadata, y):
        assert not set(train) & set(test)
        if protocol in ("P2", "P3"):
            assert unit not in set(metadata.subject.iloc[train])
            assert set(metadata.subject.iloc[test]) == {unit}
        if protocol == "P1":
            seen.add(split)
    assert any("0train_to_1test" in x for x in seen)
    assert any("1test_to_0train" in x for x in seen)


def test_source_only_scaler_pca_is_inductive():
    rng = np.random.default_rng(5)
    source = rng.normal(size=(80, 30)); target = rng.normal(size=(20, 30)) + 100
    transformed_source, transformed_target = source_transform(source, target)
    assert transformed_source.shape == (80, 20)
    assert transformed_target.shape == (20, 20)
    assert np.allclose(transformed_source.mean(axis=0), 0, atol=1e-10)
