"""Fixed PCA + t-SNE visualization pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_tsne(
    coordinates: np.ndarray, embedding_config: dict[str, Any], seed: int
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Fit the single preconfigured embedding for one representation state."""
    z = np.asarray(coordinates, dtype=np.float64)
    if z.ndim != 2 or len(z) < 3:
        raise ValueError(f"Expected at least three coordinate rows, got {z.shape}")
    requested_components = int(embedding_config["pca_components"])
    n_components = min(requested_components, z.shape[1], z.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="full")
    reduced = pca.fit_transform(z)
    tsne = TSNE(
        n_components=int(embedding_config["tsne_components"]),
        perplexity=float(embedding_config["perplexity"]),
        init=str(embedding_config["init"]),
        learning_rate=embedding_config["learning_rate"],
        max_iter=int(embedding_config["max_iter"]),
        method=str(embedding_config["method"]),
        random_state=int(seed),
        verbose=1,
    )
    embedded = tsne.fit_transform(reduced)
    metadata: dict[str, float | int | str] = {
        "n_samples": int(z.shape[0]),
        "input_dimensions": int(z.shape[1]),
        "pca_components_used": int(n_components),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "tsne_kl_divergence": float(tsne.kl_divergence_),
        "seed": int(seed),
        "standard_scaler": "false",
    }
    return embedded, metadata

