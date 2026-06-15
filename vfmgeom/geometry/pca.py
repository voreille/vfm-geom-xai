from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAResult:
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray


def fit_pca_subspace(
    x: np.ndarray,
    n_components: int,
    center: bool = True,
    random_state: int = 0,
    svd_solver: str = "randomized",
) -> PCAResult:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}.")

    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    n_components = min(n_components, x.shape[0], x.shape[1])

    if center:
        pca = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            random_state=random_state,
        )
        pca.fit(x)
        mean = pca.mean_.astype(np.float32)
    else:
        x0 = x.astype(np.float32)
        _, s, vt = np.linalg.svd(x0, full_matrices=False)
        components = vt[:n_components].astype(np.float32)
        explained = (s**2) / np.sum(s**2)
        return PCAResult(
            components=components,
            mean=np.zeros(x.shape[1], dtype=np.float32),
            explained_variance_ratio=explained[:n_components].astype(np.float32),
            singular_values=s[:n_components].astype(np.float32),
        )

    return PCAResult(
        components=pca.components_.astype(np.float32),
        mean=mean,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        singular_values=pca.singular_values_.astype(np.float32),
    )
