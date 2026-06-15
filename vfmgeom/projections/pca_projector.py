from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vfmgeom.geometry.pca import PCAResult, fit_pca_subspace
from vfmgeom.projections.linear import project_away_subspace


@dataclass
class PCASubspaceProjector:
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(
        cls,
        deltas: np.ndarray,
        n_components: int,
        center: bool = True,
        random_state: int = 0,
    ) -> "PCASubspaceProjector":
        result: PCAResult = fit_pca_subspace(
            deltas,
            n_components=n_components,
            center=center,
            random_state=random_state,
        )
        return cls(
            components=result.components,
            mean=result.mean,
            explained_variance_ratio=result.explained_variance_ratio,
        )

    def transform(self, features: np.ndarray, rank: int | None = None) -> np.ndarray:
        if rank is None:
            components = self.components
        else:
            components = self.components[:rank]

        return project_away_subspace(features, components)

    def removed_components(self, rank: int | None = None) -> np.ndarray:
        if rank is None:
            return self.components
        return self.components[:rank]
