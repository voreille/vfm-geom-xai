# vfmgeom/projections/linear.py

from __future__ import annotations

import numpy as np

from vfmgeom.geometry.subspace import orthonormalize_rows


def project_onto_subspace(
    features: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    u = orthonormalize_rows(components)
    return ((features @ u.T) @ u).astype(np.float32)


def project_away_subspace(
    features: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    projected = project_onto_subspace(features, components)
    return (features - projected).astype(np.float32)


def make_projection_matrix_from_components(
    components: np.ndarray,
    remove: bool = True,
) -> np.ndarray:
    """Create matrix P such that z_projected = z @ P.T.

    If remove=True:
        P = I - U^T U

    If remove=False:
        P = U^T U
    """
    u = orthonormalize_rows(components)
    dim = u.shape[1]

    onto = u.T @ u

    if remove:
        return (np.eye(dim, dtype=np.float32) - onto).astype(np.float32)

    return onto.astype(np.float32)


def apply_projection_matrix(
    features: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    return (features @ projection_matrix.T).astype(np.float32)


def feature_change_summary(raw: np.ndarray, projected: np.ndarray) -> dict:
    diff = projected - raw
    raw_norm = np.linalg.norm(raw, axis=1)
    diff_norm = np.linalg.norm(diff, axis=1)

    return {
        "mean_l2_change": float(diff_norm.mean()),
        "median_l2_change": float(np.median(diff_norm)),
        "mean_raw_norm": float(raw_norm.mean()),
        "median_raw_norm": float(np.median(raw_norm)),
        "mean_relative_change": float(diff_norm.mean() / (raw_norm.mean() + 1e-8)),
    }
