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
    projected_norm = np.linalg.norm(projected, axis=1)
    diff_norm = np.linalg.norm(diff, axis=1)

    relative_change = diff_norm / (raw_norm + 1e-8)

    return {
        "mean_l2_change": float(diff_norm.mean()),
        "median_l2_change": float(np.median(diff_norm)),
        "mean_raw_norm": float(raw_norm.mean()),
        "median_raw_norm": float(np.median(raw_norm)),
        "mean_projected_norm": float(projected_norm.mean()),
        "median_projected_norm": float(np.median(projected_norm)),
        "mean_relative_change": float(relative_change.mean()),
        "median_relative_change": float(np.median(relative_change)),
        "ratio_of_mean_norms": float(diff_norm.mean() / (raw_norm.mean() + 1e-8)),
    }

def delta_change_summary(
    raw_delta: np.ndarray,
    projected_delta: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, float]:
    raw_norm = np.linalg.norm(raw_delta, axis=1)
    projected_norm = np.linalg.norm(projected_delta, axis=1)

    raw_energy = np.sum(raw_delta**2, axis=1)
    projected_energy = np.sum(projected_delta**2, axis=1)

    remaining_energy_ratio = projected_energy.mean() / (raw_energy.mean() + eps)

    per_delta_norm_ratio = projected_norm / (raw_norm + eps)

    return {
        "mean_raw_delta_norm": float(raw_norm.mean()),
        "mean_projected_delta_norm": float(projected_norm.mean()),
        "remaining_delta_energy_ratio": float(remaining_energy_ratio),
        "removed_delta_energy_ratio": float(1.0 - remaining_energy_ratio),
        "mean_remaining_delta_norm_ratio": float(per_delta_norm_ratio.mean()),
        "median_remaining_delta_norm_ratio": float(np.median(per_delta_norm_ratio)),
    }
