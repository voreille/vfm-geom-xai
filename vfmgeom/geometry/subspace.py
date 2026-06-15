from __future__ import annotations

import numpy as np


def orthonormalize_rows(components: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if components.ndim != 2:
        raise ValueError(f"Expected 2D components, got {components.shape}.")

    q, r = np.linalg.qr(components.T)
    diag = np.abs(np.diag(r))
    rank = int((diag > eps).sum())

    return q[:, :rank].T.astype(np.float32)


def principal_cosines(
    components_a: np.ndarray,
    components_b: np.ndarray,
) -> np.ndarray:
    ua = orthonormalize_rows(components_a)
    ub = orthonormalize_rows(components_b)

    if len(ua) == 0 or len(ub) == 0:
        return np.array([], dtype=np.float32)

    s = np.linalg.svd(ua @ ub.T, compute_uv=False)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def subspace_overlap(
    components_a: np.ndarray,
    components_b: np.ndarray,
) -> dict:
    s = principal_cosines(components_a, components_b)

    if len(s) == 0:
        return {
            "mean_squared_cosine": None,
            "max_cosine": None,
            "cosines": [],
        }

    return {
        "mean_squared_cosine": float(np.mean(s**2)),
        "max_cosine": float(np.max(s)),
        "cosines": s.tolist(),
    }


def projection_energy(
    x: np.ndarray,
    components: np.ndarray,
    center: bool = False,
) -> float:
    """Fraction of energy of x explained by span(components)."""
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [n, d], got {x.shape}.")

    u = orthonormalize_rows(components)

    if center:
        x = x - x.mean(axis=0, keepdims=True)

    proj = (x @ u.T) @ u

    num = np.sum(proj**2, axis=1)
    den = np.sum(x**2, axis=1) + 1e-12

    return float(np.mean(num / den))


def projection_energy_per_sample(
    x: np.ndarray,
    components: np.ndarray,
    center: bool = False,
) -> np.ndarray:
    u = orthonormalize_rows(components)

    if center:
        x = x - x.mean(axis=0, keepdims=True)

    proj = (x @ u.T) @ u

    num = np.sum(proj**2, axis=1)
    den = np.sum(x**2, axis=1) + 1e-12

    return (num / den).astype(np.float32)
