from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from vfmgeom.concept_erasure.multi_paired_delta_erasers import (
    DeltaSourceSpec,
    PairedDeltaFitter,
)


def _safe_float(value: Any) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu().item()
    value = float(value)
    if math.isfinite(value):
        return value
    return float("nan")


def _safe_ratio(num: float, den: float, *, eps: float = 1e-12) -> float:
    num = float(num)
    den = float(den)
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= eps:
        return float("nan")
    return num / den


def _json_dumps_float_list(values: np.ndarray | list[float]) -> str:
    return json.dumps([_safe_float(v) for v in list(values)])


def covariance_trace_np(x: np.ndarray, *, ddof: int = 1) -> float:
    """Trace of the empirical covariance, i.e. total feature variance."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}.")
    if x.shape[0] <= ddof:
        return float("nan")
    return _safe_float(
        np.var(x.astype(np.float64, copy=False), axis=0, ddof=ddof).sum()
    )


def mean_squared_l2_np(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}.")
    if len(x) == 0:
        return float("nan")
    return _safe_float(np.mean(np.sum(x.astype(np.float64, copy=False) ** 2, axis=1)))


def probe_excess_ratio(
    *,
    raw_balanced_accuracy: float,
    projected_balanced_accuracy: float,
    chance_balanced_accuracy: float,
) -> float:
    """Scale probe BA so 1=no change and 0=chance-level nuisance probe."""
    return _safe_ratio(
        float(projected_balanced_accuracy) - float(chance_balanced_accuracy),
        float(raw_balanced_accuracy) - float(chance_balanced_accuracy),
    )


def feature_variance_metrics(
    *,
    raw: np.ndarray,
    projected: np.ndarray,
    reference_trace_A: float,
    projected_reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Feature-change metrics normalized by training-set total variance.

    ``raw`` and ``projected`` are typically the test features. ``reference_trace_A``
    should usually be the trace of the training covariance before erasure. If
    ``projected_reference`` is supplied, it is used to estimate the variance kept
    after projection; otherwise ``projected`` is used.
    """
    raw = np.asarray(raw)
    projected = np.asarray(projected)
    if raw.shape != projected.shape:
        raise ValueError(
            f"raw and projected must have identical shapes, got {raw.shape} and {projected.shape}."
        )

    diff = projected.astype(np.float64, copy=False) - raw.astype(np.float64, copy=False)
    mean_change_energy = mean_squared_l2_np(diff)
    projected_trace = covariance_trace_np(
        projected if projected_reference is None else projected_reference
    )

    return {
        "trace_A_reference": _safe_float(reference_trace_A),
        "mean_feature_change_energy": mean_change_energy,
        "feature_change_vs_A_trace": _safe_ratio(mean_change_energy, reference_trace_A),
        "projected_A_trace": projected_trace,
        "projected_A_trace_ratio": _safe_ratio(projected_trace, reference_trace_A),
    }


def delta_residual_metrics(
    *,
    raw_delta: np.ndarray,
    projected_delta: np.ndarray,
    reference_trace_A: float,
    projected_trace_A: float | None = None,
) -> dict[str, float]:
    """Residual nuisance-delta energy normalized by raw delta energy and A trace."""
    raw_delta = np.asarray(raw_delta)
    projected_delta = np.asarray(projected_delta)
    if raw_delta.shape != projected_delta.shape:
        raise ValueError(
            "raw_delta and projected_delta must have identical shapes, "
            f"got {raw_delta.shape} and {projected_delta.shape}."
        )

    raw_energy = mean_squared_l2_np(raw_delta)
    projected_energy = mean_squared_l2_np(projected_delta)
    residual_ratio = _safe_ratio(projected_energy, raw_energy)

    return {
        "raw_delta_energy_mean": raw_energy,
        "projected_delta_energy_mean": projected_energy,
        "delta_residual_energy_ratio": residual_ratio,
        "delta_removed_fraction": float("nan")
        if math.isnan(residual_ratio)
        else 1.0 - residual_ratio,
        "raw_delta_vs_A_trace": _safe_ratio(raw_energy, reference_trace_A),
        "projected_delta_vs_A_trace": _safe_ratio(projected_energy, reference_trace_A),
        "projected_delta_vs_projected_A_trace": (
            float("nan")
            if projected_trace_A is None
            else _safe_ratio(projected_energy, projected_trace_A)
        ),
    }


def _matrix_trace(matrix: Tensor) -> float:
    return _safe_float(torch.real(torch.trace(matrix)).detach().cpu())


def _matrix_frobenius(matrix: Tensor) -> float:
    return _safe_float(torch.linalg.matrix_norm(matrix, ord="fro").detach().cpu())


def _eigenvalue_summary_from_values(
    values: Tensor, *, top_k: int = 0
) -> dict[str, float | int | str]:
    values = torch.real(values).detach().cpu().clamp_min(0).to(torch.float64)
    if values.numel() == 0:
        return {}

    values_sorted = torch.sort(values, descending=True).values.numpy()
    total = float(values_sorted.sum())
    eps = 1e-12

    out: dict[str, float | int | str] = {
        "eig_trace": total,
        "eig_max": float(values_sorted[0]),
        "eig_top1_fraction": _safe_ratio(float(values_sorted[0]), total),
    }

    if total > eps:
        cumsum = np.cumsum(values_sorted)
        probs = values_sorted / total
        probs = probs[probs > eps]
        out["effective_rank"] = float(np.exp(-(probs * np.log(probs)).sum()))
        for frac in (0.5, 0.8, 0.9, 0.95, 0.99):
            out[f"rank{int(frac * 100)}"] = int(
                np.searchsorted(cumsum, frac * total) + 1
            )
    else:
        out.update(
            {
                "effective_rank": 0.0,
                "rank50": 0,
                "rank80": 0,
                "rank90": 0,
                "rank95": 0,
                "rank99": 0,
            }
        )

    if top_k and top_k > 0:
        out[f"top{int(top_k)}_eigenvalues_json"] = _json_dumps_float_list(
            values_sorted[:top_k]
        )
    return out


def matrix_moment_summary(
    matrix: Tensor,
    *,
    prefix: str,
    include_spectrum: bool = False,
    top_k: int = 32,
) -> dict[str, float | int | str]:
    """Cheap matrix diagnostics, optionally with PSD eigenspectrum summaries."""
    matrix = (matrix + matrix.mH) / 2
    out: dict[str, float | int | str] = {
        f"{prefix}_trace": _matrix_trace(matrix),
        f"{prefix}_frobenius": _matrix_frobenius(matrix),
    }
    if include_spectrum:
        eigvals = torch.linalg.eigvalsh(matrix).clamp_min(0)
        for key, value in _eigenvalue_summary_from_values(eigvals, top_k=top_k).items():
            out[f"{prefix}_{key}"] = value
    return out


def joint_moment_diagnostics(
    *,
    fitter: PairedDeltaFitter,
    source_specs: Sequence[DeltaSourceSpec | Mapping[str, Any]],
    normalize_source_weights: bool,
    joint_normalization: str = "none",
    shrink_A: bool = True,
    ridge: float = 1e-4,
    svd_tol: float = 1e-7,
    include_spectrum: bool = False,
    top_k: int = 32,
) -> dict[str, float | int | str]:
    """Diagnostics for A, B before/after joint normalization, and optionally B relative to A.

    The generalized spectrum is computed once per fold/recipe/stage and can be
    reused to interpret all lambdas through attenuation_i = 1 / (1 + lambda * mu_i).
    """
    A = fitter.covariance_x(shrinkage=shrink_A)
    B_before = fitter.combined_delta_matrix(
        source_specs,
        normalize_source_weights=normalize_source_weights,
    )
    B_after = fitter._normalize_joint_delta_matrix(  # noqa: SLF001 - intentionally shared with eraser code.
        matrix=B_before,
        reference=A,
        normalization=joint_normalization,  # type: ignore[arg-type]
        eps=1e-12,
    )

    out: dict[str, float | int | str] = {
        "joint_normalization": str(joint_normalization),
        "normalize_source_weights": bool(normalize_source_weights),
    }
    out.update(
        matrix_moment_summary(
            A, prefix="A", include_spectrum=include_spectrum, top_k=top_k
        )
    )
    out.update(
        matrix_moment_summary(
            B_before,
            prefix="B_joint_before",
            include_spectrum=include_spectrum,
            top_k=top_k,
        )
    )
    out.update(
        matrix_moment_summary(
            B_after,
            prefix="B_joint_after",
            include_spectrum=include_spectrum,
            top_k=top_k,
        )
    )
    out["B_joint_after_trace_over_A_trace"] = _safe_ratio(
        float(out["B_joint_after_trace"]),
        float(out["A_trace"]),
    )

    if include_spectrum:
        x_inv_sqrt, _ = fitter.whitening_matrices(
            shrinkage=shrink_A,
            ridge=ridge,
            svd_tol=svd_tol,
        )
        generalized = x_inv_sqrt @ B_after @ x_inv_sqrt.mH
        generalized = (generalized + generalized.mH) / 2
        mu = torch.linalg.eigvalsh(generalized).clamp_min(0)
        for key, value in _eigenvalue_summary_from_values(mu, top_k=top_k).items():
            out[f"generalized_{key}"] = value

    return out


def attenuation_metrics_from_generalized_top_values(
    *,
    generalized_eigenvalues: Sequence[float],
    lam: float,
) -> dict[str, float | int]:
    """Approximate soft-erasure attenuation summaries from generalized eigenvalues."""
    values = np.asarray(list(generalized_eigenvalues), dtype=np.float64)
    if values.size == 0:
        return {}
    attenuation = 1.0 / (1.0 + float(lam) * np.maximum(values, 0.0))
    return {
        "attenuation_min_top": float(np.min(attenuation)),
        "attenuation_median_top": float(np.median(attenuation)),
        "attenuation_mean_top": float(np.mean(attenuation)),
        "n_top_attenuation_lt_0p9": int(np.sum(attenuation < 0.9)),
        "n_top_attenuation_lt_0p5": int(np.sum(attenuation < 0.5)),
        "n_top_attenuation_lt_0p1": int(np.sum(attenuation < 0.1)),
    }
