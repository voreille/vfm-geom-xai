from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from vfmgeom.evaluation.probe import evaluate_probe_train_test
from vfmgeom.projections.linear import feature_change_summary

# Adjust this import if your LEACE code lives somewhere else.
try:
    from vfmgeom.erasers.leace import LeaceFitter
except ImportError:
    try:
        from vfmgeom.projections.leace import LeaceFitter
    except ImportError:
        from vfmgeom.leace import LeaceFitter

logger = logging.getLogger(__name__)


# =============================================================================
# Small utilities
# =============================================================================


def validate_experiment_inputs(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected features [n, d], got {features.shape}.")

    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )

    missing = [col for col in [scanner_col, group_col] if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    if metadata[scanner_col].nunique() < 2:
        raise ValueError(f"Need at least two scanners in {scanner_col}.")

    if metadata[group_col].nunique() < 2:
        raise ValueError(f"Need at least two groups in {group_col}.")


def to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    z = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for i, label in enumerate(labels.astype(str)):
        z[i, class_to_idx[label]] = 1.0
    return z


@torch.no_grad()
def apply_eraser_numpy(
    eraser: Any,
    x: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 8192,
) -> np.ndarray:
    eraser = eraser.to(device) if hasattr(eraser, "to") else eraser
    outs: list[np.ndarray] = []

    for start in range(0, len(x), batch_size):
        xb = to_tensor(x[start : start + batch_size], device=device, dtype=dtype)
        yb = eraser(xb).detach().cpu().numpy().astype(np.float32)
        outs.append(yb)

    return np.concatenate(outs, axis=0)


def save_leace_eraser_npz(path: Path, eraser: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
    }

    if hasattr(eraser, "proj_left"):
        arrays["proj_left"] = eraser.proj_left.detach().cpu().numpy().astype(np.float32)
    if hasattr(eraser, "proj_right"):
        arrays["proj_right"] = (
            eraser.proj_right.detach().cpu().numpy().astype(np.float32)
        )
    if hasattr(eraser, "bias") and eraser.bias is not None:
        arrays["bias"] = eraser.bias.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(path, **arrays)


# =============================================================================
# LEACE fitting
# =============================================================================


@torch.no_grad()
def fit_leace_eraser(
    *,
    x_train: np.ndarray,
    scanner_train: np.ndarray,
    scanner_classes: np.ndarray,
    leace_cfg: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Any:
    x_dim = x_train.shape[1]
    z_dim = len(scanner_classes)

    x_t = to_tensor(x_train, device=device, dtype=dtype)
    z_t = to_tensor(
        one_hot(scanner_train.astype(str), scanner_classes),
        device=device,
        dtype=dtype,
    )

    fitter = LeaceFitter(
        x_dim=x_dim,
        z_dim=z_dim,
        method=leace_cfg.get("method", "leace"),
        affine=bool(leace_cfg.get("affine", True)),
        constrain_cov_trace=bool(leace_cfg.get("constrain_cov_trace", True)),
        shrinkage=bool(leace_cfg.get("shrinkage", True)),
        svd_tol=float(leace_cfg.get("svd_tol", 0.01)),
        device=device,
        dtype=dtype,
    )
    fitter.update(x_t, z_t)
    return fitter.eraser


# =============================================================================
# Main experiment runner
# =============================================================================


def run_leace_scanner_erasure_experiment(
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    *,
    scanner_col: str = "scanner_id",
    group_col: str = "image_id",
    leace_cfg: dict[str, Any] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    apply_batch_size: int = 8192,
) -> dict[str, Any]:
    """Run LEACE scanner erasure with grouped CV.

    Protocol
    --------
    For each GroupKFold split:

    1. Fit LEACE only on train embeddings and train scanner labels.
    2. Apply the LEACE eraser to raw train/test embeddings.
    3. Train a linear scanner probe on erased train embeddings.
    4. Evaluate scanner prediction on erased test embeddings.

    This runner is deliberately separate from paired-delta projection because LEACE
    uses scanner labels directly rather than paired scanner deltas.
    """
    if leace_cfg is None:
        leace_cfg = {
            "method": "leace",
            "affine": True,
            "constrain_cov_trace": True,
            "shrinkage": True,
            "svd_tol": 0.01,
        }

    validate_experiment_inputs(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
    )

    device = torch.device(
        device if torch.cuda.is_available() or str(device) == "cpu" else "cpu"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    eraser_dir.mkdir(parents=True, exist_ok=True)

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    groups = metadata[group_col].astype(str).to_numpy()
    scanner_classes = np.asarray(sorted(np.unique(scanner_values))).astype(str)

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two folds/groups for GroupKFold.")

    cv = GroupKFold(n_splits=n_splits)

    fold_rows: list[dict[str, Any]] = []
    fold_diagnostics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, scanner_values, groups=groups)
    ):
        logger.info("Fold %d / %d", fold_idx + 1, n_splits)

        x_train_raw = features[train_idx].astype(np.float32, copy=False)
        x_test_raw = features[test_idx].astype(np.float32, copy=False)

        scanner_train = scanner_values[train_idx]
        scanner_test = scanner_values[test_idx]

        raw_probe = evaluate_probe_train_test(
            x_train=x_train_raw,
            x_test=x_test_raw,
            y_train=scanner_train,
            y_test=scanner_test,
        )

        logger.info(
            "Fold %d raw scanner balanced accuracy: %.4f",
            fold_idx,
            raw_probe.balanced_accuracy,
        )

        eraser = fit_leace_eraser(
            x_train=x_train_raw,
            scanner_train=scanner_train,
            scanner_classes=scanner_classes,
            leace_cfg=leace_cfg,
            device=device,
            dtype=dtype,
        )

        eraser_path = eraser_dir / f"leace_fold{fold_idx}.npz"
        effective_rank = int(getattr(eraser, "proj_left").shape[1])

        save_leace_eraser_npz(
            eraser_path,
            eraser,
            metadata={
                "fold": fold_idx,
                "leace_cfg": leace_cfg,
                "scanner_col": scanner_col,
                "group_col": group_col,
                "scanner_classes": scanner_classes.tolist(),
                "effective_rank": effective_rank,
            },
        )

        x_train_proj = apply_eraser_numpy(
            eraser,
            x_train_raw,
            device=device,
            dtype=dtype,
            batch_size=apply_batch_size,
        )
        x_test_proj = apply_eraser_numpy(
            eraser,
            x_test_raw,
            device=device,
            dtype=dtype,
            batch_size=apply_batch_size,
        )

        projected_probe = evaluate_probe_train_test(
            x_train=x_train_proj,
            x_test=x_test_proj,
            y_train=scanner_train,
            y_test=scanner_test,
        )

        change = feature_change_summary(raw=x_test_raw, projected=x_test_proj)

        fold_rows.append(
            {
                "fold": fold_idx,
                "method": "leace",
                "effective_rank": effective_rank,
                "raw_score": raw_probe.balanced_accuracy,
                "projected_score": projected_probe.balanced_accuracy,
                "raw_accuracy": raw_probe.accuracy,
                "projected_accuracy": projected_probe.accuracy,
                "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                "train_scanners": sorted(np.unique(scanner_train).tolist()),
                "test_scanners": sorted(np.unique(scanner_test).tolist()),
                "eraser_path": str(eraser_path),
                "mean_l2_change_test": change["mean_l2_change"],
                "median_l2_change_test": change["median_l2_change"],
                "mean_raw_norm_test": change["mean_raw_norm"],
                "mean_relative_change_test": change["mean_relative_change"],
            }
        )

        fold_diagnostics.append(
            {
                "fold": fold_idx,
                "raw_balanced_accuracy": raw_probe.balanced_accuracy,
                "raw_accuracy": raw_probe.accuracy,
                "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                "effective_rank": effective_rank,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                "train_scanners": sorted(np.unique(scanner_train).tolist()),
                "test_scanners": sorted(np.unique(scanner_test).tolist()),
            }
        )

    fold_scores = pd.DataFrame(fold_rows)
    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)

    summary = (
        fold_scores.groupby(["method"], dropna=False)
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            projected_score_mean=("projected_score", "mean"),
            projected_score_std=("projected_score", "std"),
            projected_accuracy_mean=("projected_accuracy", "mean"),
            effective_rank_mean=("effective_rank", "mean"),
            effective_rank_std=("effective_rank", "std"),
            mean_relative_change_test_mean=("mean_relative_change_test", "mean"),
            mean_relative_change_test_std=("mean_relative_change_test", "std"),
            mean_l2_change_test_mean=("mean_l2_change_test", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    diagnostics = {
        "experiment": "leace_scanner_erasure",
        "scanner_col": scanner_col,
        "group_col": group_col,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_splits": int(n_splits),
        "n_groups": int(len(unique_groups)),
        "n_scanners": int(len(scanner_classes)),
        "scanners": scanner_classes.tolist(),
        "seed": seed,
        "leace_cfg": leace_cfg,
        "protocol": (
            "GroupKFold by group_col. For each fold, LEACE is fitted only on "
            "train embeddings and train scanner labels. The eraser is applied to "
            "original train/test embeddings. A linear scanner probe is trained on "
            "erased train embeddings and evaluated on erased test embeddings."
        ),
        "fold_diagnostics": fold_diagnostics,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("Saved fold scores to %s", output_dir / "fold_scores.csv")
    logger.info("Saved summary to %s", output_dir / "summary.csv")
    logger.info("Saved diagnostics to %s", output_dir / "diagnostics.json")

    return diagnostics
