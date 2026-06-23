from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from vfmgeom.concept_erasure.paired_delta_erasers import PairedDeltaFitter
from vfmgeom.deltas.scanner_deltas import (
    ScannerDeltaMode,
    SignMode,
    build_scanner_deltas,
)
from vfmgeom.evaluation.probe import (
    evaluate_probe_train_test,
    summarize_probe_by_rank,
)
from vfmgeom.projections.linear import feature_change_summary, delta_change_summary

logger = logging.getLogger(__name__)


# =============================================================================
# Small utilities
# =============================================================================


def parse_ranks(ranks: str | list[int | None]) -> list[int | None]:
    """Parse integer ranks and optional full/untruncated rank."""
    if isinstance(ranks, str):
        values: list[int | None] = []
        for item in ranks.split(","):
            item = item.strip()
            if not item:
                continue
            if item.lower() in {"none", "null", "full", "untruncated"}:
                values.append(None)
            else:
                values.append(int(item))
    else:
        values = [None if rank is None else int(rank) for rank in ranks]

    if not values:
        raise ValueError("At least one rank must be provided.")

    for rank in values:
        if rank is not None and rank < 1:
            raise ValueError("Ranks must be >= 1 or None.")

    unique_ranks = sorted({rank for rank in values if rank is not None})
    return ([None] if None in values else []) + unique_ranks


def as_list(value: Any) -> list[Any]:
    """Wrap a scalar configuration value in a list."""
    return value if isinstance(value, list) else [value]


def validate_experiment_inputs(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    cv_group_col: str,
    delta_group_col: str,
    delta_mode: ScannerDeltaMode,
    pair_col: str | None,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [n, d], got {features.shape}.")

    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )

    required_columns = [scanner_col, cv_group_col, delta_group_col]

    if delta_mode in {"pair_col_pairwise", "pair_col_to_mean"}:
        if pair_col is None:
            raise ValueError(f"pair_col is required for delta_mode={delta_mode}.")
        required_columns.append(pair_col)

    missing = [column for column in required_columns if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    if metadata[scanner_col].nunique() < 2:
        raise ValueError(f"Need at least two scanners in {scanner_col!r}.")

    if metadata[cv_group_col].nunique() < 2:
        raise ValueError(f"Need at least two groups in {cv_group_col!r}.")

    if metadata[delta_group_col].nunique() < 2:
        raise ValueError(f"Need at least two groups in {delta_group_col!r}.")


def to_tensor(
    x: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def apply_delta_transform_numpy(
    eraser: Any,
    deltas: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 8192,
) -> np.ndarray:
    eraser = eraser.to(device=device, dtype=dtype)
    outputs: list[np.ndarray] = []

    for start in range(0, len(deltas), batch_size):
        batch = to_tensor(
            deltas[start : start + batch_size],
            device=device,
            dtype=dtype,
        )

        projected = (
            eraser.transform_delta(batch).detach().cpu().numpy().astype(np.float32)
        )
        outputs.append(projected)

    if not outputs:
        return np.empty_like(deltas, dtype=np.float32)

    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def apply_eraser_numpy(
    eraser: Any,
    x: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 8192,
) -> np.ndarray:
    """Apply a fitted eraser to a NumPy feature matrix in batches."""
    eraser = eraser.to(device=device, dtype=dtype)
    outputs: list[np.ndarray] = []

    for start in range(0, len(x), batch_size):
        batch = to_tensor(
            x[start : start + batch_size],
            device=device,
            dtype=dtype,
        )
        projected = eraser(batch).detach().cpu().numpy().astype(np.float32)
        outputs.append(projected)

    if not outputs:
        return np.empty_like(x, dtype=np.float32)

    return np.concatenate(outputs, axis=0)


def save_eraser_npz(
    path: Path,
    eraser: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the eraser factors and experiment metadata in a portable NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
    }

    for name in ("P", "proj_left", "proj_right", "bias", "eigenvalues"):
        value = getattr(eraser, name, None)
        if value is not None:
            arrays[name] = value.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(path, **arrays)


# =============================================================================
# Eraser-grid expansion
# =============================================================================


def expand_paired_delta_eraser_grid(
    eraser_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand the YAML eraser section into concrete method configurations."""
    method = str(eraser_cfg["method"])
    expanded: list[dict[str, Any]] = []

    if method == "paired_delta_pca":
        ranks = parse_ranks(eraser_cfg.get("ranks", [1, 2, 4, 8, 16, 32, 64]))
        ranks = [rank for rank in ranks if rank is not None]

        whitenings = as_list(eraser_cfg.get("whitening", False))
        delta_moments = as_list(eraser_cfg.get("delta_moment", "covariance"))
        shrink_As = as_list(eraser_cfg.get("shrink_A", True))
        shrink_Bs = as_list(eraser_cfg.get("shrink_B", False))

        for rank, whitening, delta_moment, shrink_A, shrink_B in product(
            ranks,
            whitenings,
            delta_moments,
            shrink_As,
            shrink_Bs,
        ):
            expanded.append(
                {
                    **eraser_cfg,
                    "method": method,
                    "rank": int(rank),
                    "whitening": bool(whitening),
                    "delta_moment": str(delta_moment),
                    "shrink_A": bool(shrink_A),
                    "shrink_B": bool(shrink_B),
                }
            )

    elif method == "soft_delta_projection":
        ranks = parse_ranks(
            eraser_cfg.get(
                "ranks",
                [None, 1, 2, 4, 8, 16, 32, 64],
            )
        )
        lambdas = [float(value) for value in as_list(eraser_cfg.get("lambdas", [1.0]))]
        delta_moments = as_list(eraser_cfg.get("delta_moment", "second_moment"))
        shrink_As = as_list(eraser_cfg.get("shrink_A", True))
        shrink_Bs = as_list(eraser_cfg.get("shrink_B", False))

        for rank, lam, delta_moment, shrink_A, shrink_B in product(
            ranks,
            lambdas,
            delta_moments,
            shrink_As,
            shrink_Bs,
        ):
            expanded.append(
                {
                    **eraser_cfg,
                    "method": method,
                    "rank": rank,
                    "lam": float(lam),
                    "delta_moment": str(delta_moment),
                    "shrink_A": bool(shrink_A),
                    "shrink_B": bool(shrink_B),
                }
            )

    else:
        raise ValueError(f"Unsupported paired-delta eraser method: {method!r}")

    return expanded


def make_eraser_from_config(
    fitter: PairedDeltaFitter,
    method_cfg: dict[str, Any],
) -> Any:
    """Build one concrete eraser from shared fitted statistics."""
    method = method_cfg["method"]

    if method == "paired_delta_pca":
        return fitter.make_pca_eraser(
            rank=int(method_cfg["rank"]),
            whitening=bool(method_cfg.get("whitening", False)),
            affine=bool(method_cfg.get("affine", True)),
            delta_moment=method_cfg.get("delta_moment", "covariance"),
            shrink_A=bool(method_cfg.get("shrink_A", True)),
            shrink_B=bool(method_cfg.get("shrink_B", False)),
            ridge=float(method_cfg.get("ridge", 1e-4)),
            svd_tol=float(method_cfg.get("svd_tol", 1e-7)),
        )

    if method == "soft_delta_projection":
        return fitter.make_soft_eraser(
            lam=float(method_cfg["lam"]),
            rank=method_cfg.get("rank"),
            affine=bool(method_cfg.get("affine", True)),
            delta_moment=method_cfg.get("delta_moment", "second_moment"),
            shrink_A=bool(method_cfg.get("shrink_A", True)),
            shrink_B=bool(method_cfg.get("shrink_B", False)),
            ridge=float(method_cfg.get("ridge", 1e-4)),
            svd_tol=float(method_cfg.get("svd_tol", 1e-7)),
        )

    raise ValueError(f"Unknown paired-delta method: {method!r}")


def make_eraser_name(
    *,
    fold_idx: int,
    method_cfg: dict[str, Any],
) -> str:
    """Create a reproducible filename stem for one eraser."""
    method = method_cfg["method"]
    rank = method_cfg.get("rank")
    delta_moment = method_cfg.get("delta_moment")

    parts = [
        method,
        f"fold{fold_idx}",
        "full" if rank is None else f"rank{rank}",
    ]

    if method == "paired_delta_pca":
        parts.append(f"white{int(bool(method_cfg.get('whitening', False)))}")
    elif method == "soft_delta_projection":
        parts.append(f"lambda{float(method_cfg['lam']):g}")

    if delta_moment is not None:
        parts.append(str(delta_moment))

    parts.append(f"shrinkA{int(bool(method_cfg.get('shrink_A', True)))}")
    parts.append(f"shrinkB{int(bool(method_cfg.get('shrink_B', False)))}")

    return "_".join(parts).replace(".", "p")


# =============================================================================
# Main experiment runner
# =============================================================================


def run_paired_delta_projection_experiment(
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    *,
    scanner_col: str = "scanner_id",
    cv_group_col: str = "slide_id",
    delta_group_col: str = "image_id",
    delta_mode: ScannerDeltaMode = "group_to_mean",
    delta_pair_col: str | None = None,
    sign_mode: SignMode = "one",
    eraser_cfg: dict[str, Any] | None = None,
    n_splits: int = 5,
    max_deltas_per_fold: int | None = None,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    apply_batch_size: int = 8192,
    run_only_one_fold: bool = False,
) -> dict[str, Any]:
    """Run grouped-CV evaluation of paired-delta PCA or soft erasure.

    For every fold, the paired-delta statistics are fitted once on train-fold
    embeddings and deltas. All erasers in the configured hyperparameter grid
    are then derived from the same fitted statistics.
    """
    if eraser_cfg is None:
        raise ValueError("eraser_cfg must be provided.")

    validate_experiment_inputs(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        cv_group_col=cv_group_col,
        delta_group_col=delta_group_col,
        delta_mode=delta_mode,
        pair_col=delta_pair_col,
    )

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU.")
        requested_device = torch.device("cpu")
    device = requested_device

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eraser_dir = output_dir / "fold_erasers"
    eraser_dir.mkdir(parents=True, exist_ok=True)

    method_grid = expand_paired_delta_eraser_grid(eraser_cfg)

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    cv_groups = metadata[cv_group_col].astype(str).to_numpy()

    unique_groups = np.unique(cv_groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two folds/groups for GroupKFold.")

    cv = GroupKFold(n_splits=n_splits)

    fold_rows: list[dict[str, Any]] = []
    fold_diagnostics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, scanner_values, groups=cv_groups)
    ):
        if run_only_one_fold and fold_idx > 0:
            break

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

        train_deltas = build_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=delta_group_col,
            delta_mode=delta_mode,
            pair_col=delta_pair_col,
            row_indices=train_idx,
            sign_mode=sign_mode,
            max_deltas=max_deltas_per_fold,
            seed=seed + fold_idx,
        ).astype(np.float32, copy=False)

        test_deltas = build_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=delta_group_col,
            delta_mode=delta_mode,
            pair_col=delta_pair_col,
            row_indices=test_idx,
            sign_mode=sign_mode,
            max_deltas=max_deltas_per_fold,
            seed=seed + fold_idx,
        ).astype(np.float32, copy=False)

        fold_diagnostics.append(
            {
                "fold": fold_idx,
                "raw_balanced_accuracy": raw_probe.balanced_accuracy,
                "raw_accuracy": raw_probe.accuracy,
                "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_delta_fit": int(len(train_deltas)),
                "n_train_groups": int(len(np.unique(cv_groups[train_idx]))),
                "n_test_groups": int(len(np.unique(cv_groups[test_idx]))),
                "train_scanners": sorted(np.unique(scanner_train).tolist()),
                "test_scanners": sorted(np.unique(scanner_test).tolist()),
            }
        )

        fitter = PairedDeltaFitter.fit(
            x=to_tensor(
                x_train_raw,
                device=device,
                dtype=dtype,
            ),
            delta=to_tensor(
                train_deltas,
                device=device,
                dtype=dtype,
            ),
        )

        for method_cfg in method_grid:
            logger.info(
                "Fold %d evaluating eraser configuration: %s",
                fold_idx,
                method_cfg,
            )

            eraser = make_eraser_from_config(
                fitter=fitter,
                method_cfg=method_cfg,
            )

            method_name = method_cfg["method"]
            rank = method_cfg.get("rank")
            lam = method_cfg.get("lam")
            whitening = method_cfg.get("whitening")
            delta_moment = method_cfg.get("delta_moment")

            eraser_name = make_eraser_name(
                fold_idx=fold_idx,
                method_cfg=method_cfg,
            )
            eraser_path = eraser_dir / f"{eraser_name}.npz"

            save_eraser_npz(
                eraser_path,
                eraser,
                metadata={
                    "fold": fold_idx,
                    "method_cfg": method_cfg,
                    "delta_mode": delta_mode,
                    "pair_col": delta_pair_col,
                    "sign_mode": sign_mode,
                    "scanner_col": scanner_col,
                    "cv_group_col": cv_group_col,
                    "delta_group_col": delta_group_col,
                    "n_deltas": int(len(train_deltas)),
                },
            )

            x_train_projected = apply_eraser_numpy(
                eraser,
                x_train_raw,
                device=device,
                dtype=dtype,
                batch_size=apply_batch_size,
            )
            x_test_projected = apply_eraser_numpy(
                eraser,
                x_test_raw,
                device=device,
                dtype=dtype,
                batch_size=apply_batch_size,
            )

            projected_probe = evaluate_probe_train_test(
                x_train=x_train_projected,
                x_test=x_test_projected,
                y_train=scanner_train,
                y_test=scanner_test,
            )

            change = feature_change_summary(
                raw=x_test_raw,
                projected=x_test_projected,
            )
            delta_change = delta_change_summary(
                raw_delta=test_deltas,
                projected_delta=apply_delta_transform_numpy(
                    eraser,
                    test_deltas,
                    device=device,
                    dtype=dtype,
                    batch_size=apply_batch_size,
                ),
            )

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "method": method_name,
                    "cv_group_col": cv_group_col,
                    "rank": -1 if rank is None else int(rank),
                    "rank_label": "full" if rank is None else str(rank),
                    "lambda": np.nan if lam is None else float(lam),
                    "whitening": (np.nan if whitening is None else bool(whitening)),
                    "delta_moment": delta_moment,
                    "affine": bool(method_cfg.get("affine", True)),
                    "shrink_A": bool(method_cfg.get("shrink_A", True)),
                    "shrink_B": bool(method_cfg.get("shrink_B", False)),
                    "ridge": float(method_cfg.get("ridge", 1e-4)),
                    "svd_tol": float(method_cfg.get("svd_tol", 1e-7)),
                    "raw_score": raw_probe.balanced_accuracy,
                    "projected_score": projected_probe.balanced_accuracy,
                    "raw_accuracy": raw_probe.accuracy,
                    "projected_accuracy": projected_probe.accuracy,
                    "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_delta_fit": int(len(train_deltas)),
                    "n_train_groups": int(len(np.unique(cv_groups[train_idx]))),
                    "n_test_groups": int(len(np.unique(cv_groups[test_idx]))),
                    "train_scanners": sorted(np.unique(scanner_train).tolist()),
                    "test_scanners": sorted(np.unique(scanner_test).tolist()),
                    "delta_mode": delta_mode,
                    "delta_pair_col": delta_pair_col,
                    "delta_group_col": delta_group_col,
                    "sign_mode": sign_mode,
                    "eraser_path": str(eraser_path),
                    "mean_l2_change_test": change["mean_l2_change"],
                    "median_l2_change_test": change["median_l2_change"],
                    "mean_raw_norm_test": change["mean_raw_norm"],
                    "mean_relative_change_test": change["mean_relative_change"],
                    "mean_raw_delta_norm": delta_change["mean_raw_delta_norm"],
                    "mean_projected_delta_norm": delta_change[
                        "mean_projected_delta_norm"
                    ],
                    "remaining_delta_energy_ratio": delta_change[
                        "remaining_delta_energy_ratio"
                    ],
                    "removed_delta_energy_ratio": delta_change[
                        "removed_delta_energy_ratio"
                    ],
                    "mean_remaining_delta_norm_ratio": delta_change[
                        "mean_remaining_delta_norm_ratio"
                    ],
                    "median_remaining_delta_norm_ratio": delta_change[
                        "median_remaining_delta_norm_ratio"
                    ],
                }
            )

    fold_scores = pd.DataFrame(fold_rows)
    fold_scores.to_csv(
        output_dir / "fold_scores.csv",
        index=False,
    )

    group_columns = [
        "method",
        "rank",
        "rank_label",
        "lambda",
        "whitening",
        "delta_moment",
        "affine",
        "shrink_A",
        "shrink_B",
        "ridge",
        "svd_tol",
    ]

    summary = (
        fold_scores.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            projected_score_mean=("projected_score", "mean"),
            projected_score_std=("projected_score", "std"),
            projected_accuracy_mean=("projected_accuracy", "mean"),
            mean_relative_change_test_mean=(
                "mean_relative_change_test",
                "mean",
            ),
            mean_relative_change_test_std=(
                "mean_relative_change_test",
                "std",
            ),
            mean_l2_change_test_mean=(
                "mean_l2_change_test",
                "mean",
            ),
            n_delta_fit_mean=("n_delta_fit", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        output_dir / "summary_by_method.csv",
        index=False,
    )

    if "rank" in fold_scores.columns:
        try:
            summary_by_rank = summarize_probe_by_rank(
                fold_scores,
                projected_col="projected_score",
                raw_col="raw_score",
            )
            summary_by_rank.to_csv(
                output_dir / "summary_by_rank.csv",
                index=False,
            )
        except Exception as exc:
            logger.warning(
                "Could not write summary_by_rank.csv: %s",
                exc,
            )

    diagnostics = {
        "experiment": "paired_delta_projection",
        "scanner_col": scanner_col,
        "cv_group_col": cv_group_col,
        "delta_group_col": delta_group_col,
        "delta_mode": delta_mode,
        "delta_pair_col": delta_pair_col,
        "sign_mode": sign_mode,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_splits": int(n_splits),
        "n_groups": int(len(unique_groups)),
        "n_scanners": int(len(np.unique(scanner_values))),
        "scanners": sorted(np.unique(scanner_values).tolist()),
        "max_deltas_per_fold": max_deltas_per_fold,
        "seed": seed,
        "run_only_one_fold": run_only_one_fold,
        "eraser_cfg": eraser_cfg,
        "expanded_method_grid": method_grid,
        "protocol": (
            "GroupKFold by group_col. For each fold, scanner deltas are built "
            "only from training groups. One PairedDeltaFitter is fitted from "
            "train embeddings and train-fold deltas. All configured PCA or soft "
            "erasers are derived from the same fitted statistics. Each eraser is "
            "applied to the original train/test embeddings, and a linear scanner "
            "probe is trained on projected train embeddings and evaluated on "
            "projected test embeddings."
        ),
        "fold_diagnostics": fold_diagnostics,
    }

    with open(
        output_dir / "diagnostics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            diagnostics,
            handle,
            indent=2,
        )

    logger.info(
        "Saved fold scores to %s",
        output_dir / "fold_scores.csv",
    )
    logger.info(
        "Saved summary to %s",
        output_dir / "summary_by_method.csv",
    )
    logger.info(
        "Saved diagnostics to %s",
        output_dir / "diagnostics.json",
    )

    return diagnostics
