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

from vfmgeom.concept_erasure.paired_delta_pca import PairedDeltaPcaFitter
from vfmgeom.concept_erasure.soft_delta_projection import SoftDeltaProjectionFitter
from vfmgeom.deltas.scanner_deltas import (
    ScannerDeltaMode,
    SignMode,
    build_scanner_deltas,
)
from vfmgeom.evaluation.scanner_probe import (
    evaluate_scanner_probe_train_test,
    summarize_probe_by_rank,
)
from vfmgeom.projections.linear import feature_change_summary

logger = logging.getLogger(__name__)


# =============================================================================
# Small utilities
# =============================================================================


def parse_ranks(ranks: str | list[int | None]) -> list[int | None]:
    if isinstance(ranks, str):
        values: list[int | None] = []
        for item in ranks.split(","):
            item = item.strip()
            if not item:
                continue
            if item.lower() in {"none", "null", "full"}:
                values.append(None)
            else:
                values.append(int(item))
    else:
        values = [None if r is None else int(r) for r in ranks]

    if not values:
        raise ValueError("At least one rank must be provided.")

    for r in values:
        if r is not None and r < 1:
            raise ValueError("Ranks must be >= 1 or None.")

    # Keep None first, then sorted integer ranks.
    unique_ints = sorted({r for r in values if r is not None})
    return ([None] if None in values else []) + unique_ints


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def validate_experiment_inputs(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    delta_mode: ScannerDeltaMode,
    pair_col: str | None,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected features [n, d], got {features.shape}.")

    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )

    required = [scanner_col, group_col]

    if delta_mode in {"pair_col_pairwise", "pair_col_to_mean"}:
        if pair_col is None:
            raise ValueError(f"pair_col is required for delta_mode={delta_mode}.")
        required.append(pair_col)

    missing = [col for col in required if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    if metadata[scanner_col].nunique() < 2:
        raise ValueError(f"Need at least two scanners in {scanner_col}.")

    if metadata[group_col].nunique() < 2:
        raise ValueError(f"Need at least two groups in {group_col}.")


def to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


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


def save_eraser_npz(path: Path, eraser: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
    }

    # PCA-style eraser.
    if hasattr(eraser, "proj_left") and getattr(eraser, "proj_left") is not None:
        arrays["proj_left"] = eraser.proj_left.detach().cpu().numpy().astype(np.float32)
    if hasattr(eraser, "proj_right") and getattr(eraser, "proj_right") is not None:
        arrays["proj_right"] = (
            eraser.proj_right.detach().cpu().numpy().astype(np.float32)
        )

    # Soft full-rank eraser.
    if hasattr(eraser, "P") and getattr(eraser, "P") is not None:
        arrays["P"] = eraser.P.detach().cpu().numpy().astype(np.float32)

    if hasattr(eraser, "bias") and getattr(eraser, "bias") is not None:
        arrays["bias"] = eraser.bias.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(path, **arrays)


# =============================================================================
# Eraser grid and fitting
# =============================================================================


def expand_paired_delta_eraser_grid(eraser_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the eraser section of the YAML into concrete method configs.

    Examples
    --------
    PCA:
        eraser:
          method: paired_delta_pca
          ranks: [1, 2, 4, 8]
          whitening: [false, true]

    Soft:
        eraser:
          method: soft_delta_projection
          ranks: [null, 1, 2, 4, 8]
          lambdas: [0.1, 1, 10]
    """
    method = eraser_cfg["method"]
    expanded: list[dict[str, Any]] = []

    if method == "paired_delta_pca":
        ranks = parse_ranks(eraser_cfg.get("ranks", [1, 2, 4, 8, 16, 32, 64]))
        ranks = [r for r in ranks if r is not None]
        whitenings = as_list(eraser_cfg.get("whitening", False))
        shrinkages = as_list(eraser_cfg.get("shrinkage", True))

        for rank, whitening, shrinkage in product(ranks, whitenings, shrinkages):
            cfg = dict(eraser_cfg)
            cfg.update(
                {
                    "method": method,
                    "rank": int(rank),
                    "whitening": bool(whitening),
                    "shrinkage": bool(shrinkage),
                }
            )
            expanded.append(cfg)

    elif method == "soft_delta_projection":
        ranks = parse_ranks(eraser_cfg.get("ranks", [None, 1, 2, 4, 8, 16, 32, 64]))
        lambdas = [float(x) for x in as_list(eraser_cfg.get("lambdas", [1.0]))]
        shrink_As = as_list(eraser_cfg.get("shrink_A", True))
        shrink_Bs = as_list(eraser_cfg.get("shrink_B", False))

        for rank, lam, shrink_A, shrink_B in product(
            ranks, lambdas, shrink_As, shrink_Bs
        ):
            cfg = dict(eraser_cfg)
            cfg.update(
                {
                    "method": method,
                    "rank": rank,
                    "lam": float(lam),
                    "shrink_A": bool(shrink_A),
                    "shrink_B": bool(shrink_B),
                }
            )
            expanded.append(cfg)

    else:
        raise ValueError(f"Unsupported paired-delta eraser method: {method!r}")

    return expanded


@torch.no_grad()
def build_eraser_fitter(
    method_cfg: dict[str, Any],
    x_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Any:
    method = method_cfg["method"]

    if method == "paired_delta_pca":
        fitter = PairedDeltaPcaFitter(
            x_dim=x_dim,
            rank=int(method_cfg["rank"]),
            whitening=bool(method_cfg.get("whitening", False)),
            affine=bool(method_cfg.get("affine", True)),
            shrinkage=bool(method_cfg.get("shrinkage", True)),
            ridge=float(method_cfg.get("ridge", 1e-4)),
            svd_tol=float(method_cfg.get("svd_tol", 1e-7)),
            device=device,
            dtype=dtype,
        )
        return fitter

    if method == "soft_delta_projection":
        fitter = SoftDeltaProjectionFitter(
            x_dim=x_dim,
            affine=bool(method_cfg.get("affine", True)),
            shrink_A=bool(method_cfg.get("shrink_A", True)),
            shrink_B=bool(method_cfg.get("shrink_B", False)),
            device=device,
            dtype=dtype,
        )
        return fitter

    raise ValueError(f"Unsupported paired-delta eraser method: {method!r}")


# =============================================================================
# Main experiment runner
# =============================================================================


def run_paired_delta_projection_experiment(
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    *,
    scanner_col: str = "scanner_id",
    group_col: str = "image_id",
    delta_mode: ScannerDeltaMode = "group_to_mean",
    pair_col: str | None = None,
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
    """Run paired-delta PCA/soft projection with the same CV protocol.

    Protocol
    --------
    For each GroupKFold split:

    1. Use only train groups to build scanner-induced deltas.
    2. Fit the selected paired-delta eraser on train embeddings and train deltas.
    3. Apply the eraser to raw train/test embeddings.
    4. Train a linear scanner probe on projected train embeddings.
    5. Evaluate scanner prediction on projected test embeddings.

    This avoids fitting the eraser on test biological groups.
    """
    if eraser_cfg is None:
        raise ValueError("eraser_cfg must be provided.")

    validate_experiment_inputs(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
        delta_mode=delta_mode,
        pair_col=pair_col,
    )

    device = torch.device(
        device if torch.cuda.is_available() or str(device) == "cpu" else "cpu"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eraser_dir = output_dir / "fold_erasers"
    eraser_dir.mkdir(parents=True, exist_ok=True)

    method_grid = expand_paired_delta_eraser_grid(eraser_cfg)

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    groups = metadata[group_col].astype(str).to_numpy()

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
        if run_only_one_fold and fold_idx > 0:
            break

        logger.info("Fold %d / %d", fold_idx + 1, n_splits)

        x_train_raw = features[train_idx].astype(np.float32, copy=False)
        x_test_raw = features[test_idx].astype(np.float32, copy=False)

        scanner_train = scanner_values[train_idx]
        scanner_test = scanner_values[test_idx]

        raw_probe = evaluate_scanner_probe_train_test(
            x_train=x_train_raw,
            x_test=x_test_raw,
            scanner_train=scanner_train,
            scanner_test=scanner_test,
        )

        logger.info(
            "Fold %d raw scanner balanced accuracy: %.4f",
            fold_idx,
            raw_probe.balanced_accuracy,
        )

        deltas = build_scanner_deltas(
            features=features,
            metadata=metadata,
            scanner_col=scanner_col,
            group_col=group_col,
            delta_mode=delta_mode,
            pair_col=pair_col,
            row_indices=train_idx,
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
                "n_delta_fit": int(len(deltas)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                "train_scanners": sorted(np.unique(scanner_train).tolist()),
                "test_scanners": sorted(np.unique(scanner_test).tolist()),
            }
        )

        fitter = build_eraser_fitter(
            method_cfg=method_grid[0],
            x_dim=x_train_raw.shape[1],
            device=device,
            dtype=dtype,
        )
        fitter.update(
            x=to_tensor(x_train_raw, device=device, dtype=dtype),
            delta=to_tensor(deltas, device=device, dtype=dtype),
        )

        for method_cfg in method_grid:
            logger.info("Fold %d fitting eraser: %s", fold_idx, method_cfg)
            eraser = fitter.make_eraser(
                lam=method_cfg.get("lam", None),
                rank=method_cfg.get("rank", None),
                ridge=method_cfg.get("ridge", None),
            )

            method_name = method_cfg["method"]
            rank = method_cfg.get("rank", None)
            lam = method_cfg.get("lam", None)
            whitening = method_cfg.get("whitening", None)

            eraser_name_parts = [method_name, f"fold{fold_idx}"]
            if rank is not None:
                eraser_name_parts.append(f"rank{rank}")
            else:
                eraser_name_parts.append("fullrank")
            if lam is not None:
                eraser_name_parts.append(f"lambda{lam:g}")
            if whitening is not None:
                eraser_name_parts.append(f"white{int(bool(whitening))}")
            eraser_name = "_".join(eraser_name_parts).replace(".", "p")
            eraser_path = eraser_dir / f"{eraser_name}.npz"

            save_eraser_npz(
                eraser_path,
                eraser,
                metadata={
                    "fold": fold_idx,
                    "method_cfg": method_cfg,
                    "delta_mode": delta_mode,
                    "pair_col": pair_col,
                    "sign_mode": sign_mode,
                    "scanner_col": scanner_col,
                    "group_col": group_col,
                    "n_deltas": int(len(deltas)),
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

            projected_probe = evaluate_scanner_probe_train_test(
                x_train=x_train_proj,
                x_test=x_test_proj,
                scanner_train=scanner_train,
                scanner_test=scanner_test,
            )

            change = feature_change_summary(raw=x_test_raw, projected=x_test_proj)

            row = {
                "fold": fold_idx,
                "method": method_name,
                "rank": -1 if rank is None else int(rank),
                "rank_label": "full" if rank is None else str(rank),
                "lambda": np.nan if lam is None else float(lam),
                "whitening": np.nan if whitening is None else bool(whitening),
                "shrinkage": method_cfg.get("shrinkage", np.nan),
                "shrink_A": method_cfg.get("shrink_A", np.nan),
                "shrink_B": method_cfg.get("shrink_B", np.nan),
                "raw_score": raw_probe.balanced_accuracy,
                "projected_score": projected_probe.balanced_accuracy,
                "raw_accuracy": raw_probe.accuracy,
                "projected_accuracy": projected_probe.accuracy,
                "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_delta_fit": int(len(deltas)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                "train_scanners": sorted(np.unique(scanner_train).tolist()),
                "test_scanners": sorted(np.unique(scanner_test).tolist()),
                "delta_mode": delta_mode,
                "pair_col": pair_col,
                "sign_mode": sign_mode,
                "eraser_path": str(eraser_path),
                "mean_l2_change_test": change["mean_l2_change"],
                "median_l2_change_test": change["median_l2_change"],
                "mean_raw_norm_test": change["mean_raw_norm"],
                "mean_relative_change_test": change["mean_relative_change"],
            }
            fold_rows.append(row)

    fold_scores = pd.DataFrame(fold_rows)
    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)

    group_cols = [
        "method",
        "rank",
        "rank_label",
        "lambda",
        "whitening",
        "shrinkage",
        "shrink_A",
        "shrink_B",
    ]
    group_cols = [c for c in group_cols if c in fold_scores.columns]

    summary = (
        fold_scores.groupby(group_cols, dropna=False)
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            projected_score_mean=("projected_score", "mean"),
            projected_score_std=("projected_score", "std"),
            projected_accuracy_mean=("projected_accuracy", "mean"),
            mean_relative_change_test_mean=("mean_relative_change_test", "mean"),
            mean_relative_change_test_std=("mean_relative_change_test", "std"),
            mean_l2_change_test_mean=("mean_l2_change_test", "mean"),
            n_delta_fit_mean=("n_delta_fit", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "summary_by_method.csv", index=False)

    # Also write a PCA-compatible summary if only rank matters downstream.
    if "rank" in fold_scores.columns:
        try:
            summary_by_rank = summarize_probe_by_rank(
                fold_scores,
                projected_col="projected_score",
                raw_col="raw_score",
            )
            summary_by_rank.to_csv(output_dir / "summary_by_rank.csv", index=False)
        except Exception as exc:  # keep main result robust to helper API changes
            logger.warning("Could not write summary_by_rank.csv: %s", exc)

    diagnostics = {
        "experiment": "paired_delta_projection",
        "scanner_col": scanner_col,
        "group_col": group_col,
        "delta_mode": delta_mode,
        "pair_col": pair_col,
        "sign_mode": sign_mode,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_splits": int(n_splits),
        "n_groups": int(len(unique_groups)),
        "n_scanners": int(len(np.unique(scanner_values))),
        "scanners": sorted(np.unique(scanner_values).tolist()),
        "max_deltas_per_fold": max_deltas_per_fold,
        "seed": seed,
        "eraser_cfg": eraser_cfg,
        "expanded_method_grid": method_grid,
        "protocol": (
            "GroupKFold by group_col. For each fold, scanner deltas are built "
            "only from training groups. A paired-delta eraser is fitted on train "
            "embeddings and train-fold deltas. The eraser is applied to original "
            "train/test embeddings. A linear scanner probe is trained on projected "
            "train embeddings and evaluated on projected test embeddings."
        ),
        "fold_diagnostics": fold_diagnostics,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("Saved fold scores to %s", output_dir / "fold_scores.csv")
    logger.info("Saved summary to %s", output_dir / "summary_by_method.csv")
    logger.info("Saved diagnostics to %s", output_dir / "diagnostics.json")

    return diagnostics
