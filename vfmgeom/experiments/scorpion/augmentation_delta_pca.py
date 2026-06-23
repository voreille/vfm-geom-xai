from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from vfmgeom.deltas.augmentation_deltas import (
    AugmentationDeltaConfig,
    get_or_compute_augmentation_deltas,
)
from vfmgeom.evaluation.probe import (
    evaluate_probe_train_test,
    summarize_probe_by_rank,
)
from vfmgeom.geometry.pca import fit_pca_subspace
from vfmgeom.projections.io import save_projection_npz
from vfmgeom.projections.linear import (
    feature_change_summary,
    project_away_subspace,
)

logger = logging.getLogger(__name__)


def parse_ranks(ranks: str | list[int]) -> list[int]:
    if isinstance(ranks, str):
        values = sorted({int(x.strip()) for x in ranks.split(",") if x.strip()})
    else:
        values = sorted({int(x) for x in ranks})

    if not values:
        raise ValueError("At least one rank must be provided.")

    if min(values) < 1:
        raise ValueError("Ranks must be >= 1.")

    return values


def run_augmentation_delta_pca(
    features: np.ndarray,
    metadata: pd.DataFrame,
    tile_dir: Path,
    delta_cache: Path,
    output_dir: Path,
    encoder_id: str,
    device: torch.device,
    token_mode: str,
    augmentation_config: AugmentationDeltaConfig,
    scanner_col: str = "scanner_id",
    group_col: str = "image_id",
    path_col: str = "path",
    filename_col: str = "filename",
    ranks: str | list[int] = "1,2,4,8,16,32,64",
    n_splits: int = 5,
    seed: int = 0,
    force_deltas: bool = False,
    pca_center: bool = True,
    pca_svd_solver: str = "randomized",
) -> dict[str, Any]:
    if len(features) != len(metadata):
        raise ValueError(
            f"Features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )

    for col in [scanner_col, group_col]:
        if col not in metadata.columns:
            raise ValueError(f"Missing metadata column: {col}")

    output_dir.mkdir(parents=True, exist_ok=True)
    projector_dir = output_dir / "fold_projectors"
    projector_dir.mkdir(parents=True, exist_ok=True)

    rank_values = parse_ranks(ranks)
    max_rank = max(rank_values)

    deltas, delta_row_indices = get_or_compute_augmentation_deltas(
        delta_cache=delta_cache,
        force=force_deltas,
        tile_dir=tile_dir,
        metadata=metadata,
        original_features=features,
        encoder_id=encoder_id,
        device=device,
        token_mode=token_mode,
        config=augmentation_config,
        path_col=path_col,
        filename_col=filename_col,
    )

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
        logger.info("Fold %d / %d", fold_idx + 1, n_splits)

        x_train_raw = features[train_idx]
        x_test_raw = features[test_idx]

        scanner_train = scanner_values[train_idx]
        scanner_test = scanner_values[test_idx]

        raw_probe = evaluate_probe_train_test(
            x_train=x_train_raw,
            x_test=x_test_raw,
            y_train=scanner_train,
            y_test=scanner_test,
        )

        train_delta_mask = np.isin(delta_row_indices, train_idx)
        fold_deltas = deltas[train_delta_mask]

        if len(fold_deltas) == 0:
            raise ValueError(f"No augmentation deltas available for fold {fold_idx}.")

        n_components = min(max_rank, features.shape[1], len(fold_deltas))

        pca = fit_pca_subspace(
            fold_deltas,
            n_components=n_components,
            center=pca_center,
            random_state=seed + fold_idx,
            svd_solver=pca_svd_solver,
        )

        projector_path = projector_dir / f"augmentation_delta_pca_fold{fold_idx}.npz"

        save_projection_npz(
            projector_path,
            components=pca.components,
            mean=pca.mean,
            explained_variance_ratio=pca.explained_variance_ratio,
            metadata={
                "fold": fold_idx,
                "backend": augmentation_config.backend,
                "preset": augmentation_config.preset,
                "delta_mode": augmentation_config.delta_mode,
                "n_augmentations_per_image": (
                    augmentation_config.n_augmentations_per_image
                ),
                "n_deltas": int(len(fold_deltas)),
                "pca_center": pca_center,
                "pca_svd_solver": pca_svd_solver,
            },
        )

        fold_diag = {
            "fold": fold_idx,
            "raw_balanced_accuracy": raw_probe.balanced_accuracy,
            "raw_accuracy": raw_probe.accuracy,
            "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_delta_fit": int(len(fold_deltas)),
            "n_train_groups": int(len(np.unique(groups[train_idx]))),
            "n_test_groups": int(len(np.unique(groups[test_idx]))),
            "train_scanners": sorted(np.unique(scanner_train).tolist()),
            "test_scanners": sorted(np.unique(scanner_test).tolist()),
            "projector_path": str(projector_path),
            "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        }

        for rank in rank_values:
            if rank > len(pca.components):
                continue

            components = pca.components[:rank]

            x_train_proj = project_away_subspace(x_train_raw, components)
            x_test_proj = project_away_subspace(x_test_raw, components)

            projected_probe = evaluate_probe_train_test(
                x_train=x_train_proj,
                x_test=x_test_proj,
                y_train=scanner_train,
                y_test=scanner_test,
            )

            change = feature_change_summary(
                raw=x_test_raw,
                projected=x_test_proj,
            )

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "rank": rank,
                    "raw_score": raw_probe.balanced_accuracy,
                    "augmentation_delta_pca_score": (
                        projected_probe.balanced_accuracy
                    ),
                    "raw_accuracy": raw_probe.accuracy,
                    "projected_accuracy": projected_probe.accuracy,
                    "chance_balanced_accuracy": raw_probe.chance_balanced_accuracy,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_delta_fit": int(len(fold_deltas)),
                    "n_train_groups": int(len(np.unique(groups[train_idx]))),
                    "n_test_groups": int(len(np.unique(groups[test_idx]))),
                    "train_scanners": sorted(np.unique(scanner_train).tolist()),
                    "test_scanners": sorted(np.unique(scanner_test).tolist()),
                    "backend": augmentation_config.backend,
                    "preset": augmentation_config.preset,
                    "delta_mode": augmentation_config.delta_mode,
                    "projector_path": str(projector_path),
                    "mean_l2_change_test": change["mean_l2_change"],
                    "median_l2_change_test": change["median_l2_change"],
                    "mean_raw_norm_test": change["mean_raw_norm"],
                    "mean_relative_change_test": change["mean_relative_change"],
                    "explained_variance_ratio_sum": float(
                        pca.explained_variance_ratio[:rank].sum()
                    ),
                }
            )

        fold_diagnostics.append(fold_diag)

    fold_scores = pd.DataFrame(fold_rows)
    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)

    summary_by_rank = summarize_probe_by_rank(
        fold_scores,
        projected_col="augmentation_delta_pca_score",
        raw_col="raw_score",
    )

    extra_summary = (
        fold_scores.groupby("rank")
        .agg(
            mean_relative_change_test_mean=("mean_relative_change_test", "mean"),
            mean_relative_change_test_std=("mean_relative_change_test", "std"),
            explained_variance_ratio_sum_mean=(
                "explained_variance_ratio_sum",
                "mean",
            ),
            explained_variance_ratio_sum_std=(
                "explained_variance_ratio_sum",
                "std",
            ),
        )
        .reset_index()
    )

    summary_by_rank = summary_by_rank.merge(
        extra_summary,
        on="rank",
        how="left",
    )

    summary_by_rank.to_csv(output_dir / "summary_by_rank.csv", index=False)

    diagnostics = {
        "experiment": "augmentation_delta_pca",
        "scanner_col": scanner_col,
        "group_col": group_col,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_deltas": int(len(deltas)),
        "n_splits": int(n_splits),
        "n_groups": int(len(unique_groups)),
        "n_scanners": int(len(np.unique(scanner_values))),
        "scanners": sorted(np.unique(scanner_values).tolist()),
        "ranks": rank_values,
        "seed": seed,
        "pca_center": pca_center,
        "pca_svd_solver": pca_svd_solver,
        "augmentation_config": {
            "backend": augmentation_config.backend,
            "preset": augmentation_config.preset,
            "delta_mode": augmentation_config.delta_mode,
            "n_augmentations_per_image": (
                augmentation_config.n_augmentations_per_image
            ),
            "batch_size": augmentation_config.batch_size,
            "num_workers": augmentation_config.num_workers,
            "use_amp": augmentation_config.use_amp,
            "augmentation_kwargs": augmentation_config.augmentation_kwargs,
        },
        "protocol": (
            "GroupKFold by group_col. Augmentation deltas are computed as "
            "f(aug(x)) - f(x). For each fold, PCA is fitted only on deltas whose "
            "source images belong to training groups. The top-k augmentation-delta "
            "directions are projected away from original train/test embeddings, "
            "then a linear scanner probe is trained and evaluated."
        ),
        "fold_diagnostics": fold_diagnostics,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("Saved fold scores to %s", output_dir / "fold_scores.csv")
    logger.info("Saved summary to %s", output_dir / "summary_by_rank.csv")

    return diagnostics