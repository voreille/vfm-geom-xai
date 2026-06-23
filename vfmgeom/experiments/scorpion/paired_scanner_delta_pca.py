from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from vfmgeom.deltas.scanner_deltas import (
    ScannerDeltaMode,
    SignMode,
    build_scanner_deltas,
)
from vfmgeom.evaluation.probe import (
    evaluate_probe_train_test,
    summarize_probe_by_rank,
)
from vfmgeom.geometry.pca import fit_pca_subspace
from vfmgeom.geometry.subspace import subspace_overlap
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


def scanner_centroid_subspace(
    features: np.ndarray,
    scanner_labels: np.ndarray,
    max_rank: int,
) -> np.ndarray:
    """Small diagnostic scanner-centroid basis.

    This is not LEACE. It is only used to compare the PCA delta subspace with
    the linear subspace spanned by scanner centroids in the train fold.
    """
    scanner_labels = scanner_labels.astype(str)

    global_mean = features.mean(axis=0, keepdims=True)

    centered_centroids = []
    for scanner in sorted(np.unique(scanner_labels)):
        idx = scanner_labels == scanner
        centroid = features[idx].mean(axis=0, keepdims=True)
        centered_centroids.append((centroid - global_mean).ravel())

    centroid_matrix = np.stack(centered_centroids, axis=0)
    centroid_matrix -= centroid_matrix.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(centroid_matrix, full_matrices=False)
    rank = min(max_rank, int((singular_values > 1e-8).sum()))

    return vt[:rank].astype(np.float32)


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


def run_paired_scanner_delta_pca(
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    scanner_col: str = "scanner_id",
    group_col: str = "image_id",
    delta_mode: ScannerDeltaMode = "group_to_mean",
    pair_col: str | None = None,
    sign_mode: SignMode = "one",
    ranks: str | list[int] = "1,2,4,8,16,32,64",
    n_splits: int = 5,
    max_deltas_per_fold: int | None = None,
    seed: int = 0,
    pca_center: bool = True,
    pca_svd_solver: str = "randomized",
) -> dict[str, Any]:
    """Run paired scanner-delta PCA erasure with grouped CV.

    Protocol
    --------
    For each GroupKFold split:

    1. Use only train groups to build scanner-induced deltas.
    2. Fit PCA on those train-fold deltas.
    3. For each rank k, project away the first k PCA directions.
    4. Train a linear scanner probe on projected train embeddings.
    5. Evaluate scanner prediction on projected test embeddings.

    This avoids fitting the PCA projector on test biological groups.
    """
    validate_experiment_inputs(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
        delta_mode=delta_mode,
        pair_col=pair_col,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    projector_dir = output_dir / "fold_projectors"
    projector_dir.mkdir(parents=True, exist_ok=True)

    rank_values = parse_ranks(ranks)
    max_rank = max(rank_values)

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
        )

        n_components = min(max_rank, features.shape[1], len(deltas))

        if n_components < max_rank:
            logger.warning(
                "Fold %d: requested max rank %d, but fitting only %d components "
                "because n_deltas=%d and feature_dim=%d.",
                fold_idx,
                max_rank,
                n_components,
                len(deltas),
                features.shape[1],
            )

        pca = fit_pca_subspace(
            deltas,
            n_components=n_components,
            center=pca_center,
            random_state=seed + fold_idx,
            svd_solver=pca_svd_solver,
        )

        projector_path = projector_dir / (
            f"paired_scanner_delta_pca_fold{fold_idx}.npz"
        )

        save_projection_npz(
            projector_path,
            components=pca.components,
            mean=pca.mean,
            explained_variance_ratio=pca.explained_variance_ratio,
            metadata={
                "fold": fold_idx,
                "delta_mode": delta_mode,
                "pair_col": pair_col,
                "sign_mode": sign_mode,
                "scanner_col": scanner_col,
                "group_col": group_col,
                "n_deltas": int(len(deltas)),
                "pca_center": pca_center,
                "pca_svd_solver": pca_svd_solver,
            },
        )

        scanner_basis = scanner_centroid_subspace(
            features=x_train_raw,
            scanner_labels=scanner_train,
            max_rank=min(len(np.unique(scanner_train)) - 1, max_rank),
        )

        fold_diag = {
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

            overlap = subspace_overlap(
                components_a=components,
                components_b=scanner_basis,
            )

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "rank": rank,
                    "raw_score": raw_probe.balanced_accuracy,
                    "paired_scanner_delta_pca_score": (
                        projected_probe.balanced_accuracy
                    ),
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
                    "projector_path": str(projector_path),
                    "mean_l2_change_test": change["mean_l2_change"],
                    "median_l2_change_test": change["median_l2_change"],
                    "mean_raw_norm_test": change["mean_raw_norm"],
                    "mean_relative_change_test": change["mean_relative_change"],
                    "explained_variance_ratio_sum": float(
                        pca.explained_variance_ratio[:rank].sum()
                    ),
                    "scanner_overlap_mean_squared_cosine": overlap[
                        "mean_squared_cosine"
                    ],
                    "scanner_overlap_max_cosine": overlap["max_cosine"],
                    "scanner_overlap_cosines": overlap["cosines"],
                }
            )

        fold_diagnostics.append(fold_diag)

    fold_scores = pd.DataFrame(fold_rows)
    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)

    summary_by_rank = summarize_probe_by_rank(
        fold_scores,
        projected_col="paired_scanner_delta_pca_score",
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
            scanner_overlap_mean_squared_cosine_mean=(
                "scanner_overlap_mean_squared_cosine",
                "mean",
            ),
            scanner_overlap_max_cosine_mean=(
                "scanner_overlap_max_cosine",
                "mean",
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
        "experiment": "paired_scanner_delta_pca",
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
        "ranks": rank_values,
        "max_deltas_per_fold": max_deltas_per_fold,
        "seed": seed,
        "pca_center": pca_center,
        "pca_svd_solver": pca_svd_solver,
        "protocol": (
            "GroupKFold by group_col. For each fold, scanner deltas are built "
            "only from training groups. PCA is fitted on the train-fold scanner "
            "deltas. The top-k directions are projected away from original "
            "train/test embeddings. A linear scanner probe is trained on "
            "projected train embeddings and evaluated on projected test embeddings."
        ),
        "fold_diagnostics": fold_diagnostics,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("Saved fold scores to %s", output_dir / "fold_scores.csv")
    logger.info("Saved summary to %s", output_dir / "summary_by_rank.csv")
    logger.info("Saved diagnostics to %s", output_dir / "diagnostics.json")

    return diagnostics
