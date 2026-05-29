#!/usr/bin/env python
from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from vfmgeom.scripts.analyze_scorpion_perturbation_delta_pca import (
    get_or_compute_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# IO
# =============================================================================


def load_npz_embeddings(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)

    metadata = pd.DataFrame(
        {key: data[key].astype(str) for key in data.files if key != "features"}
    )

    if len(metadata) != len(features):
        raise ValueError(
            f"Metadata/features length mismatch: {len(metadata)} vs {len(features)}"
        )

    return features, metadata


# =============================================================================
# Probe and projection utilities
# =============================================================================


def parse_ranks(ranks: str) -> list[int]:
    out = sorted({int(x.strip()) for x in ranks.split(",") if x.strip()})
    if not out:
        raise ValueError("At least one rank must be provided.")
    if min(out) < 1:
        raise ValueError("Ranks must be >= 1.")
    return out


def make_scanner_probe_classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )


def project_away(features: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project away the row span of components.

    components is expected to have shape (rank, dim) and orthonormal rows, as in
    sklearn PCA.components_.
    """
    if components.ndim != 2:
        raise ValueError(f"Expected 2D components, got {components.shape}")
    return features - (features @ components.T) @ components


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


def scanner_centroid_subspace(
    features: np.ndarray,
    scanner_labels: np.ndarray,
    max_rank: int,
) -> np.ndarray:
    """Return an orthonormal scanner-centroid contrast basis.

    This is not LEACE. It is only a small diagnostic proxy for the supervised
    scanner separation subspace.
    """
    scanner_labels = scanner_labels.astype(str)
    global_mean = features.mean(axis=0, keepdims=True)

    centered_centroids = []
    for scanner in sorted(np.unique(scanner_labels)):
        idx = scanner_labels == scanner
        centroid = features[idx].mean(axis=0, keepdims=True)
        centered_centroids.append((centroid - global_mean).ravel())

    mat = np.stack(centered_centroids, axis=0)
    mat -= mat.mean(axis=0, keepdims=True)

    _, s, vt = np.linalg.svd(mat, full_matrices=False)
    rank = min(max_rank, int((s > 1e-8).sum()))
    return vt[:rank].astype(np.float32)


def subspace_overlap(components_a: np.ndarray, components_b: np.ndarray) -> dict:
    """Principal-angle style overlap between two orthonormal row bases."""
    if len(components_a) == 0 or len(components_b) == 0:
        return {
            "scanner_overlap_mean_squared_cosine": None,
            "scanner_overlap_max_cosine": None,
            "scanner_overlap_cosines": [],
        }

    m = components_a @ components_b.T
    s = np.linalg.svd(m, compute_uv=False)

    return {
        "scanner_overlap_mean_squared_cosine": float(np.mean(s**2)),
        "scanner_overlap_max_cosine": float(np.max(s)),
        "scanner_overlap_cosines": s.tolist(),
    }


# =============================================================================
# Paired scanner delta construction
# =============================================================================


def _sample_rows(
    x: np.ndarray,
    max_rows: Optional[int],
    seed: int,
) -> np.ndarray:
    if max_rows is None or len(x) <= max_rows:
        return x

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


def build_group_mean_to_mean_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    row_indices: np.ndarray,
    scanner_col: str,
    group_col: str,
    sign_mode: str,
) -> np.ndarray:
    """Build deltas from scanner-specific group means to the group mean.

    For each biological group, compute one mean embedding per scanner.
    Then compute:

        delta_s = mean_embedding(group, scanner=s)
                  - mean_embedding(group, all scanners)

    This estimates scanner-specific deviations at the group/slide level.
    """
    df = metadata.iloc[row_indices].copy()
    df["_feature_index"] = row_indices

    deltas: list[np.ndarray] = []

    for _, group_df in df.groupby(group_col, sort=False):
        scanner_vectors = []

        for scanner, scanner_df in group_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append((scanner, features[idx].mean(axis=0)))

        if len(scanner_vectors) < 2:
            continue

        scanner_matrix = np.stack([v for _, v in scanner_vectors], axis=0)
        mean_vector = scanner_matrix.mean(axis=0)

        for _, scanner_vector in scanner_vectors:
            delta = scanner_vector - mean_vector
            deltas.append(delta.astype(np.float32))

            if sign_mode == "both":
                deltas.append((-delta).astype(np.float32))

    if not deltas:
        raise ValueError("No group-mean-to-mean scanner deltas were built.")

    return np.stack(deltas, axis=0).astype(np.float32)


def build_group_mean_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    row_indices: np.ndarray,
    scanner_col: str,
    group_col: str,
    sign_mode: str,
) -> np.ndarray:
    """Build deltas between scanner-specific group means.

    For each biological group, compute one mean embedding per scanner and add all
    pairwise scanner deltas.
    """
    df = metadata.iloc[row_indices].copy()
    df["_feature_index"] = row_indices

    deltas: list[np.ndarray] = []

    for _, group_df in df.groupby(group_col, sort=False):
        scanner_vectors = []

        for scanner, scanner_df in group_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append((scanner, features[idx].mean(axis=0)))

        if len(scanner_vectors) < 2:
            continue

        for (_, xa), (_, xb) in combinations(scanner_vectors, 2):
            delta = xb - xa
            deltas.append(delta.astype(np.float32))
            if sign_mode == "both":
                deltas.append((-delta).astype(np.float32))

    if not deltas:
        raise ValueError("No paired scanner group-mean deltas were built.")

    return np.stack(deltas, axis=0).astype(np.float32)


def build_pair_col_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    row_indices: np.ndarray,
    scanner_col: str,
    group_col: str,
    pair_col: str,
    sign_mode: str,
) -> np.ndarray:
    """Build deltas between scanner-specific embeddings for matched pair keys.

    This is useful if metadata contains a patch/location identifier shared across
    scanners. If duplicate rows exist for a scanner within a pair key, they are
    averaged before building deltas.
    """
    df = metadata.iloc[row_indices].copy()
    df["_feature_index"] = row_indices

    deltas: list[np.ndarray] = []

    for _, pair_df in df.groupby([group_col, pair_col], sort=False):
        scanner_vectors = []

        for scanner, scanner_df in pair_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append((scanner, features[idx].mean(axis=0)))

        if len(scanner_vectors) < 2:
            continue

        for (_, xa), (_, xb) in combinations(scanner_vectors, 2):
            delta = xb - xa
            deltas.append(delta.astype(np.float32))
            if sign_mode == "both":
                deltas.append((-delta).astype(np.float32))

    if not deltas:
        raise ValueError(
            "No paired scanner deltas were built with --delta-unit pair_col. "
            "Check that --pair-col identifies matched locations across scanners."
        )

    return np.stack(deltas, axis=0).astype(np.float32)


def build_pair_col_to_mean_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    row_indices: np.ndarray,
    scanner_col: str,
    group_col: str,
    pair_col: str,
    sign_mode: str,
) -> np.ndarray:
    """Build deltas between scanner-specific embeddings for matched pair keys.

    This is useful if metadata contains a patch/location identifier shared across
    scanners. If duplicate rows exist for a scanner within a pair key, they are
    averaged before building deltas.
    """
    df = metadata.iloc[row_indices].copy()
    df["_feature_index"] = row_indices

    deltas: list[np.ndarray] = []

    for _, pair_df in df.groupby([group_col, pair_col], sort=False):
        scanner_vectors = []

        for scanner, scanner_df in pair_df.groupby(scanner_col, sort=True):
            idx = scanner_df["_feature_index"].to_numpy(dtype=int)
            scanner_vectors.append((scanner, features[idx].mean(axis=0)))

        if len(scanner_vectors) < 2:
            continue

        mean_vector = np.stack([v for _, v in scanner_vectors], axis=0).mean(axis=0)

        for scanner_vector in scanner_vectors:
            delta = scanner_vector[1] - mean_vector
            deltas.append(delta.astype(np.float32))

    if not deltas:
        raise ValueError(
            "No paired scanner deltas were built with --delta-unit pair_col. "
            "Check that --pair-col identifies matched locations across scanners."
        )

    return np.stack(deltas, axis=0).astype(np.float32)


def build_paired_scanner_deltas(
    features: np.ndarray,
    metadata: pd.DataFrame,
    row_indices: np.ndarray,
    scanner_col: str,
    group_col: str,
    delta_mode: str,
    pair_col: str | None,
    sign_mode: str,
    max_deltas: Optional[int],
    seed: int,
) -> np.ndarray:
    deltas = None
    if delta_mode == "group_pairwise":
        deltas = build_group_mean_scanner_deltas(
            features=features,
            metadata=metadata,
            row_indices=row_indices,
            scanner_col=scanner_col,
            group_col=group_col,
            sign_mode=sign_mode,
        )

    if delta_mode == "group_to_mean":
        return build_group_mean_to_mean_scanner_deltas(
            features=features,
            metadata=metadata,
            row_indices=row_indices,
            scanner_col=scanner_col,
            group_col=group_col,
            sign_mode=sign_mode,
        )

    if pair_col is None:
        raise ValueError(f"--pair-col is required for delta_mode={delta_mode}")

    if delta_mode == "pair_col_pairwise":
        return build_pair_col_scanner_deltas(
            features=features,
            metadata=metadata,
            row_indices=row_indices,
            scanner_col=scanner_col,
            group_col=group_col,
            pair_col=pair_col,
            sign_mode=sign_mode,
        )

    if delta_mode == "pair_col_to_mean":
        return build_pair_col_to_mean_scanner_deltas(
            features=features,
            metadata=metadata,
            row_indices=row_indices,
            scanner_col=scanner_col,
            group_col=group_col,
            pair_col=pair_col,
            sign_mode=sign_mode,
        )

    if deltas is not None:
        deltas = _sample_rows(deltas, max_rows=max_deltas, seed=seed)
        return deltas.astype(np.float32)

    raise ValueError(f"Unknown delta_mode: {delta_mode}")


# =============================================================================
# Main analysis
# =============================================================================


def run_paired_scanner_delta_pca(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanner_col: str,
    group_col: str,
    output_dir: Path,
    ranks: list[int],
    n_splits: int,
    delta_unit: str,
    pair_col: Optional[str],
    sign_mode: str,
    max_deltas_per_fold: Optional[int],
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    projector_dir = output_dir / "fold_projectors"
    projector_dir.mkdir(parents=True, exist_ok=True)

    scanner_values = metadata[scanner_col].astype(str).to_numpy()
    groups = metadata[group_col].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(scanner_values)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two groups for GroupKFold.")

    n_splits = min(n_splits, len(unique_groups))
    max_rank = max(ranks)

    cv = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict] = []
    fold_diags: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, y_all, groups=groups)
    ):
        logger.info("Fold %d / %d", fold_idx + 1, n_splits)

        x_train_raw = features[train_idx]
        x_test_raw = features[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        train_classes = sorted(np.unique(scanner_values[train_idx]).tolist())
        test_classes = sorted(np.unique(scanner_values[test_idx]).tolist())

        raw_clf = make_scanner_probe_classifier()
        raw_clf.fit(x_train_raw, y_train)
        y_pred_raw = raw_clf.predict(x_test_raw)
        raw_score = balanced_accuracy_score(y_test, y_pred_raw)

        deltas = build_paired_scanner_deltas(
            features=features,
            metadata=metadata,
            row_indices=train_idx,
            scanner_col=scanner_col,
            group_col=group_col,
            delta_mode=delta_unit,
            pair_col=pair_col,
            sign_mode=sign_mode,
            max_deltas=max_deltas_per_fold,
            seed=seed + fold_idx,
        )

        n_components = min(features.shape[1], len(deltas))
        if n_components < max_rank:
            logger.warning(
                "Requested max rank %d, but only fitting %d components.",
                max_rank,
                n_components,
            )

        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        pca.fit(deltas)

        components_all = pca.components_.astype(np.float32)
        explained_all = pca.explained_variance_ratio_.astype(float)

        projector_path = projector_dir / f"paired_scanner_delta_pca_fold{fold_idx}.npz"
        np.savez_compressed(
            projector_path,
            components=components_all,
            explained_variance_ratio=explained_all,
            mean=pca.mean_.astype(np.float32),
            delta_unit=np.array(delta_unit),
            pair_col=np.array(pair_col if pair_col is not None else ""),
            sign_mode=np.array(sign_mode),
        )

        scanner_basis = scanner_centroid_subspace(
            features=x_train_raw,
            scanner_labels=scanner_values[train_idx],
            max_rank=min(len(train_classes) - 1, max_rank),
        )

        fold_diag = {
            "fold": fold_idx,
            "raw_score": float(raw_score),
            "n_delta_fit": int(len(deltas)),
            "delta_unit": delta_unit,
            "pair_col": pair_col,
            "sign_mode": sign_mode,
            "projector_path": str(projector_path),
            "train_classes": train_classes,
            "test_classes": test_classes,
            "explained_variance_ratio": explained_all.tolist(),
        }

        for rank in ranks:
            if rank > len(components_all):
                continue

            components = components_all[:rank]

            x_train_proj = project_away(x_train_raw, components)
            x_test_proj = project_away(x_test_raw, components)

            clf = make_scanner_probe_classifier()
            clf.fit(x_train_proj, y_train)
            y_pred = clf.predict(x_test_proj)
            score = balanced_accuracy_score(y_test, y_pred)

            change = feature_change_summary(x_test_raw, x_test_proj)
            overlap = subspace_overlap(components, scanner_basis)

            row = {
                "fold": fold_idx,
                "rank": rank,
                "raw_score": float(raw_score),
                "paired_scanner_delta_pca_score": float(score),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_delta_fit": int(len(deltas)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                "train_classes": train_classes,
                "test_classes": test_classes,
                "delta_unit": delta_unit,
                "pair_col": pair_col,
                "sign_mode": sign_mode,
                "projector_path": str(projector_path),
                "mean_relative_change_test": change["mean_relative_change"],
                "explained_variance_ratio_sum": float(explained_all[:rank].sum()),
                **overlap,
            }
            fold_rows.append(row)

        fold_diag["rank_results"] = [r for r in fold_rows if r["fold"] == fold_idx]
        fold_diags.append(fold_diag)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_scores.csv", index=False)

    summary_df = (
        fold_df.groupby("rank")
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_std=("raw_score", "std"),
            paired_scanner_delta_pca_score_mean=(
                "paired_scanner_delta_pca_score",
                "mean",
            ),
            paired_scanner_delta_pca_score_std=(
                "paired_scanner_delta_pca_score",
                "std",
            ),
            mean_relative_change_test_mean=("mean_relative_change_test", "mean"),
            explained_variance_ratio_sum_mean=(
                "explained_variance_ratio_sum",
                "mean",
            ),
            scanner_overlap_mean_squared_cosine_mean=(
                "scanner_overlap_mean_squared_cosine",
                "mean",
            ),
            scanner_overlap_max_cosine_mean=("scanner_overlap_max_cosine", "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(output_dir / "summary_by_rank.csv", index=False)

    diagnostics = {
        "scanner_col": scanner_col,
        "group_col": group_col,
        "n_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_splits": int(n_splits),
        "classes": label_encoder.classes_.tolist(),
        "chance_balanced_accuracy": float(1.0 / len(label_encoder.classes_)),
        "ranks": ranks,
        "delta_unit": delta_unit,
        "pair_col": pair_col,
        "sign_mode": sign_mode,
        "max_deltas_per_fold": max_deltas_per_fold,
        "protocol": (
            "GroupKFold by group_col. For each fold, paired scanner deltas are "
            "built only from training groups, PCA is fitted on these deltas, and "
            "the top-k PCA directions are projected away from both train and test "
            "embeddings before fitting/evaluating the scanner probe."
        ),
        "fold_diagnostics": fold_diags,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("Saved fold scores to %s", output_dir / "fold_scores.csv")
    logger.info("Saved summary to %s", output_dir / "summary_by_rank.csv")

    return diagnostics


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--embeddings-cache",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="NPZ file containing cached embeddings and metadata columns. Created if missing.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory where PCA results and run config will be written.",
)
@click.option(
    "--tile-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing tiles. Required if embeddings must be computed.",
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV file containing tile metadata. Required if embeddings must be computed.",
)
@click.option(
    "--encoder-id",
    type=str,
    default=None,
    help="Encoder/model identifier used to compute embeddings if cache is missing.",
)
@click.option(
    "--token-mode",
    type=click.Choice(["cls", "gap"], case_sensitive=False),
    default="cls",
    show_default=True,
    help="Token aggregation mode used when computing embeddings.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    show_default=True,
    help="Torch device used when computing embeddings.",
)
@click.option(
    "--embedding-batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Batch size used when computing embeddings.",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of dataloader workers used when computing embeddings.",
)
@click.option(
    "--use-amp/--no-use-amp",
    default=True,
    show_default=True,
    help="Use automatic mixed precision when computing embeddings.",
)
@click.option(
    "--force-embeddings",
    is_flag=True,
    help="Force recomputation of embeddings even if the cache exists.",
)
@click.option("--scanner-col", type=str, default="scanner_id", show_default=True)
@click.option("--group-col", type=str, default="image_id", show_default=True)
@click.option(
    "--delta-unit",
    type=click.Choice(
        [
            "group_pairwise",
            "group_to_mean",
            "pair_col_pairwise",
            "pair_col_to_mean",
        ]
    ),
    default="pair_col",
    show_default=True,
    help=(
        "How to build scanner deltas. 'group_pairwise' uses pairwise differences "
        "between scanner-specific means inside each group_col. 'group_to_mean' uses "
        "differences between scanner-specific means and the group mean. 'pair_col_pairwise' uses "
        "matched locations identified by --pair-col and computes pairwise differences. 'pair_col_to_mean' uses "
        "matched locations identified by --pair-col and computes differences to the mean."
    ),
)
@click.option(
    "--pair-col",
    type=str,
    default=None,
    help="Column identifying matched patches/locations across scanners. Required for --delta-unit pair_col.",
)
@click.option(
    "--sign-mode",
    type=click.Choice(["one", "both"]),
    default="one",
    show_default=True,
    help="Use one arbitrary pairwise sign, or both delta and -delta.",
)
@click.option(
    "--ranks",
    type=str,
    default="1,2,4,8,16,32,64",
    show_default=True,
    help="Comma-separated PCA ranks to evaluate.",
)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option(
    "--max-deltas-per-fold",
    type=int,
    default=None,
    show_default=True,
    help="Optional random subsampling of scanner deltas per fold, useful for pair_col mode.",
)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    embeddings_cache: Path,
    output_dir: Path,
    tile_dir: Optional[Path],
    metadata_csv: Optional[Path],
    encoder_id: Optional[str],
    token_mode: str,
    device: str,
    embedding_batch_size: int,
    num_workers: int,
    use_amp: bool,
    force_embeddings: bool,
    scanner_col: str,
    group_col: str,
    delta_unit: str,
    pair_col: Optional[str],
    sign_mode: str,
    ranks: str,
    n_splits: int,
    max_deltas_per_fold: Optional[int],
    seed: int,
) -> None:
    torch_device = torch.device(device)

    features, metadata = get_or_compute_embeddings(
        embeddings_cache=embeddings_cache,
        force_embeddings=force_embeddings,
        tile_dir=tile_dir,
        metadata_csv=metadata_csv,
        encoder_id=encoder_id,
        device=torch_device,
        token_mode=token_mode,
        embedding_batch_size=embedding_batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )

    if scanner_col not in metadata.columns:
        raise ValueError(f"Missing scanner column: {scanner_col}")

    if group_col not in metadata.columns:
        raise ValueError(f"Missing group column: {group_col}")

    if delta_unit == "pair_col":
        if pair_col is None:
            raise ValueError("--pair-col is required when --delta-unit=pair_col")
        if pair_col not in metadata.columns:
            raise ValueError(f"Missing pair column: {pair_col}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rank_values = parse_ranks(ranks)

    run_paired_scanner_delta_pca(
        features=features,
        metadata=metadata,
        scanner_col=scanner_col,
        group_col=group_col,
        output_dir=output_dir,
        ranks=rank_values,
        n_splits=n_splits,
        delta_unit=delta_unit,
        pair_col=pair_col,
        sign_mode=sign_mode,
        max_deltas_per_fold=max_deltas_per_fold,
        seed=seed,
    )

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "embeddings_cache": str(embeddings_cache),
                "output_dir": str(output_dir),
                "tile_dir": str(tile_dir) if tile_dir is not None else None,
                "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
                "encoder_id": encoder_id,
                "token_mode": token_mode,
                "device": device,
                "embedding_batch_size": embedding_batch_size,
                "num_workers": num_workers,
                "use_amp": use_amp,
                "force_embeddings": force_embeddings,
                "scanner_col": scanner_col,
                "group_col": group_col,
                "delta_unit": delta_unit,
                "pair_col": pair_col,
                "sign_mode": sign_mode,
                "ranks": rank_values,
                "n_splits": n_splits,
                "max_deltas_per_fold": max_deltas_per_fold,
                "seed": seed,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
